"""FP32 epoch-boundary training/recovery; no claim of mid-epoch recovery."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import platform
import random
import subprocess
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import Batch, ManifestDataset, Normalizer, collate, group_key, sha256
from .metrics import summarize
from .model import ModelConfig, SignalModel, count_parameters, patchify
from .objectives import (EMATarget, MultiTeacherLoss, contrastive_loss, latent_loss,
                         logit_distillation, make_mask, reconstruction_loss)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def set_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def code_record():
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "src").rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, encoding="utf-8", errors="replace")
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    return dict(source_sha256=digest.hexdigest(), commit=git("rev-parse", "HEAD"),
                diff_sha256=hashlib.sha256(git("diff", "HEAD").encode("utf-8")).hexdigest())


def snapshot_source(directory):
    """Keep actual local source, including untracked modules, with each new run."""
    root = Path(__file__).resolve().parents[2]
    path = Path(directory) / "source.zip"
    files = sorted((root / "src").rglob("*.py")) + sorted((root / "scripts").glob("*.py"))
    files += [root / "pyproject.toml"]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in files:
            archive.write(source, str(source.relative_to(root)))
    return dict(path=path.name, sha256=sha256(path))


def load_checkpoint(path, device="cpu"):
    # Full recovery contains Python/numpy RNG objects. Load ONLY locally trusted files.
    return torch.load(path, map_location=device, weights_only=False)


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def signal_contract(manifest):
    fields = ("sample_rate", "unit", "coordinate_unit", "coordinate_frame", "channel_vocabulary")
    return {k: manifest[k] for k in fields}


def training_class_weights(dataset, outputs, device):
    """Fit weights on labeled training targets only, including valid sequence labels."""
    counts = torch.zeros(outputs, dtype=torch.long)
    for sample in dataset:
        if sample.target is None:
            raise ValueError("Class balancing requires training labels")
        labels = sample.target.reshape(-1)
        labels = labels[labels != -100]
        if ((labels < 0) | (labels >= outputs)).any():
            raise ValueError("Training label outside output vocabulary")
        counts += torch.bincount(labels, minlength=outputs)
    if (counts == 0).any():
        raise ValueError("Class balancing requires all classes in training")
    weights = counts.sum().float() / (outputs * counts.float())
    return weights.to(device), counts.tolist()


def collect_predictions(output, batch, task):
    if not task.startswith("sequence_"):
        return (output["logits"].detach().cpu().tolist(),
                batch.target.cpu().tolist() if batch.target is not None else [], batch.metadata)
    valid = output["sequence_valid"].clone()
    if batch.target is not None:
        valid &= (batch.target != -100) if task.endswith("classification") else torch.isfinite(batch.target).all(-1)
    prediction = output["sequence_logits"][valid].detach().cpu().tolist()
    target = batch.target[valid].cpu().tolist() if batch.target is not None else []
    metadata = [dict(batch.metadata[b], patch_index=p) for b, p in valid.nonzero().cpu().tolist()]
    return prediction, target, metadata


def validate_config(config):
    allowed = {"model", "training", "teachers", "distillation", "labels"}
    if set(config) - allowed:
        raise ValueError(f"Unknown config fields: {set(config) - allowed}")
    cfg = ModelConfig(**config["model"])
    t = config["training"]
    valid_keys = {"objective", "epochs", "batch_size", "lr", "weight_decay", "seed", "device", "threads",
                  "tuning", "mask_ratio", "mask_kind", "band_weight", "spectral_weight", "ema_weight", "ema_decay",
                  "clip_grad", "max_seconds", "selection_metric", "direction", "class_balance", "batch_by_dataset",
                  "channel_dropout"}
    if set(t) - valid_keys:
        raise ValueError(f"Unknown training fields: {set(t) - valid_keys}")
    if not 0. <= t.get("channel_dropout", 0.) < 1.:
        raise ValueError("channel_dropout must be in [0, 1)")
    if t["objective"] not in {"supervised", "masked", "ema", "contrastive", "distill"}:
        raise ValueError("Unknown training objective")
    if t.get("class_balance", "none") not in {"none", "inverse_frequency"}:
        raise ValueError("Unknown class balance strategy")
    if not isinstance(t.get("batch_by_dataset", False), bool):
        raise ValueError("batch_by_dataset must be boolean")
    if t.get("class_balance", "none") != "none" and (not cfg.task.endswith("classification") or t["objective"] not in {"supervised", "distill"}):
        raise ValueError("Class balancing is for supervised classification")
    if t.get("tuning", "full") != "full" and t["objective"] != "supervised":
        raise ValueError("Linear/adapter tuning is for downstream supervision")
    if min(t["epochs"], t["batch_size"], t["lr"]) <= 0:
        raise ValueError("Positive epochs/batch_size/lr required")
    if t["objective"] == "contrastive" and t["batch_size"] < 2:
        raise ValueError("Contrastive batches require at least two samples")
    if cfg.task.endswith("classification"):
        labels = config.get("labels", [])
        if len(labels) != cfg.outputs or len(set(labels)) != len(labels):
            raise ValueError("Ordered unique classification labels required")
    return cfg


def drop_channels(batch, rate, generator=None):
    """训练期随机屏蔽整通道（导联丢弃增广）。

    直接针对 T1 的实测结论做增广：同一权重在 8 导 / 3 导上掉 13–14 个百分点。
    只改 `valid` 与对应信号，不改通道 ID 与坐标——模型看到的仍是"这顶帽子上少了几个电极"，
    而不是"换了一套编码"。每个样本至少保留一个通道。
    """
    if rate <= 0:
        return batch
    shape = batch.valid.shape
    # 按"通道"整列丢弃，而不是按时间点随机涂抹：模拟的是电极没贴/没戴。
    drawn = torch.rand((shape[0], shape[1], 1), generator=generator)
    keep = (drawn >= rate) & batch.valid
    empty = ~keep.any(-1).any(-1)
    if empty.any():
        first = batch.valid.any(-1).float().argmax(-1)
        rows = torch.arange(shape[0])[empty]
        keep[rows, first[empty]] = True
    valid = batch.valid & keep
    return Batch(batch.x.masked_fill(~valid, 0), valid, batch.channel_ids, batch.coords,
                 batch.coord_valid, batch.target, batch.metadata)


class Runner:
    def __init__(self, config, manifest, device=None):
        self.config, self.cfg = config, validate_config(config)
        self.training = config["training"]
        self.device = torch.device(device or self.training.get("device", "cpu"))
        torch.set_num_threads(self.training.get("threads", 2))
        set_seed(self.training.get("seed", 0))
        self.data = {s: ManifestDataset(manifest, s, self.cfg.sample_rate, self.cfg.task)
                     for s in ("train", "validation")}
        self.manifest_path = Path(manifest).resolve()
        self.manifest_hash = sha256(self.manifest_path)
        self.contract = signal_contract(self.data["train"].manifest)
        if self.cfg.task.endswith("classification") and self.config["labels"] != self.data["train"].manifest.get("labels"):
            raise ValueError("Manifest and model ordered labels differ")
        if self.cfg.task.endswith("regression") and (len(self.data["train"].manifest.get("target_schema", {}).get("names", [])) != self.cfg.outputs or not self.data["train"].manifest.get("target_schema", {}).get("unit")):
            raise ValueError("Regression target names/unit must be documented in target_schema")
        self.target_contract = (self.config["labels"] if self.cfg.task.endswith("classification")
                                else self.data["train"].manifest["target_schema"])
        if any(int(k) < 0 or int(k) >= self.cfg.channel_vocab for k in self.contract["channel_vocabulary"]):
            raise ValueError("Manifest channel vocabulary exceeds model vocabulary")
        self.normalizer = Normalizer.fit(self.data["train"])
        self.class_weights, self.class_counts = None, None
        if self.training.get("class_balance") == "inverse_frequency":
            self.class_weights, self.class_counts = training_class_weights(self.data["train"], self.cfg.outputs, self.device)
        self.group_by = self.data["train"].manifest["group_by"]
        self.exposure_groups = {group_key(r, self.group_by) for r in self.data["train"].records}
        self.exposure_hashes = set(self.data["train"].train_signal_hashes)
        self.selection_groups = {group_key(r, self.group_by) for r in self.data["validation"].records}
        self.model = SignalModel(self.cfg).to(self.device)
        self.model.configure_tuning(self.training.get("tuning", "full"))
        self.ema = EMATarget(self.model.encoder, self.training.get("ema_decay", .99)).to(self.device) \
            if self.training["objective"] == "ema" else None
        self.teachers, self.teacher_normalizers, self.teacher_states = [], [], []
        self.kd = None
        if self.training["objective"] == "distill":
            self._load_teachers()
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.kd:
            params += list(self.kd.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.training["lr"],
                                           weight_decay=self.training.get("weight_decay", .01))
        self.sampler = torch.Generator().manual_seed(self.training.get("seed", 0))
        self.start_epoch, self.best, self.initialization_record = 0, None, None

    def import_provenance(self, state):
        """Reject transitive pretraining/teacher contamination before using any weights."""
        if state["group_by"] != self.group_by:
            raise ValueError("Source grouping differs; provenance cannot be compared")
        groups = {tuple(g) for g in state["training_groups"]}
        heldout = [r for r in self.data["train"].manifest["samples"] if r["split"] != "train"]
        if any(group_key(r, self.group_by) in groups for r in heldout):
            raise ValueError("Source training groups overlap held-out groups")
        hashes = set(state["training_signal_hashes"])
        heldout_hashes = {h for h, split in self.data["train"].signal_hashes.items() if split != "train"}
        if hashes & heldout_hashes:
            raise ValueError("Source training waveforms overlap held-out signals")
        selected = {tuple(g) for g in state["selection_groups"]}
        if any(group_key(r, self.group_by) in selected for r in heldout if r["split"] == "test"):
            raise ValueError("Source model selection groups overlap reserved test")
        self.exposure_groups.update(groups)
        self.exposure_hashes.update(hashes)
        self.selection_groups.update(selected)

    def _load_teachers(self):
        specifications = self.config.get("teachers", [])
        if not specifications:
            raise ValueError("Distillation needs teacher checkpoint records")
        dims, weights = [], []
        for spec in specifications:
            if spec.get("use_status") != "approved" or not spec.get("provenance") or "overlap" not in spec:
                raise ValueError("Teacher use approval/provenance/pretraining-overlap record required")
            path = Path(spec["checkpoint"])
            if sha256(path) != spec["sha256"]:
                raise ValueError("Teacher checkpoint hash changed")
            state = load_checkpoint(path, self.device)
            if state["signal_contract"] != self.contract:
                raise ValueError("Teacher input contract differs; implement a verified native-view adapter first")
            self.import_provenance(state)
            cfg = ModelConfig(**state["config"]["model"])
            teacher = SignalModel(cfg).to(self.device).eval().requires_grad_(False)
            teacher.load_state_dict(state["model"])
            self.teachers.append(teacher)
            self.teacher_normalizers.append(Normalizer(**state["normalizer"]))
            self.teacher_states.append(state)
            dims.append(cfg.dim)
            weights.append(spec.get("weight", 1.))
        d = self.config.get("distillation", {})
        self.kd = MultiTeacherLoss(self.cfg.dim, dims, weights, d.get("learned", False)).to(self.device)

    def initialize(self, checkpoint):
        state = load_checkpoint(checkpoint, self.device)
        if self.contract != state["signal_contract"]:
            raise ValueError("Pretrained signal contract mismatch")
        source = asdict(ModelConfig(**state["config"]["model"]))
        target = asdict(self.cfg)
        for k in ("task", "outputs"):
            source.pop(k, None)
            target.pop(k, None)
        # JSON tuples and lists have the same documented meaning.
        if json.dumps(source, sort_keys=True) != json.dumps(target, sort_keys=True):
            raise ValueError("Encoder config mismatch")
        self.import_provenance(state)
        encoder = {k[len("encoder."):]: v for k, v in state["model"].items() if k.startswith("encoder.")}
        self.model.encoder.load_state_dict(encoder)
        self.initialization_record = dict(checkpoint_sha256=sha256(Path(checkpoint)),
                                          manifest_sha256=state["manifest_sha256"],
                                          training_groups=state["training_groups"], group_by=state["group_by"])
        # Preserve pretraining statistics, rather than silently altering the pretrained input.
        self.normalizer = Normalizer(**state["normalizer"])
        if self.ema:
            self.ema = EMATarget(self.model.encoder, self.training.get("ema_decay", .99)).to(self.device)

    def loader(self, split, train=False):
        if self.training.get("batch_by_dataset", False):
            groups = {}
            for index, record in enumerate(self.data[split].records):
                groups.setdefault(record["dataset"], []).append(index)
            batches = []
            size = self.training["batch_size"]
            generator = self.sampler if train else torch.Generator().manual_seed(0)
            for _, indices in sorted(groups.items()):
                if train:
                    indices = [indices[i] for i in torch.randperm(len(indices), generator=generator).tolist()]
                for start in range(0, len(indices), size):
                    batch = indices[start:start + size]
                    if train and self.training["objective"] == "contrastive" and len(batch) < size:
                        continue
                    batches.append(batch)
            if train:
                batches = [batches[i] for i in torch.randperm(len(batches), generator=generator).tolist()]
            return DataLoader(self.data[split], batch_sampler=batches, num_workers=0, collate_fn=collate)
        return DataLoader(self.data[split], batch_size=self.training["batch_size"], shuffle=train,
                          generator=self.sampler if train else torch.Generator().manual_seed(0),
                          num_workers=0, collate_fn=collate,
                          drop_last=train and self.training["objective"] == "contrastive")

    def supervised_loss(self, output, target):
        if target is None:
            raise ValueError("Supervision requires targets")
        if self.cfg.task.startswith("sequence_"):
            logits, valid = output["sequence_logits"], output["sequence_valid"]
            if self.cfg.task.endswith("classification"):
                if target.shape != valid.shape:
                    raise ValueError("Sequence labels must align with patch end times [B,P]")
                valid = valid & (target != -100)
                if not valid.any():
                    raise ValueError("No labeled sequence targets")
                return F.cross_entropy(logits[valid], target[valid], weight=self.class_weights)
            if target.shape != logits.shape:
                raise ValueError("Sequence regression labels must align with patches [B,P,O]")
            valid = valid & torch.isfinite(target).all(-1)
            if not valid.any():
                raise ValueError("No finite sequence targets")
            return F.mse_loss(logits[valid], target[valid])
        if self.cfg.task == "classification":
            if target.ndim != 1:
                raise ValueError("One integer class per sample required")
            return F.cross_entropy(output["logits"], target, weight=self.class_weights)
        if output["logits"].shape != target.shape:
            raise ValueError("Regression target must be [B,outputs]")
        return F.mse_loss(output["logits"], target)

    def drop_channels(self, batch):
        """按 `training.channel_dropout` 施加导联丢弃；验证集不增广。"""
        return drop_channels(batch, self.training.get("channel_dropout", 0.))

    def loss(self, raw_batch):
        t, objective = self.training, self.training["objective"]
        batch = self.normalizer.apply(raw_batch)
        if objective == "supervised":
            return self.supervised_loss(self.model(batch), batch.target), {}
        if objective == "distill":
            output = self.model(batch)
            with torch.no_grad():
                teachers = [teacher(normalizer.apply(raw_batch))
                            for teacher, normalizer in zip(self.teachers, self.teacher_normalizers)]
            d = self.config.get("distillation", {})
            kd, weights = self.kd(output["features"], [o["features"] for o in teachers],
                                  entropy_weight=d.get("entropy_weight", .01))
            total = kd
            if d.get("supervised_weight", 0):
                total = total + d["supervised_weight"] * self.supervised_loss(output, batch.target)
            if d.get("logit_weight", 0):
                if self.cfg.task != "classification":
                    raise ValueError("Logit KL supports shared classification semantics only")
                kl = 0
                for i, (teacher, state) in enumerate(zip(teachers, self.teacher_states)):
                    # Aggregate per teacher under verified common labels; no feature averaging.
                    if state["config"]["model"]["task"] != "classification":
                        raise ValueError("Teacher output task mismatch")
                    per_sample = logit_distillation(output["logits"], teacher["logits"],
                                                self.config["labels"], state["config"].get("labels"),
                                                d.get("temperature", 2.), reduction="none")
                    kl = kl + (per_sample * weights[:, i]).mean()
                total = total + d["logit_weight"] * kl
            return total, {"teacher_weights": weights.mean(0).tolist()}
        if objective == "contrastive":
            # Additive noise and channel-preserving gain; do not scramble physiological channel identities.
            def augment():
                from .data import Batch
                noise = torch.randn_like(batch.x) * .05
                gain = torch.empty(batch.x.shape[0], 1, 1, device=batch.x.device).uniform_(.9, 1.1)
                x = (batch.x * gain + noise).masked_fill(~batch.valid, 0)
                return Batch(x, batch.valid, batch.channel_ids, batch.coords, batch.coord_valid,
                             batch.target, batch.metadata)
            loss = contrastive_loss(self.model(augment())["features"], self.model(augment())["features"])
            return loss, {}
        _, valid = patchify(batch, self.cfg.patch_samples)
        mask = make_mask(valid.any(-1), t.get("mask_ratio", .5), t.get("mask_kind", "joint"))
        output = self.model(batch, mask)
        loss, parts = reconstruction_loss(output, mask, self.cfg, t.get("band_weight", 0), t.get("spectral_weight", 0))
        if objective == "ema":
            loss = loss + t.get("ema_weight", 1.) * latent_loss(output["tokens"], self.ema(batch), mask)
        parts = {k: float(v) for k, v in parts.items()}
        parts["actual_mask_ratio"] = float(mask.sum() / valid.any(-1).sum())
        return loss, parts

    @torch.no_grad()
    def validation(self, predictions=False):
        self.model.eval()
        if self.kd:
            self.kd.eval()
        # Fixed validation corruption; validation must not consume the training RNG stream.
        state = rng_state()
        set_seed(314159)
        losses, outputs, targets, metadata = [], [], [], []
        try:
            for batch in self.loader("validation"):
                raw = batch.to(self.device)
                loss, _ = self.loss(raw)
                losses.append((float(loss), len(batch.metadata)))
                if predictions or self.training["objective"] in {"supervised", "distill"}:
                    o = self.model(self.normalizer.apply(raw))
                    pp, yy, mm = collect_predictions(o, raw, self.cfg.task)
                    outputs.extend(pp)
                    targets.extend(yy)
                    metadata.extend(mm)
        finally:
            restore_rng(state)
        result = {"loss": sum(v * n for v, n in losses) / sum(n for _, n in losses)}
        if targets:
            result.update(summarize(outputs, targets, metadata, self.cfg.task))
        return result

    def checkpoint(self, epoch):
        return dict(epoch=epoch, config=self.config, model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(), ema=self.ema.state_dict() if self.ema else None,
                    kd=self.kd.state_dict() if self.kd else None, sampler=self.sampler.get_state(),
                    rng=rng_state(), best=self.best, normalizer=self.normalizer.__dict__,
                    manifest_sha256=self.manifest_hash, signal_contract=self.contract, target_contract=self.target_contract,
                    initialization=self.initialization_record, code=code_record(),
                    group_by=self.group_by,
                    training_groups=[list(key) for key in sorted(self.exposure_groups)],
                    training_signal_hashes=sorted(self.exposure_hashes),
                    selection_groups=[list(key) for key in sorted(self.selection_groups)],
                    class_counts=self.class_counts)

    def resume(self, checkpoint):
        state = load_checkpoint(checkpoint, self.device)
        previous = json.loads(json.dumps(state["config"]))
        current = json.loads(json.dumps(self.config))
        previous["training"].pop("epochs")
        current["training"].pop("epochs")
        if previous != current or state["manifest_sha256"] != self.manifest_hash:
            raise ValueError("Resume changes config/data; use initialization for a new experiment")
        if state["code"]["source_sha256"] != code_record()["source_sha256"]:
            raise ValueError("Source changed; exact recovery cannot be claimed")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.ema:
            self.ema.load_state_dict(state["ema"])
        if self.kd:
            self.kd.load_state_dict(state["kd"])
        self.normalizer = Normalizer(**state["normalizer"])
        self.initialization_record = state["initialization"]
        self.exposure_groups = {tuple(g) for g in state["training_groups"]}
        self.exposure_hashes = set(state["training_signal_hashes"])
        self.selection_groups = {tuple(g) for g in state["selection_groups"]}
        self.sampler.set_state(state["sampler"].cpu())
        restore_rng(state["rng"])
        self.start_epoch, self.best = state["epoch"], state["best"]
        if self.start_epoch >= self.training["epochs"]:
            raise ValueError("Resume epochs must exceed completed epochs")

    def fit(self, directory, resume=None, initialize=None):
        if resume and initialize:
            raise ValueError("Resume and initialize are mutually exclusive")
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()) and not resume:
            raise ValueError("Use an empty output directory for a new run")
        directory.mkdir(parents=True, exist_ok=True)
        if resume:
            if Path(resume).resolve() != (directory / "last.pt").resolve():
                raise ValueError("Resume the same run's last.pt; prevents stale history")
            self.resume(resume)
            history = [json.loads(line) for line in (directory / "epochs.jsonl").read_text(encoding="utf-8").splitlines()]
            if len(history) != self.start_epoch:
                raise ValueError("Log/checkpoint epoch mismatch; repair provenance before resuming")
        if initialize:
            self.initialize(initialize)
        snapshot = dict(path="source.zip", sha256=sha256(directory / "source.zip")) if resume else snapshot_source(directory)
        write_json(directory / "config.json", self.config)
        environment = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                           device=str(self.device), hardware=torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else platform.processor(),
                           params=count_parameters(self.model), trainable=count_parameters(self.model, True), code=code_record(),
                           evidence="synthetic_engineering_only" if self.data["train"].manifest.get("evidence") == "synthetic_engineering_only" else "uninterpreted_run",
                           manifest_sha256=self.manifest_hash, initialization=self.initialization_record,
                           class_counts=self.class_counts,
                           class_weights=self.class_weights.tolist() if self.class_weights is not None else None,
                           source_snapshot=snapshot,
                           precision="fp32", scheduler=None, scaler=None, recovery="epoch_boundary_only")
        if self.kd:
            environment["distillation_trainable_parameters"] = count_parameters(self.kd, True)
        write_json(directory / "environment.json", environment)
        direction = self.training.get("direction", "min")
        if direction not in {"min", "max"}:
            raise ValueError("Selection direction must be min/max")
        start = time.monotonic()
        for epoch in range(self.start_epoch, self.training["epochs"]):
            self.model.train()
            if self.training.get("tuning") in {"linear", "adapter"}:
                self.model.encoder.eval()  # frozen dropout is not an adaptation source
                if self.training["tuning"] == "adapter":
                    self.model.encoder.adapter.train()
            if self.kd:
                self.kd.train()
            values, details = [], []
            for batch in self.loader("train", True):
                if time.monotonic() - start > self.training.get("max_seconds", float("inf")):
                    raise TimeoutError("Budget exceeded; resume the last complete epoch")
                self.optimizer.zero_grad(set_to_none=True)
                loss, detail = self.loss(self.drop_channels(batch.to(self.device)))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss; affected stage stopped")
                loss.backward()
                params = [p for group in self.optimizer.param_groups for p in group["params"]]
                torch.nn.utils.clip_grad_norm_(params, self.training.get("clip_grad", 1.), error_if_nonfinite=True)
                self.optimizer.step()
                if self.ema:
                    self.ema.update(self.model.encoder)
                values.append((float(loss.detach()), len(batch.metadata)))
                details.append(detail)
            if not values:
                raise ValueError("No training batches; reduce batch size")
            validation = self.validation()
            metric = self.training.get("selection_metric", "loss")
            score = validation["loss"] if metric == "loss" else validation["subject_macro"][metric]
            improved = self.best is None or (score < self.best if direction == "min" else score > self.best)
            if improved:
                self.best = score
            record = dict(epoch=epoch + 1, train_loss=sum(v * n for v, n in values) / sum(n for _, n in values),
                          validation=validation, details=details, elapsed_seconds=time.monotonic() - start)
            with (directory / "epochs.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            state = self.checkpoint(epoch + 1)
            save_checkpoint(directory / "last.pt", state)
            if improved:
                save_checkpoint(directory / "best.pt", state)
        return dict(directory=str(directory.resolve()), epochs=self.training["epochs"], best_validation=self.best,
                    params=count_parameters(self.model), trainable=count_parameters(self.model, True))


@torch.no_grad()
def evaluate(checkpoint, manifest, split, output, device="cpu"):
    """No training/model selection; every test access is appended to the run ledger."""
    state = load_checkpoint(checkpoint, device)
    cfg = ModelConfig(**state["config"]["model"])
    data = ManifestDataset(manifest, split, cfg.sample_rate, cfg.task)
    if data.manifest["group_by"] != state["group_by"]:
        raise ValueError("Evaluation grouping differs from checkpoint protocol")
    if signal_contract(data.manifest) != state["signal_contract"]:
        raise ValueError("Evaluation signal contract mismatch")
    if cfg.task.endswith("classification") and state["config"]["labels"] != data.manifest.get("labels"):
        raise ValueError("Evaluation label semantics mismatch")
    target_contract = data.manifest.get("labels") if cfg.task.endswith("classification") else data.manifest.get("target_schema")
    if target_contract != state["target_contract"]:
        raise ValueError("Evaluation target names/units/semantics mismatch")
    training_groups = {tuple(key) for key in state["training_groups"]}
    if any(group_key(r, state["group_by"]) in training_groups for r in data.records):
        raise ValueError("Evaluation groups overlap checkpoint training groups")
    if split == "test" and any(group_key(r, state["group_by"]) in {tuple(g) for g in state["selection_groups"]} for r in data.records):
        raise ValueError("Evaluation test overlaps checkpoint model selection groups")
    split_hashes = {key for key, value in data.signal_hashes.items() if value == split}
    if split_hashes & set(state["training_signal_hashes"]):
        raise ValueError("Evaluation waveforms overlap checkpoint training signals")
    model = SignalModel(cfg).to(device).eval()
    model.load_state_dict(state["model"])
    normalizer = Normalizer(**state["normalizer"])
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    logits, targets, metadata = [], [], []
    started = time.monotonic()
    if split == "test":
        with (Path(checkpoint).parent / "test_access.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(timestamp=time.time(), checkpoint_sha256=sha256(Path(checkpoint)),
                                    manifest_sha256=sha256(Path(manifest)), output=str(output.resolve()))) + "\n")
    for batch in DataLoader(data, batch_size=8, collate_fn=collate, num_workers=0):
        raw = batch.to(device)
        prediction = model(normalizer.apply(raw))
        pp, yy, mm = collect_predictions(prediction, raw, cfg.task)
        logits.extend(pp)
        targets.extend(yy)
        metadata.extend(mm)
    report = dict(split=split, checkpoint_sha256=sha256(Path(checkpoint)), manifest_sha256=sha256(Path(manifest)),
                  predictions=len(metadata), samples=len(data), prediction_unit="patch" if cfg.task.startswith("sequence_") else "window",
                  elapsed_seconds=time.monotonic() - started, code=code_record(),
                  evidence=data.manifest.get("evidence", "uninterpreted_evaluation"))
    if targets:
        report.update(summarize(logits, targets, metadata, cfg.task))
    write_json(output / "metrics.json", report)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for i, (prediction, meta) in enumerate(zip(logits, metadata)):
            f.write(json.dumps(dict(**meta, prediction=prediction, target=targets[i] if targets else None)) + "\n")
    return report
