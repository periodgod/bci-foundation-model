"""Direct real-EEG architecture validation; no ordinary-model benchmarks."""
import argparse
import copy
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from eeg_mime_foundation.data import Batch, ManifestDataset, Normalizer, collate, sha256
from eeg_mime_foundation.engine import Runner, evaluate, load_checkpoint, set_seed, write_json
from eeg_mime_foundation.model import ModelConfig, SignalModel, count_parameters, patchify
from eeg_mime_foundation.objectives import make_mask, reconstruction_loss


def config_for(kind, seed=17, architecture="fusion", objective="supervised", tuning="full"):
    sleep = kind == "sleep"
    training = dict(objective=objective, tuning=tuning, epochs=5, batch_size=8,
                    lr=.0003, weight_decay=.01, seed=seed, device="cuda", threads=2,
                    clip_grad=1., max_seconds=1800, selection_metric="balanced_accuracy", direction="max")
    if objective in {"masked", "ema"}:
        training.update(epochs=3, selection_metric="loss", direction="min", mask_ratio=.5,
                        mask_kind="joint", band_weight=.01, spectral_weight=.1, ema_weight=.1)
    else:
        training["class_balance"] = "inverse_frequency"
    return dict(model=dict(architecture=architecture, sample_rate=100 if sleep else 160,
                           patch_samples=100 if sleep else 80, dim=256, depth=5, heads=4,
                           mlp_ratio=3, dropout=.1, task="classification", outputs=5 if sleep else 2),
                labels=["W", "N1", "N2", "N3", "REM"] if sleep else ["left_hand_imagery", "right_hand_imagery"],
                training=training)


def bounded_manifest(source, output, train_limit=96, validation_limit=160):
    """Label-independent per-person support; complete reserved test kept unchanged."""
    source = Path(source).resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    buckets = {}
    for record in manifest["samples"]:
        buckets.setdefault((record["split"], record["subject"]), []).append(record)
    selected = []
    rng = np.random.default_rng(20261011)
    for (split, subject), records in sorted(buckets.items()):
        limit = train_limit if split == "train" else validation_limit if split == "validation" else len(records)
        indices = sorted(rng.choice(len(records), min(limit, len(records)), replace=False).tolist())
        selected.extend(dict(records[i], path=str((source.parent / records[i]["path"]).resolve())) for i in indices)
    manifest.update(samples=selected, parent_manifest_sha256=sha256(source),
                    study_scope="label-independent bounded foundation feasibility; not official benchmark")
    write_json(output, manifest)
    return dict(parent=str(source), parent_sha256=sha256(source), selected_sha256=sha256(output),
                counts={split: sum(r["split"] == split for r in selected) for split in ("train", "validation", "test")},
                selection_seed=20261011, train_per_subject_cap=train_limit, validation_per_subject_cap=validation_limit)


def real_architecture_checks(kind, manifest, directory):
    """Assert physiology metadata, masking, permutation and branch gradients on real inputs."""
    cfg = config_for(kind)
    data = ManifestDataset(manifest, "train", cfg["model"]["sample_rate"], "classification")
    batch = Normalizer.fit(data).apply(collate([data[i] for i in range(4)]).to("cuda"))
    reports = []
    for architecture in ("fusion", "factorized", "compact", "latent", "hierarchical", "causal"):
        set_seed(17)
        model_cfg = ModelConfig(**dict(cfg["model"], architecture=architecture))
        model = SignalModel(model_cfg).cuda().eval()
        torch.cuda.reset_peak_memory_stats()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
        started = time.perf_counter()
        with torch.no_grad():
            output = model(batch)
            if not torch.isfinite(output["tokens"]).all():
                raise AssertionError("Nonfinite encoder output")
            permutation = torch.arange(batch.x.shape[1] - 1, -1, -1, device="cuda")
            permuted = Batch(batch.x[:, permutation], batch.valid[:, permutation], batch.channel_ids[:, permutation],
                             batch.coords[:, permutation], batch.coord_valid[:, permutation], batch.target, batch.metadata)
            torch.testing.assert_close(output["logits"], model(permuted)["logits"], rtol=2e-4, atol=2e-5)
            _, valid = patchify(batch, model_cfg.patch_samples)
            mask = make_mask(valid.any(-1), .5)
            changed = batch.x.clone()
            sample_mask = mask.repeat_interleave(model_cfg.patch_samples, -1)[..., :changed.shape[-1]]
            changed[sample_mask] += 123.
            corrupted = Batch(changed, batch.valid, batch.channel_ids, batch.coords,
                              batch.coord_valid, batch.target, batch.metadata)
            first = model(batch, mask)
            second = model(corrupted, mask)
            torch.testing.assert_close(first["tokens"], second["tokens"], rtol=0, atol=0)
        model.train()  # CUDA GRU backward requires a training-mode forward.
        optimizer.zero_grad(set_to_none=True)
        output = model(batch, mask)
        loss, parts = reconstruction_loss(output, mask, model_cfg, .01, .1)
        loss = loss + torch.nn.functional.cross_entropy(model(batch)["logits"], batch.target)
        loss.backward()
        norms = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise AssertionError(f"Nonfinite gradient: {name}")
                component = name.split(".")[1] if name.startswith("encoder.") else name.split(".")[0]
                norms[component] = norms.get(component, 0.) + float(parameter.grad.square().sum())
        expected = {"raw", "ids", "head", "reconstruction", "bandpower", "adapter"}
        expected.add("recurrent" if architecture == "causal" else "blocks")
        if model.encoder.use_spectrum:
            expected.update(("wave", "spectrum", "fusion"))
        if architecture == "latent":
            expected.update(("queries", "compress", "retrieve"))
        if architecture == "hierarchical":
            expected.add("epoch_gru")
        if any(norms.get(k, 0.) <= 0 for k in expected):
            raise AssertionError(f"Disconnected branch: {expected - {k for k,v in norms.items() if v > 0}}")
        optimizer.step()
        torch.cuda.synchronize()
        report = dict(architecture=architecture, parameters=count_parameters(model), batch=list(batch.x.shape),
                      loss=float(loss.detach()), components={k: v**.5 for k, v in norms.items()},
                      masked_target_isolation=True, channel_permutation_invariance=True,
                      cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      elapsed_seconds=time.perf_counter()-started)
        reports.append(report)
        write_json(directory / "architecture_checks.json", reports)
        print(f"{kind} architecture {architecture}: gradients/masks/permutation passed", flush=True)
        del model, optimizer, output, first, second, loss
        gc.collect()
        torch.cuda.empty_cache()
    return reports


def run_one(config, manifest, directory, initialize=None, normalization_source=None):
    runner = Runner(config, manifest)
    if normalization_source:
        if initialize:
            raise ValueError("Use the initializer's statistics when loading encoder weights")
        state = load_checkpoint(normalization_source)
        if state["signal_contract"] != runner.contract:
            raise ValueError("Normalization source contract mismatch")
        runner.import_provenance(state)
        runner.normalizer = Normalizer(**state["normalizer"])
        runner.initialization_record = dict(normalization_only=True,
            checkpoint_sha256=sha256(Path(normalization_source)),manifest_sha256=state["manifest_sha256"],
            training_groups=state["training_groups"],group_by=state["group_by"])
    if runner.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    frozen = {n: p.detach().cpu().clone() for n,p in runner.model.named_parameters() if not p.requires_grad}
    # Initialization changes encoder weights before tuning; capture after initialization.
    if initialize:
        runner.initialize(initialize)
        frozen = {n: p.detach().cpu().clone() for n,p in runner.model.named_parameters() if not p.requires_grad}
    teacher_before = [{n: p.detach().cpu().clone() for n,p in t.named_parameters()} for t in runner.teachers]
    result = runner.fit(directory)
    for name, param in runner.model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(param.detach().cpu(), frozen[name], rtol=0, atol=0)
    for teacher, before in zip(runner.teachers, teacher_before):
        for name, param in teacher.named_parameters():
            torch.testing.assert_close(param.detach().cpu(), before[name], rtol=0, atol=0)
            if param.grad is not None:
                raise AssertionError("Teacher acquired gradients")
    peak = torch.cuda.max_memory_allocated()/2**20
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    checkpoint = directory / "best.pt"
    metrics = evaluate(checkpoint, manifest, "validation", directory / "validation", "cuda")
    history = [json.loads(line) for line in (directory / "epochs.jsonl").read_text(encoding="utf-8").splitlines()]
    result.update(validation=metrics, cuda_peak_allocated_mib=peak, frozen_parameters_unchanged=True,
                  teachers_frozen=True, checkpoint_sha256=sha256(checkpoint),
                  first_train_loss=history[0]["train_loss"], last_train_loss=history[-1]["train_loss"])
    write_json(directory / "result.json", result)
    print(f"completed {directory.name}: subject BA={metrics['subject_macro']['balanced_accuracy']:.4f}, peak={peak:.1f} MiB", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mi", "sleep"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    selected = root / "manifest.json"
    intake = bounded_manifest(args.manifest, selected)
    launch = dict(hypothesis="Direct validation of the project foundation architecture and learning chain",
                  intake=intake, seeds=[17,29,43], main_config=config_for(args.dataset),
                  selection="validation subject-mean balanced accuracy; SSL checkpoint by fixed-corruption validation loss",
                  planned_variants=["fusion_scratch","factorized_scratch","masked_pretrain","masked_full","masked_linear","masked_adapter"],
                  engineering_variants_seed17=["ema_pretrain","ema_full","two_teacher_fixed","two_teacher_learned"],
                  inference="same native dataset contract; no cross-dataset harmonization claim",
                  budget="5 supervised / 3 SSL epochs, 1800 seconds per run, FP32 batch8; no leaderboard access",
                  ordinary_model_baselines=False, test_predictions=0, claim="exploratory feasibility; no superiority conclusion")
    write_json(root / "launch.json", launch)
    real_architecture_checks(args.dataset, selected, root)
    if args.checks_only:
        return
    results = {}
    write_json(root / "summary.json", results)
    def execute(name, config, initialize=None):
        results[name] = run_one(config, selected, root/name, initialize)
        write_json(root / "summary.json", results)
    for seed in launch["seeds"]:
        for arch in ("fusion", "factorized"):
            execute(f"{arch}_scratch_s{seed}", config_for(args.dataset, seed, arch))
        execute(f"masked_pretrain_s{seed}", config_for(args.dataset, seed, objective="masked"))
        source = root/f"masked_pretrain_s{seed}"/"best.pt"
        for tuning in ("full", "linear", "adapter"):
            execute(f"masked_{tuning}_s{seed}", config_for(args.dataset, seed, tuning=tuning), source)
    execute("ema_pretrain_s17", config_for(args.dataset, objective="ema"))
    execute("ema_full_s17", config_for(args.dataset), root/"ema_pretrain_s17"/"best.pt")
    for learned in (False, True):
        config = config_for(args.dataset, objective="distill")
        config["teachers"] = []
        for name in ("fusion_scratch_s17", "masked_full_s17"):
            checkpoint = root/name/"best.pt"
            config["teachers"].append(dict(checkpoint=str(checkpoint), sha256=sha256(checkpoint),
                use_status="approved", provenance="self-built local scratch lineage, no public weights",
                overlap="subject and waveform audit enforced by Runner", weight=1.))
        config["distillation"] = dict(learned=learned, entropy_weight=.01, supervised_weight=1.,
                                      logit_weight=.1, temperature=2.)
        execute(f"two_teacher_{'learned' if learned else 'fixed'}_s17", config)
    write_json(root/"completion.json", dict(completed_runs=len(results), test_predictions=0,
                remaining="official base/data/I-O, cross-dataset vocabulary/rate alignment, confirmatory population study"))


if __name__ == "__main__":
    main()
