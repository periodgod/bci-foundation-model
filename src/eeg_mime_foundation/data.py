"""Strict sample contract, grouped splits and train-only normalization.

Each NPZ is one independent window. Split parent recordings BEFORE windowing.
Filtering, epoching and physical unit conversion belong to dataset ingestion.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class Sample:
    x: Tensor                     # [channels, samples], finite, in documented units
    channel_ids: Tensor           # global electrode vocabulary IDs, -1 = unknown
    coords: Tensor                # [channels, 3], meters in a documented frame
    coord_valid: Tensor           # [channels]; coordinates optional if IDs known
    target: Tensor | None
    sample_id: str
    subject: str
    session: str
    trial: str
    dataset: str


@dataclass
class Batch:
    x: Tensor
    valid: Tensor                 # [B,C,T], padding validity, NOT SSL masking
    channel_ids: Tensor
    coords: Tensor
    coord_valid: Tensor
    target: Tensor | None
    metadata: list[dict]

    def to(self, device):
        return Batch(**{k: v.to(device) if isinstance(v, Tensor) else v
                        for k, v in self.__dict__.items()})


def collate(samples: list[Sample]) -> Batch:
    if not samples:
        raise ValueError("Empty batch")
    b, c, t = len(samples), max(s.x.shape[0] for s in samples), max(s.x.shape[1] for s in samples)
    x, valid = torch.zeros(b, c, t), torch.zeros(b, c, t, dtype=torch.bool)
    ids = torch.full((b, c), -1, dtype=torch.long)
    coords, cv = torch.zeros(b, c, 3), torch.zeros(b, c, dtype=torch.bool)
    metadata = []
    for i, s in enumerate(samples):
        nc, nt = s.x.shape
        if nc == 0 or nt == 0 or not torch.isfinite(s.x).all():
            raise ValueError(f"Invalid signal: {s.sample_id}")
        if s.channel_ids.shape != (nc,) or s.coords.shape != (nc, 3) or s.coord_valid.shape != (nc,):
            raise ValueError("Invalid channel metadata shape")
        if (s.channel_ids < -1).any() or not torch.isfinite(s.coords[s.coord_valid]).all():
            raise ValueError("Invalid channel ID/coordinate")
        if ((s.channel_ids < 0) & ~s.coord_valid).any():
            raise ValueError("Each real channel needs a genuine vocabulary ID or coordinate")
        x[i, :nc, :nt], valid[i, :nc, :nt] = s.x, True
        ids[i, :nc], cv[i, :nc] = s.channel_ids, s.coord_valid
        coords[i, :nc] = torch.where(s.coord_valid[:, None], s.coords, 0)
        metadata.append({k: getattr(s, k) for k in ("sample_id", "subject", "session", "trial", "dataset")})
    targets = [s.target for s in samples]
    if any(y is None for y in targets) and not all(y is None for y in targets):
        raise ValueError("Mixed labeled/unlabeled samples in a batch")
    target = None
    if targets[0] is not None:
        if targets[0].ndim == 1 and not targets[0].is_floating_point():
            target = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
        elif targets[0].ndim == 2 and targets[0].is_floating_point():
            target = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=float("nan"))
        else:
            target = torch.stack(targets)
    return Batch(x, valid, ids, coords, cv, target, metadata)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ManifestDataset(Dataset):
    """JSON manifest + numeric NPZ; never enable pickle loading."""
    def __init__(self, path, split, sample_rate, task, require_permission=True):
        self.path = Path(path).resolve()
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        m = self.manifest
        if m["sample_rate"] != sample_rate or m["unit"] != "volt":
            raise ValueError("Ingest/resample explicitly: sample_rate must match and unit must be volt")
        if m.get("coordinate_unit") != "meter" or not m.get("coordinate_frame") or not m.get("channel_vocabulary"):
            raise ValueError("Coordinate frame/unit and channel vocabulary must be documented")
        if require_permission and m.get("use_status") != "approved":
            raise ValueError("Dataset use_status must be approved in the intake record")
        records = m["samples"]
        audit_splits(records, m["group_by"])
        self.records = [r for r in records if r["split"] == split]
        if not self.records:
            raise ValueError(f"No samples for split {split}")
        self.task = task
        self.content_hashes = {}
        self.signal_hashes = {}
        self.train_signal_hashes = set()
        for r in records:
            p = (self.path.parent / r["path"]).resolve()
            actual = sha256(p)
            if actual != r["sha256"]:
                raise ValueError(f"Data hash changed: {r['sample_id']}")
            if actual in self.content_hashes and self.content_hashes[actual] != r["split"]:
                raise ValueError("Duplicate NPZ content crosses partitions")
            self.content_hashes[actual] = r["split"]
            with np.load(p, allow_pickle=False) as arrays:
                signal = arrays["x"]
                signal_hash = hashlib.sha256(str((signal.shape, str(signal.dtype))).encode() + signal.tobytes()).hexdigest()
            if signal_hash in self.signal_hashes and self.signal_hashes[signal_hash] != r["split"]:
                raise ValueError("Identical waveform crosses partitions")
            self.signal_hashes[signal_hash] = r["split"]
            if r["split"] == "train":
                self.train_signal_hashes.add(signal_hash)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        with np.load(self.path.parent / r["path"], allow_pickle=False) as a:
            x = torch.tensor(a["x"], dtype=torch.float32)
            ids = torch.tensor(a["channel_ids"], dtype=torch.long)
            coords = torch.tensor(a["coords"], dtype=torch.float32)
            cv = torch.tensor(a["coord_valid"], dtype=torch.bool)
            target = None
            if "target" in a:
                if self.task.endswith("classification") and not np.issubdtype(a["target"].dtype, np.integer):
                    raise ValueError("Classification targets must be stored as integers")
                target = torch.tensor(a["target"], dtype=torch.long if self.task.endswith("classification") else torch.float32)
            channel_ids = ids.tolist()
            if any(i >= 0 and str(i) not in self.manifest["channel_vocabulary"] for i in channel_ids):
                raise ValueError("Channel ID is absent from documented vocabulary")
        return Sample(x, ids, coords, cv, target, **{k: r[k] for k in
                      ("sample_id", "subject", "session", "trial", "dataset")})


def group_key(record, group_by):
    if group_by == "subject":
        fields = ("dataset", "subject")
    elif group_by == "session":
        fields = ("dataset", "subject", "session")
    elif group_by == "trial":
        fields = ("dataset", "subject", "session", "trial")
    else:
        raise ValueError("group_by must be subject, session or trial")
    return tuple(record[f] for f in fields)


def audit_splits(records, group_by):
    groups, ids, trials = {}, set(), {}
    for r in records:
        if r["split"] not in {"train", "validation", "test"} or r["sample_id"] in ids:
            raise ValueError("Invalid split or duplicate sample_id")
        ids.add(r["sample_id"])
        # Trial parents must stay together even when protocol groups at another level.
        for table, key in ((groups, group_key(r, group_by)), (trials, group_key(r, "trial"))):
            if key in table and table[key] != r["split"]:
                raise ValueError(f"Group leaks across partitions: {key}")
            table[key] = r["split"]


def assign_grouped_splits(records, group_by="subject", seed=0):
    keys = sorted({group_key(r, group_by) for r in records})
    if len(keys) < 3:
        raise ValueError("Need at least three independent groups")
    order = np.random.default_rng(seed).permutation(len(keys))
    nval = max(1, len(keys) // 5)
    ntest = max(1, len(keys) // 5)
    mapping = {keys[i]: "validation" if j < nval else "test" if j < nval + ntest else "train"
               for j, i in enumerate(order)}
    return [dict(r, split=mapping[group_key(r, group_by)]) for r in records]


class Normalizer:
    """One global train-only mean/std; keeps channel amplitude ratios meaningful."""
    def __init__(self, mean=0.0, std=1.0):
        self.mean, self.std = float(mean), float(std)

    @classmethod
    def fit(cls, dataset):
        count, total, squares = 0, 0.0, 0.0
        for s in dataset:
            x = s.x.double()
            if not torch.isfinite(x).all():
                raise ValueError("Non-finite training signal")
            count += x.numel()
            total += x.sum().item()
            squares += x.square().sum().item()
        if not count:
            raise ValueError("Empty normalization fit")
        mean = total / count
        variance = squares / count - mean * mean
        if variance <= 0:
            raise ValueError("Constant/degenerate training signal")
        return cls(mean, variance ** 0.5)

    def apply(self, batch):
        return Batch((batch.x - self.mean).div(self.std).masked_fill(~batch.valid, 0),
                     batch.valid, batch.channel_ids, batch.coords, batch.coord_valid,
                     batch.target, batch.metadata)


def make_synthetic(directory, seed=7, task="classification", subjects=6):
    """Diagnostic signals only. Numeric IDs represent documented synthetic sensors."""
    directory = Path(directory)
    if task not in {"classification", "regression", "sequence_classification", "sequence_regression"}:
        raise ValueError("Unknown synthetic task")
    directory.mkdir(parents=True, exist_ok=False)
    rng, records, fs = np.random.default_rng(seed), [], 64
    for subject in range(subjects):
        for trial in range(6):
            c, t = int(rng.integers(2, 5)), int(rng.integers(97, 161))
            label = trial % 2
            time = np.arange(t) / fs
            x = (np.sin(2 * np.pi * (6 + label * 8) * time)[None, :] +
                 rng.normal(0, .2, (c, t))).astype(np.float32) * 1e-5
            target = np.array(label if task == "classification" else [6 + label * 8])
            if task.startswith("sequence_"):
                patches = (t + 31) // 32
                target = np.full(patches, label, dtype=np.int64) if task.endswith("classification") else np.full((patches, 1), 6 + label * 8, dtype=np.float32)
            coords = np.stack((np.linspace(-.05, .05, c), np.zeros(c), np.full(c, .08)), axis=1)
            name = f"subject{subject}_trial{trial}.npz"
            np.savez(directory / name, x=x, channel_ids=np.arange(c), coords=coords,
                     coord_valid=np.ones(c, dtype=bool), target=target)
            records.append(dict(path=name, sha256=sha256(directory / name), sample_id=name,
                                subject=str(subject), session="0", trial=str(trial), dataset="synthetic"))
    manifest = dict(sample_rate=fs, unit="volt", coordinate_unit="meter", coordinate_frame="synthetic_cartesian",
                    channel_vocabulary={str(i): f"synthetic_sensor_{i}" for i in range(4)},
                    use_status="approved", evidence="synthetic_engineering_only", group_by="subject",
                    samples=assign_grouped_splits(records, seed=seed))
    if task.endswith("classification"):
        manifest["labels"] = ["synthetic_low_frequency", "synthetic_high_frequency"]
    else:
        manifest["target_schema"] = {"names": ["synthetic_frequency"], "unit": "hertz"}
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
