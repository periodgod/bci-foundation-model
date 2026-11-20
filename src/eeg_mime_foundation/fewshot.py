"""Few-shot support selection from TRAIN trials only; never edits query/test sets."""
from pathlib import Path
import json

import numpy as np

from .data import audit_splits, group_key, sha256


def support_manifest(source, output, trials_per_class, seed=0):
    source, output = Path(source).resolve(), Path(output).resolve()
    if trials_per_class <= 0 or output.exists():
        raise ValueError("Positive support budget and new output file required")
    manifest = json.loads(source.read_text(encoding="utf-8"))
    audit_splits(manifest["samples"], manifest["group_by"])
    trials, labels = {}, {}
    for record in manifest["samples"]:
        if record["split"] != "train":
            continue
        path = source.parent / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError("Data changed")
        with np.load(path, allow_pickle=False) as arrays:
            label = arrays["target"]
            if label.ndim != 0 or not np.issubdtype(label.dtype, np.integer):
                raise ValueError("Few-shot selector expects one classification label per window")
            label = int(label)
        key = group_key(record, "trial")
        if key in labels and labels[key] != label:
            raise ValueError("Trial contains conflicting labels")
        labels[key] = label
        trials.setdefault(label, {})[key] = True
    rng, selected = np.random.default_rng(seed), set()
    for label, keys in sorted(trials.items()):
        keys = sorted(keys)
        if len(keys) < trials_per_class:
            raise ValueError(f"Insufficient train trials for class {label}")
        for i in rng.choice(len(keys), trials_per_class, replace=False):
            selected.add(keys[i])
    records = []
    for record in manifest["samples"]:
        if record["split"] != "train" or group_key(record, "trial") in selected:
            records.append(dict(record, path=str((source.parent / record["path"]).resolve())))
    manifest["samples"] = records
    manifest["support_selection"] = dict(parent_sha256=sha256(source), seed=seed,
                                          trials_per_class=trials_per_class, unit="trial", selected_trials=len(selected))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest["support_selection"]
