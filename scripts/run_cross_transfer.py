"""跨设备冻结迁移：用 A 库训练的检查点直接评 B 库，不做任何微调。

为什么不走 `engine.evaluate`：它把 `channel_vocabulary` 算进 `signal_contract`，
跨库必然不一致。这个限制是对的——两套词表直接对齐等于把同一电极当成两个传感器。
这里改成按**电极名**把目标库映射到训练库的 id 空间，映射不到的通道写 -1
（模型走未知电极 + 坐标编码），并把逐库映射率写进报告。

归一化统计一律取检查点里保存的 train-only 统计，重算目标库统计会让结果偏向目标库。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_mime_foundation.data import ManifestDataset, Normalizer, Sample, collate, sha256  # noqa: E402
from eeg_mime_foundation.engine import collect_predictions, load_checkpoint  # noqa: E402
from eeg_mime_foundation.metrics import summarize  # noqa: E402
from eeg_mime_foundation.model import ModelConfig, SignalModel  # noqa: E402
from eeg_mime_foundation.montage import normalise  # noqa: E402


class Remapped(Dataset):
    """把目标库的 local channel id 换成训练库的 id；映射不到的保留 -1。"""

    def __init__(self, dataset, mapping):
        self.dataset, self.mapping = dataset, mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        ids = torch.tensor([self.mapping.get(int(value), -1) for value in sample.channel_ids],
                           dtype=torch.long)
        return Sample(x=sample.x, channel_ids=ids, coords=sample.coords,
                      coord_valid=sample.coord_valid, target=sample.target,
                      sample_id=sample.sample_id, subject=sample.subject,
                      session=sample.session, trial=sample.trial, dataset=sample.dataset)


def build_mapping(train_vocabulary, target_vocabulary):
    """电极名 → 训练库 id。返回 (local→train, 明细)。"""
    by_name = {}
    for key, name in train_vocabulary.items():
        by_name.setdefault(normalise(name), int(key))
    mapping, detail = {}, {}
    for key, name in target_vocabulary.items():
        local = int(key)
        target_id = by_name.get(normalise(name), -1)
        mapping[local] = target_id
        detail[name] = target_id
    return mapping, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, help="目标库的 manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--library", default=None, help="报告里标注目标库名字，默认取 manifest 的 dataset")
    args = parser.parse_args()

    state = load_checkpoint(args.checkpoint, args.device)
    cfg = ModelConfig(**state["config"]["model"])
    contract = state["signal_contract"]
    manifest_path = Path(args.manifest).resolve()
    target = ManifestDataset(manifest_path, args.split, cfg.sample_rate, cfg.task)
    for field in ("sample_rate", "unit", "coordinate_unit", "coordinate_frame"):
        if target.manifest[field] != contract[field]:
            raise ValueError(f"跨库评测要求 {field} 一致："
                             f"训练={contract[field]} 目标={target.manifest[field]}")
    if cfg.task.endswith("classification") and state["config"]["labels"] != target.manifest.get("labels"):
        raise ValueError("标签语义不一致，迁移评测没有意义")
    mapping, detail = build_mapping(contract["channel_vocabulary"], target.manifest["channel_vocabulary"])
    matched = sum(1 for value in mapping.values() if value >= 0)
    model = SignalModel(cfg).to(args.device).eval()
    model.load_state_dict(state["model"])
    normalizer = Normalizer(**state["normalizer"])

    data = Remapped(target, mapping)
    logits, targets, metadata = [], [], []
    started = time.monotonic()
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=args.batch_size, collate_fn=collate, num_workers=0):
            raw = batch.to(args.device)
            prediction = model(normalizer.apply(raw))
            pp, yy, mm = collect_predictions(prediction, raw, cfg.task)
            logits.extend(pp)
            targets.extend(yy)
            metadata.extend(mm)
    report = dict(
        protocol="cross-library frozen transfer; no fine-tuning, no target statistics",
        library=args.library or target.manifest.get("dataset"),
        split=args.split,
        checkpoint_sha256=sha256(Path(args.checkpoint)),
        manifest_sha256=sha256(manifest_path),
        checkpoint_training_groups=len(state["training_groups"]),
        channel_mapping=detail,
        channels_matched=matched,
        channels_total=len(mapping),
        match_rate=round(matched / max(len(mapping), 1), 4),
        samples=len(data), predictions=len(metadata),
        elapsed_seconds=round(time.monotonic() - started, 1))
    if targets:
        report.update(summarize(logits, targets, metadata, cfg.task))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, prediction in enumerate(logits):
            handle.write(json.dumps(dict(**metadata[index], prediction=prediction,
                                         target=targets[index] if targets else None)) + "\n")
    print(json.dumps({k: report[k] for k in ("library", "split", "match_rate", "samples")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
