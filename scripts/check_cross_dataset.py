"""跨库拼 batch 的可用性检查：词表映射、坐标一致性、一次真实前向。

只做"能不能拼、拼完形状对不对"，不产出任何指标——迁移结论要等正式实验，
这里的作用是把"数据接不进来"和"模型效果不好"分开。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_mime_foundation.data import Sample, collate  # noqa: E402
from eeg_mime_foundation.model import ModelConfig, SignalModel  # noqa: E402


def load(directory, label, limit):
    directory = Path(directory)
    vocabulary = json.loads((directory / "shared_vocabulary.json").read_text(encoding="utf-8"))
    remap = json.loads((directory / f"remap-{label}.json").read_text(encoding="utf-8"))
    return vocabulary, {int(local): int(global_) for local, global_ in remap.items()}, remap


def samples(manifest_path, label, remap, limit):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = [r for r in manifest["samples"] if r["split"] == "train"][:limit]
    if not records:
        raise ValueError(f"{manifest_path} 没有 train 样本")
    output = []
    for record in records:
        with np.load(Path(manifest_path).parent / record["path"], allow_pickle=False) as arrays:
            ids = [remap.get(int(value), -1) for value in arrays["channel_ids"]]
            output.append(Sample(
                x=torch.tensor(arrays["x"], dtype=torch.float32),
                channel_ids=torch.tensor(ids, dtype=torch.long),
                coords=torch.tensor(arrays["coords"], dtype=torch.float32),
                coord_valid=torch.tensor(arrays["coord_valid"], dtype=torch.bool),
                target=torch.tensor(int(arrays["target"]), dtype=torch.long),
                sample_id=record["sample_id"], subject=record["subject"], session=record["session"],
                trial=record["trial"], dataset=f"{label}:{record['dataset']}"))
    return output


def electrode_positions(batch, names):
    """同一全局 id 在不同库里是否落在同一坐标；差超过 1 mm 说明词表映射错了。"""
    seen = {}
    clashes = []
    for row in range(batch.channel_ids.shape[0]):
        for column in range(batch.channel_ids.shape[1]):
            channel = int(batch.channel_ids[row, column])
            if channel < 0 or not bool(batch.coord_valid[row, column]):
                continue
            position = batch.coords[row, column].numpy()
            if channel in seen:
                if float(np.linalg.norm(seen[channel] - position)) > 1e-3:
                    clashes.append(dict(channel=channel, name=names[channel],
                                        existing=seen[channel].tolist(), new=position.tolist()))
            else:
                seen[channel] = position
    return clashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocabulary", required=True, help="build_shared_vocabulary.py 的输出目录")
    parser.add_argument("--manifest", action="append", required=True,
                        help="库名=manifest.json，可重复")
    parser.add_argument("--config", default="configs/mi_corpus_supervised.json")
    parser.add_argument("--limit", type=int, default=2, help="每个库取多少条 train 样本")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    directory = Path(args.vocabulary).resolve()
    names = json.loads((directory / "shared_vocabulary.json").read_text(encoding="utf-8"))["names"]
    collected, report = [], dict(libraries={}, clashes=[], forward={})
    for item in args.manifest:
        label, path = item.split("=", 1)
        _, mapping, raw = load(directory, label, args.limit)
        batch_samples = samples(path, label, mapping, args.limit)
        collected.extend(batch_samples)
        report["libraries"][label] = dict(manifest=str(Path(path).resolve()), sampled=len(batch_samples),
                                          remap_size=len(raw),
                                          channel_ids=sorted({int(i) for s in batch_samples
                                                              for i in s.channel_ids if int(i) >= 0}))
    batch = collate(collected)
    report["clashes"] = electrode_positions(batch, names)
    report["batch"] = dict(size=batch.x.shape[0], channels=int(batch.x.shape[1]),
                           samples=int(batch.x.shape[2]),
                           labels_tensor=list(batch.target.shape) if batch.target is not None else None)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))["model"]
    model = SignalModel(ModelConfig(**dict(config, graph=True))).eval()
    with torch.no_grad():
        output = model(batch)
    report["forward"] = dict(logits=list(output["logits"].shape),
                             finite=bool(torch.isfinite(output["logits"]).all()),
                             params=sum(p.numel() for p in model.parameters()))
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
