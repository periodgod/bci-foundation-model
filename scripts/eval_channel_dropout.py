"""通道丢弃鲁棒性评测（E4）。

在固定检查点上做推理期通道缺失评测：保留 k 个通道、其余整列标记为无效，
由模型自身的 valid 掩码与加权池化处理（不是零填充）。每个 k 重复多次随机抽样，
报告平衡准确率的均值与标准差。

用法（在仓库根目录）：
    python scripts/eval_channel_dropout.py \
        --checkpoint <best.pt> --manifest <manifest.json> \
        --output reports/channel_dropout_mi.json --reps 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_mime_foundation.data import ManifestDataset, Normalizer, collate  # noqa: E402
from eeg_mime_foundation.engine import load_checkpoint  # noqa: E402
from eeg_mime_foundation.model import ModelConfig, SignalModel  # noqa: E402

LEVELS = (64, 56, 48, 40, 32, 24, 16, 8)


def balanced_accuracy(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    classes = np.unique(target)
    recalls = [(pred[target == c] == c).mean() for c in classes if (target == c).any()]
    return float(np.mean(recalls)), float((pred == target).mean())


@torch.no_grad()
def run(model, normalizer, batches, keep: np.ndarray | None):
    preds, targets = [], []
    for batch in batches:
        raw = batch.to("cpu")
        masked = normalizer.apply(raw)
        if keep is not None:
            device_mask = torch.as_tensor(keep, device=masked.valid.device)[None, :, None]
            masked = type(masked)(
                x=masked.x, valid=masked.valid & device_mask, channel_ids=masked.channel_ids,
                coords=masked.coords, coord_valid=masked.coord_valid, target=masked.target,
                metadata=masked.metadata,
            )
        logits = model(masked)["logits"]
        preds.append(logits.argmax(-1).cpu().numpy())
        targets.append(masked.target.cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--output", required=True)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    state = load_checkpoint(args.checkpoint, "cpu")
    cfg = ModelConfig(**state["config"]["model"])
    dataset = ManifestDataset(args.manifest, args.split, cfg.sample_rate, cfg.task)
    if dataset.manifest["group_by"] != state["group_by"]:
        raise ValueError("group_by mismatch with checkpoint")
    model = SignalModel(cfg).eval()
    model.load_state_dict(state["model"])
    normalizer = Normalizer(**state["normalizer"])

    n_channels = int(dataset[0].x.shape[0])
    batches = list(DataLoader(dataset, batch_size=8, collate_fn=collate, num_workers=0))
    print(f"channels={n_channels} samples={len(dataset)} batches={len(batches)}")

    pred, target = run(model, normalizer, batches, None)
    base_ba, base_acc = balanced_accuracy(pred, target)
    print(f"baseline BA={base_ba:.4f} acc={base_acc:.4f}")

    rng = random.Random(args.seed)
    levels = [k for k in LEVELS if k <= n_channels]
    rows = []
    for k in levels:
        if k == n_channels:
            rows.append(dict(keep=k, mean=base_ba, std=0.0, acc=base_acc, reps=1))
            print(f"keep={k:3d} BA={base_ba:.4f} (baseline)")
            continue
        scores, accs = [], []
        for _ in range(args.reps):
            keep_idx = rng.sample(range(n_channels), k)
            mask = np.zeros(n_channels, dtype=bool)
            mask[keep_idx] = True
            p, t = run(model, normalizer, batches, mask)
            ba, acc = balanced_accuracy(p, t)
            scores.append(ba)
            accs.append(acc)
        rows.append(dict(keep=k, mean=float(np.mean(scores)), std=float(np.std(scores)),
                         acc=float(np.mean(accs)), reps=args.reps))
        print(f"keep={k:3d} BA={np.mean(scores):.4f}±{np.std(scores):.4f}")

    report = dict(
        checkpoint=str(Path(args.checkpoint).resolve()), manifest=str(Path(args.manifest).resolve()),
        split=args.split, samples=int(len(dataset)), channels=n_channels, reps=args.reps,
        baseline=dict(balanced_accuracy=base_ba, accuracy=base_acc),
        levels=rows,
        protocol="channel validity mask (not zero fill); random uniform subset per repetition",
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()
