"""导联子集迁移评测（T1）。

回答一个具体问题：**同一套权重，换成一顶更稀疏的电极帽，性能剩多少。**

三种条件：
1. 固定导联子集 —— 10-10(32 导) / 10-20(19 导) / 感觉运动区(8 导) / BCIV-2b 布局(3 导)，
   这些是真实商用设备的典型布局，比随机抽通道更接近"换设备"。
2. 随机通道丢弃 —— 保留 k 导的均匀随机子集，k 逐档下降，评估鲁棒性下界。
3. 脑区整块移除 —— 按解剖区域整块置为无效，检验模型是否真的依赖生理相关区域。

通道失效通过 `valid=False` 表达（而非零填充），由模型自身的掩码与加权池化处理；
坐标与电极 ID 一律保留，因此这是"导联变稀疏"而不是"信号被抹掉"。

用法（仓库根目录）：
    python scripts/eval_montage_transfer.py \
        --checkpoint experiments/outputs/compact-latent/best.pt \
        --manifest <mi manifest> --output reports/montage_transfer_mi.json --reps 20
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

MONTAGES = {
    "10-10-32": [
        "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ", "FC4", "FT8",
        "T7", "C3", "CZ", "C4", "T8", "TP7", "CP3", "CPZ", "CP4", "TP8",
        "P7", "P3", "PZ", "P4", "P8", "PO3", "POZ", "PO4", "O1", "O2",
    ],
    "10-20-19": [
        "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", "CZ", "C4", "T8",
        "P7", "P3", "PZ", "P4", "P8", "O1", "O2",
    ],
    "sensorimotor-8": ["FC3", "FC4", "C3", "CZ", "C4", "CP3", "CPZ", "CP4"],
    "bciv2b-3": ["C3", "CZ", "C4"],
}

REGIONS = {
    "central": ["C5", "C3", "C1", "CZ", "C2", "C4", "C6"],
    "occipital": ["PO7", "PO3", "POZ", "PO4", "PO8", "O1", "OZ", "O2", "IZ"],
    "frontal": ["FP1", "FPZ", "FP2", "AF7", "AF3", "AFZ", "AF4", "AF8",
                "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8"],
}


def scores(pred: np.ndarray, target: np.ndarray) -> dict:
    classes = np.unique(target)
    recall = {int(c): float((pred[target == c] == c).mean()) for c in classes}
    return dict(
        balanced_accuracy=float(np.mean(list(recall.values()))),
        accuracy=float((pred == target).mean()),
        recall=recall,
        predicted_classes=sorted(int(c) for c in np.unique(pred)),
    )


@torch.no_grad()
def infer(model, normalizer, batches, keep: np.ndarray | None):
    preds, targets = [], []
    for batch in batches:
        data = normalizer.apply(batch.to("cpu"))
        if keep is not None:
            column = torch.as_tensor(keep, device=data.valid.device)[None, :, None]
            data = type(data)(
                x=data.x, valid=data.valid & column, channel_ids=data.channel_ids,
                coords=data.coords, coord_valid=data.coord_valid,
                target=data.target, metadata=data.metadata,
            )
        preds.append(model(data)["logits"].argmax(-1).cpu().numpy())
        targets.append(data.target.cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def bootstrap_ci(pred: np.ndarray, target: np.ndarray, n_boot: int = 2000,
                 seed: int = 0) -> tuple[float, float]:
    """单条件 BA 的 95% bootstrap 置信区间。"""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(target))
    vals = [scores(pred[s], target[s])["balanced_accuracy"]
            for s in (rng.choice(idx, len(idx), replace=True) for _ in range(n_boot))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_delta_ci(pred_a: np.ndarray, true_a: np.ndarray, pred_b: np.ndarray,
                    true_b: np.ndarray, n_boot: int = 2000,
                    seed: int = 0) -> tuple[float, float]:
    """两条件配对的 ΔBA 95% 置信区间；跨 0 即表示差异不显著。"""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(true_a))
    vals = []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        vals.append(scores(pred_b[s], true_b[s])["balanced_accuracy"]
                    - scores(pred_a[s], true_a[s])["balanced_accuracy"])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mask_from(names: list[str], index: dict[str, int], n: int) -> np.ndarray:
    keep = np.zeros(n, dtype=bool)
    missing = [x for x in names if x not in index]
    if missing:
        raise KeyError(f"电极不在当前布局中: {missing}")
    for name in names:
        keep[index[name]] = True
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--output", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-levels", default="56,48,40,32,24,16,8")
    args = ap.parse_args()

    state = load_checkpoint(args.checkpoint, "cpu")
    cfg = ModelConfig(**state["config"]["model"])
    dataset = ManifestDataset(args.manifest, args.split, cfg.sample_rate, cfg.task)
    if dataset.manifest["group_by"] != state["group_by"]:
        raise ValueError("group_by mismatch with checkpoint")

    model = SignalModel(cfg).eval()
    model.load_state_dict(state["model"])
    normalizer = Normalizer(**state["normalizer"])
    batches = list(DataLoader(dataset, batch_size=8, collate_fn=collate, num_workers=0))

    sample = dataset[0]
    n_ch = int(sample.x.shape[0])
    vocab = dataset.manifest["channel_vocabulary"]
    index = {vocab[str(int(cid))]: i for i, cid in enumerate(sample.channel_ids) if int(cid) >= 0}

    base_pred, base_true = infer(model, normalizer, batches, None)
    base = scores(base_pred, base_true)
    base["ci95"] = bootstrap_ci(base_pred, base_true, seed=args.seed)
    print(f"channels={n_ch} samples={len(dataset)} baseline BA={base['balanced_accuracy']:.4f} "
          f"CI95=[{base['ci95'][0]:.4f}, {base['ci95'][1]:.4f}]")
    print(f"baseline predicted classes: {base['predicted_classes']}")

    montages = []
    for name, names in MONTAGES.items():
        keep = mask_from(names, index, n_ch)
        pred, true = infer(model, normalizer, batches, keep)
        row = scores(pred, true)
        row.update(montage=name, kept=int(keep.sum()),
                   delta=row["balanced_accuracy"] - base["balanced_accuracy"],
                   delta_ci95=paired_delta_ci(base_pred, base_true, pred, true, seed=args.seed))
        montages.append(row)
        lo, hi = row["delta_ci95"]
        verdict = "n.s." if lo <= 0 <= hi else "significant"
        print(f"{name:16s} kept={keep.sum():3d} BA={row['balanced_accuracy']:.4f} "
              f"Δ={row['delta']:+.4f} CI95=[{lo:+.4f},{hi:+.4f}] {verdict} "
              f"classes={row['predicted_classes']}")

    rng = random.Random(args.seed)
    random_curve = []
    for k in [int(x) for x in args.random_levels.split(",") if 0 < int(x) <= n_ch]:
        vals = []
        for _ in range(args.reps):
            keep = np.zeros(n_ch, dtype=bool)
            keep[rng.sample(range(n_ch), k)] = True
            pred, true = infer(model, normalizer, batches, keep)
            vals.append(scores(pred, true)["balanced_accuracy"])
        random_curve.append(dict(keep=k, mean=float(np.mean(vals)), std=float(np.std(vals)),
                                 reps=args.reps))
        print(f"random keep={k:3d} BA={np.mean(vals):.4f}±{np.std(vals):.4f}")

    regions = []
    for name, names in REGIONS.items():
        present = [x for x in names if x in index]
        if not present:
            continue
        drop = mask_from(present, index, n_ch)
        keep = ~drop
        pred, true = infer(model, normalizer, batches, keep)
        row = scores(pred, true)
        row.update(region=name, dropped=int(drop.sum()),
                   delta=row["balanced_accuracy"] - base["balanced_accuracy"],
                   delta_ci95=paired_delta_ci(base_pred, base_true, pred, true, seed=args.seed))
        regions.append(row)
        lo, hi = row["delta_ci95"]
        print(f"drop {name:12s} -{drop.sum():2d}ch BA={row['balanced_accuracy']:.4f} "
              f"Δ={row['delta']:+.4f} CI95=[{lo:+.4f},{hi:+.4f}]")

    report = dict(
        task="T1_montage_transfer",
        checkpoint=str(Path(args.checkpoint).resolve()),
        manifest=str(Path(args.manifest).resolve()),
        split=args.split,
        samples=int(len(dataset)),
        channels=n_ch,
        baseline=base,
        montages=montages,
        random_dropout=random_curve,
        region_removal=regions,
        protocol=(
            "dropped channels set valid=False (not zero-filled); channel IDs and coordinates "
            "preserved so the model sees a sparser montage rather than corrupted signal"
        ),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()
