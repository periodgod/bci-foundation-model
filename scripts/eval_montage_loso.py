"""基于 LOSO 折的导联子集迁移评测。

T1 的第一版用的是单一划分（6 被试训练 / 3 被试验证），基线只有 0.55，
落在机会水平的噪声里，导联子集的差异因此无法判别。跨被试留一结果显示
真实基线约 0.70——那份划分只是恰好抽到了难被试。

本脚本按正确协议重做：每一折的模型只在自己留出的那名被试上评测导联子集，
再把 10 折的结果汇总成均值与离散度。这样每个条件都有各自折内的基线做配对比较。

用法（仓库根目录，先跑 run_loso.py）：
    python scripts/eval_montage_loso.py --loso-root experiments/outputs/loso-fusion \
        --output reports/montage_transfer_loso.json --reps 10
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from eeg_mime_foundation.data import ManifestDataset, Normalizer, collate  # noqa: E402
from eeg_mime_foundation.engine import load_checkpoint  # noqa: E402
from eeg_mime_foundation.model import ModelConfig, SignalModel  # noqa: E402
from eval_montage_transfer import REGIONS, MONTAGES, infer, mask_from, scores  # noqa: E402


def evaluate_fold(checkpoint: Path, manifest: Path, reps: int, seed: int) -> dict:
    state = load_checkpoint(str(checkpoint), "cpu")
    cfg = ModelConfig(**state["config"]["model"])
    dataset = ManifestDataset(str(manifest), "validation", cfg.sample_rate, cfg.task)
    model = SignalModel(cfg).eval()
    model.load_state_dict(state["model"])
    normalizer = Normalizer(**state["normalizer"])
    batches = list(DataLoader(dataset, batch_size=8, collate_fn=collate, num_workers=0))

    sample = dataset[0]
    n_ch = int(sample.x.shape[0])
    vocab = dataset.manifest["channel_vocabulary"]
    index = {vocab[str(int(cid))]: i for i, cid in enumerate(sample.channel_ids) if int(cid) >= 0}

    pred, true = infer(model, normalizer, batches, None)
    base = scores(pred, true)["balanced_accuracy"]

    out = {"baseline": base, "montages": {}, "regions": {}, "random": {}}
    for name, names in MONTAGES.items():
        keep = mask_from(names, index, n_ch)
        p, t = infer(model, normalizer, batches, keep)
        out["montages"][name] = scores(p, t)["balanced_accuracy"]

    for name, names in REGIONS.items():
        present = [x for x in names if x in index]
        if not present:
            continue
        keep = ~mask_from(present, index, n_ch)
        p, t = infer(model, normalizer, batches, keep)
        out["regions"][name] = scores(p, t)["balanced_accuracy"]

    rng = random.Random(seed)
    for k in (32, 16, 8):
        vals = []
        for _ in range(reps):
            keep = np.zeros(n_ch, dtype=bool)
            keep[rng.sample(range(n_ch), k)] = True
            p, t = infer(model, normalizer, batches, keep)
            vals.append(scores(p, t)["balanced_accuracy"])
        out["random"][str(k)] = float(np.mean(vals))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loso-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.loso_root)
    folds = sorted((root / "folds").glob("manifest-*.json"))
    if not folds:
        raise SystemExit(f"未找到折清单：{root / 'folds'}")

    per_fold = {}
    for manifest in folds:
        subject = manifest.stem.replace("manifest-", "")
        checkpoint = root / f"fold-{subject}" / "best.pt"
        if not checkpoint.exists():
            print(f"跳过 {subject}：缺少 {checkpoint}")
            continue
        result = evaluate_fold(checkpoint, manifest, args.reps, args.seed)
        per_fold[subject] = result
        print(f"{subject} base={result['baseline']:.4f} "
              f"19导={result['montages']['10-20-19']:.4f} "
              f"3导={result['montages']['bciv2b-3']:.4f}")

    def aggregate(getter) -> dict:
        values = [getter(r) for r in per_fold.values() if getter(r) is not None]
        if not values:
            return {}
        return dict(mean=float(statistics.mean(values)),
                    std=float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
                    n=len(values))

    bases = [r["baseline"] for r in per_fold.values()]
    baseline = dict(mean=float(statistics.mean(bases)),
                    std=float(statistics.pstdev(bases)) if len(bases) > 1 else 0.0,
                    n=len(bases))

    montages = {}
    for name in MONTAGES:
        deltas = [r["montages"][name] - r["baseline"] for r in per_fold.values()
                  if name in r["montages"]]
        montages[name] = dict(score=aggregate(lambda r, n=name: r["montages"].get(n)),
                              delta=dict(mean=float(statistics.mean(deltas)),
                                         std=float(statistics.pstdev(deltas)) if len(deltas) > 1 else 0.0,
                                         folds_worse=sum(1 for d in deltas if d < 0),
                                         folds=len(deltas)))

    regions = {}
    for name in REGIONS:
        deltas = [r["regions"][name] - r["baseline"] for r in per_fold.values()
                  if name in r["regions"]]
        if deltas:
            regions[name] = dict(score=aggregate(lambda r, n=name: r["regions"].get(n)),
                                 delta=dict(mean=float(statistics.mean(deltas)),
                                            folds_worse=sum(1 for d in deltas if d < 0),
                                            folds=len(deltas)))

    random_curve = {k: aggregate(lambda r, kk=k: r["random"].get(kk)) for k in ("32", "16", "8")}

    report = dict(
        task="T1_montage_transfer_loso",
        protocol="each fold model evaluated on its own held-out subject; 10-fold mean",
        baseline=baseline, montages=montages, regions=regions, random_dropout=random_curve,
        per_fold=per_fold,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nLOSO 基线 BA = {baseline['mean']:.4f} ± {baseline['std']:.4f} (n={baseline['n']})")
    for name, row in montages.items():
        d = row["delta"]
        print(f"{name:16s} BA={row['score']['mean']:.4f} Δ={d['mean']:+.4f} "
              f"（{d['folds_worse']}/{d['folds']} 折下降）")
    for name, row in regions.items():
        print(f"drop {name:11s} BA={row['score']['mean']:.4f} Δ={row['delta']['mean']:+.4f} "
              f"（{row['delta']['folds_worse']}/{row['delta']['folds']} 折下降）")
    print(f"written {out}")


if __name__ == "__main__":
    main()
