"""被试分组 K 折交叉验证：在大语料上做结构/策略对照。

与 ``run_loso.py`` 的区别只有一处：留一法在 109 名被试上要跑 109 折，成本不可接受，
这里改成把被试随机分成 K 组、每次留一组。划分仍严格按被试分组，组内所有试次不跨折。

保留集不参与：清单里 split=="test" 的被试整折排除，只在其上做最终评测时才会用到。

用法（仓库根目录）：
    python scripts/run_grouped_cv.py --name cv-fusion --arch fusion \
        --manifest D:/wyu-bci-data/processed/corpus/mi-corpus-v1/manifest.json \
        --base-config configs/fusion_supervised.json --folds 10 --epochs 40
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_mime_foundation.engine import Runner, evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", default="fusion")
    ap.add_argument("--graph", action="store_true", help="启用电极距离偏置（图条件注意力）")
    ap.add_argument("--channel-dropout", type=float, default=0.,
                    help="训练期整通道丢弃概率，0 表示关闭")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-root", default="experiments/outputs")
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_of = {}
    for record in manifest["samples"]:
        split_of.setdefault(record["subject"], record["split"])
    subjects = sorted(s for s, split in split_of.items() if split != "test")
    if len(subjects) < args.folds * 2:
        raise SystemExit(f"可用被试 {len(subjects)} 太少，无法做 {args.folds} 折")
    order = np.random.default_rng(args.seed).permutation(len(subjects))
    folds = [sorted(subjects[i] for i in order[rank::args.folds]) for rank in range(args.folds)]
    print(f"subjects={len(subjects)} (保留测试集 {len(split_of) - len(subjects)} 名不参与) "
          f"folds={args.folds}", flush=True)

    base = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    root = Path(args.out_root) / args.name
    fold_dir = root / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, held in enumerate(folds):
        fold_manifest = copy.deepcopy(manifest)
        for record in fold_manifest["samples"]:
            record["split"] = "validation" if record["subject"] in held else "train"
            record["path"] = str((manifest_path.parent / record["path"]).resolve())
        fold_manifest["samples"] = [r for r in fold_manifest["samples"]
                                    if r["subject"] in held or split_of[r["subject"]] != "test"]
        path = fold_dir / f"manifest-fold{index:02d}.json"
        path.write_text(json.dumps(fold_manifest, ensure_ascii=False), encoding="utf-8")

        cfg = copy.deepcopy(base)
        cfg["model"]["architecture"] = args.arch
        cfg["model"]["graph"] = args.graph
        cfg["training"].update(objective="supervised", tuning="full", epochs=args.epochs,
                               seed=args.seed, device="cpu", threads=args.threads,
                               channel_dropout=args.channel_dropout,
                               selection_metric="balanced_accuracy", direction="max",
                               max_seconds=7200)
        out = root / f"fold{index:02d}"
        Runner(cfg, str(path)).fit(out)
        report = evaluate(str(out / "best.pt"), str(path), "validation", str(out / "eval"))
        macro = report.get("subject_macro") or {}
        row = dict(fold=index, subjects=held, balanced_accuracy=macro.get("balanced_accuracy"),
                   accuracy=macro.get("accuracy"), samples=report.get("samples"))
        rows.append(row)
        print(f"[fold {index:02d}] BA={row['balanced_accuracy']} n={row['samples']} "
              f"held={len(held)}", flush=True)

    values = [r["balanced_accuracy"] for r in rows if r["balanced_accuracy"] is not None]
    summary = dict(protocol="subject-grouped-kfold", arch=args.arch, graph=args.graph,
                   channel_dropout=args.channel_dropout, folds=args.folds, epochs=args.epochs,
                   seed=args.seed, subjects=len(subjects), results=rows,
                   mean_balanced_accuracy=float(statistics.mean(values)) if values else None,
                   std_balanced_accuracy=float(statistics.pstdev(values)) if len(values) > 1 else None,
                   min_balanced_accuracy=float(min(values)) if values else None,
                   max_balanced_accuracy=float(max(values)) if values else None)
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    if values:
        print(f"CV mean BA={summary['mean_balanced_accuracy']:.4f} "
              f"±{summary['std_balanced_accuracy']:.4f} over {len(values)} folds")


if __name__ == "__main__":
    main()
