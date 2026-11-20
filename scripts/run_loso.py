"""跨被试留一（LOSO）评测。

把原有的一次性划分换成标准协议：每一折留出 1 名被试做验证，其余被试全部参与训练，
10 名被试轮流留出。这样得到的是"面对一个全新的人"的期望表现与离散度，
而不是单次划分的偶然值。

用法（仓库根目录）：
    python scripts/run_loso.py --name loso-fusion --arch fusion \
        --manifest <mi manifest> --base-config <config.json> --epochs 40

输出：<out-root>/<name>/folds/fold-<subject>.json 与 summary.json
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_mime_foundation.engine import Runner, evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", default="fusion")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--subjects", nargs="*", default=None, help="只跑指定被试（调试用）")
    ap.add_argument("--out-root", default="experiments/outputs")
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    subjects = sorted({r["subject"] for r in manifest["samples"]})
    if args.subjects:
        subjects = [s for s in subjects if s in args.subjects]
    print(f"subjects={len(subjects)}: {subjects}")

    base = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    root = Path(args.out_root) / args.name
    fold_dir = root / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for subject in subjects:
        fold_manifest = copy.deepcopy(manifest)
        for record in fold_manifest["samples"]:
            record["split"] = "validation" if record["subject"] == subject else "train"
            # 清单落到 folds/ 子目录后，相对路径会失效，统一改成绝对路径
            record["path"] = str((manifest_path.parent / record["path"]).resolve())
        path = fold_dir / f"manifest-{subject}.json"
        path.write_text(json.dumps(fold_manifest, ensure_ascii=False), encoding="utf-8")

        cfg = copy.deepcopy(base)
        cfg["model"]["architecture"] = args.arch
        cfg["training"].update(objective="supervised", tuning="full", epochs=args.epochs,
                               seed=args.seed, device="cpu", threads=4,
                               selection_metric="balanced_accuracy", direction="max",
                               max_seconds=7200)

        # 样本路径相对 manifest 所在目录，折内清单必须放回原目录旁
        out = root / f"fold-{subject}"
        Runner(cfg, str(path)).fit(out)
        report = evaluate(str(out / "best.pt"), str(path), "validation", str(out / "eval"))
        macro = report.get("subject_macro") or {}
        row = dict(subject=subject,
                   balanced_accuracy=macro.get("balanced_accuracy"),
                   accuracy=macro.get("accuracy"),
                   samples=report.get("samples"))
        rows.append(row)
        (fold_dir / f"fold-{subject}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{subject}] BA={row['balanced_accuracy']} acc={row['accuracy']} n={row['samples']}")

    values = [r["balanced_accuracy"] for r in rows if r["balanced_accuracy"] is not None]
    summary = dict(
        protocol="leave-one-subject-out",
        arch=args.arch, epochs=args.epochs, seed=args.seed,
        subjects=len(rows), folds=rows,
        mean_balanced_accuracy=float(statistics.mean(values)) if values else None,
        std_balanced_accuracy=float(statistics.pstdev(values)) if len(values) > 1 else None,
        min_balanced_accuracy=float(min(values)) if values else None,
        max_balanced_accuracy=float(max(values)) if values else None,
    )
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    mean, std = summary["mean_balanced_accuracy"], summary["std_balanced_accuracy"]
    spread = f"±{std:.4f}" if std is not None else "(单折无离散度)"
    print(f"LOSO mean BA={mean:.4f} {spread} over {len(rows)} subjects" if mean is not None
          else "LOSO 无有效折结果")


if __name__ == "__main__":
    main()
