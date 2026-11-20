"""结构消融（E5）：同一划分、同一预算下横向比较 6 种编码器结构。

与 ``configs/*.json`` 的区别是：本脚本固定数据清单、固定 epoch 预算与种子集合，
只改 ``model.architecture``，因此差异可归因到结构本身。

用法：
    python scripts/run_arch_ablation.py \
        --manifest <manifest.json> --base-config <config.json> \
        --output reports/arch_ablation_mi.json --seeds 17 29 43 --epochs 5
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_mime_foundation.engine import Runner, evaluate  # noqa: E402

ARCHS = ("fusion", "factorized", "compact", "latent", "hierarchical", "causal")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base-config", required=True, help="训练配置模板（取采样率/patch/维度等）")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--archs", nargs="+", default=list(ARCHS))
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    base = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    work = Path(args.workdir or tempfile.mkdtemp(prefix="arch-ablation-"))
    work.mkdir(parents=True, exist_ok=True)

    rows = []
    for arch in args.archs:
        scores, params = [], None
        for seed in args.seeds:
            cfg = copy.deepcopy(base)
            cfg["model"]["architecture"] = arch
            cfg["model"]["task"] = "classification"
            cfg["training"].update(objective="supervised", tuning="full", epochs=args.epochs,
                                   seed=seed, device="cpu", threads=2,
                                   selection_metric="balanced_accuracy", direction="max",
                                   max_seconds=900)
            run_dir = work / f"{arch}-s{seed}"
            result = Runner(cfg, args.manifest).fit(run_dir)
            params = result.get("params")
            report = evaluate(run_dir / "best.pt", args.manifest, "validation",
                              run_dir / "validation")
            metrics = report.get("subject_macro") or report.get("overall") or {}
            ba = metrics.get("balanced_accuracy")
            acc = metrics.get("accuracy")
            scores.append(float(ba))
            print(f"{arch:13s} seed={seed} BA={ba:.4f} acc={acc:.4f}", flush=True)
        rows.append(dict(architecture=arch, params=params, scores=scores,
                         mean=statistics.mean(scores),
                         std=statistics.pstdev(scores) if len(scores) > 1 else 0.0))
        print(f"== {arch}: {rows[-1]['mean']:.4f} ± {rows[-1]['std']:.4f} (params {params})", flush=True)

    out = dict(manifest=str(Path(args.manifest).resolve()), seeds=args.seeds, epochs=args.epochs,
               split="validation", metric="balanced_accuracy", rows=rows)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written {target}")


if __name__ == "__main__":
    main()
