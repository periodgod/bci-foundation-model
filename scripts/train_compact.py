"""训练紧凑骨干，用于导联子集评测（T1）。

大容量候选（dim=256, depth=5, ~4.9M）在 MI 开发清单（270 训练窗口）上会塌缩为
常数预测，无法用于鲁棒性评测。本脚本训练一个 dim/depth 可调的紧凑模型，
在相同划分上取得一个不塌缩、可用于后续导联子集评测的检查点。

用法（仓库根目录）：
    python scripts/train_compact.py --name compact-latent --arch latent \
        --manifest <mi manifest> --base-config <config.json> --epochs 40

输出：experiments/outputs/<name>/{best.pt,config.json,metrics.json}
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_mime_foundation.engine import Runner, evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base-config", required=True, help="参考训练配置（只取 model 段作骨架）")
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", default="latent")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out-root", default="experiments/outputs")
    args = ap.parse_args()

    base = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    cfg = copy.deepcopy(base)
    cfg["model"].update(architecture=args.arch, dim=args.dim, depth=args.depth, heads=args.heads)
    cfg["training"].update(
        objective="supervised",
        tuning="full",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device="cpu",
        threads=4,
        selection_metric="balanced_accuracy",
        direction="max",
        max_seconds=7200,
    )

    out = Path(args.out_root) / args.name
    result = Runner(cfg, args.manifest).fit(out)
    print(f"[{args.name}] params={result.get('params')}")

    report = evaluate(str(out / "best.pt"), args.manifest, "validation", str(out / "validation"))
    macro = report.get("subject_macro") or {}
    print(f"[{args.name}] validation BA={macro.get('balanced_accuracy')} acc={macro.get('accuracy')}")

    (out / "train_summary.json").write_text(
        json.dumps(
            dict(name=args.name, arch=args.arch, dim=args.dim, depth=args.depth,
                 heads=args.heads, epochs=args.epochs, lr=args.lr, seed=args.seed,
                 params=result.get("params"), validation=macro),
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
