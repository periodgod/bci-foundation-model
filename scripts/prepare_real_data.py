"""Thin entry point for bounded, hashed real EEG intake.

`--scope corpus` 读取 `scripts/download_corpora.py` 下到 --root 的全量语料，
产出带真实电极坐标的 manifest；`smoke` / `development` 仍走原来的受限预检路径。
"""
import argparse
import json

from eeg_mime_foundation.realdata import download_data, ingest_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["download", "ingest"])
    p.add_argument("--dataset", choices=["mi", "sleep", "all"], default="all")
    p.add_argument("--root", default="data")
    p.add_argument("--scope", choices=["smoke", "development", "corpus"], default="smoke")
    p.add_argument("--epoch-stride", type=int, default=4,
                   help="corpus 睡眠每 N 个 30s 片段取一个，控制窗口总量")
    p.add_argument("--runs", nargs="*", default=None,
                   help="corpus MI 只取这些 run（默认 R04 R08 R12，纯想象）")
    args = p.parse_args()
    if args.stage == "download" and args.scope == "corpus":
        raise SystemExit("corpus 语料请用 scripts/download_corpora.py 下载")
    operation = download_data if args.stage == "download" else ingest_data
    for kind in ["sleep", "mi"] if args.dataset == "all" else [args.dataset]:
        result = (operation(kind, args.root, scope=args.scope) if args.stage == "download"
                  else operation(kind, args.root, scope=args.scope, epoch_stride=args.epoch_stride,
                                 runs=args.runs if kind == "mi" else None))
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
