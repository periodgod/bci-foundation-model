"""Thin entry point for bounded, hashed real EEG intake.

`--scope corpus` 读取 `scripts/download_corpora.py` 下到 --root 的全量语料，
产出带真实电极坐标的 manifest；`smoke` / `development` 仍走原来的受限预检路径。
BCIC IV 2a/2b 不在 PhysioNet 链路上，必须用 `--raw` 显式指向本地 gdf 目录。
"""
import argparse
import json

from eeg_mime_foundation.realdata import BCIC_SOURCES, download_data, ingest_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["download", "ingest"])
    p.add_argument("--dataset", choices=["mi", "sleep", "bcic2a", "bcic2b", "all"], default="all")
    p.add_argument("--root", default="data")
    p.add_argument("--scope", choices=["smoke", "development", "corpus"], default="smoke")
    p.add_argument("--raw", default=None,
                   help="BCIC IV 原始 gdf 目录；只允许与单个 --dataset 一起用")
    p.add_argument("--labels", default=None,
                   help="BCIC IV 官方真值 .mat 目录；E session 没有事件码，必须靠它才有标签")
    p.add_argument("--epoch-stride", type=int, default=4,
                   help="corpus 睡眠每 N 个 30s 片段取一个，控制窗口总量")
    p.add_argument("--runs", nargs="*", default=None,
                   help="corpus MI 只取这些 run（默认 R04 R08 R12，纯想象）")
    args = p.parse_args()
    if args.stage == "download" and args.scope == "corpus":
        raise SystemExit("corpus 语料请用 scripts/download_corpora.py 下载")
    if args.dataset in BCIC_SOURCES and args.stage == "download":
        raise SystemExit("BCIC IV 是本地语料，不需要下载；直接用 ingest + --raw")
    if args.raw and args.dataset == "all":
        raise SystemExit("--raw 只能配合单个 --dataset 使用")
    if args.dataset in BCIC_SOURCES and not args.raw:
        raise SystemExit(f"--dataset {args.dataset} 必须用 --raw 指定 gdf 目录")
    operation = download_data if args.stage == "download" else ingest_data
    kinds = ["sleep", "mi"] if args.dataset == "all" else [args.dataset]
    for kind in kinds:
        if args.stage == "download":
            result = operation(kind, args.root, scope=args.scope)
        else:
            result = operation(kind, args.root, scope=args.scope, epoch_stride=args.epoch_stride,
                               runs=args.runs if kind == "mi" else None, raw=args.raw,
                               labels=args.labels)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
