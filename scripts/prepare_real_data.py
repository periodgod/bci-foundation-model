"""Thin entry point for bounded, hashed real EEG intake."""
import argparse
import json

from eeg_mime_foundation.realdata import download_data, ingest_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["download", "ingest"])
    p.add_argument("--dataset", choices=["mi", "sleep", "all"], default="all")
    p.add_argument("--root", default="data")
    p.add_argument("--scope", choices=["smoke", "development"], default="smoke")
    args = p.parse_args()
    operation = download_data if args.stage == "download" else ingest_data
    for kind in ["mi", "sleep"] if args.dataset == "all" else [args.dataset]:
        print(json.dumps(operation(kind, args.root, scope=args.scope), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
