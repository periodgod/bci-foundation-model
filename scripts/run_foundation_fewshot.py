"""Paired labeled-support studies; retain the frozen validation/test queries."""
import argparse
import json
from pathlib import Path

from eeg_mime_foundation.data import sha256
from eeg_mime_foundation.engine import write_json
from eeg_mime_foundation.fewshot import support_manifest
from validate_foundation import config_for, run_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=["mi","sleep"],required=True)
    parser.add_argument("--study",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    study = Path(args.study).resolve()
    required = [study/f"masked_pretrain_s{seed}"/"best.pt" for seed in (17,29,43)]
    if not all(path.is_file() for path in required):
        raise ValueError("Complete all three frozen pretraining sources before launching few-shot studies")
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    parent = study/"manifest.json"
    queries = [r for r in json.loads(parent.read_text(encoding="utf-8"))["samples"] if r["split"]!="train"]
    write_json(output/"launch.json",dict(parent_manifest_sha256=sha256(parent),
        seeds=[17,29,43],training_trials_per_class=8,variants=["scratch","masked_full"],
        unlabeled_pretraining="all selected TRAIN subjects/windows in parent; sources never see held-out",
        labeled_unit="MI event trial" if args.dataset=="mi" else "30s sleep epoch; not independent participant",
        normalization="Both variants use the same unlabeled parent-TRAIN normalization statistics",
        claim="bounded labeled-support feasibility; not target-subject calibration",test_predictions=0))
    results = {}
    for seed in (17,29,43):
        selected = output/f"support_s{seed}.json"
        support_manifest(parent,selected,8,seed)
        child = json.loads(selected.read_text(encoding="utf-8"))
        if queries != [r for r in child["samples"] if r["split"]!="train"]:
            raise AssertionError("Few-shot query changed")
        for variant in ("scratch","masked_full"):
            name = f"{variant}_s{seed}"
            checkpoint = study/f"masked_pretrain_s{seed}"/"best.pt"
            source = checkpoint if variant=="masked_full" else None
            results[name] = run_one(config_for(args.dataset,seed),selected,output/name,source,
                                   normalization_source=checkpoint if variant=="scratch" else None)
            write_json(output/"summary.json",results)
    write_json(output/"completion.json",dict(runs=6,queries_identical=True,test_predictions=0))


if __name__ == "__main__":
    main()
