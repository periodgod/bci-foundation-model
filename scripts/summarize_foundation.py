"""Compute paired exploratory summaries from retained foundation run records."""
import argparse
import json
from pathlib import Path

import numpy as np

from eeg_mime_foundation.engine import write_json


def summarize(study,controls=None,distillation=None):
    study = Path(study)
    runs = json.loads((study/"summary.json").read_text(encoding="utf-8"))
    controls_runs = json.loads((Path(controls)/"summary.json").read_text(encoding="utf-8")) if controls else {}
    distillation_runs = json.loads((Path(distillation)/"summary.json").read_text(encoding="utf-8")) if distillation else {}
    rows,paired = {},[]
    for variant in ("fusion_scratch","factorized_scratch","masked_full","masked_linear","masked_adapter"):
        selected = [runs[f"{variant}_s{seed}"] for seed in (17,29,43)]
        means = [r["validation"]["subject_macro"]["balanced_accuracy"] for r in selected]
        f1 = [r["validation"]["subject_macro"]["macro_f1"] for r in selected]
        rows[variant] = dict(subject_balanced_accuracy_mean=float(np.mean(means)),
                            seed_sd=float(np.std(means,ddof=1)),seeds=[17,29,43],
                            subject_macro_f1_mean=float(np.mean(f1)),
                            cuda_peak_allocated_mib=max(r["cuda_peak_allocated_mib"] for r in selected))
        if variant != "fusion_scratch":
            for seed in (17,29,43):
                base = runs[f"fusion_scratch_s{seed}"]["validation"]["by_subject"]
                candidate = runs[f"{variant}_s{seed}"]["validation"]["by_subject"]
                for subject in base:
                    paired.append(dict(variant=variant,seed=seed,subject=subject,
                                       scratch=base[subject]["balanced_accuracy"],
                                       candidate=candidate[subject]["balanced_accuracy"],
                                       delta=candidate[subject]["balanced_accuracy"]-base[subject]["balanced_accuracy"]))
    single_seed = {}
    for name,run in {**runs,**controls_runs,**distillation_runs}.items():
        if name.endswith("s17") and not name.startswith(("masked_pretrain","ema_pretrain","contrastive_pretrain")):
            single_seed[name] = dict(**run["validation"]["subject_macro"],cuda_peak_allocated_mib=run["cuda_peak_allocated_mib"])
    recalls = {}
    for name,run in runs.items():
        if not name.startswith(("masked_pretrain","ema_pretrain")):
            recalls[name] = {}
            for subject,metric in run["validation"]["by_subject"].items():
                matrix = np.array(metric["confusion"])
                support = matrix.sum(1)
                recalls[name][subject] = np.divide(matrix.diagonal(),support,out=np.zeros(len(support)),where=support>0).tolist()
    report = dict(three_seed_results=rows,single_seed_engineering=single_seed,paired_subject_rows=paired,
                  class_recalls=recalls,test_predictions=0,
                  interpretation="Seed SD is optimizer variation, not population uncertainty. Three validation subjects and selected best epochs are development evidence only. SSL heads are not transfer results.")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study",required=True)
    parser.add_argument("--controls")
    parser.add_argument("--distillation")
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    result = summarize(args.study,args.controls,args.distillation)
    write_json(args.output,result)
    print(json.dumps(result["three_seed_results"],indent=2),flush=True)


if __name__ == "__main__":
    main()
