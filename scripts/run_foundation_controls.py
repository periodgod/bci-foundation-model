"""Train additional foundation controls on an already frozen real-data study."""
import argparse
import json
from pathlib import Path

from eeg_mime_foundation.data import sha256
from eeg_mime_foundation.engine import write_json
from validate_foundation import config_for, run_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mi","sleep"], required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    study = Path(args.study).resolve()
    if not all((study/name/"best.pt").is_file() for name in ("fusion_scratch_s17","masked_full_s17")):
        raise ValueError("Complete both frozen teacher sources before launching controls")
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    manifest = study/"manifest.json"
    write_json(output/"launch.json",dict(manifest_sha256=sha256(manifest),seed=17,
               purpose="single-teacher attribution, contrastive real-data chain and transfer",
               ordinary_model_baselines=False,teacher_names=["fusion_scratch_s17","masked_full_s17"],
               test_predictions=0,scope="engineering controls, not three-seed teacher advantage"))
    results = {}
    for teacher_name in ("fusion_scratch_s17","masked_full_s17"):
        checkpoint = study/teacher_name/"best.pt"
        config = config_for(args.dataset,objective="distill")
        config["teachers"] = [dict(checkpoint=str(checkpoint),sha256=sha256(checkpoint),
            use_status="approved",provenance="local scratch lineage",overlap="actual group/waveform enforced",weight=1.)]
        config["distillation"] = dict(learned=False,entropy_weight=.01,supervised_weight=1.,logit_weight=.1,temperature=2.)
        name = "single_"+teacher_name
        results[name] = run_one(config,manifest,output/name)
        write_json(output/"summary.json",results)
    config = config_for(args.dataset)
    config["training"].update(objective="contrastive",epochs=3,selection_metric="loss",direction="min")
    config["training"].pop("class_balance")
    name = "contrastive_pretrain_s17"
    results[name] = run_one(config,manifest,output/name)
    write_json(output/"summary.json",results)
    name = "contrastive_full_s17"
    results[name] = run_one(config_for(args.dataset),manifest,output/name,output/"contrastive_pretrain_s17"/"best.pt")
    write_json(output/"summary.json",results)


if __name__ == "__main__":
    main()
