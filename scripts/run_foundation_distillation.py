"""Complete the two planned teacher variants with verified metric selection."""
import argparse
from pathlib import Path

from eeg_mime_foundation.data import sha256
from eeg_mime_foundation.engine import write_json
from validate_foundation import config_for, run_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=["mi","sleep"],required=True)
    parser.add_argument("--study",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    study = Path(args.study).resolve()
    checkpoints = [study/name/"best.pt" for name in ("fusion_scratch_s17","masked_full_s17")]
    if not all(path.is_file() for path in checkpoints):
        raise ValueError("Both frozen sources must exist before launching distillation")
    root = Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    manifest = study/"manifest.json"
    write_json(root/"launch.json",dict(parent_launch=str(study/"launch.json"),
        manifest_sha256=sha256(manifest),seed=17,variants=["two_teacher_fixed","two_teacher_learned"],
        change="Repair selection metadata for distillation; architecture/loss/budget unchanged",test_predictions=0))
    results = {}
    for learned in (False,True):
        config = config_for(args.dataset,objective="distill")
        config["teachers"] = [dict(checkpoint=str(path),sha256=sha256(path),
            use_status="approved",provenance="self-built local scratch lineage, no public weights",
            overlap="transitive subject and waveform enforcement",weight=1.) for path in checkpoints]
        config["distillation"] = dict(learned=learned,entropy_weight=.01,supervised_weight=1.,logit_weight=.1,temperature=2.)
        name = f"two_teacher_{'learned' if learned else 'fixed'}_s17"
        results[name] = run_one(config,manifest,root/name)
        write_json(root/"summary.json",results)
    write_json(root/"completion.json",dict(completed_runs=2,teachers_frozen=True,test_predictions=0))


if __name__ == "__main__":
    main()
