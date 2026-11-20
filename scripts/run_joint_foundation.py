"""Bounded multi-source scratch pretraining and a shared checkpoint's two-task transfer."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from eeg_mime_foundation.data import sha256
from eeg_mime_foundation.engine import Runner, write_json
from validate_foundation import config_for, run_one


def prepare(mi_manifest, sleep_manifest, root):
    parents = {kind: Path(path).resolve() for kind,path in (("mi",mi_manifest),("sleep",sleep_manifest))}
    manifests = {kind: json.loads(path.read_text(encoding="utf-8")) for kind,path in parents.items()}
    names = sorted({name for manifest in manifests.values() for name in manifest["channel_vocabulary"].values()})
    vocabulary = {str(index): name for index,name in enumerate(names)}
    lookup = {name:index for index,name in enumerate(names)}
    contract = dict(sample_rate=200,unit="volt",coordinate_unit="meter",
                    coordinate_frame="unknown_not_used; channel identities documented",channel_vocabulary=vocabulary,
                    use_status="approved",group_by="subject",evidence="real_data_bounded_joint_development",
                    use_scope="local research; not competition-authorized weights")
    unlabelled, tasks = [], {}
    (root/"signals").mkdir()
    for kind,parent in manifests.items():
        records = []
        native = parent["sample_rate"]
        divisor = math.gcd(200,native)
        for record in parent["samples"]:
            source = parents[kind].parent/record["path"]
            if sha256(source) != record["sha256"]:
                raise ValueError("Parent signal file changed")
            with np.load(source,allow_pickle=False) as arrays:
                original = arrays["x"]
                signal = resample_poly(original,200//divisor,native//divisor,axis=-1,padtype="line").astype(np.float32)
                ids = np.array([lookup[parent["channel_vocabulary"][str(int(i))]] for i in arrays["channel_ids"]])
                payload = dict(x=signal,channel_ids=ids,coords=arrays["coords"],coord_valid=arrays["coord_valid"])
                target = arrays["target"].copy()
            if not np.isfinite(signal).all() or signal.shape[-1]*native != original.shape[-1]*200:
                raise ValueError("Resampling timebase/finite check failed")
            path = root/"signals"/f"{kind}_{record['sample_id']}.npz"
            np.savez(path,**payload,target=target)
            base = {key:record[key] for key in ("sample_id","dataset","subject","session","trial","split")}
            base.update(parent_file_sha256=record["sha256"],native_rate=native,
                        parent_start_sample=record.get("start_sample"),parent_stop_sample=record.get("stop_sample"))
            records.append(dict(base,path=str(path.resolve()),sha256=sha256(path)))
            path = root/"signals"/f"unlabelled_{kind}_{record['sample_id']}.npz"
            np.savez(path,**payload)
            unlabelled.append(dict(base,path=str(path.resolve()),sha256=sha256(path)))
        manifest = dict(contract,labels=parent["labels"],samples=records,parent_manifest_sha256=sha256(parents[kind]),
                        reference_policy="native reference preserved; bipolar leads remain distinct vocabulary identities")
        task_path = root/f"{kind}_manifest.json"
        write_json(task_path,manifest)
        tasks[kind] = task_path
    joint = root/"unlabelled_manifest.json"
    write_json(joint,dict(contract,labels=["unused_ssl_head"],samples=unlabelled,
                         target_policy="No target arrays in any joint pretraining NPZ",
                         parent_manifest_sha256={kind:sha256(path) for kind,path in parents.items()}))
    return joint,tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mi-manifest",required=True)
    parser.add_argument("--sleep-manifest",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/"launch.json",dict(seed=17,scope="same scratch base, unified physical inputs, two-task adaptation",
        parents={"mi":sha256(Path(args.mi_manifest)),"sleep":sha256(Path(args.sleep_manifest))},
        target_rate=200,patch_samples=100,method="scipy resample_poly, segmented offline, line padding",
        vocabulary="union of verified native channel names; bipolar identities kept separate; no invented coordinates",
        pretraining="3 epochs masked, labels removed, grouped homogeneous dataset batches",
        transfer="full and frozen linear, each task5epochs; same source checkpoint/normalizer; reset heads",
        independent_validation="same six validation subjects and two reserved subjects as parent studies",
        test_predictions=0,claim="engineering transfer feasibility; single seed; resampled protocol differs from native experiments"))
    joint,tasks = prepare(args.mi_manifest,args.sleep_manifest,root)
    cfg = config_for("sleep",objective="masked")
    cfg["model"].update(sample_rate=200,patch_samples=100,outputs=1)
    cfg["labels"] = ["unused_ssl_head"]
    cfg["training"]["batch_by_dataset"] = True
    runner = Runner(cfg,joint)
    if any(sample.target is not None for sample in runner.data["train"]):
        raise AssertionError("Joint pretraining must not consume class labels")
    torch.cuda.reset_peak_memory_stats()
    result = runner.fit(root/"shared_pretrain")
    result["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated()/2**20
    result["label_arrays_absent"] = True
    write_json(root/"shared_pretrain"/"result.json",result)
    del runner
    torch.cuda.empty_cache()
    results = {"shared_pretrain":result}
    write_json(root/"summary.json",results)
    source = root/"shared_pretrain"/"best.pt"
    for kind,manifest in tasks.items():
        for tuning in ("full","linear"):
            cfg = config_for(kind,tuning=tuning)
            cfg["model"].update(sample_rate=200,patch_samples=100)
            name = f"{kind}_{tuning}"
            results[name] = run_one(cfg,manifest,root/name,source)
            write_json(root/"summary.json",results)
    write_json(root/"completion.json",dict(shared_checkpoint_sha256=sha256(source),
               adapted_tasks=["mi","sleep"],transfer_uses_identical_base=True,
               input_contract_shared=True,test_predictions=0,
               pending="three seeds, unlabeled budget controls, official datasets/heads, geometry and whole-night context"))


if __name__ == "__main__":
    main()
