"""Fixed train-only memorization and CUDA epoch recovery for the foundation candidate."""
import argparse
import copy
import json
from pathlib import Path

import torch

from eeg_mime_foundation.data import collate, sha256
from eeg_mime_foundation.engine import Runner, load_checkpoint, write_json
from validate_foundation import config_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mi","sleep"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    config = config_for(args.dataset)
    runner = Runner(config,args.manifest)
    selected = {i: [] for i in range(runner.cfg.outputs)}
    for sample in runner.data["train"]:
        label = int(sample.target)
        if len(selected[label]) < 2:
            selected[label].append(sample)
        if all(len(v)==2 for v in selected.values()):
            break
    if any(len(v)!=2 for v in selected.values()):
        raise ValueError("Two training windows per class required")
    samples = [sample for bucket in selected.values() for sample in bucket]
    batch = collate(samples).to("cuda")
    def measure():
        runner.model.eval()
        with torch.no_grad():
            predicted = runner.model(runner.normalizer.apply(batch))
            return dict(loss=float(runner.supervised_loss(predicted,batch.target)),
                        accuracy=float((predicted["logits"].argmax(-1)==batch.target).float().mean()),
                        feature_std=float(predicted["features"].std(0).mean()))
    before = measure()
    # This diagnostic has its own fixed optimizer; no weights enter the study.
    optimizer = torch.optim.AdamW(runner.model.parameters(),lr=.001,weight_decay=.01)
    trace = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(100):
        runner.model.train()
        optimizer.zero_grad(set_to_none=True)
        loss,_ = runner.loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite memorization loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(runner.model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if (step+1)%10==0:
            trace.append(dict(step=step+1,**measure()))
    report = dict(before=before,after=measure(),trace=trace,steps=100,
                  params=sum(p.numel() for p in runner.model.parameters()),
                  sample_ids=[s.sample_id for s in samples],input="training windows only",
                  cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  diagnostic_weights_discarded=True,test_predictions=0)
    write_json(output/"memorization.json",report)
    del runner,optimizer,loss,batch
    torch.cuda.empty_cache()
    # Use real native data and dropout, but just one batch per partition.
    parent = Path(args.manifest).resolve()
    manifest = json.loads(parent.read_text(encoding="utf-8"))
    records = []
    for split in ("train","validation","test"):
        rows = [r for r in manifest["samples"] if r["split"]==split]
        records.extend(dict(r,path=str((parent.parent/r["path"]).resolve())) for r in rows[:8])
    manifest["samples"] = records
    small = output/"recovery_manifest.json"
    write_json(small,manifest)
    config = config_for(args.dataset,objective="ema")
    config["training"]["epochs"] = 2
    Runner(config,small).fit(output/"continuous")
    short = copy.deepcopy(config)
    short["training"]["epochs"] = 1
    Runner(short,small).fit(output/"recovered")
    Runner(config,small).fit(output/"recovered",resume=output/"recovered"/"last.pt")
    first,second = [load_checkpoint(output/name/"last.pt") for name in ("continuous","recovered")]
    for component in ("model","ema"):
        for name in first[component]:
            torch.testing.assert_close(first[component][name],second[component][name],rtol=0,atol=0)
    if first["best"] != second["best"]:
        raise AssertionError("Selection differs after recovery")
    write_json(output/"recovery.json",dict(real_data=True,device="cuda",dropout=.1,
               exact_model_and_ema_match=True,manifest_sha256=sha256(small),epochs=2,
               scope="epoch-boundary equivalence; eight windows per partition; no test prediction"))
    print(json.dumps(dict(memorization=report["after"],exact_cuda_recovery=True)),flush=True)


if __name__ == "__main__":
    main()
