"""Verify causal MODEL semantics on real EEG; preprocessing is separately audited."""
import argparse
import copy
from pathlib import Path

import torch

from eeg_mime_foundation.data import ManifestDataset, Normalizer, collate, sha256
from eeg_mime_foundation.engine import set_seed,write_json
from eeg_mime_foundation.model import ModelConfig,SignalModel
from eeg_mime_foundation.stream import CausalStream
from validate_foundation import config_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=["mi","sleep"],required=True)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    config = config_for(args.dataset,architecture="causal")
    torch.set_num_threads(2)
    set_seed(17)
    cfg = ModelConfig(**config["model"])
    data = ManifestDataset(args.manifest,"train",cfg.sample_rate,cfg.task)
    normalizer = Normalizer.fit(data)
    sample = data[0]
    batch = collate([sample]).to("cuda")
    model = SignalModel(cfg).cuda().eval()
    completed = sample.x.shape[-1]//cfg.patch_samples
    length = completed*cfg.patch_samples
    batch.x,batch.valid = batch.x[...,:length],batch.valid[...,:length]
    with torch.no_grad():
        expected = model(normalizer.apply(batch))["sequence_logits"]
        changed = copy.deepcopy(batch)
        boundary = cfg.patch_samples*3
        changed.x[...,boundary:] += 1.
        torch.testing.assert_close(expected[:,:3],model(normalizer.apply(changed))["sequence_logits"][:,:3],rtol=0,atol=0)
        stream = CausalStream(model,normalizer)
        edges = [0,cfg.patch_samples//3,cfg.patch_samples*2+7,length]
        outputs = [stream.push(batch.x[0,:,lo:hi],batch.channel_ids[0],batch.coords[0],batch.coord_valid[0])
                   for lo,hi in zip(edges,edges[1:])]
        torch.testing.assert_close(torch.cat(outputs),expected[0],rtol=2e-5,atol=2e-5)
        stream.reset()
        torch.testing.assert_close(stream.push(batch.x[0],batch.channel_ids[0],batch.coords[0],batch.coord_valid[0]),expected[0],rtol=0,atol=0)
    write_json(args.output,dict(real_data=True,sample_id=sample.sample_id,manifest_sha256=sha256(Path(args.manifest)),
        future_change_does_not_change_past=True,chunk_and_batch_equivalent=True,recording_reset_equivalent=True,
        patch_wait_seconds=cfg.patch_samples/cfg.sample_rate,emitted_completed_patches=completed,
        test_predictions=0,trained_task_accuracy_claim=False,
        limitations="Prefix replay, no bounded cache or latency claim. MI data were zero-phase filtered offline; this proves model causality only, not online preprocessing."))
    print(f"{args.dataset}: real causal model/chunk/reset checks passed",flush=True)


if __name__ == "__main__":
    main()
