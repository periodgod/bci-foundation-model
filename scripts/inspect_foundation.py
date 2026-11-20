"""Representation collapse and robustness diagnostics on validation EEG only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eeg_mime_foundation.data import Batch, ManifestDataset, Normalizer, collate, sha256
from eeg_mime_foundation.engine import load_checkpoint, set_seed, write_json
from eeg_mime_foundation.metrics import summarize
from eeg_mime_foundation.model import ModelConfig, SignalModel, patchify
from eeg_mime_foundation.objectives import make_mask, reconstruction_loss


@torch.no_grad()
def inspect(checkpoint, manifest, output):
    state = load_checkpoint(checkpoint, "cuda")
    cfg = ModelConfig(**state["config"]["model"])
    model = SignalModel(cfg).cuda().eval()
    model.load_state_dict(state["model"])
    ds = ManifestDataset(manifest, "validation", cfg.sample_rate, cfg.task)
    normalizer = Normalizer(**state["normalizer"])
    set_seed(20261011)
    torch.set_num_threads(2)
    groups = {}
    for index,record in enumerate(ds.records):
        groups.setdefault(record["subject"],[]).append(index)
    per_group = (32+len(groups)-1)//len(groups)
    candidates = {subject: [indices[int(i)] for i in np.linspace(0,len(indices)-1,min(per_group,len(indices)))]
                  for subject,indices in sorted(groups.items())}
    selected = [indices[position] for position in range(per_group) for indices in candidates.values() if position<len(indices)][:32]
    base = normalizer.apply(collate([ds[i] for i in selected]).to("cuda"))
    features = model(base)["features"].double()
    singular = torch.linalg.svdvals(features - features.mean(0))
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(1e-20)
    effective_rank = (-(probability*probability.clamp_min(1e-20).log()).sum()).exp()
    pair = torch.nn.functional.normalize(features, dim=-1)
    similarity = pair@pair.T
    off_diagonal = ~torch.eye(len(features), dtype=torch.bool, device="cuda")
    objective = state["config"]["training"]["objective"]
    decoder_trained = objective in {"masked","ema"}
    head_trained = objective == "supervised" or (objective == "distill" and any(
        state["config"].get("distillation",{}).get(key,0)>0 for key in ("supervised_weight","logit_weight")))
    reconstructed,zero_error = None,None
    if decoder_trained:
        _, valid = patchify(base, cfg.patch_samples)
        mask = make_mask(valid.any(-1), .5)
        decoded = model(base, mask)
        _, parts = reconstruction_loss(decoded, mask, cfg, .01, .1)
        samples = mask[...,None] & decoded["sample_valid"]
        zero_error = float(decoded["raw"].square()[samples].mean())
        reconstructed = float(parts["waveform"])
    # Padding values are irrelevant; changing valid content is intentionally not invariant.
    padded = Batch(torch.nn.functional.pad(base.x,(0,cfg.patch_samples),value=999.),
                   torch.nn.functional.pad(base.valid,(0,cfg.patch_samples),value=False),
                   base.channel_ids,base.coords,base.coord_valid,base.target,base.metadata)
    torch.testing.assert_close(model(base)["logits"],model(padded)["logits"],rtol=2e-4,atol=2e-5)
    reports = {}
    conditions = ("clean","half_channels","noise_0.1_train_std","first_75_percent") if head_trained else ()
    for condition in conditions:
        predictions, targets, metadata = [], [], []
        set_seed(20261011)
        for start in range(0,len(ds),8):
            raw = collate([ds[i] for i in range(start,min(start+8,len(ds)))]).to("cuda")
            batch = normalizer.apply(raw)
            if condition == "half_channels":
                keep = torch.arange(0,batch.x.shape[1],2,device="cuda")
                batch = Batch(batch.x[:,keep],batch.valid[:,keep],batch.channel_ids[:,keep],
                              batch.coords[:,keep],batch.coord_valid[:,keep],batch.target,batch.metadata)
            elif condition == "noise_0.1_train_std":
                batch.x = (batch.x + torch.randn_like(batch.x)*.1).masked_fill(~batch.valid,0)
            elif condition == "first_75_percent":
                length = batch.x.shape[-1]*3//4
                batch.x, batch.valid = batch.x[...,:length], batch.valid[...,:length]
            predictions.extend(model(batch)["logits"].cpu().tolist())
            targets.extend(batch.target.cpu().tolist())
            metadata.extend(batch.metadata)
        reports[condition] = summarize(predictions,targets,metadata,cfg.task)
    report = dict(checkpoint_sha256=sha256(Path(checkpoint)), manifest_sha256=sha256(Path(manifest)),
                  selection_data="validation only; exploratory diagnostic",
                  representation_samples=len(features), mean_feature_std=float(features.std(0).mean()),
                  representation_sample_ids=[ds.records[i]["sample_id"] for i in selected],
                  representation_subjects=sorted(groups), representation_selection="label-independent evenly spaced per-person round robin, cap32",
                  effective_rank=float(effective_rank), mean_pair_cosine=float(similarity[off_diagonal].mean()),
                  head_supervised=head_trained, decoder_trained=decoder_trained,
                  masked_waveform_mse=reconstructed, normalized_zero_target_mse=zero_error,
                  reconstruction_beats_zero=reconstructed<zero_error if decoder_trained else None, invalid_padding_invariance=True,
                  perturbations=reports, test_predictions=0,
                  limitations="No universal collapse threshold; 32-window rank capped at31; stress changes input and is not official scoring; reconstruction gain is not transfer gain")
    write_json(output,report)
    print(json.dumps({k:report[k] for k in ("mean_feature_std","effective_rank","mean_pair_cosine","reconstruction_beats_zero")}),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inspect(args.checkpoint,args.manifest,args.output)


if __name__ == "__main__":
    main()
