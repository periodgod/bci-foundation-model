"""Load a local trusted checkpoint from its saved source in an isolated process."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import torch

from eeg_mime_foundation.data import ManifestDataset, Normalizer, collate, sha256
from eeg_mime_foundation.engine import load_checkpoint, write_json
from eeg_mime_foundation.model import ModelConfig, SignalModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = checkpoint.parent / "source.zip"
    environment = json.loads((checkpoint.parent/"environment.json").read_text(encoding="utf-8"))
    if sha256(source) != environment["source_snapshot"]["sha256"]:
        raise ValueError("Source snapshot changed")
    state = load_checkpoint(checkpoint)
    config = ModelConfig(**state["config"]["model"])
    dataset = ManifestDataset(manifest, "validation", config.sample_rate, config.task)
    torch.set_num_threads(2)
    model = SignalModel(config).eval()
    model.load_state_dict(state["model"])
    batch = Normalizer(**state["normalizer"]).apply(collate([dataset[i] for i in range(4)]))
    with torch.no_grad():
        expected = model(batch)["logits"]
    frozen = dict(model=state["model"], config=state["config"]["model"], normalizer=state["normalizer"],
                  signal_contract=state["signal_contract"], target_contract=state["target_contract"],
                  group_by=state["group_by"], training_groups=state["training_groups"],
                  selection_groups=state["selection_groups"], training_signal_hashes=state["training_signal_hashes"],
                  source_checkpoint_sha256=sha256(checkpoint), source_snapshot_sha256=sha256(source))
    frozen_path = output/"inference.pt"
    torch.save(frozen, frozen_path)
    with tempfile.TemporaryDirectory(prefix="eeg-frozen-") as td:
        root = Path(td).resolve()
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                if not (root/member.filename).resolve().is_relative_to(root):
                    raise ValueError("Archive member escapes extraction root")
            archive.extractall(root)
        code = """
import json, sys, torch
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from eeg_mime_foundation.data import ManifestDataset, Normalizer, collate
from eeg_mime_foundation.model import ModelConfig, SignalModel
torch.set_num_threads(2)
state = torch.load(sys.argv[2], map_location='cpu', weights_only=True)
cfg = ModelConfig(**state['config'])
data = ManifestDataset(sys.argv[3], 'validation', cfg.sample_rate, cfg.task)
model = SignalModel(cfg).eval()
model.load_state_dict(state['model'])
batch = Normalizer(**state['normalizer']).apply(collate([data[i] for i in range(4)]))
with torch.no_grad():
    logits = model(batch)['logits']
Path(sys.argv[4]).write_text(json.dumps(dict(predictions=logits.tolist(),
    imported_source=__import__('eeg_mime_foundation').__file__)), encoding='utf-8')
"""
        subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", code, str(root/"src"),
                        str(frozen_path), str(manifest), str(output/"fresh_process.json")],
                       cwd=root, check=True, timeout=120)
    actual = torch.tensor(json.loads((output/"fresh_process.json").read_text(encoding="utf-8"))["predictions"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    write_json(output/"acceptance.json", dict(exact_cpu_prediction_match=True, samples=4,
               optimizer_teacher_ema_not_exported=True, source_checkpoint_sha256=sha256(checkpoint),
               inference_sha256=sha256(frozen_path), inference_bytes=frozen_path.stat().st_size,
               scope="isolated Python -I, saved source, existing torch/numpy; no network operations; not official container"))
    print("Saved-source fresh-process inference: exact match", flush=True)


if __name__ == "__main__":
    main()
