"""Run a bounded supervised real-data loop, task baseline and real-size profile."""
import argparse
import json
import importlib.metadata
from pathlib import Path
import time

import numpy as np
import torch

from eeg_mime_foundation.baselines import run_baseline
from eeg_mime_foundation.data import ManifestDataset, collate
from eeg_mime_foundation.engine import Runner, evaluate, write_json
from eeg_mime_foundation.profile import profile


def plot_intake(kind, manifest, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch
    m = json.loads(Path(manifest).read_text(encoding="utf-8"))
    ds = ManifestDataset(manifest, "train", m["sample_rate"], "classification")
    if kind == "sleep":
        index = next(i for i, r in enumerate(ds.records) if int(np.load(ds.path.parent / r["path"])["target"]) == 2)
    else:
        index = 0
    sample = ds[index]
    names = [m["channel_vocabulary"][str(int(i))] for i in sample.channel_ids]
    selected = [names.index(n) for n in ("C3", "CZ", "C4")] if kind == "mi" else list(range(len(names)))
    x, fs = sample.x.numpy()[selected], m["sample_rate"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    for offset, (signal, index) in enumerate(zip(x, selected)):
        axes[0].plot(np.arange(len(signal)) / fs, signal * 1e6 + offset * 150, label=names[index], lw=.7)
        f, psd = welch(signal, fs=fs, nperseg=min(len(signal), fs * 2))
        axes[1].semilogy(f, psd * 1e12, label=names[index])
    axes[0].set(xlabel="Time (s)", ylabel="Amplitude (uV), offsets 150 uV",
                title=f"{kind}: TRAIN ONLY {sample.sample_id}, target={m['labels'][int(sample.target)]}")
    axes[1].set(xlabel="Frequency (Hz)", ylabel="PSD (uV^2/Hz)", xlim=(0, 40))
    for axis in axes:
        axis.legend()
        axis.grid(alpha=.2)
    path = directory / "train_signal_qc.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path.resolve())


def config_for(kind):
    sleep = kind == "sleep"
    return dict(model=dict(architecture="compact", sample_rate=100 if sleep else 160,
                           patch_samples=100 if sleep else 80, dim=32, depth=1, heads=4,
                           mlp_ratio=2, dropout=.1, task="classification", outputs=5 if sleep else 2),
                labels=["W", "N1", "N2", "N3", "REM"] if sleep else ["left_hand_imagery", "right_hand_imagery"],
                training=dict(objective="supervised", tuning="full", epochs=5, batch_size=16 if sleep else 8,
                              lr=.001, weight_decay=.01, seed=17, device="cuda", threads=2,
                              clip_grad=1., max_seconds=600, selection_metric="balanced_accuracy", direction="max"))


def tiny_training_check(runner, directory):
    """Up to 16 fixed TRAIN windows; no validation/test access during this check."""
    dataset = runner.data["train"]
    # Round-robin per class, train only, to avoid a single-class debug batch.
    classes = runner.cfg.outputs
    by_class = {i: [] for i in range(classes)}
    for index in range(len(dataset)):
        sample = dataset[index]
        if len(by_class[int(sample.target)]) < 4:
            by_class[int(sample.target)].append(sample)
        if all(len(v) >= 4 for v in by_class.values()):
            break
    samples = [s for position in range(4) for label in range(classes)
               for s in by_class[label][position:position + 1]][:16]
    raw = collate(samples).to(runner.device)
    optimizer = torch.optim.AdamW(runner.model.parameters(), lr=.003)
    def measure():
        runner.model.eval()
        with torch.no_grad():
            output = runner.model(runner.normalizer.apply(raw))
            loss = runner.supervised_loss(output, raw.target)
            accuracy = (output["logits"].argmax(-1) == raw.target).float().mean()
        return dict(loss=float(loss), accuracy=float(accuracy))
    before = measure()
    for _ in range(100):
        runner.model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = runner.loss(raw)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite tiny real-data training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(runner.model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    after = measure()
    report = dict(windows=len(samples), steps=100, before=before, after=after,
                  all_training_windows=True, validation_test_predictions=0,
                  signal="learning-chain diagnostic only; not a generalization estimate",
                  sample_ids=[s.sample_id for s in samples])
    write_json(directory / "tiny_training_check.json", report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mi", "sleep", "all"], default="all")
    p.add_argument("--data-root", default="data")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "dependencies.json", {n: importlib.metadata.version(n)
               for n in ("torch", "numpy", "mne", "scipy", "scikit-learn", "requests", "matplotlib")})
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8")) if (root / "summary.json").exists() else {}
    for kind in ["mi", "sleep"] if args.dataset == "all" else [args.dataset]:
        started = time.monotonic()
        directory = root / kind
        directory.mkdir(parents=True, exist_ok=False)
        manifest = Path(args.data_root) / "processed" / "precheck" / f"{kind}-v1" / "manifest.json"
        cfg = config_for(kind)
        write_json(directory / "config.json", cfg)
        print(f"{kind}: baseline", flush=True)
        baseline = run_baseline(kind, manifest, directory / "domain-baseline")
        print(f"{kind}: tiny training check", flush=True)
        tiny = tiny_training_check(Runner(cfg, manifest), directory)
        # Fresh initialization: debug overfit parameters never enter the actual run.
        torch.cuda.empty_cache()
        print(f"{kind}: fresh supervised run", flush=True)
        runner = Runner(cfg, manifest)
        torch.cuda.reset_peak_memory_stats()
        training = runner.fit(directory / "supervised")
        training_peak = torch.cuda.max_memory_allocated() / 2**20
        del runner
        torch.cuda.empty_cache()
        validation = evaluate(directory / "supervised" / "best.pt", manifest, "validation",
                              directory / "validation", "cuda")
        print(f"{kind}: real-size candidate profile", flush=True)
        candidate = json.loads(json.dumps(cfg))
        candidate["model"].update(architecture="fusion", dim=192, depth=4, mlp_ratio=3)
        candidate["training"]["batch_size"] = 1
        write_json(directory / "candidate_profile_config.json", candidate)
        resource = profile(candidate, manifest, "cuda")
        write_json(directory / "candidate_resource.json", resource)
        summary[kind] = dict(training=training, validation=validation, baseline=baseline, tiny_training=tiny,
                             actual_supervised_peak_allocated_mib=training_peak, candidate_resource=resource,
                             qc_plot=plot_intake(kind, manifest, directory), elapsed_seconds=time.monotonic() - started,
                             test_predictions=0, evidence="real_data_pipeline_smoke_not_matched_paper_reproduction")
        write_json(directory / "summary.json", summary[kind])
        write_json(root / "summary.json", summary)
        print(f"{kind}: completed in {summary[kind]['elapsed_seconds']:.1f}s", flush=True)
    print(json.dumps({k: dict(validation_balanced_accuracy=v["validation"]["overall"]["balanced_accuracy"],
                               windows=v["validation"]["samples"], seconds=v["elapsed_seconds"]) for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
