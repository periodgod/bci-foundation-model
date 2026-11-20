"""Bounded diagnostics, never use these scores to choose a competition model."""
from copy import deepcopy
from pathlib import Path

from .data import make_synthetic, sha256
from .decision import decide
from .engine import Runner, evaluate, write_json


def base_config(architecture="fusion", objective="supervised", task="classification", outputs=2):
    config = dict(model=dict(architecture=architecture, sample_rate=64, patch_samples=32,
                             dim=24, depth=1, heads=4, local_patches=3, latent_channels=2,
                             task=task, outputs=outputs),
                  training=dict(objective=objective, epochs=1, batch_size=4, lr=.001, seed=17,
                                threads=2, device="cpu", max_seconds=120, tuning="full"))
    if task.endswith("classification"):
        config["labels"] = ["synthetic_low_frequency", "synthetic_high_frequency"]
    if objective in {"masked", "ema"}:
        config["training"].update(mask_ratio=.5, mask_kind="joint", band_weight=.01, spectral_weight=.1)
    return config


def run(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = make_synthetic(directory / "classification_data")
    results = []
    for architecture in ("fusion", "factorized", "compact", "latent", "hierarchical", "causal"):
        config = base_config(architecture)
        results.append(Runner(config, manifest).fit(directory / architecture))
        evaluate(directory / architecture / "best.pt", manifest, "validation", directory / f"{architecture}_eval")
    for objective in ("masked", "ema", "contrastive"):
        results.append(Runner(base_config(objective=objective), manifest).fit(directory / objective))
    for tuning in ("linear", "adapter"):
        config = base_config()
        config["training"]["tuning"] = tuning
        results.append(Runner(config, manifest).fit(directory / tuning, initialize=directory / "masked" / "best.pt"))
    teachers = []
    for name in ("fusion", "compact"):
        path = (directory / name / "best.pt").resolve()
        teachers.append(dict(checkpoint=str(path), sha256=sha256(path), weight=1,
                             use_status="approved", provenance="local synthetic diagnostic teacher",
                             overlap="same synthetic train subjects; not a generalization experiment"))
    for learned in (False, True):
        config = base_config(objective="distill")
        config["teachers"] = teachers
        config["distillation"] = dict(learned=learned, entropy_weight=.01, supervised_weight=.2, logit_weight=.1)
        name = "distill_learned" if learned else "distill_equal"
        results.append(Runner(config, manifest).fit(directory / name))
    regression = make_synthetic(directory / "regression_data", task="regression")
    config = base_config(task="regression", outputs=1)
    results.append(Runner(config, regression).fit(directory / "regression"))
    evaluate(directory / "regression" / "best.pt", regression, "validation", directory / "regression_eval")
    for task in ("sequence_classification", "sequence_regression"):
        manifest_sequence = make_synthetic(directory / f"{task}_data", task=task)
        config = base_config("causal", task=task, outputs=2 if task.endswith("classification") else 1)
        results.append(Runner(config, manifest_sequence).fit(directory / task))
        evaluate(directory / task / "best.pt", manifest_sequence, "validation", directory / f"{task}_eval")
    config = base_config()
    Runner(config, manifest).fit(directory / "recovery")
    config = deepcopy(config)
    config["training"]["epochs"] = 2
    results.append(Runner(config, manifest).fit(directory / "recovery", resume=directory / "recovery" / "last.pt"))
    result = dict(evidence="synthetic_engineering_only", runs=results, decision=decide({}),
                  test_accesses=0, interpretation="Checks execution and interfaces; establishes no EEG effectiveness")
    write_json(directory / "smoke_report.json", result)
    return result
