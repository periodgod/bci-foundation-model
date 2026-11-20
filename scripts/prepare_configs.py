"""Generate executable diagnostic recipes without checkpoint placeholders."""
from pathlib import Path

from eeg_mime_foundation.engine import write_json
from eeg_mime_foundation.smoke import base_config


def main():
    output = Path(__file__).resolve().parents[1] / "configs" / "recipes"
    output.mkdir(parents=True, exist_ok=True)
    for architecture in ("fusion", "factorized", "compact", "latent", "hierarchical", "causal"):
        write_json(output / f"{architecture}_supervised.json", base_config(architecture))
    for objective in ("masked", "ema", "contrastive"):
        write_json(output / f"fusion_{objective}.json", base_config(objective=objective))
    for tuning in ("linear", "adapter", "full"):
        config = base_config()
        config["training"]["tuning"] = tuning
        write_json(output / f"fusion_tune_{tuning}.json", config)
    for task in ("regression", "sequence_classification", "sequence_regression"):
        config = base_config("causal" if task.startswith("sequence_") else "fusion", task=task,
                             outputs=2 if task.endswith("classification") else 1)
        write_json(output / f"{task}.json", config)


if __name__ == "__main__":
    main()
