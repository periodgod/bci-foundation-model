"""Create a concrete KD recipe from existing locally trusted teacher checkpoints."""
import argparse
from pathlib import Path

from eeg_mime_foundation.data import sha256
from eeg_mime_foundation.engine import write_json
from eeg_mime_foundation.smoke import base_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teachers", nargs="+", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--overlap", required=True)
    parser.add_argument("--use-status", choices=["pending", "approved"], default="pending")
    parser.add_argument("--weights", nargs="+", type=float)
    parser.add_argument("--learned", action="store_true")
    parser.add_argument("--student-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import json
    config = json.loads(Path(args.student_config).read_text(encoding="utf-8"))
    config["training"]["objective"] = "distill"
    weights = args.weights or [1.] * len(args.teachers)
    if len(weights) != len(args.teachers):
        parser.error("one weight per teacher required")
    config["teachers"] = [dict(checkpoint=str(Path(path).resolve()), sha256=sha256(Path(path)),
                              weight=weight, use_status=args.use_status, provenance=args.provenance,
                              overlap=args.overlap) for path, weight in zip(args.teachers, weights)]
    config["distillation"] = dict(learned=args.learned, entropy_weight=.01,
                                  supervised_weight=.2, logit_weight=0.)
    write_json(args.output, config)


if __name__ == "__main__":
    main()
