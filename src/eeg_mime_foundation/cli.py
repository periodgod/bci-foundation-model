import argparse
import json
from pathlib import Path

from .data import make_synthetic, sha256
from .decision import BRANCHES, decide
from .engine import Runner, evaluate, write_json
from .model import ModelConfig, SignalModel, count_parameters


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="EEG/time-series research preparation")
    sub = parser.add_subparsers(dest="command", required=True)
    synthetic = sub.add_parser("synthetic", help="Create diagnostic NPZ and a grouped manifest")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--task", choices=["classification", "regression", "sequence_classification", "sequence_regression"], default="classification")
    fit = sub.add_parser("train")
    fit.add_argument("--config", required=True)
    fit.add_argument("--manifest", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--resume")
    fit.add_argument("--initialize")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--manifest", required=True)
    ev.add_argument("--split", choices=["validation", "test"], default="validation")
    ev.add_argument("--output", required=True)
    ev.add_argument("--device", default="cpu")
    prune = sub.add_parser("decide")
    prune.add_argument("--evidence", required=True)
    prune.add_argument("--output", required=True)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--config", required=True)
    few = sub.add_parser("fewshot", help="Select labeled train support trials; preserve validation/test")
    few.add_argument("--manifest", required=True)
    few.add_argument("--output", required=True)
    few.add_argument("--trials-per-class", type=int, required=True)
    few.add_argument("--seed", type=int, default=0)
    prof = sub.add_parser("profile", help="Measure this batch's latency and CUDA allocated memory")
    prof.add_argument("--config", required=True)
    prof.add_argument("--manifest", required=True)
    prof.add_argument("--device", default="cpu")
    prof.add_argument("--output", required=True)
    smoke = sub.add_parser("smoke", help="Bounded synthetic-only integration matrix")
    smoke.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "synthetic":
        result = {"manifest": str(make_synthetic(args.output, task=args.task).resolve())}
    elif args.command == "train":
        result = Runner(read(args.config), args.manifest).fit(args.output, args.resume, args.initialize)
    elif args.command == "evaluate":
        result = evaluate(args.checkpoint, args.manifest, args.split, args.output, args.device)
    elif args.command == "decide":
        result = decide(read(args.evidence))
        write_json(args.output, result)
    elif args.command == "inventory":
        model = SignalModel(ModelConfig(**read(args.config)["model"]))
        result = dict(config=model.config_dict(), parameters=count_parameters(model), branches=BRANCHES)
    elif args.command == "fewshot":
        from .fewshot import support_manifest
        result = support_manifest(args.manifest, args.output, args.trials_per_class, args.seed)
    elif args.command == "profile":
        from .profile import profile
        result = profile(read(args.config), args.manifest, args.device)
        write_json(args.output, result)
    else:
        from .smoke import run
        result = run(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
