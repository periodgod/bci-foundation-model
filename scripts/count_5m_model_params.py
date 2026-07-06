from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_mime_foundation import BalancedSignedGraphMIME, count_parameters


def main() -> None:
    model = BalancedSignedGraphMIME()
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    print(f"total_parameters={total:,}")
    print(f"trainable_parameters={trainable:,}")


if __name__ == "__main__":
    main()
