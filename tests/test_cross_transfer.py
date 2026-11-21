"""跨库评测的电极名映射。

这里唯一要保证的事：目标库的 C3 必须落到训练库 C3 那一行 embedding，而不是
"目标库里编号刚好也是 1 的那一行"。映射错了照样能跑出准确率，不会报错。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_cross_transfer as transfer  # noqa: E402


class Mapping(unittest.TestCase):
    def test_local_ids_do_not_decide_the_row(self):
        train = {"5": "C3", "6": "C4", "9": "CZ"}
        target = {"0": "C3", "1": "CZ", "2": "C4"}
        mapping, detail = transfer.build_mapping(train, target)
        self.assertEqual(mapping[0], 5)
        self.assertEqual(mapping[1], 9)
        self.assertEqual(mapping[2], 6)
        self.assertEqual(detail["C3"], 5)

    def test_unseen_electrode_becomes_unknown_not_a_random_row(self):
        train = {"0": "C3", "1": "C4"}
        target = {"0": "C3", "1": "PZ"}
        mapping, detail = transfer.build_mapping(train, target)
        self.assertEqual(mapping[0], 0)
        self.assertEqual(mapping[1], -1)
        self.assertEqual(detail["PZ"], -1)

    def test_names_are_normalised_before_lookup(self):
        train = {"0": "FP1", "1": "CZ"}
        target = {"0": "Fp1.", "1": "cz"}
        mapping, _ = transfer.build_mapping(train, target)
        self.assertEqual(mapping, {0: 0, 1: 1})

    def test_match_rate_counts_only_resolved_channels(self):
        train = {"0": "C3", "1": "C4", "2": "CZ"}
        target = {"0": "C3", "1": "C4", "2": "CZ"}
        mapping, _ = transfer.build_mapping(train, target)
        self.assertEqual(sum(1 for v in mapping.values() if v >= 0), 3)


if __name__ == "__main__":
    unittest.main()
