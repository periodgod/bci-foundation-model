"""BCIC 标签交叉校验脚本的判定逻辑。

真实数据里 2a 的 2592 条、2b 的 3680 条都逐条一致。这里用替身覆盖判定本身：
一致就通过、不一致必须报出来——否则这个脚本只会打印"全部通过"。
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import verify_bcic_labels as verify  # noqa: E402


class Checking(unittest.TestCase):
    def run_with(self, codes, truth):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            (root / "labels").mkdir()
            (root / "raw" / "A01T.gdf").write_bytes(b"")
            (root / "labels" / "A01T.mat").write_bytes(b"")
            output = root / "report.json"
            argv = ["x", "--raw", str(root / "raw"), "--labels", str(root / "labels"),
                    "--output", str(output)]
            with mock.patch.object(verify, "session_codes", return_value=codes), \
                    mock.patch.object(verify, "bcic_truth_labels", return_value=truth), \
                    mock.patch.object(sys, "argv", argv):
                try:
                    verify.main()
                except SystemExit as error:
                    return json.loads(output.read_text(encoding="utf-8")), str(error)
                return json.loads(output.read_text(encoding="utf-8")), None

    def test_identical_sequences_pass(self):
        report, error = self.run_with([1, 2, 3, 4], [1, 2, 3, 4])
        self.assertIsNone(error)
        self.assertEqual(report["matched_labels"], 4)
        self.assertEqual(report["mismatches"], [])

    def test_shifted_sequence_is_reported(self):
        report, error = self.run_with([1, 2, 3, 4], [2, 3, 4, 1])
        self.assertIsNotNone(error)
        self.assertEqual(report["mismatches"], ["A01T"])

    def test_truncated_sequence_is_not_treated_as_a_match(self):
        report, _ = self.run_with([1, 2], [1, 2, 3, 4])
        self.assertFalse(report["rows"][0]["identical"])

    def test_sessions_without_codes_are_marked_unverifiable(self):
        report, error = self.run_with([], [1, 2])
        self.assertIsNone(error)
        self.assertEqual(report["rows"][0]["note"], "unverifiable")
        self.assertEqual(report["verified_labels"], 0)


if __name__ == "__main__":
    unittest.main()
