"""共享通道词表的合并规则。

要防的不是"排序不好看"，而是同一个电极在不同库里拿到不同 id：那样跨设备迁移
测的是两个互不相干的传感器，而且训练指标不会报错。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_shared_vocabulary as shared  # noqa: E402


class Build(unittest.TestCase):
    def test_same_electrode_gets_one_global_id_across_libraries(self):
        vocabulary, remaps, summary = shared.build({
            "mi": ["FC3", "C3", "CZ", "C4"],
            "bcic2a": ["FZ", "C3", "CZ", "C4", "PZ"],
            "bcic2b": ["C3", "CZ", "C4"],
        })
        self.assertEqual(remaps["mi"]["1"], remaps["bcic2a"]["1"])
        self.assertEqual(remaps["bcic2a"]["1"], remaps["bcic2b"]["0"])
        self.assertEqual(vocabulary["names"][remaps["bcic2b"]["0"]], "C3")

    def test_shared_is_the_intersection_of_all_libraries(self):
        _, _, summary = shared.build({
            "mi": ["FZ", "C3", "CZ", "C4", "PZ"],
            "bcic2b": ["C3", "CZ", "C4"],
        })
        self.assertEqual(summary["shared"], ["C3", "C4", "CZ"])
        self.assertEqual(summary["libraries"]["bcic2b"]["mappable"], 3)
        self.assertEqual(summary["global_size"], 5)

    def test_missing_lists_what_a_library_cannot_supply(self):
        _, _, summary = shared.build({"a": ["C3"], "b": ["C3", "C4"]})
        self.assertEqual(summary["missing"]["a"], ["C4"])
        self.assertEqual(summary["missing"]["b"], [])

    def test_global_ids_are_dense_and_unique(self):
        vocabulary, _, _ = shared.build({"a": ["PZ", "C3", "CZ"]})
        self.assertEqual(vocabulary["names"], ["C3", "CZ", "PZ"])
        self.assertEqual(sorted(vocabulary["ids"].values()), [0, 1, 2])


class Loading(unittest.TestCase):
    def write(self, directory, vocabulary):
        import json
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps({"channel_vocabulary": vocabulary}), encoding="utf-8")
        return path

    def test_gapped_ids_are_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, {"0": "C3", "2": "C4"})
            with self.assertRaises(ValueError):
                shared.load_vocabulary(path)

    def test_duplicate_names_are_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, {"0": "C3", "1": "C3"})
            with self.assertRaises(ValueError):
                shared.load_vocabulary(path)


if __name__ == "__main__":
    unittest.main()
