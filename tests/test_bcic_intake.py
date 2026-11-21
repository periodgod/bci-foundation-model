"""BCIC IV 2a/2b 接入约定。

真实踩到的坑：2a 的 gdf 只有 5 个导联带电极名，其余 17 个是 `EEG-0`…`EEG-16`。
若按位置硬补成 FC3/CP3 之类，坐标会错、图偏置会错，而训练指标不会报错。
所以这里断言的是"宁可丢通道也不猜"。
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_mime_foundation import realdata  # noqa: E402


class Channels(unittest.TestCase):
    def test_2a_keeps_only_self_labelled_electrodes(self):
        names = ["EEG-Fz"] + [f"EEG-{i}" for i in range(17)] + \
                ["EEG-C3", "EEG-Cz", "EEG-C4", "EEG-Pz"] + ["EOG-left", "EOG-central", "EOG-right"]
        # 真实文件顺序：Fz, C3, Cz, C4, Pz 夹在占位符之间。
        names = ["EEG-Fz", "EEG-0", "EEG-1", "EEG-2", "EEG-3", "EEG-4", "EEG-5", "EEG-C3",
                 "EEG-6", "EEG-Cz", "EEG-7", "EEG-C4", "EEG-8", "EEG-9", "EEG-10", "EEG-11",
                 "EEG-12", "EEG-13", "EEG-14", "EEG-Pz", "EEG-15", "EEG-16",
                 "EOG-left", "EOG-central", "EOG-right"]
        keep, labels, placeholders, others = realdata.bcic_channels(names, "EEG-")
        self.assertEqual(labels, ["FZ", "C3", "CZ", "C4", "PZ"])
        self.assertEqual(keep, [0, 7, 9, 11, 19])
        self.assertEqual(len(placeholders), 17)
        self.assertEqual(others, ["EOG-left", "EOG-central", "EOG-right"])

    def test_2b_keeps_three_named_channels(self):
        names = ["EEG:C3", "EEG:Cz", "EEG:C4", "EOG:ch01", "EOG:ch02", "EOG:ch03"]
        keep, labels, placeholders, others = realdata.bcic_channels(names, "EEG:")
        self.assertEqual(labels, ["C3", "CZ", "C4"])
        self.assertEqual(keep, [0, 1, 2])
        self.assertEqual(placeholders, [])
        self.assertEqual(len(others), 3)


class Intervals(unittest.TestCase):
    def annotations(self, onsets, descriptions):
        return SimpleNamespace(onset=np.array(onsets, dtype=float), description=descriptions)

    def test_beep_is_the_anchor_and_hand_codes_become_labels(self):
        # cue 在 0 s，beep 在 3 s；窗口从 beep 起 4 s。
        ann = self.annotations([0, 3, 10, 13], ["769", "768", "770", "768"])
        intervals, excluded, source = realdata.bcic_trials(
            ann, 160, 160 * 40, 0.0, 4.0, {"769": 0, "770": 1})
        self.assertEqual(source, "gdf_event_code")
        self.assertEqual(excluded, [])
        self.assertEqual([item[3] for item in intervals], [0, 1])
        self.assertEqual(intervals[0][1:3], (480, 1120))

    def test_foot_and_tongue_are_dropped_not_relabelled(self):
        ann = self.annotations([0, 3, 10, 13], ["771", "768", "772", "768"])
        intervals, excluded, _ = realdata.bcic_trials(
            ann, 160, 160 * 40, 0.0, 4.0, {"769": 0, "770": 1})
        self.assertEqual(intervals, [])
        self.assertEqual([item["reason"] for item in excluded], ["not_a_hand_class"] * 2)

    def test_window_past_record_end_is_excluded_not_truncated(self):
        # 33 s 记录 + 30 s 的 beep：窗口落到 30–34 s，整段剔除。
        ann = self.annotations([27, 30], ["769", "768"])
        intervals, excluded, _ = realdata.bcic_trials(
            ann, 160, 160 * 33, 0.0, 4.0, {"769": 0, "770": 1})
        self.assertEqual(intervals, [])
        self.assertEqual(excluded[0]["reason"], "window_outside_record")

    def test_truth_file_drives_labels_when_event_codes_are_absent(self):
        # E session：只有 768，标签来自 classlabel（1..4，+768 就是事件码）。
        ann = self.annotations([3, 13, 23, 33], ["768"] * 4)
        intervals, excluded, source = realdata.bcic_trials(
            ann, 160, 160 * 40, 0.0, 4.0, {"769": 0, "770": 1}, truth=[1, 2, 3, 1])
        self.assertEqual(source, "official_truth_mat")
        self.assertEqual([item[3] for item in intervals], [0, 1, 0])
        self.assertEqual([item["reason"] for item in excluded], ["not_a_hand_class"])

    def test_label_count_mismatch_raises_instead_of_shifting(self):
        ann = self.annotations([3, 13], ["768", "768"])
        with self.assertRaises(ValueError):
            realdata.bcic_trials(ann, 160, 160 * 40, 0.0, 4.0, {"769": 0, "770": 1}, truth=[1])


class Specs(unittest.TestCase):
    def test_subject_and_session_are_split_from_the_stem(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("A01T.gdf", "A01E.gdf", "A02T.gdf", "A02E.gdf", "A03T.gdf",
                         "A03E.gdf", "A04T.gdf", "A04E.gdf", "A05T.gdf", "A05E.gdf"):
                (root / name).write_bytes(b"")
            spec = realdata.corpus_spec("bcic2a", root, seed=0)
        self.assertEqual(len(spec["files"]), 10)
        self.assertEqual(sorted(spec["subjects"]), ["A01", "A02", "A03", "A04", "A05"])
        self.assertEqual(set(spec["subjects"].values()), {"train", "validation", "test"})

    def test_missing_gdf_raises_instead_of_empty_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                realdata.corpus_spec("bcic2b", directory, seed=0)

    def test_bcic_only_accepts_corpus_scope(self):
        with self.assertRaises(ValueError):
            realdata.source_spec("bcic2a", "smoke")


if __name__ == "__main__":
    unittest.main()
