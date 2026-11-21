"""导联坐标解析：把"坐标编码"这条路径从恒零变成真实几何。

这里断言的是几何事实而不是分数：EEGMMIDB 的 64 个导联必须全部解析到标准头表位置，
C3–C4 间距落在解剖合理区间，双极导联取中点，未收录名字必须如实标为无效。
"""
import unittest

import numpy as np

from eeg_mime_foundation.montage import (bipolar_coordinates, coordinates_for, distance_matrix,
                                         normalise, standard_coordinates)

EEGMMIDB_64 = [
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "FP1", "FPZ", "FP2", "AF7", "AF3", "AFZ",
    "AF4", "AF8", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FT8", "T7", "T8",
    "T9", "T10", "TP7", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO3",
    "POZ", "PO4", "PO8", "O1", "OZ", "O2", "IZ",
]


class CoordinatesTests(unittest.TestCase):
    def test_all_eegmmidb_channels_resolve(self):
        coords, valid = standard_coordinates(EEGMMIDB_64)
        missing = [n for n, ok in zip(EEGMMIDB_64, valid) if not ok]
        self.assertEqual(missing, [])
        self.assertEqual(coords.shape, (64, 3))
        self.assertTrue(np.isfinite(coords).all())

    def test_geometry_is_anatomically_plausible(self):
        coords, valid = standard_coordinates(EEGMMIDB_64)
        distance = distance_matrix(coords, valid)
        c3, c4 = EEGMMIDB_64.index("C3"), EEGMMIDB_64.index("C4")
        self.assertAlmostEqual(float(distance[c3, c4]), .13, delta=.02)
        self.assertLess(float(distance.max()), .25)          # 头表直径上限
        self.assertEqual(float(np.abs(np.diag(distance)).max()), 0.)

    def test_bipolar_uses_midpoint(self):
        coords, valid = bipolar_coordinates(["Fpz-Cz", "Pz-Oz"])
        self.assertTrue(valid.all())
        single, _ = standard_coordinates(["Fpz", "Cz"])
        np.testing.assert_allclose(coords[0], (single[0] + single[1]) / 2, atol=1e-6)

    def test_unknown_channel_is_invalid_not_guessed(self):
        coords, valid = standard_coordinates(["C3", "NOT_AN_ELECTRODE"])
        self.assertTrue(valid[0])
        self.assertFalse(valid[1])
        self.assertEqual(float(np.abs(coords[1]).max()), 0.)

    def test_kind_selects_parsing_rule(self):
        coords, valid = coordinates_for(["Fpz-Cz"], "bipolar_derivation")
        self.assertTrue(valid[0])
        self.assertFalse(coordinates_for(["Fpz-Cz"], "scalp_acquisition_channel")[1][0])

    def test_name_normalisation(self):
        self.assertEqual(normalise("Fp1."), "FP1")
        self.assertEqual(normalise(" eeg c3 "), "EEGC3")


if __name__ == "__main__":
    unittest.main()
