"""Event-boundary and label semantics tests for the real-data importers."""
import unittest

import mne
import numpy as np

from eeg_mime_foundation.realdata import mi_trial_intervals, sleep_epoch_intervals
from eeg_mime_foundation.baselines import sleep_features


class RealDataTests(unittest.TestCase):
    def test_mi_respects_event_and_run_boundaries(self):
        annotation = mne.Annotations([0, 5, 10, 18], [4.1, 3.9, 4.1, 4.1], ["T1", "T2", "T0", "T2"])
        kept, excluded = mi_trial_intervals(annotation, 160, 20 * 160)
        self.assertEqual(kept, [(0, 80, 640, 0)])
        self.assertEqual(len(excluded), 2)

    def test_sleep_stage_merge_and_unknown_exclusion(self):
        annotation = mne.Annotations([0, 30, 60, 90, 120, 150, 180], [30] * 7,
            ["Sleep stage W", "Sleep stage 1", "Sleep stage 2", "Sleep stage 3", "Sleep stage 4", "Sleep stage R", "Sleep stage ?"])
        kept, excluded, _ = sleep_epoch_intervals(annotation, 100, 210 * 100)
        self.assertEqual([item[-1] for item in kept], [0, 1, 2, 3, 3, 4])
        self.assertTrue(all(item[3] - item[2] == 3000 for item in kept))
        self.assertEqual(len(excluded), 1)

    def test_sleep_rejects_overlapping_scores(self):
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            sleep_epoch_intervals(mne.Annotations([0, 15], [30, 30], ["Sleep stage 2"] * 2), 100, 6000)

    def test_sleep_crops_long_wake_to_thirty_minutes(self):
        annotation = mne.Annotations([0, 3600, 3660], [3600, 60, 3600],
                                     ["Sleep stage W", "Sleep stage 2", "Sleep stage W"])
        kept, _, crop = sleep_epoch_intervals(annotation, 100, 726000)
        self.assertEqual(crop, [1800, 5460])
        self.assertEqual(len(kept), 122)

    def test_sleep_feature_frequency_semantics(self):
        t = np.arange(3000) / 100
        x = np.sin(2 * np.pi * 10 * t)[None, None, :] * 1e-5
        f = sleep_features(x)
        self.assertEqual(f.shape, (1, 11))
        self.assertTrue(np.isfinite(f).all())
        self.assertGreater(f[0, 7], .99)  # Relative alpha power, not an arbitrary channel index.


if __name__ == "__main__":
    unittest.main()


class WindowIssueTests(unittest.TestCase):
    """全平窗口要被剔除，单通道平直要保留。"""

    def test_flat_across_all_channels_is_rejected(self):
        import numpy as np
        from eeg_mime_foundation.realdata import window_issue
        x = np.full((2, 3000), -1.831e-07)
        self.assertEqual(window_issue(x), "flat_channels")

    def test_single_flat_channel_is_kept(self):
        import numpy as np
        from eeg_mime_foundation.realdata import window_issue
        x = np.zeros((2, 3000))
        x[0] = np.random.default_rng(0).normal(size=3000)
        self.assertIsNone(window_issue(x))

    def test_non_finite_is_rejected(self):
        import numpy as np
        from eeg_mime_foundation.realdata import window_issue
        x = np.zeros((2, 3000))
        x[0, 5] = np.nan
        self.assertEqual(window_issue(x), "non_finite")
