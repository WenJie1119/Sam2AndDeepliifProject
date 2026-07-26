import unittest

import numpy as np

from cd34_pipeline.cell.extraction import (
    compute_marker_multi_otsu_thresholds,
    compute_marker_range_multi_otsu_thresholds,
    compute_marker_range_otsu_threshold,
    compute_marker_two_stage_multi_otsu_details,
    compute_marker_two_stage_multi_otsu_threshold,
)


class MarkerThresholdTests(unittest.TestCase):
    def test_multi_otsu_ignores_zero_background(self) -> None:
        marker = np.zeros((30, 30), dtype=np.uint8)
        marker[0:10, :] = 10
        marker[10:20, :] = 80
        marker[20:30, :] = 180

        low_threshold, high_threshold = compute_marker_multi_otsu_thresholds(
            marker)

        self.assertEqual(low_threshold, 10)
        self.assertEqual(high_threshold, 80)
        self.assertEqual(int((marker > low_threshold).sum()), 600)
        self.assertEqual(int((marker > high_threshold).sum()), 300)

    def test_range_otsu_splits_inside_requested_bounds(self) -> None:
        marker = np.zeros((30, 30), dtype=np.uint8)
        marker[0:12, :] = 18
        marker[12:20, :] = 26
        marker[20:30, :] = 38

        threshold = compute_marker_range_otsu_threshold(
            marker,
            min_intensity=18,
            max_intensity=38,
        )

        self.assertEqual(threshold, 26)
        self.assertEqual(int(((marker >= 18) & (marker <= threshold)).sum()),
                         600)
        self.assertEqual(int(((marker > threshold) & (marker <= 38)).sum()),
                         300)

    def test_range_multi_otsu_splits_inside_requested_bounds(self) -> None:
        marker = np.zeros((30, 30), dtype=np.uint8)
        marker[0:10, :] = 18
        marker[10:20, :] = 26
        marker[20:30, :] = 38

        low_threshold, high_threshold = (
            compute_marker_range_multi_otsu_thresholds(
                marker,
                min_intensity=18,
                max_intensity=38,
            )
        )

        self.assertEqual(low_threshold, 18)
        self.assertEqual(high_threshold, 26)
        self.assertEqual(int(((marker >= 18)
                              & (marker <= low_threshold)).sum()), 300)
        self.assertEqual(int(((marker > low_threshold)
                              & (marker <= high_threshold)).sum()), 300)
        self.assertEqual(int(((marker > high_threshold)
                              & (marker <= 38)).sum()), 300)

    def test_two_stage_multi_otsu_keeps_orange_and_above(self) -> None:
        marker = np.zeros((50, 20), dtype=np.uint8)
        marker[0:10, :] = 5
        marker[10:20, :] = 18
        marker[20:30, :] = 24
        marker[30:40, :] = 31
        marker[40:50, :] = 60

        details = compute_marker_two_stage_multi_otsu_details(marker)
        threshold = compute_marker_two_stage_multi_otsu_threshold(marker)

        self.assertEqual(details["outer_thresholds"], [5, 31])
        self.assertEqual(details["middle_thresholds"], [18, 24])
        self.assertEqual(threshold, 24)
        self.assertEqual(int((marker > threshold).sum()), 400)

    def test_two_stage_threshold_keeps_at_least_marker_20(self) -> None:
        marker = np.zeros((8, 8), dtype=np.uint8)
        marker[0:4, 0:4] = 9
        marker[6, 6] = 40

        details = compute_marker_two_stage_multi_otsu_details(marker)
        threshold = compute_marker_two_stage_multi_otsu_threshold(marker)

        self.assertEqual(details["marker_min_keep_intensity"], 20)
        self.assertLess(details["raw_keep_threshold"], 19)
        self.assertEqual(details["keep_threshold"], 19)
        self.assertEqual(threshold, 19)
        self.assertFalse(19 > threshold)
        self.assertTrue(20 > threshold)
        self.assertTrue(bool((marker > threshold)[6, 6]))


if __name__ == "__main__":
    unittest.main()
