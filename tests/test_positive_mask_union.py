import unittest

import numpy as np

from cd34_pipeline.cell.extraction import (
    compute_marker_peak_threshold,
    compute_positive_masks,
    extract_connected_positive_regions,
)


class PositiveMaskUnionTests(unittest.TestCase):
    def test_marker_peak_threshold_uses_dominant_nonzero_mode(self) -> None:
        marker = np.zeros((8, 8), dtype=np.uint8)
        marker[0:4, 0:4] = 9
        marker[5, 5] = 40
        marker[6, 6] = 80

        self.assertEqual(compute_marker_peak_threshold(marker), 9)

    def test_masks_use_seg_marker_union(self) -> None:
        seg = np.zeros((4, 4, 3), dtype=np.uint8)
        marker = np.zeros((4, 4), dtype=np.uint8)

        seg[0, 0] = [180, 0, 10]  # Seg branch
        seg[0, 1] = [160, 0, 10]  # no minimum-R filter
        marker[3, 3] = 200         # Marker supplements outside Seg foreground

        masks = compute_positive_masks(
            seg,
            marker,
            seg_thresh=120,
            marker_thresh=100,
        )

        self.assertTrue(masks["seg_positive"][0, 0])
        self.assertTrue(masks["seg_positive"][0, 1])
        self.assertTrue(masks["marker_positive"][3, 3])
        self.assertTrue(masks["combined_positive"][0, 0])
        self.assertTrue(masks["combined_positive"][0, 1])
        self.assertTrue(masks["combined_positive"][3, 3])

    def test_seg_only_and_marker_only_regions_enter_extraction(self) -> None:
        seg = np.zeros((14, 14, 3), dtype=np.uint8)
        marker = np.zeros((14, 14), dtype=np.uint8)
        seg[1:5, 1:5] = [180, 0, 10]
        marker[9:13, 9:13] = 200

        regions = extract_connected_positive_regions(
            seg,
            marker,
            seg_thresh=120,
            marker_thresh=100,
            morphology_kernel=1,
            min_area=1,
        )

        self.assertEqual(len(regions), 2)
        centers = {region["center"] for region in regions}
        self.assertEqual(centers, {(2, 2), (10, 10)})

    def test_min_area_does_not_filter_connected_regions(self) -> None:
        seg = np.zeros((8, 8, 3), dtype=np.uint8)
        marker = np.zeros((8, 8), dtype=np.uint8)
        seg[2:5, 2:5] = [180, 0, 10]

        regions = extract_connected_positive_regions(
            seg,
            marker,
            seg_thresh=120,
            marker_thresh=100,
            morphology_kernel=1,
            min_area=1000,
        )

        self.assertEqual(len(regions), 1)
        self.assertGreater(regions[0]["pixel_count"], 0)
        self.assertLess(regions[0]["pixel_count"], 1000)


if __name__ == "__main__":
    unittest.main()
