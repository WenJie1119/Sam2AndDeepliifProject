import unittest

import numpy as np

from cd34_pipeline.sam2_wrapper.weighted_prompt import (
    WeightedPromptConfig,
    _dab_lumen_points,
    _near_white_value_threshold_details,
    build_weighted_prompt,
)
from cell.sam2 import SAM2Processor
from cell.utils import BucketItem


class WeightedPromptBuilderTests(unittest.TestCase):
    def test_seg_marker_logits_and_strong_point(self) -> None:
        seg = np.zeros((12, 12, 3), dtype=np.uint8)
        marker = np.zeros((12, 12), dtype=np.uint8)
        seg[0, 0] = [160, 0, 0]
        seg[0, 1] = [180, 0, 0]
        seg[0, 2] = [200, 0, 0]
        seg[0, 3] = [220, 0, 0]
        seg[5:10, 5:10] = [240, 0, 0]
        seg[0, 4] = [200, 0, 190]  # ambiguous: logit 3 -> 2
        marker[1, 0] = 110

        result = build_weighted_prompt(
            seg,
            marker,
            WeightedPromptConfig(
                marker_thresh=100,
                marker_max=120,
                repair_kernel=1,
                repair_iterations=0,
                uncertain_iterations=0,
                enable_artifact_filter=False,
                enable_small_fragment_filter=False,
                enable_isolated_fragment_filter=False,
                point_min_area=20,
            ),
        )

        self.assertEqual(result.logits[0, :5].tolist(), [1, 2, 3, 4, 2])
        self.assertEqual(int(result.logits[1, 0]), 3)
        self.assertEqual(int(result.logits[6, 6]), 5)
        self.assertEqual(len(result.point_coords), 1)
        self.assertEqual(result.mask_input.shape, (1, 256, 256))

    def test_default_marker_threshold_uses_two_stage_multi_otsu(self) -> None:
        seg = np.zeros((8, 8, 3), dtype=np.uint8)
        marker = np.zeros((8, 8), dtype=np.uint8)
        marker[0:4, 0:4] = 9
        marker[6, 6] = 40

        result = build_weighted_prompt(
            seg,
            marker,
            WeightedPromptConfig(
                repair_kernel=1,
                repair_iterations=0,
                uncertain_iterations=0,
                enable_artifact_filter=False,
                enable_small_fragment_filter=False,
                enable_isolated_fragment_filter=False,
            ),
        )

        self.assertEqual(result.stats["marker_thresh"], 19)
        self.assertEqual(
            result.stats["marker_threshold_source"],
            "auto_two_stage_multiotsu",
        )
        self.assertEqual(result.stats["marker_min_keep_intensity"], 20)
        self.assertEqual(
            result.stats["marker_effective_keep_min_intensity"], 20)
        self.assertEqual(
            result.stats["marker_two_stage_outer_thresholds"], [9, 9])
        self.assertEqual(
            result.stats["marker_two_stage_middle_thresholds"], [10, 10])
        self.assertEqual(result.stats["marker_positive_px"], 1)
        self.assertGreater(int(result.logits[6, 6]), 0)
        self.assertEqual(int(result.logits[0, 0]), -5)

    def test_marker_threshold_keeps_minimum_intensity_20(self) -> None:
        seg = np.zeros((4, 4, 3), dtype=np.uint8)
        marker = np.zeros((4, 4), dtype=np.uint8)
        marker[0, 0] = 19
        marker[0, 1] = 20
        marker[0, 2] = 21

        result = build_weighted_prompt(
            seg,
            marker,
            WeightedPromptConfig(
                marker_thresh=0,
                marker_max=25,
                repair_kernel=1,
                repair_iterations=0,
                uncertain_iterations=0,
                enable_artifact_filter=False,
                enable_small_fragment_filter=False,
                enable_isolated_fragment_filter=False,
            ),
        )

        self.assertEqual(result.stats["marker_thresh"], 19)
        self.assertEqual(
            result.stats["marker_effective_keep_min_intensity"], 20)
        self.assertEqual(result.stats["marker_positive_px"], 2)
        self.assertEqual(int(result.logits[0, 0]), -5)
        self.assertGreaterEqual(int(result.logits[0, 1]), 1)
        self.assertGreaterEqual(int(result.logits[0, 2]), 1)

    def test_near_white_threshold_uses_post_peak_multi_otsu_high_cut(
            self) -> None:
        values = np.asarray(
            [186] * 200
            + [190] * 120
            + [196] * 80
            + [202] * 50
            + [212] * 20,
            dtype=np.int16,
        )
        balanced = np.ones(values.shape, dtype=bool)

        details = _near_white_value_threshold_details(
            values, balanced, fallback_value_min=210)

        self.assertEqual(details["source"], "auto_peak_tail_multiotsu")
        self.assertEqual(details["peak_intensity"], 186)
        self.assertEqual(
            details["threshold"],
            details["multiotsu_thresholds"][1],
        )
        self.assertLess(details["threshold"], 210)

    def test_small_fragment_filter_includes_logit_zero(self) -> None:
        seg = np.zeros((8, 8, 3), dtype=np.uint8)
        marker = np.zeros((8, 8), dtype=np.uint8)
        seg[3, 3] = [140, 0, 0]  # logit 0
        seg[3, 4] = [160, 0, 0]  # logit 1

        result = build_weighted_prompt(
            seg,
            marker,
            WeightedPromptConfig(
                repair_kernel=1,
                repair_iterations=0,
                uncertain_iterations=0,
                enable_artifact_filter=False,
                enable_small_fragment_filter=True,
                small_fragment_max_area=2,
                small_fragment_max_logit=3,
                enable_isolated_fragment_filter=False,
            ),
        )

        self.assertEqual(int(result.logits[3, 3]), -5)
        self.assertEqual(int(result.logits[3, 4]), -5)
        self.assertEqual(result.stats["small_fragment_removed_count"], 1)

    def test_border_foreground_does_not_fill_entire_tile(self) -> None:
        seg = np.zeros((32, 32, 3), dtype=np.uint8)
        marker = np.zeros((32, 32), dtype=np.uint8)
        seg[0:4, 0:4] = [240, 0, 0]

        result = build_weighted_prompt(
            seg,
            marker,
            WeightedPromptConfig(
                repair_kernel=1,
                enable_artifact_filter=False,
                enable_small_fragment_filter=False,
                enable_isolated_fragment_filter=False,
                uncertain_iterations=0,
            ),
        )

        self.assertLess(result.stats["final_nonnegative_px"], 32 * 32 // 2)
        self.assertEqual(int(result.logits[-1, -1]), -5)

    def test_isolated_fragment_filter_removes_small_strong_island(self) -> None:
        seg = np.zeros((32, 32, 3), dtype=np.uint8)
        marker = np.zeros((32, 32), dtype=np.uint8)
        seg[2:4, 2:4] = [240, 0, 0]
        seg[10:16, 10:16] = [240, 0, 0]
        seg[10:12, 20:22] = [240, 0, 0]

        result = build_weighted_prompt(
            seg,
            marker,
            WeightedPromptConfig(
                repair_kernel=1,
                repair_iterations=0,
                uncertain_iterations=0,
                enable_artifact_filter=False,
                enable_small_fragment_filter=False,
                enable_isolated_fragment_filter=True,
                isolated_fragment_max_area=6,
                isolated_fragment_min_gap=6,
                isolated_fragment_neighbor_min_area=20,
            ),
        )

        self.assertEqual(int(result.logits[2, 2]), -5)
        self.assertEqual(int(result.logits[10, 10]), 5)
        self.assertEqual(int(result.logits[10, 20]), 5)
        self.assertEqual(result.stats["isolated_fragment_removed_count"], 1)

    def test_dab_filter_removes_low_dab_prompt_support(self) -> None:
        seg = np.zeros((12, 12, 3), dtype=np.uint8)
        marker = np.zeros((12, 12), dtype=np.uint8)
        tile = np.full((12, 12, 3), 245, dtype=np.uint8)

        seg[2:5, 2:5] = [240, 0, 0]
        tile[2:5, 2:5] = [110, 65, 20]  # strong DAB
        seg[7:10, 7:10] = [240, 0, 0]
        tile[7:10, 7:10] = [205, 185, 145]  # low normalized DAB

        result = build_weighted_prompt(
            seg,
            marker,
            WeightedPromptConfig(
                enable_dab_filter=True,
                dab_min_intensity=160,
                repair_kernel=1,
                repair_iterations=0,
                uncertain_iterations=0,
                enable_artifact_filter=False,
                enable_small_fragment_filter=False,
                enable_isolated_fragment_filter=False,
                enable_dab_lumen_fill=False,
            ),
            tile_rgb=tile,
        )

        self.assertEqual(int(result.logits[3, 3]), 5)
        self.assertEqual(int(result.logits[8, 8]), -5)
        self.assertEqual(result.stats["dab_prompt_kept_px"], 9)
        self.assertEqual(result.stats["dab_prompt_removed_px"], 9)

    def test_dab_filter_requires_hsv_brown_confirmation(self) -> None:
        seg = np.zeros((12, 12, 3), dtype=np.uint8)
        marker = np.zeros((12, 12), dtype=np.uint8)
        tile = np.full((12, 12, 3), 245, dtype=np.uint8)

        seg[2:5, 2:5] = [240, 0, 0]
        tile[2:5, 2:5] = [180, 180, 180]  # HED-DAB positive, not HSV brown.

        result = build_weighted_prompt(
            seg,
            marker,
            WeightedPromptConfig(
                enable_dab_filter=True,
                enable_dab_strong_support=False,
                dab_min_intensity=160,
                repair_kernel=1,
                repair_iterations=0,
                uncertain_iterations=0,
                enable_artifact_filter=False,
                enable_small_fragment_filter=False,
                enable_isolated_fragment_filter=False,
                enable_lumen_points=False,
                enable_dab_lumen_fill=False,
            ),
            tile_rgb=tile,
        )

        self.assertTrue(result.debug["dab_hed_intensity_keep_mask"][3, 3])
        self.assertFalse(result.debug["dab_hsv_brown_keep_mask"][3, 3])
        self.assertFalse(result.debug["dab_keep_mask"][3, 3])
        self.assertEqual(int(result.logits[3, 3]), -5)
        self.assertEqual(result.stats["dab_prompt_removed_px"], 9)

    def test_dab_filter_excludes_seg_blue_even_when_brown(self) -> None:
        seg = np.zeros((8, 8, 3), dtype=np.uint8)
        marker = np.zeros((8, 8), dtype=np.uint8)
        tile = np.full((8, 8, 3), 245, dtype=np.uint8)

        seg[3, 3] = [0, 0, 240]
        marker[3, 3] = 100
        tile[3, 3] = [110, 65, 20]

        result = build_weighted_prompt(
            seg,
            marker,
            WeightedPromptConfig(
                marker_thresh=10,
                marker_max=100,
                enable_dab_filter=True,
                enable_dab_strong_support=False,
                dab_min_intensity=160,
                dab_hsv_brown_seg_blue_dilate_kernel=1,
                repair_kernel=1,
                repair_iterations=0,
                uncertain_iterations=0,
                enable_artifact_filter=False,
                enable_small_fragment_filter=False,
                enable_isolated_fragment_filter=False,
                enable_lumen_points=False,
                enable_dab_lumen_fill=False,
            ),
            tile_rgb=tile,
        )

        self.assertTrue(result.debug["dab_hed_intensity_keep_mask"][3, 3])
        self.assertTrue(result.debug["dab_hsv_brown_keep_mask"][3, 3])
        self.assertTrue(result.debug["dab_seg_blue_excluded_mask"][3, 3])
        self.assertFalse(result.debug["dab_keep_mask"][3, 3])
        self.assertEqual(int(result.logits[3, 3]), -5)

    def test_dapi_lumen_rescue_keeps_low_dab_wrapping_prompt(self) -> None:
        seg = np.zeros((64, 64, 3), dtype=np.uint8)
        marker = np.zeros((64, 64), dtype=np.uint8)
        tile = np.full((64, 64, 3), 245, dtype=np.uint8)
        dapi = np.full((64, 64, 3), 120, dtype=np.uint8)
        tile[0:8, 0:8] = [80, 40, 10]

        seg[18:46, 18:22] = [240, 0, 0]
        seg[18:22, 18:46] = [240, 0, 0]
        seg[42:46, 18:46] = [240, 0, 0]
        seg[18:46, 42:46] = [240, 0, 0]
        dapi[24:40, 24:40] = [0, 0, 0]

        result = build_weighted_prompt(
            seg,
            marker,
            WeightedPromptConfig(
                enable_dab_filter=True,
                enable_dab_strong_support=False,
                dab_min_intensity=160,
                dapi_lumen_dark_max=20,
                repair_kernel=1,
                repair_iterations=0,
                uncertain_iterations=0,
                enable_artifact_filter=False,
                enable_small_fragment_filter=False,
                enable_isolated_fragment_filter=False,
                enable_lumen_points=False,
                dab_lumen_near_wall_kernel=15,
                dab_lumen_ring_kernel=7,
                dab_lumen_min_area=80,
                dab_lumen_min_wall_ratio=0.10,
                dab_lumen_min_boundary_ratio=0.30,
                max_dab_lumen_points=3,
            ),
            tile_rgb=tile,
            dapi=dapi,
        )

        self.assertEqual(result.stats["dapi_lumen_accepted_count"], 1)
        self.assertGreater(result.stats["dab_prompt_kept_by_dapi_lumen_px"], 0)
        self.assertGreater(int(result.logits[20, 20]), 0)
        self.assertGreaterEqual(int(result.logits[32, 32]), 2)
        self.assertTrue(any(component.get("kind") == "dapi_lumen"
                            for component in result.point_components))

    def test_low_dab_wrapping_prompt_is_removed_without_dapi_lumen(self) -> None:
        seg = np.zeros((64, 64, 3), dtype=np.uint8)
        marker = np.zeros((64, 64), dtype=np.uint8)
        tile = np.full((64, 64, 3), 245, dtype=np.uint8)
        tile[0:8, 0:8] = [80, 40, 10]

        seg[18:46, 18:22] = [240, 0, 0]
        seg[18:22, 18:46] = [240, 0, 0]
        seg[42:46, 18:46] = [240, 0, 0]
        seg[18:46, 42:46] = [240, 0, 0]

        result = build_weighted_prompt(
            seg,
            marker,
            WeightedPromptConfig(
                enable_dab_filter=True,
                enable_dab_strong_support=False,
                dab_min_intensity=160,
                repair_kernel=1,
                repair_iterations=0,
                uncertain_iterations=0,
                enable_artifact_filter=False,
                enable_small_fragment_filter=False,
                enable_isolated_fragment_filter=False,
                enable_lumen_points=False,
                enable_dab_lumen_fill=False,
            ),
            tile_rgb=tile,
        )

        self.assertEqual(result.stats["dapi_lumen_accepted_count"], 0)
        self.assertEqual(int(result.logits[20, 20]), -5)
        self.assertEqual(result.stats["final_nonnegative_px"], 0)

    def test_strong_dab_support_adds_only_near_prompt_pixels(self) -> None:
        seg = np.zeros((20, 20, 3), dtype=np.uint8)
        marker = np.zeros((20, 20), dtype=np.uint8)
        tile = np.full((20, 20, 3), 245, dtype=np.uint8)
        seg[3, 3] = [240, 0, 0]
        tile[2:5, 2:5] = [110, 65, 20]
        tile[14:17, 14:17] = [110, 65, 20]

        result = build_weighted_prompt(
            seg,
            marker,
            WeightedPromptConfig(
                enable_dab_filter=True,
                enable_dab_strong_support=True,
                dab_min_intensity=160,
                dab_strong_support_neighborhood_kernel=5,
                repair_kernel=1,
                repair_iterations=0,
                uncertain_iterations=0,
                enable_artifact_filter=False,
                enable_small_fragment_filter=False,
                enable_isolated_fragment_filter=False,
                enable_lumen_points=False,
                enable_dab_lumen_fill=False,
            ),
            tile_rgb=tile,
        )

        self.assertEqual(result.stats["dab_prompt_candidate_px"], 1)
        self.assertEqual(result.stats["dab_prompt_added_px"], 8)
        self.assertEqual(result.stats["dab_prompt_blocked_by_context_px"], 9)
        self.assertEqual(int(result.logits[3, 3]), 5)
        self.assertGreater(int(result.logits[2, 2]), 0)
        self.assertEqual(int(result.logits[15, 15]), -5)

    def test_enabled_dab_filter_requires_original_tile(self) -> None:
        seg = np.zeros((4, 4, 3), dtype=np.uint8)
        marker = np.zeros((4, 4), dtype=np.uint8)

        with self.assertRaises(ValueError):
            build_weighted_prompt(
                seg,
                marker,
                WeightedPromptConfig(enable_dab_filter=True),
            )

    def test_lumen_point_added_from_lowres_closed_candidate(self) -> None:
        seg = np.zeros((256, 256, 3), dtype=np.uint8)
        marker = np.zeros((256, 256), dtype=np.uint8)

        seg[40:100, 40:110] = [240, 0, 0]
        seg[50:90, 50:100] = [0, 0, 0]
        seg[68:74, 40:55] = [0, 0, 0]  # small opening to the exterior

        result = build_weighted_prompt(
            seg,
            marker,
            WeightedPromptConfig(
                repair_kernel=1,
                repair_iterations=0,
                uncertain_iterations=0,
                enable_artifact_filter=False,
                enable_small_fragment_filter=False,
                enable_isolated_fragment_filter=False,
                lumen_point_closing_kernel=9,
                lumen_point_min_area=100,
                lumen_point_max_area=3000,
                lumen_point_min_wall_ratio=0.30,
            ),
        )

        lumen_points = [
            component for component in result.point_components
            if component.get("kind") == "lumen"
        ]
        self.assertEqual(result.stats["lumen_point_accepted_count"], 1)
        self.assertEqual(len(lumen_points), 1)
        x, y = lumen_points[0]["point_xy"]
        self.assertTrue(50 <= x < 100)
        self.assertTrue(50 <= y < 90)
        low_x, low_y = lumen_points[0]["point_xy_256"]
        self.assertGreaterEqual(int(result.mask_input[0, low_y, low_x]), 2)
        self.assertGreater(result.stats["lumen_point_filled_px"], 0)
        self.assertEqual(result.point_labels[-1], 1)

    def test_open_lowres_lumen_candidate_is_not_pointed(self) -> None:
        seg = np.zeros((256, 256, 3), dtype=np.uint8)
        marker = np.zeros((256, 256), dtype=np.uint8)

        seg[40:100, 40:110] = [240, 0, 0]
        seg[50:90, 50:100] = [0, 0, 0]
        seg[60:82, 40:66] = [0, 0, 0]  # opening too wide to close

        result = build_weighted_prompt(
            seg,
            marker,
            WeightedPromptConfig(
                repair_kernel=1,
                repair_iterations=0,
                uncertain_iterations=0,
                enable_artifact_filter=False,
                enable_small_fragment_filter=False,
                enable_isolated_fragment_filter=False,
                lumen_point_closing_kernel=7,
                lumen_point_min_area=100,
                lumen_point_max_area=3000,
                lumen_point_min_wall_ratio=0.30,
            ),
        )

        lumen_points = [
            component for component in result.point_components
            if component.get("kind") == "lumen"
        ]
        self.assertEqual(result.stats["lumen_point_accepted_count"], 0)
        self.assertEqual(lumen_points, [])

    def test_dab_lumen_accepts_tile_border_clipped_candidate(self) -> None:
        dab = np.full((64, 64), 30, dtype=np.uint8)
        dab[20:56, 42:47] = 220
        dab[20:25, 42:64] = 220
        dab[51:56, 42:64] = 220

        coords, labels, components, stats, debug = _dab_lumen_points(
            dab,
            existing_lumen_mask=np.zeros(dab.shape, dtype=bool),
            forbidden_mask=np.zeros(dab.shape, dtype=bool),
            config=WeightedPromptConfig(
                enable_dab_lumen_fill=True,
                dab_lumen_near_wall_kernel=9,
                dab_lumen_ring_kernel=7,
                dab_lumen_min_area=20,
                dab_lumen_max_area=2000,
                dab_lumen_min_wall_ratio=0.10,
                dab_lumen_min_border_boundary_ratio=0.20,
                max_dab_lumen_points=2,
            ),
        )

        selected = [component for component in components
                    if component["selected"]]
        self.assertGreaterEqual(stats["dab_lumen_accepted_count"], 1)
        self.assertGreaterEqual(len(coords), 1)
        self.assertTrue(any(component["touches_border"]
                            for component in selected))
        self.assertTrue(np.any(debug["dab_lumen_accepted_mask"][:, -1]))
        self.assertTrue(np.all(labels == 1))

    def test_dab_lumen_accepts_rgb_white_border_candidate(self) -> None:
        dab = np.zeros((64, 64), dtype=np.uint8)
        tile = np.full((64, 64, 3), [180, 170, 155], dtype=np.uint8)
        tile[24:54, 44:64] = [246, 246, 242]

        wall = np.zeros(dab.shape, dtype=bool)
        wall[20:58, 40:45] = True
        wall[20:25, 40:64] = True
        wall[53:58, 40:64] = True

        coords, labels, components, stats, debug = _dab_lumen_points(
            dab,
            existing_lumen_mask=np.zeros(dab.shape, dtype=bool),
            forbidden_mask=np.zeros(dab.shape, dtype=bool),
            config=WeightedPromptConfig(
                enable_dab_lumen_fill=True,
                dab_lumen_near_wall_kernel=11,
                dab_lumen_ring_kernel=7,
                dab_lumen_min_area=100,
                dab_lumen_max_area=2000,
                dab_lumen_min_wall_ratio=0.10,
                dab_lumen_min_border_boundary_ratio=0.20,
                max_dab_lumen_points=2,
            ),
            tile_rgb=tile,
            wall_mask=wall,
        )

        selected = [component for component in components
                    if component["selected"]]
        self.assertEqual(stats["dab_lumen_candidate_source"], "rgb_white")
        self.assertGreaterEqual(stats["dab_lumen_accepted_count"], 1)
        self.assertGreaterEqual(len(coords), 1)
        self.assertTrue(any(component["touches_border"]
                            for component in selected))
        self.assertTrue(any(component["fill_kind"] == "rgb_white"
                            for component in selected))
        self.assertTrue(np.any(debug["dab_lumen_white_mask"][:, -1]))
        self.assertTrue(np.any(debug["dab_lumen_accepted_mask"][:, -1]))
        self.assertTrue(np.all(labels == 1))

    def test_dab_lumen_accepts_macro_closed_candidate(self) -> None:
        dab = np.full((128, 128), 120, dtype=np.uint8)
        dab[45:85, 50:80] = 30
        dab[36:94, 42:47] = 220
        dab[36:41, 42:88] = 220
        dab[89:94, 42:88] = 220
        dab[36:94, 83:88] = 220

        coords, labels, components, stats, debug = _dab_lumen_points(
            dab,
            existing_lumen_mask=np.zeros(dab.shape, dtype=bool),
            forbidden_mask=np.zeros(dab.shape, dtype=bool),
            config=WeightedPromptConfig(
                enable_dab_lumen_fill=True,
                dab_lumen_near_wall_kernel=31,
                dab_lumen_ring_kernel=5,
                dab_lumen_min_area=500,
                dab_lumen_max_area=3000,
                dab_lumen_min_wall_ratio=0.18,
                dab_lumen_min_boundary_ratio=0.45,
                dab_lumen_macro_closing_kernel=31,
                dab_lumen_macro_min_overlap=0.50,
                dab_lumen_macro_min_wall_ratio=0.30,
                max_dab_lumen_points=2,
            ),
        )

        selected = [component for component in components
                    if component["selected"]]
        self.assertEqual(stats["dab_lumen_accepted_count"], 1)
        self.assertEqual(len(coords), 1)
        self.assertTrue(selected[0]["macro_supported"])
        self.assertLess(selected[0]["boundary_ratio"], 0.45)
        self.assertGreater(selected[0]["macro_overlap_ratio"], 0.50)
        self.assertTrue(np.any(debug["dab_lumen_macro_hole_mask"]))
        self.assertTrue(np.all(labels == 1))

    def test_dab_lumen_protects_border_candidate_from_artifact_filter(
            self) -> None:
        seg = np.zeros((96, 96, 3), dtype=np.uint8)
        marker = np.zeros((96, 96), dtype=np.uint8)
        tile = np.full((96, 96, 3), [180, 170, 155], dtype=np.uint8)
        tile[28:78, 66:96] = [246, 246, 242]

        seg[20:86, 56:62] = [180, 0, 0]
        seg[20:26, 56:96] = [180, 0, 0]
        seg[80:86, 56:96] = [180, 0, 0]

        result = build_weighted_prompt(
            seg,
            marker,
            WeightedPromptConfig(
                enable_dab_filter=True,
                enable_dab_strong_support=False,
                dab_min_intensity=125,
                repair_kernel=1,
                repair_iterations=0,
                uncertain_iterations=0,
                artifact_min_area=100,
                artifact_score_threshold=5,
                enable_small_fragment_filter=False,
                enable_isolated_fragment_filter=False,
                enable_lumen_points=False,
                dab_lumen_near_wall_kernel=15,
                dab_lumen_ring_kernel=7,
                dab_lumen_min_area=100,
                dab_lumen_max_area=3000,
                dab_lumen_min_wall_ratio=0.05,
                dab_lumen_min_border_boundary_ratio=0.15,
                dab_lumen_white_value_min=190,
                dab_lumen_white_saturation_max=0.25,
                dab_lumen_white_channel_delta_max=50,
                max_dab_lumen_points=3,
            ),
            tile_rgb=tile,
        )

        self.assertGreaterEqual(result.stats["dab_lumen_accepted_count"], 1)
        self.assertGreater(result.stats["dab_lumen_filter_rescue_px"], 0)
        self.assertEqual(result.stats["artifact_selected_count"], 0)
        self.assertGreaterEqual(int(result.logits[52, 82]), 2)
        self.assertTrue(np.any(
            result.debug["dab_lumen_accepted_mask"][:, -1]))
        self.assertTrue(any(component.get("kind") == "dab_lumen"
                            for component in result.point_components))


class _WeightedPredictor:
    def __init__(self) -> None:
        self.image = None
        self.predict_args = None

    def set_image(self, image: np.ndarray) -> None:
        self.image = image

    def predict(self, **kwargs):
        self.predict_args = kwargs
        height, width = self.image.shape[:2]
        masks = np.zeros((3, height, width), dtype=bool)
        masks[0, 0, 0] = True
        masks[1, 1:3, 1:3] = True
        masks[2, 2:5, 2:5] = True
        scores = np.array([0.2, 0.9, 0.5], dtype=np.float32)
        low_res = np.zeros((3, 256, 256), dtype=np.float32)
        return masks, scores, low_res


class WeightedSam2Tests(unittest.TestCase):
    def test_segment_batch_uses_mask_and_points(self) -> None:
        predictor = _WeightedPredictor()
        processor = SAM2Processor.__new__(SAM2Processor)
        processor._predictor = predictor
        processor.reuse_cache_dir = None
        processor.cache_dir = None
        processor.score_threshold = 0.1

        item = BucketItem(
            tile_np=np.zeros((8, 8, 3), dtype=np.uint8),
            positive_cells_info=[],
            tile_info={},
            tile_name="tile_0_0_0_0",
            mask_input=np.zeros((1, 256, 256), dtype=np.float32),
            point_coords=np.array([[2, 2]], dtype=np.float32),
            point_labels=np.ones((1,), dtype=np.int32),
        )

        [(mask, scores)] = processor.segment_batch([item])

        self.assertIn("mask_input", predictor.predict_args)
        self.assertIn("point_coords", predictor.predict_args)
        self.assertEqual(int((mask > 0).sum()), 4)
        self.assertEqual(scores[0][0], 1)
        self.assertAlmostEqual(scores[0][1], 0.9, places=5)


if __name__ == "__main__":
    unittest.main()
