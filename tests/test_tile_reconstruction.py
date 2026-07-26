import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cd34_pipeline.io.tile_reconstruction import (
    _extract_tile_polygons,
    export_geojson,
)
from cell.postprocess import PostProcessor, _fill_instance_holes


class CenterValidStitchingTests(unittest.TestCase):
    def test_fill_instance_holes_fills_only_enclosed_background(self) -> None:
        mask = np.zeros((16, 16), dtype=np.uint16)
        mask[3:13, 3:13] = 1
        mask[6:10, 6:10] = 0
        mask[0:4, 14:16] = 2

        filled, filled_px = _fill_instance_holes(mask)

        self.assertEqual(filled_px, 16)
        self.assertTrue(np.all(filled[6:10, 6:10] == 1))
        self.assertEqual(int(filled[0, 0]), 0)
        self.assertTrue(np.all(filled[0:4, 14:16] == 2))

    def test_postprocessor_keeps_uncropped_mask_for_stitching(self) -> None:
        pp = PostProcessor(
            output_dir=".",
            tile_size=16,
            overlap=4,
            stitch_mode="center-valid",
            tile_records=[
                {"row": 0, "col": 0},
                {"row": 0, "col": 1},
            ],
        )
        sam_mask = np.zeros((16, 16), dtype=np.uint16)
        sam_mask[4:12, 8:16] = 1

        processed = pp.merge_and_process(
            sam_mask,
            [(1, 0.9)],
            [],
            "tile_0_0_0_0",
            tile_info={"actual_w": 16, "actual_h": 16},
        )

        self.assertTrue(processed)
        self.assertTrue(np.any(pp.masks[(0, 0)][:, 14:16] > 0))
        poly_bounds = [
            poly.bounds
            for polys in pp.poly_map.values()
            for poly in polys
        ]
        self.assertTrue(poly_bounds)
        self.assertLessEqual(max(bounds[2] for bounds in poly_bounds), 13)

    def test_center_valid_export_uses_overlap_masks_to_merge_seam(self) -> None:
        tile_size = 16
        stride = 12
        mask_a = np.zeros((tile_size, tile_size), dtype=np.uint8)
        mask_b = np.zeros((tile_size, tile_size), dtype=np.uint8)
        mask_a[4:12, 8:16] = 1
        mask_b[4:12, 0:10] = 1

        cropped_a = mask_a.copy()
        cropped_a[:, 14:] = 0
        cropped_b = mask_b.copy()
        cropped_b[:, :2] = 0

        poly_map = {}
        for inst_id, poly in _extract_tile_polygons(cropped_a, 0, 0):
            poly_map[(0, 0, inst_id)] = [poly]
        for inst_id, poly in _extract_tile_polygons(cropped_b, stride, 0):
            poly_map[(0, 1, inst_id)] = [poly]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "merged.geojson"
            export_geojson(
                tile_dir=None,
                output_path=str(output_path),
                tile_size=tile_size,
                stride=stride,
                contour_tolerance=0,
                min_area=0,
                poly_map=poly_map,
                masks={(0, 0): mask_a, (0, 1): mask_b},
                merge_mode="center-valid",
            )

            with output_path.open() as f:
                features = json.load(f)

        self.assertEqual(len(features), 1)
        coords = features[0]["geometry"]["coordinates"][0]
        xs = [pt[0] for pt in coords]
        self.assertEqual(min(xs), 8)
        self.assertEqual(max(xs), 21)

    def test_center_valid_raw_skips_cross_tile_merge(self) -> None:
        tile_size = 16
        stride = 12
        mask_a = np.zeros((tile_size, tile_size), dtype=np.uint8)
        mask_b = np.zeros((tile_size, tile_size), dtype=np.uint8)
        mask_a[4:12, 8:16] = 1
        mask_b[4:12, 0:10] = 1

        cropped_a = mask_a.copy()
        cropped_a[:, 14:] = 0
        cropped_b = mask_b.copy()
        cropped_b[:, :2] = 0

        poly_map = {}
        for inst_id, poly in _extract_tile_polygons(cropped_a, 0, 0):
            poly_map[(0, 0, inst_id)] = [poly]
        for inst_id, poly in _extract_tile_polygons(cropped_b, stride, 0):
            poly_map[(0, 1, inst_id)] = [poly]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "raw.geojson"
            export_geojson(
                tile_dir=None,
                output_path=str(output_path),
                tile_size=tile_size,
                stride=stride,
                contour_tolerance=0,
                min_area=0,
                poly_map=poly_map,
                masks={(0, 0): mask_a, (0, 1): mask_b},
                merge_mode="center-valid-raw",
            )

            with output_path.open() as f:
                features = json.load(f)

        self.assertEqual(len(features), 2)

    def test_strong_overlap_merges_large_pieces_despite_far_centroids(self) -> None:
        tile_size = 32
        stride = 24
        overlap = tile_size - stride
        mask_a = np.ones((tile_size, tile_size), dtype=np.uint8)
        mask_b = np.ones((tile_size, tile_size), dtype=np.uint8)

        cropped_a = mask_a.copy()
        cropped_a[:, tile_size - overlap // 2:] = 0
        cropped_b = mask_b.copy()
        cropped_b[:, :overlap // 2] = 0

        poly_map = {}
        for inst_id, poly in _extract_tile_polygons(cropped_a, 0, 0):
            poly_map[(0, 0, inst_id)] = [poly]
        for inst_id, poly in _extract_tile_polygons(cropped_b, stride, 0):
            poly_map[(0, 1, inst_id)] = [poly]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "merged.geojson"
            export_geojson(
                tile_dir=None,
                output_path=str(output_path),
                tile_size=tile_size,
                stride=stride,
                contour_tolerance=0,
                min_area=0,
                poly_map=poly_map,
                masks={(0, 0): mask_a, (0, 1): mask_b},
                merge_mode="center-valid",
            )

            with output_path.open() as f:
                features = json.load(f)

        self.assertEqual(len(features), 1)


if __name__ == "__main__":
    unittest.main()
