from types import SimpleNamespace
import unittest
from threading import Event

import numpy as np

from cell.main import Producer, _seg_positive_pixel_count


class _DummyReader:
    def get_tile_filename(self, tile_info):
        return f"tile_{tile_info['row']}_{tile_info['col']}_0_0.png"


class _DummyBucket:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class _DummyDeepLIIF:
    def __init__(self):
        self.cached = []

    def cache_result(self, tile_name, seg_np, marker_np, dapi_np):
        self.cached.append((tile_name, seg_np, marker_np, dapi_np))


class PipelineSegSkipTests(unittest.TestCase):
    def test_seg_positive_count_matches_debug_curve_rule(self) -> None:
        seg = np.zeros((4, 4, 3), dtype=np.uint8)
        seg[0, 0] = [130, 0, 0]

        self.assertEqual(_seg_positive_pixel_count(seg, 120), 1)

    def test_producer_skips_tile_when_seg_has_no_positive_pixels(self) -> None:
        args = SimpleNamespace(debug_region_um=None, seg_thresh=120)
        bucket = _DummyBucket()
        producer = Producer(
            _DummyReader(),
            [],
            args,
            bucket,
            Event(),
            {},
        )
        deepliif = _DummyDeepLIIF()
        tile_info = {"row": 0, "col": 0}
        tile_np = np.zeros((8, 8, 3), dtype=np.uint8)
        seg = np.zeros((8, 8, 3), dtype=np.uint8)
        marker = np.full((8, 8), 255, dtype=np.uint8)

        produced = producer._extract_one(
            deepliif,
            tile_info,
            tile_np,
            {"Seg": seg, "Marker": marker},
        )

        self.assertFalse(produced)
        self.assertEqual(bucket.items, [])
        self.assertEqual(len(deepliif.cached), 1)


if __name__ == "__main__":
    unittest.main()
