import unittest
from types import SimpleNamespace

import numpy as np

from cell.sam3 import SAM3Processor


class _FakeImageProcessor:
    def __init__(self):
        self.image = None

    def set_image(self, image):
        self.image = image
        return {"image": "state"}


class _FakeModel:
    def __init__(self):
        self.predict_args = None

    def predict_inst(self, **kwargs):
        self.predict_args = kwargs
        masks = np.zeros((3, 4, 4), dtype=bool)
        masks[1, 1:3, 1:3] = True
        scores = np.array([0.2, 0.9, 0.4], dtype=np.float32)
        logits = np.zeros((3, 256, 256), dtype=np.float32)
        return masks, scores, logits


class SAM3BackendTests(unittest.TestCase):
    def test_weighted_prompt_selects_best_candidate(self):
        processor = SAM3Processor.__new__(SAM3Processor)
        processor.score_threshold = 0.1
        processor._processor = _FakeImageProcessor()
        processor._model = _FakeModel()
        item = SimpleNamespace(
            tile_np=np.zeros((4, 4, 3), dtype=np.uint8),
            mask_input=np.zeros((1, 256, 256), dtype=np.float32),
            point_coords=np.array([[2.0, 2.0]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
        )

        mask, scores = processor._segment_weighted_points(item)

        self.assertEqual(mask.dtype, np.uint16)
        self.assertEqual(int(mask.sum()), 4)
        self.assertAlmostEqual(scores[0][1], 0.9, places=6)
        self.assertIn("mask_input", processor._model.predict_args)
        self.assertIn("point_coords", processor._model.predict_args)

    def test_low_score_returns_empty_mask(self):
        masks = np.ones((1, 3, 3), dtype=bool)
        mask, scores = SAM3Processor._select_best_mask(
            masks,
            np.array([0.05], dtype=np.float32),
            score_threshold=0.1,
        )
        self.assertFalse(mask.any())
        self.assertEqual(scores, [])


if __name__ == "__main__":
    unittest.main()
