import unittest

import numpy as np

from cd34_pipeline.sam2_wrapper.inference import run_sam2_segmentation
from cell.sam2 import SAM2Processor


class _FakePredictor:
    def __init__(self) -> None:
        self.image_shape = None
        self.predict_calls = 0

    def set_image(self, image: np.ndarray) -> None:
        self.image_shape = image.shape[:2]

    def predict(self, mask_input: np.ndarray,
                multimask_output: bool) -> tuple:
        self.predict_calls += 1
        h, w = self.image_shape
        masks = np.zeros((3, h, w), dtype=bool)
        masks[0, 0, 0] = True
        scores = np.array([0.9, 0.2, 0.1], dtype=np.float32)
        low_res_masks = np.zeros((3, 256, 256), dtype=np.float32)
        return masks, scores, low_res_masks


class Sam2FilteringTests(unittest.TestCase):
    def test_small_cluster_is_filtered_before_predictor(self) -> None:
        predictor = _FakePredictor()
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        cluster = np.array([[0, 0]], dtype=np.int64)

        mask, scores, filtered = run_sam2_segmentation(
            predictor,
            image,
            [cluster],
            min_area=1000,
            score_threshold=0.0,
        )

        self.assertEqual(predictor.predict_calls, 0)
        self.assertEqual(mask.dtype, np.uint16)
        self.assertEqual(int(mask.max()), 0)
        self.assertEqual(scores, [])
        self.assertEqual(filtered, [])

    def test_zero_score_threshold_disables_filtering(self) -> None:
        mask = np.array([[1]], dtype=np.uint16)
        scores = [(1, -0.2)]

        filtered_mask, filtered_scores = SAM2Processor._apply_threshold(
            mask, scores, threshold=0.0)

        self.assertIs(filtered_mask, mask)
        self.assertEqual(filtered_scores, scores)

    def test_positive_score_threshold_filters_low_score_mask(self) -> None:
        predictor = _FakePredictor()
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        cluster = np.array([[0, 0]], dtype=np.int64)

        mask, scores, filtered = run_sam2_segmentation(
            predictor,
            image,
            [cluster],
            min_area=1,
            score_threshold=0.95,
        )

        self.assertEqual(int(mask.max()), 0)
        self.assertEqual(scores, [])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0][0], 1)


if __name__ == "__main__":
    unittest.main()
