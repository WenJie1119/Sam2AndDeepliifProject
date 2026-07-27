"""SAM3 interactive-image backend for weighted mask and point prompts."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


class SAM3Processor:
    """Adapt SAM3 instance interactivity to the pipeline backend contract."""

    def __init__(
        self,
        config: str,
        checkpoint: str,
        device: str,
        batch_size: int = 1,
        score_threshold: float = 0.1,
        cache_dir: Optional[str] = None,
        reuse_cache_dir: Optional[str] = None,
    ):
        del config
        if cache_dir is not None or reuse_cache_dir is not None:
            raise NotImplementedError(
                "SAM3 cache/reuse is not implemented yet; run without "
                "--cache-sam2 and --reuse-sam2-cache"
            )
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(
            device="cpu",
            checkpoint_path=checkpoint,
            load_from_HF=False,
            enable_inst_interactivity=True,
        )
        model = model.to(device)
        model.eval()

        self.batch_size = batch_size
        self.score_threshold = score_threshold
        self._model = model
        self._processor = Sam3Processor(model)
        print(f"[SAM3] Loaded on {device}")

    def segment_batch(self, items: list) -> list[tuple[np.ndarray, list]]:
        """Segment tiles serially through the common batch interface."""
        return [self._segment_weighted_points(item) for item in items]

    def _segment_weighted_points(self, item) -> tuple[np.ndarray, list]:
        if item.mask_input is None:
            raise ValueError("SAM3 weighted-points mode requires mask_input")

        state = self._processor.set_image(item.tile_np)
        predict_args = {
            "inference_state": state,
            "mask_input": item.mask_input,
            "multimask_output": True,
        }
        if item.point_coords is not None and len(item.point_coords) > 0:
            predict_args["point_coords"] = item.point_coords
            predict_args["point_labels"] = item.point_labels

        masks, candidate_scores, _ = self._model.predict_inst(**predict_args)
        return self._select_best_mask(
            masks,
            candidate_scores,
            score_threshold=self.score_threshold,
        )

    @staticmethod
    def _select_best_mask(
        masks: np.ndarray,
        candidate_scores: np.ndarray,
        *,
        score_threshold: float,
    ) -> tuple[np.ndarray, list]:
        scores_array = np.asarray(candidate_scores, dtype=np.float32)
        if scores_array.size == 0:
            raise ValueError("SAM3 returned no candidate masks")
        best_index = int(np.argmax(scores_array))
        best_score = float(scores_array[best_index])
        best_mask = np.asarray(masks[best_index], dtype=bool)
        instance_mask = np.zeros(best_mask.shape, dtype=np.uint16)
        if best_score >= score_threshold:
            instance_mask[best_mask] = 1
            return instance_mask, [(1, best_score)]
        return instance_mask, []

    def shutdown(self) -> None:
        """Release model references held by the backend."""
        self._processor = None
        self._model = None
