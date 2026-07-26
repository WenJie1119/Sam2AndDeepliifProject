"""
cell/sam2.py -- SAM2 batch segmentation processor.
"""

import os
from typing import Optional

import numpy as np

from cell.utils import AsyncSaver


class SAM2Processor:
    """Wraps SAM2 model for weighted mask + point-prompt segmentation."""

    def __init__(self, config: str, checkpoint: str, device: str,
                 batch_size: int = 32, score_threshold: float = 0.1,
                 cache_dir: Optional[str] = None,
                 reuse_cache_dir: Optional[str] = None):
        self.batch_size = batch_size
        self.score_threshold = score_threshold
        self.cache_dir = cache_dir
        self.reuse_cache_dir = reuse_cache_dir
        self._saver: Optional[AsyncSaver] = None

        if cache_dir is not None and reuse_cache_dir is not None:
            raise ValueError("--cache-sam2 and --reuse-sam2-cache cannot be used together")

        if reuse_cache_dir is not None:
            if not os.path.isdir(reuse_cache_dir):
                raise FileNotFoundError(f"SAM2 cache directory not found: {reuse_cache_dir}")
            self._predictor = None
            print(f"[SAM2] Reusing cached masks from -> {reuse_cache_dir}")
            return

        from cd34_pipeline.sam2_wrapper.model_loader import load_sam2
        self._predictor = load_sam2(config, checkpoint, device)

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            self._saver = AsyncSaver(num_workers=1)
            print(f"[SAM2] Cache enabled -> {cache_dir}")
        print(f"[SAM2] Loaded on {device}")

    def segment_batch(self, items: list) -> list[tuple[np.ndarray, list]]:
        """Run weighted mask + point prompts for each tile."""
        return [self._segment_weighted_points(item) for item in items]

    def _segment_weighted_points(self, item) -> tuple[np.ndarray, list]:
        """Run one tile with a dense weighted mask plus positive points."""
        if item.mask_input is None:
            raise ValueError("weighted-points mode requires mask_input")
        if self.reuse_cache_dir is not None:
            return self._load_cached_weighted_result(item.tile_name)

        self._predictor.set_image(item.tile_np)
        predict_args = {
            "mask_input": item.mask_input,
            "multimask_output": True,
        }
        point_coords = item.point_coords
        if point_coords is not None and len(point_coords) > 0:
            predict_args["point_coords"] = point_coords
            predict_args["point_labels"] = item.point_labels

        masks, candidate_scores, low_res_masks = self._predictor.predict(
            **predict_args)
        candidate_scores = np.asarray(candidate_scores, dtype=np.float32)
        best_idx = int(np.argmax(candidate_scores))
        best_score = float(candidate_scores[best_idx])
        best_mask = masks[best_idx].astype(bool)
        instance_mask = np.zeros(best_mask.shape, dtype=np.uint16)
        instance_mask[best_mask] = 1
        scores = [(1, best_score)]

        if item.prompt_debug_dir is not None:
            self._save_weighted_prediction_debug(
                item, masks, candidate_scores, low_res_masks, best_idx)

        if self.cache_dir is not None:
            self._cache_weighted_result(item.tile_name, instance_mask, scores)
        if self.score_threshold > 0 and best_score < self.score_threshold:
            instance_mask.fill(0)
            scores = []
        return instance_mask, scores

    @staticmethod
    def _save_weighted_prediction_debug(item, masks: np.ndarray,
                                        scores: np.ndarray,
                                        low_res_masks: np.ndarray,
                                        best_idx: int) -> None:
        import cv2
        import json

        output_dir = item.prompt_debug_dir
        os.makedirs(output_dir, exist_ok=True)
        for name in os.listdir(output_dir):
            if name.startswith("step4_weighted_"):
                path = os.path.join(output_dir, name)
                if os.path.isfile(path):
                    os.remove(path)

        image = item.tile_np[:, :, :3]
        for index, (mask, score) in enumerate(zip(masks, scores)):
            pixels = mask.astype(bool)
            overlay = image.copy()
            overlay[pixels] = (
                overlay[pixels].astype(np.float32) * 0.45
                + np.array([255, 0, 0], dtype=np.float32) * 0.55
            ).clip(0, 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    f"step4_weighted_candidate_{index}_score_{score:.4f}.png",
                ),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

        np.save(
            os.path.join(output_dir, "step4_weighted_low_res_masks.npy"),
            low_res_masks,
        )
        summary = {
            "prompt_mode": "weighted-points",
            "point_count": int(
                len(item.point_coords) if item.point_coords is not None else 0),
            "scores": [float(score) for score in scores],
            "candidate_areas": [int(mask.astype(bool).sum()) for mask in masks],
            "best_idx": int(best_idx),
            "best_score": float(scores[best_idx]),
            "best_area": int(masks[best_idx].astype(bool).sum()),
        }
        with open(
            os.path.join(output_dir, "step4_weighted_summary.json"),
            "w", encoding="utf-8",
        ) as output:
            json.dump(summary, output, indent=2)

    def _load_cached_weighted_result(self, tile_name: str) -> tuple[np.ndarray, list]:
        cache_path = os.path.join(self.reuse_cache_dir, f"{tile_name}.npy")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"SAM2 cache file not found: {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        if isinstance(cached, np.ndarray) and cached.shape == ():
            cached = cached.item()
        if not isinstance(cached, dict):
            raise ValueError(f"Invalid SAM2 cache format: {cache_path}")
        if cached.get("prompt_mode") != "weighted-points":
            raise ValueError(
                f"SAM2 cache was not produced by weighted-points mode: "
                f"{cache_path}")
        mask = cached.get("sam_mask")
        scores = cached.get("scores")
        if mask is None or scores is None:
            raise ValueError(f"Invalid SAM2 cache format: {cache_path}")
        return self._apply_threshold(mask, list(scores), self.score_threshold)

    def _cache_weighted_result(self, tile_name: str, sam_mask: np.ndarray,
                               scores: list) -> None:
        self._saver.submit(
            {
                "prompt_mode": "weighted-points",
                "sam_mask": sam_mask,
                "scores": scores,
            },
            os.path.join(self.cache_dir, f"{tile_name}.npy"),
            allow_pickle=True,
        )

    @staticmethod
    def _apply_threshold(mask: np.ndarray, scores: list,
                         threshold: float = None) -> tuple[np.ndarray, list]:
        """Remove instances below score threshold from mask."""
        if threshold is None or threshold <= 0:
            return mask, scores
        kept_scores = []
        mask = mask.copy()
        for inst_id, score in scores:
            if score < threshold:
                mask[mask == inst_id] = 0
            else:
                kept_scores.append((inst_id, score))
        return mask, kept_scores

    def segment_batch_apply_threshold(self, mask: np.ndarray,
                                      scores: list) -> tuple[np.ndarray, list]:
        """Apply this processor's score_threshold to a cached result."""
        return self._apply_threshold(mask, scores, self.score_threshold)

    def shutdown(self) -> None:
        """Flush pending cache writes."""
        if self._saver is not None:
            self._saver.shutdown()
