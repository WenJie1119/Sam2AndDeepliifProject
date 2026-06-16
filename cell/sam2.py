"""
cell/sam2.py -- SAM2 batch segmentation processor.
"""

import os
from typing import Optional

import numpy as np

from cell.utils import AsyncSaver


class SAM2Processor:
    """Wraps SAM2 model for batch point-prompt segmentation."""

    def __init__(self, config: str, checkpoint: str, device: str,
                 batch_size: int = 32, min_area: int = 50,
                 score_threshold: float = 0.1,
                 cache_dir: Optional[str] = None,
                 reuse_cache_dir: Optional[str] = None):
        self.batch_size = batch_size
        self.min_area = min_area
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

    def segment(self, tile_np: np.ndarray, clusters: list,
                tile_name: str = None,
                positive_cells_info: list = None,
                debug_dir: Optional[str] = None) -> tuple[np.ndarray, list]:
        """
        Run SAM2 batch segmentation on a single tile.

        Args:
            tile_np: RGB image (H, W, 3)
            clusters: list of (N,2) coordinate arrays
            debug_dir: If set, switches to the per-prompt path
                (run_sam2_segmentation) so it can write
                ``{debug_dir}/sam2_steps/instance_XXX/*`` for each cluster.
                Slower; only used in single-tile --debug-vis mode.

        Returns:
            (sam_mask, scores): raw mask array and score list
        """
        from cd34_pipeline.sam2_wrapper.inference import (
            run_sam2_segmentation_batch, run_sam2_segmentation,
        )

        if self.reuse_cache_dir is not None:
            if tile_name is None or positive_cells_info is None:
                raise ValueError("tile_name and positive_cells_info are required for SAM2 cache reuse")
            return self._load_cached_result(tile_name, positive_cells_info)

        if debug_dir is not None:
            sam_mask, scores, _ = run_sam2_segmentation(
                self._predictor, tile_np, clusters,
                min_area=self.min_area,
                set_image=True,
                score_threshold=self.score_threshold,
                debug_dir=debug_dir,
            )
            return sam_mask, scores

        sam_mask, scores, _ = run_sam2_segmentation_batch(
            self._predictor, tile_np, clusters,
            min_area=self.min_area,
            set_image=True,
            batch_size=self.batch_size,
            score_threshold=self.score_threshold,
        )
        return sam_mask, scores

    def segment_batch(self, items: list) -> list[tuple[np.ndarray, list]]:
        """
        Run SAM2 multi-image batch segmentation.

        Uses set_image_batch() to encode all images in one ViT forward,
        then processes each image's prompts through the decoder.

        When cache is enabled, inference runs with score_threshold=0 so that
        all instances (including low-confidence ones) are preserved in the
        cached .npy files.  The configured threshold is then re-applied before
        returning results to the pipeline.

        Args:
            items: list of BucketItem (each has .tile_np and .clusters)

        Returns:
            list of (sam_mask, scores) tuples, one per item
        """
        from cd34_pipeline.sam2_wrapper.inference import run_sam2_multi_image_batch

        if self.reuse_cache_dir is not None:
            return [
                self._load_cached_result(item.tile_name, item.positive_cells_info)
                for item in items
            ]

        images = [item.tile_np for item in items]
        clusters_list = [item.clusters for item in items]

        # When caching, run without score filtering to preserve all instances
        infer_threshold = 0.0 if self.cache_dir else self.score_threshold

        raw_results = run_sam2_multi_image_batch(
            self._predictor, images, clusters_list,
            min_area=self.min_area,
            prompt_batch_size=self.batch_size,
            score_threshold=infer_threshold,
        )

        if self.cache_dir is None:
            return [(mask, scores) for mask, scores, _ in raw_results]

        # Cache unfiltered results, then apply threshold for pipeline
        pipeline_results = []
        for item, (mask, scores, _) in zip(items, raw_results):
            self._cache_result(item.tile_name, mask, scores)
            if self.score_threshold > 0:
                mask, scores = self._apply_threshold(
                    mask, scores, self.score_threshold)
            pipeline_results.append((mask, scores))
        return pipeline_results

    def _cache_result(self, tile_name: str, sam_mask: np.ndarray,
                      scores: list) -> None:
        """Save unfiltered SAM2 mask + scores to cache directory."""
        self._saver.submit(
            {'sam_mask': sam_mask, 'scores': scores},
            os.path.join(self.cache_dir, f"{tile_name}.npy"),
            allow_pickle=True,
        )

    def _load_cached_result(self, tile_name: str,
                            positive_cells_info: list) -> tuple[np.ndarray, list]:
        """Load old SAM2 cache and remap selected original region IDs."""
        cache_path = os.path.join(self.reuse_cache_dir, f"{tile_name}.npy")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"SAM2 cache file not found: {cache_path}")

        cached = np.load(cache_path, allow_pickle=True)
        if isinstance(cached, np.ndarray) and cached.shape == ():
            cached = cached.item()
        if not isinstance(cached, dict) or 'sam_mask' not in cached or 'scores' not in cached:
            raise ValueError(f"Invalid SAM2 cache format: {cache_path}")

        old_mask = cached['sam_mask']
        old_scores = cached['scores']
        score_by_old_id = {int(inst_id): float(score) for inst_id, score in old_scores}

        remapped = np.zeros_like(old_mask)
        remapped_scores = []
        selected_old_ids = []

        for new_id, cell in enumerate(positive_cells_info, start=1):
            old_id = int(cell.get('id', new_id))
            score = score_by_old_id.get(old_id)
            if score is None:
                print(f"      [SAM2 cache] {tile_name}: region {old_id} missing in cache [SKIPPED]")
                continue
            if score < self.score_threshold:
                print(f"      [SAM2 cache] {tile_name}: region {old_id} score={score:.4f} "
                      f"[FILTERED: score<{self.score_threshold}]")
                continue

            remapped[old_mask == old_id] = new_id
            remapped_scores.append((new_id, score))
            selected_old_ids.append(old_id)

        if not remapped_scores:
            remapped, remapped_scores, selected_old_ids = self._remap_cached_by_overlap(
                old_mask, score_by_old_id, positive_cells_info)

        print(f"      [SAM2 cache] {tile_name}: reused {len(remapped_scores)} "
              f"instances from old region IDs {selected_old_ids}")
        return remapped, remapped_scores

    def _remap_cached_by_overlap(self, old_mask: np.ndarray,
                                 score_by_old_id: dict[int, float],
                                 positive_cells_info: list) -> tuple[np.ndarray, list, list]:
        """
        Fallback for caches produced by an older extraction order.

        Keep cached SAM2 instances that spatially overlap the new candidate
        prompt union. This avoids trusting stale instance IDs when the candidate
        extraction changed between runs.
        """
        prompt_union = np.zeros(old_mask.shape, dtype=bool)
        for cell in positive_cells_info:
            coords = cell['coords']
            prompt_union[coords[:, 0], coords[:, 1]] = True

        remapped = np.zeros_like(old_mask)
        remapped_scores = []
        selected_old_ids = []
        next_id = 1

        for old_id in sorted(int(i) for i in np.unique(old_mask) if int(i) > 0):
            score = score_by_old_id.get(old_id)
            if score is None or score < self.score_threshold:
                continue
            inst_mask = old_mask == old_id
            overlap_pixels = int(np.logical_and(inst_mask, prompt_union).sum())
            if overlap_pixels < self.min_area:
                continue

            remapped[inst_mask] = next_id
            remapped_scores.append((next_id, score))
            selected_old_ids.append(old_id)
            next_id += 1

        if selected_old_ids:
            print("      [SAM2 cache] IDs did not match current candidate "
                  "regions; used spatial-overlap fallback")

        return remapped, remapped_scores, selected_old_ids

    @staticmethod
    def _apply_threshold(mask: np.ndarray, scores: list,
                         threshold: float = None) -> tuple[np.ndarray, list]:
        """Remove instances below score threshold from mask."""
        if threshold is None:
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
