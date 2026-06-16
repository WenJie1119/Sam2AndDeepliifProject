"""
cell/deepliif.py -- DeepLIIF batch inference processor.
"""

import os
from typing import Optional

import numpy as np

from cell.utils import AsyncSaver


class DeepLIIFProcessor:
    """Wraps DeepLIIFBatchInference for batch Seg/Marker generation."""

    def __init__(self, model_dir: str, device: str,
                 cache_dir: Optional[str] = None):
        from cd34_pipeline.deepliif.batch_inference import DeepLIIFBatchInference

        self._engine = DeepLIIFBatchInference(
            model_dir=model_dir, device=device,
        )
        self.cache_dir = cache_dir
        self._saver: Optional[AsyncSaver] = None
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            self._saver = AsyncSaver(num_workers=1)
            print(f"[DeepLIIF] Cache enabled -> {cache_dir}")
        print(f"[DeepLIIF] Loaded on {device}")

    def process_batch(self, tile_pils: list, batch_size: int,
                      resolution: str = "40x") -> list[dict]:
        """
        Run DeepLIIF batch inference on a list of PIL images.

        Args:
            tile_pils: list of PIL Images
            batch_size: DeepLIIF batch size
            resolution: "10x", "20x", or "40x"

        Returns:
            list of dicts, each with 'Seg' and 'Marker' PIL Images
        """
        return self._engine.inference_batch(
            tile_pils,
            batch_size=batch_size,
            resolution=resolution,
        )

    def cache_result(self, tile_name: str, seg_np: np.ndarray,
                     marker_np: np.ndarray) -> None:
        """Save DeepLIIF Seg/Marker result to cache directory (async)."""
        if self._saver is None:
            return
        self._saver.submit(
            {'Seg': seg_np, 'Marker': marker_np},
            os.path.join(self.cache_dir, f"{tile_name}.npy"),
            allow_pickle=True,
        )

    def shutdown(self) -> None:
        """Flush pending cache writes."""
        if self._saver is not None:
            self._saver.shutdown()
