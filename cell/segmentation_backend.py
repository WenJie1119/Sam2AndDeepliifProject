"""Backend contract and factory for prompt-driven segmentation models."""

from __future__ import annotations

from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np


SegmentationResult: TypeAlias = tuple[np.ndarray, list]


@runtime_checkable
class SegmentationBackend(Protocol):
    """Common interface consumed by the WSI pipeline."""

    def segment_batch(self, items: list) -> list[SegmentationResult]:
        """Segment a batch of prompt-bearing tile items."""

    def shutdown(self) -> None:
        """Release resources and flush pending writes."""


def create_segmentation_backend(
    backend: str,
    *,
    config: str,
    checkpoint: str,
    device: str,
    batch_size: int,
    cache_dir: str | None = None,
    reuse_cache_dir: str | None = None,
) -> SegmentationBackend:
    """Create a segmentation backend without importing optional models early."""
    normalized = backend.strip().lower()
    if normalized == "sam2":
        from cell.sam2 import SAM2Processor

        return SAM2Processor(
            config=config,
            checkpoint=checkpoint,
            device=device,
            batch_size=batch_size,
            cache_dir=cache_dir,
            reuse_cache_dir=reuse_cache_dir,
        )
    raise ValueError(
        f"Unsupported segmentation backend: {backend!r}. "
        "Available backends: sam2"
    )
