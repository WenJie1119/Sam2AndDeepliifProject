"""
cell/segmentation.py -- Cell segmentation (connected-region extraction).
"""

import numpy as np


class CellSegmentor:
    """Extracts positive cell regions from DeepLIIF Seg/Marker outputs."""

    def __init__(self, seg_thresh: int = 120,
                 marker_thresh: int = None,
                 marker_percentile_factor: float = 0.9,
                 morphology_kernel: int = 11,
                 min_area: int = 50):
        self.seg_thresh = seg_thresh
        self.marker_thresh = marker_thresh
        self.marker_percentile_factor = marker_percentile_factor
        self.morphology_kernel = morphology_kernel
        self.min_area = min_area

    def extract(self, seg_np: np.ndarray,
                marker_np: np.ndarray) -> tuple[list, list]:
        """
        Extract positive cell regions and compute point-prompt clusters.

        Args:
            seg_np: Segmentation image as numpy array
            marker_np: Marker image as numpy array

        Returns:
            (positive_cells_info, clusters):
                positive_cells_info: list of cell dicts
                clusters: list of (N,2) coordinate arrays for SAM2
                Returns ([], []) if no valid cells found.
        """
        from cd34_pipeline.cell.extraction import (
            extract_connected_positive_regions,
            get_clusters_from_cells,
        )

        positive_cells_info = extract_connected_positive_regions(
            seg_np, marker_np,
            seg_thresh=self.seg_thresh,
            marker_thresh=self.marker_thresh,
            marker_percentile_factor=self.marker_percentile_factor,
            morphology_kernel=self.morphology_kernel,
            min_area=self.min_area,
        )

        if len(positive_cells_info) == 0:
            return [], []

        clusters = get_clusters_from_cells(positive_cells_info)
        return positive_cells_info, clusters
