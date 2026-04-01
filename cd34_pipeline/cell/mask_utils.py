#!/usr/bin/env python3
"""
mask_utils.py — 掩码与几何操作模块

包含：
- 颜色生成
- 连通域分析
- 掩码提取
- Bounding box 计算与合并
- 细胞重叠检测与合并
"""

import numpy as np
import cv2
from typing import Optional


def generate_distinct_colors(n: int) -> list[tuple[int, int, int]]:
    """
    Generate n colors - all red (255, 0, 0) for uniform mask visualization.
    
    Args:
        n: Number of colors to generate
        
    Returns:
        List of (R, G, B) tuples
    """
    if n == 0:
        return []
    # Return all red colors instead of distinct colors
    return [(255, 0, 0)] * n


def get_clusters_from_mask_image(mask_array: np.ndarray, min_area: int = 10) -> list[np.ndarray]:
    """
    Generate clusters from a binary mask array using connected components.
    
    Args:
        mask_array: Binary mask array (H, W) or (H, W, C)
        min_area: Minimum area threshold for clusters
        
    Returns:
        List of cluster coordinate arrays, each is np.ndarray of shape (N, 2) with (y, x) coords
    """
    # Handle dimensions (H, W) or (H, W, C)
    if mask_array.ndim == 3:
        if mask_array.shape[2] == 4:  # RGBA
            mask_array = mask_array[..., 3]
        elif mask_array.shape[2] == 3:  # RGB
            mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
            
    # Threshold
    _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
    
    # Connected components
    num_labels, labels = cv2.connectedComponents(binary_mask)
    
    clusters = []
    # Label 0 is background
    for i in range(1, num_labels):
        points = np.argwhere(labels == i)
        if len(points) >= min_area:
            clusters.append(points)
            
    return clusters


def extract_red_mask_from_overlaid(overlaid_array: np.ndarray, 
                                    red_thresh: int = 150, 
                                    other_thresh: int = 100) -> np.ndarray:
    """
    Extract red regions from SegOverlaid image.
    Red pixels indicate positive cells in DeepLIIF output.
    
    Args:
        overlaid_array: RGB array from SegOverlaid
        red_thresh: Minimum R channel value for red detection
        other_thresh: Maximum G and B channel values
    
    Returns:
        Binary mask (uint8) where red regions are 255
    """
    if overlaid_array.ndim != 3 or overlaid_array.shape[2] < 3:
        raise ValueError("Expected RGB image for red mask extraction")
    
    r, g, b = overlaid_array[:,:,0], overlaid_array[:,:,1], overlaid_array[:,:,2]
    
    # Red pixels: high R, low G and B
    red_mask = (r > red_thresh) & (g < other_thresh) & (b < other_thresh)
    
    return (red_mask * 255).astype(np.uint8)


def get_clusters_from_red_regions(overlaid_array: np.ndarray, 
                                   min_area: int = 10, 
                                   red_thresh: int = 150, 
                                   other_thresh: int = 100) -> list[np.ndarray]:
    """
    Extract red regions from SegOverlaid and return as clusters.
    
    Args:
        overlaid_array: RGB array from SegOverlaid
        min_area: Minimum area threshold
        red_thresh: Minimum R channel value
        other_thresh: Maximum G and B channel values
        
    Returns:
        List of cluster coordinate arrays
    """
    red_mask = extract_red_mask_from_overlaid(overlaid_array, red_thresh, other_thresh)
    
    # Connected components on red mask
    num_labels, labels = cv2.connectedComponents(red_mask)
    
    clusters = []
    for i in range(1, num_labels):
        points = np.argwhere(labels == i)
        if len(points) >= min_area:
            clusters.append(points)
            
    return clusters


def generate_mask_from_cluster(cluster_coords: np.ndarray, 
                                image_shape: tuple, 
                                target_size: int = 256) -> np.ndarray:
    """
    Generate low-res mask for SAM2 prompt.
    
    IMPORTANT: SAM2 expects mask_input as LOGITS, not binary masks.
    - Negative values = background (e.g., -10)
    - Positive values = foreground (e.g., +10)
    
    Args:
        cluster_coords: Array of (y, x) coordinates
        image_shape: Original image shape (H, W) or (H, W, C)
        target_size: Target size for low-res mask (default 256)
        
    Returns:
        Low-res mask array of shape (1, target_size, target_size)
    """
    h, w = image_shape[:2]
    # Initialize with negative logits (background)
    # Using ±5 instead of ±10 to give SAM2 more flexibility in adjusting boundaries
    full_mask = np.full((h, w), -5.0, dtype=np.float32)
    
    # Ensure coordinates are integer type (fix for float coords from some sources)
    if not np.issubdtype(cluster_coords.dtype, np.integer):
        cluster_coords = cluster_coords.astype(np.int64)
    
    rows = cluster_coords[:, 0].astype(np.intp)
    cols = cluster_coords[:, 1].astype(np.intp)
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    
    # Set foreground pixels to positive logits
    full_mask[rows[valid], cols[valid]] = 5.0
    
    low_res_mask = cv2.resize(full_mask, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return low_res_mask[np.newaxis, :, :]


def get_bounding_box_from_cluster(cluster_coords: np.ndarray, 
                                   padding: int = 10) -> Optional[np.ndarray]:
    """
    Get bounding box (x1, y1, x2, y2) from cluster coordinates.
    
    Args:
        cluster_coords: Array of (y, x) coordinates
        padding: Padding around the bounding box
        
    Returns:
        Bounding box as np.array([x1, y1, x2, y2]) or None if empty
    """
    if len(cluster_coords) == 0:
        return None
    min_row = int(cluster_coords[:, 0].min())
    max_row = int(cluster_coords[:, 0].max())
    min_col = int(cluster_coords[:, 1].min())
    max_col = int(cluster_coords[:, 1].max())
    return np.array([
        max(0, min_col - padding),
        max(0, min_row - padding),
        max_col + padding,
        max_row + padding
    ], dtype=np.int64)


def boxes_overlap(box1: np.ndarray, box2: np.ndarray) -> bool:
    """
    Check if two boxes overlap. Boxes are [x1, y1, x2, y2].
    
    Args:
        box1: First bounding box
        box2: Second bounding box
        
    Returns:
        True if boxes overlap
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    # Check if one box is to the left/right/above/below the other
    if x2_1 < x1_2 or x2_2 < x1_1:  # No horizontal overlap
        return False
    if y2_1 < y1_2 or y2_2 < y1_1:  # No vertical overlap
        return False
    return True


def merge_two_boxes(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    Merge two boxes into one larger box.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        Merged bounding box
    """
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    return np.array([x1, y1, x2, y2], dtype=np.int64)


def merge_overlapping_boxes(boxes: list, cell_indices: list) -> tuple[list, list]:
    """
    Merge overlapping boxes into larger unified boxes using Union-Find.
    
    Args:
        boxes: List of bounding boxes [x1, y1, x2, y2]
        cell_indices: List of cell indices corresponding to each box
    
    Returns:
        merged_boxes: List of merged bounding boxes
        merged_indices: List of lists, where each inner list contains the cell indices merged into that box
    """
    if len(boxes) == 0:
        return [], []
    
    # Create groups of overlapping boxes using Union-Find
    n = len(boxes)
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Find all overlapping pairs and union them
    for i in range(n):
        for j in range(i + 1, n):
            if boxes_overlap(boxes[i], boxes[j]):
                union(i, j)
    
    # Group boxes by their root
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    # Merge each group into a single box
    merged_boxes = []
    merged_indices = []
    for root, members in groups.items():
        merged_box = boxes[members[0]]
        member_cells = [cell_indices[members[0]]]
        for m in members[1:]:
            merged_box = merge_two_boxes(merged_box, boxes[m])
            member_cells.append(cell_indices[m])
        merged_boxes.append(merged_box)
        merged_indices.append(member_cells)
    
    return merged_boxes, merged_indices


def merge_overlapping_cells(cells_info: list[dict], padding: int = 10) -> list[dict]:
    """
    Detect overlapping cells based on their bounding boxes and merge them.
    
    Args:
        cells_info: List of cell info dicts (must contain 'coords', 'center' keys)
        padding: Bounding box padding (same as used in SAM2)
    
    Returns:
        merged_cells: List of merged cell info dicts, each containing:
            - 'coords': combined coordinates of all merged cells
            - 'center': center of the merged region
            - 'box': merged bounding box [x1, y1, x2, y2]
            - 'member_ids': list of original cell IDs that were merged
            - 'is_merged': True if this contains multiple cells
    """
    if not cells_info or len(cells_info) == 0:
        return []
    
    # Compute bounding boxes for all cells
    boxes = []
    for cell in cells_info:
        coords = cell['coords']
        min_row = int(coords[:, 0].min())
        max_row = int(coords[:, 0].max())
        min_col = int(coords[:, 1].min())
        max_col = int(coords[:, 1].max())
        x1, y1 = min_col - padding, min_row - padding
        x2, y2 = max_col + padding, max_row + padding
        boxes.append(np.array([x1, y1, x2, y2], dtype=np.int64))
    
    # Use merge_overlapping_boxes to find groups
    cell_indices = list(range(len(cells_info)))
    merged_boxes, merged_indices = merge_overlapping_boxes(boxes, cell_indices)
    
    # Create merged cell info
    merged_cells = []
    for merged_box, member_indices in zip(merged_boxes, merged_indices):
        # Combine coordinates from all member cells
        all_coords = np.vstack([cells_info[i]['coords'] for i in member_indices])
        
        # Calculate center of merged region
        center_y = int(all_coords[:, 0].mean())
        center_x = int(all_coords[:, 1].mean())
        
        merged_cells.append({
            'coords': all_coords,
            'center': (center_y, center_x),
            'box': merged_box,
            'member_ids': [i + 1 for i in member_indices],  # 1-based IDs
            'is_merged': len(member_indices) > 1,
            'pixel_count': len(all_coords)
        })
    
    return merged_cells
