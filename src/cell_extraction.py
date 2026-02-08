#!/usr/bin/env python3
"""
cell_extraction.py — 细胞提取与分类模块

包含：
- 从 DeepLIIF Seg 输出提取细胞
- 正/负细胞分类
- 细胞过滤
"""

import numpy as np
import cv2
from typing import Optional


def extract_positive_cells_info(seg_refined_array: np.ndarray, 
                                 marker_array: np.ndarray, 
                                 min_area: int = 10) -> list[dict]:
    """
    Extract positive cells from SegRefined and collect marker intensity information.
    Similar to DeepLIIF's compute_cell_mapping but only for positive (red) cells.
    
    Args:
        seg_refined_array: RGB array from SegRefined (Red=positive, Blue=negative, Green=border)
        marker_array: Grayscale or RGB marker image
        min_area: Minimum cell area threshold
    
    Returns:
        List of dicts with cell info: 
        {
            'id': cell_id,
            'coords': np.ndarray of (y, x) coordinates,
            'pixel_count': number of pixels,
            'marker_sum': sum of marker values,
            'marker_mean': mean marker value,
            'marker_max': max marker value,
            'marker_min': min marker value,
            'center': (cy, cx) center coordinates
        }
    """
    # Convert marker to grayscale if needed
    if marker_array.ndim == 3:
        marker_gray = cv2.cvtColor(marker_array, cv2.COLOR_RGB2GRAY)
    else:
        marker_gray = marker_array
    
    # Extract red mask (positive cells) from SegRefined
    # Red pixels: high R, low G and B
    r, g, b = seg_refined_array[:,:,0], seg_refined_array[:,:,1], seg_refined_array[:,:,2]
    red_mask = (r > 150) & (g < 100) & (b < 100)
    binary_mask = (red_mask * 255).astype(np.uint8)
    
    # Connected components analysis
    num_labels, labels = cv2.connectedComponents(binary_mask)
    
    cells_info = []
    for cell_id in range(1, num_labels):
        # Get coordinates of this cell
        coords = np.argwhere(labels == cell_id)
        pixel_count = len(coords)
        
        if pixel_count < min_area:
            continue
        
        # Extract marker values for this cell
        marker_values = marker_gray[coords[:, 0], coords[:, 1]]
        
        # Calculate center
        center_y = int(coords[:, 0].mean())
        center_x = int(coords[:, 1].mean())
        
        cells_info.append({
            'id': len(cells_info) + 1,  # 1-based ID
            'coords': coords,
            'pixel_count': pixel_count,
            'marker_sum': int(marker_values.sum()),
            'marker_mean': float(marker_values.mean()),
            'marker_max': int(marker_values.max()),
            'marker_min': int(marker_values.min()),
            'center': (center_y, center_x),
            'is_positive': True  # All cells from this function are positive
        })
    
    return cells_info


def extract_cells_from_seg(seg_array: np.ndarray, 
                            marker_array: np.ndarray, 
                            min_area: int = 10, 
                            seg_thresh: int = 120,
                            marker_thresh: Optional[int] = None) -> list[dict]:
    """
    从原始 Seg 图像中提取所有细胞并分类为阳性/阴性。
    
    该函数完全模拟 DeepLIIF 的后处理逻辑：
    - 第1步：从 Seg RGB 创建阳性/阴性掩码（R >= B 表示阳性像素）
    - 第2步：对每个细胞，统计阳性像素 vs 阴性像素的数量
    - 第3步：最终分类：(多数为阳性) 或 (marker_max > marker_thresh)
    
    Args:
        seg_array: 原始 Seg 输出的 RGB 数组
        marker_array: 灰度或 RGB Marker 图像，用于阳性/阴性分类
        min_area: 最小细胞面积阈值
        seg_thresh: 前景检测阈值（类似 DeepLIIF 的 seg_thresh）
        marker_thresh: 阳性/阴性分类阈值。如果为 None，则自动从 Marker 图像计算。
    
    Returns:
        包含细胞信息的字典列表（包含 'is_positive' 标志）
    """
    # Convert marker to grayscale if needed
    if marker_array.ndim == 3:
        marker_gray = cv2.cvtColor(marker_array, cv2.COLOR_RGB2GRAY)
    else:
        marker_gray = marker_array.copy()
    
    # === Step 1: Create pos/neg mask from Seg RGB (like DeepLIIF's create_posneg_mask) ===
    if seg_array.ndim == 3:
        r_channel = seg_array[:,:,0].astype(int)
        g_channel = seg_array[:,:,1].astype(int)
        b_channel = seg_array[:,:,2].astype(int)
        
        sum_rb = r_channel + b_channel
        
        # Foreground condition: (R + B > seg_thresh) and (G <= 80)
        is_foreground = (sum_rb > seg_thresh) & (g_channel <= 80)
        
        # Positive pixel condition: R >= B
        is_pos_pixel = r_channel >= b_channel
        
        # Create labeled mask: 0=background, 1=negative, 2=positive
        posneg_mask = np.zeros_like(r_channel, dtype=np.uint8)
        posneg_mask[is_foreground & is_pos_pixel] = 2  # Positive
        posneg_mask[is_foreground & ~is_pos_pixel] = 1  # Negative
    else:
        # Grayscale fallback - treat all foreground as unknown
        _, binary_mask = cv2.threshold(seg_array, seg_thresh, 255, cv2.THRESH_BINARY)
        posneg_mask = (binary_mask > 0).astype(np.uint8)
    
    # Binary mask for connected components (all foreground)
    binary_mask = ((posneg_mask > 0) * 255).astype(np.uint8)
    
    # Connected components to find individual cells
    num_labels, labels = cv2.connectedComponents(binary_mask)
    
    # Auto-calculate marker threshold if not provided (like DeepLIIF)
    if marker_thresh is None:
        # Use 90th percentile of non-zero marker values as threshold
        nonzero_marker = marker_gray[marker_gray > 0]
        if len(nonzero_marker) > 0:
            marker_range_min = np.percentile(nonzero_marker, 0.1)
            marker_range_max = np.percentile(nonzero_marker, 99.9)
            marker_thresh = int((marker_range_max - marker_range_min) * 0.9 + marker_range_min)
            print(f"    [DEBUG] Auto marker_thresh: {marker_thresh} (range: {marker_range_min:.1f} - {marker_range_max:.1f})")
        else:
            marker_thresh = 128  # fallback
            print(f"    [DEBUG] Fallback marker_thresh: {marker_thresh}")
    
    cells_info = []
    for cell_id in range(1, num_labels):
        # Get coordinates of this cell
        coords = np.argwhere(labels == cell_id)
        pixel_count = len(coords)
        
        if pixel_count < min_area:
            continue
        
        # Extract marker values for this cell
        marker_values = marker_gray[coords[:, 0], coords[:, 1]]
        marker_max = int(marker_values.max())
        marker_mean = float(marker_values.mean())
        
        # === Step 2: Count positive vs negative pixels in this cell ===
        cell_posneg = posneg_mask[coords[:, 0], coords[:, 1]]
        count_positive = np.sum(cell_posneg == 2)
        count_negative = np.sum(cell_posneg == 1)
        
        # Initial classification based on Seg RGB (majority vote)
        initial_positive = count_positive >= count_negative
        
        # Calculate center
        center_y = int(coords[:, 0].mean())
        center_x = int(coords[:, 1].mean())
        
        # === Step 3: Final classification (like DeepLIIF's create_cell_classification) ===
        # [MODIFIED] 只用 Seg 判断，禁用 Marker 阈值逻辑
        # 原逻辑: is_pos = initial_positive OR (max_marker > marker_thresh)
        is_positive = initial_positive  # 只使用 Seg 的 R>=B 多数票
        # is_positive = initial_positive or (marker_max > marker_thresh)  # Marker 逻辑已禁用
        
        cells_info.append({
            'id': len(cells_info) + 1,  # 1-based ID
            'coords': coords,
            'pixel_count': pixel_count,
            'marker_sum': int(marker_values.sum()),
            'marker_mean': marker_mean,
            'marker_max': marker_max,
            'marker_min': int(marker_values.min()),
            'center': (center_y, center_x),
            'is_positive': is_positive,
            'count_positive_pixels': int(count_positive),
            'count_negative_pixels': int(count_negative)
        })
    
    return cells_info


def filter_positive_cells(cells_info: list[dict], 
                           marker_sum_thresh: int = 2000, 
                           marker_max_thresh: int = 30) -> list[dict]:
    """
    从所有细胞中过滤出满足条件的正向细胞。
    
    Args:
        cells_info: 所有细胞信息列表
        marker_sum_thresh: Marker Sum 阈值
        marker_max_thresh: Marker Max 阈值
        
    Returns:
        满足条件的正向细胞列表
    """
    positive_cells = [
        c for c in cells_info 
        if c.get('is_positive', False) 
        and c['marker_sum'] > marker_sum_thresh 
        and c['marker_max'] >= marker_max_thresh
    ]
    return positive_cells


def get_clusters_from_cells(cells_info: list[dict]) -> list[np.ndarray]:
    """
    从细胞信息列表中提取坐标簇。
    
    Args:
        cells_info: 细胞信息列表
        
    Returns:
        坐标数组列表
    """
    return [cell['coords'] for cell in cells_info]


def create_binary_mask_from_cells(cells_info: list[dict], 
                                   image_shape: tuple) -> np.ndarray:
    """
    从细胞信息创建二值掩码。
    
    Args:
        cells_info: 细胞信息列表
        image_shape: 图像形状 (H, W) 或 (H, W, C)
        
    Returns:
        二值掩码 (H, W) uint8
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for cell in cells_info:
        coords = cell['coords']
        mask[coords[:, 0], coords[:, 1]] = 255
    
    return mask


def renumber_cells(cells_info: list[dict]) -> list[dict]:
    """
    重新编号细胞 ID (1, 2, 3...)
    
    Args:
        cells_info: 细胞信息列表
        
    Returns:
        重新编号后的细胞信息列表
    """
    for new_id, cell in enumerate(cells_info, start=1):
        cell['id'] = new_id
    return cells_info


def group_cells_by_distance(cells_info: list[dict], 
                            distance_threshold: float = 50.0) -> list[dict]:
    """
    根据细胞中心点之间的距离进行分组。
    
    距离小于 threshold 的细胞会被归为同一组。
    使用 Union-Find 算法进行聚类。
    
    Args:
        cells_info: 细胞信息列表（必须包含 'center' 和 'coords' 字段）
        distance_threshold: 距离阈值（像素），中心点距离小于此值的细胞归为一组
        
    Returns:
        分组后的列表，每组是一个 dict:
        {
            'group_id': int,
            'member_cells': list[dict],  # 组内所有细胞信息
            'member_ids': list[int],     # 组内细胞的原始 ID
            'merged_coords': np.ndarray, # 合并后的所有坐标
            'center': (cy, cx),          # 组的中心点
            'total_pixels': int          # 组内总像素数
        }
    """
    if len(cells_info) == 0:
        return []
    
    if len(cells_info) == 1:
        # 只有一个细胞，直接返回
        cell = cells_info[0]
        return [{
            'group_id': 1,
            'member_cells': [cell],
            'member_ids': [cell['id']],
            'merged_coords': cell['coords'],
            'center': cell['center'],
            'total_pixels': cell['pixel_count']
        }]
    
    n = len(cells_info)
    
    # Union-Find 数据结构
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # 计算每对细胞中心点之间的距离，距离小于阈值则合并
    for i in range(n):
        center_i = cells_info[i]['center']
        for j in range(i + 1, n):
            center_j = cells_info[j]['center']
            # 欧氏距离
            dist = np.sqrt((center_i[0] - center_j[0])**2 + (center_i[1] - center_j[1])**2)
            if dist < distance_threshold:
                union(i, j)
    
    # 根据 Union-Find 结果分组
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    # 构建分组结果
    result = []
    for group_id, (root, member_indices) in enumerate(groups.items(), start=1):
        member_cells = [cells_info[i] for i in member_indices]
        member_ids = [cells_info[i]['id'] for i in member_indices]
        
        # 合并所有成员的坐标
        all_coords = np.vstack([cells_info[i]['coords'] for i in member_indices])
        
        # 计算组的中心点
        center_y = int(all_coords[:, 0].mean())
        center_x = int(all_coords[:, 1].mean())
        
        # 总像素数
        total_pixels = sum(cells_info[i]['pixel_count'] for i in member_indices)
        
        result.append({
            'group_id': group_id,
            'member_cells': member_cells,
            'member_ids': member_ids,
            'merged_coords': all_coords,
            'center': (center_y, center_x),
            'total_pixels': total_pixels
        })
    
    return result


def visualize_cell_extraction(seg_array: np.ndarray,
                               marker_array: np.ndarray,
                               cells_info: list[dict],
                               output_path: Optional[str] = None,
                               seg_thresh: int = 120,
                               show_labels: bool = True,
                               show_plot: bool = False,
                               figsize: tuple = (16, 12)) -> None:
    """
    Visualize the cell extraction process and results.
    
    Generates a 2x3 subplot showing:
    1. Original Seg Image
    2. Positive/Negative Pixel Mask (Red>=Blue is Positive)
    3. Connected Components Analysis
    4. Marker Image
    5. Final Cell Classification (Red=Positive, Blue=Negative)
    6. Cell Statistics Table
    
    Args:
        seg_array: Original Seg output RGB array
        marker_array: Marker image
        cells_info: List of cell info dicts from extract_cells_from_seg
        output_path: Optional path to save the visualization
        seg_thresh: Foreground threshold
        show_labels: Whether to show cell IDs on the plot
        show_plot: Whether to display the plot (plt.show())
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    # Re-calculate intermediate results for visualization
    if seg_array.ndim == 3:
        r_channel = seg_array[:,:,0].astype(int)
        g_channel = seg_array[:,:,1].astype(int)
        b_channel = seg_array[:,:,2].astype(int)
        
        sum_rb = r_channel + b_channel
        is_foreground = (sum_rb > seg_thresh) & (g_channel <= 80)
        is_pos_pixel = r_channel >= b_channel
        
        # Positive/Negative mask visualization
        posneg_vis = np.zeros((*seg_array.shape[:2], 3), dtype=np.uint8)
        posneg_vis[is_foreground & is_pos_pixel] = [255, 0, 0]   # Red = Positive pixel
        posneg_vis[is_foreground & ~is_pos_pixel] = [0, 0, 255]  # Blue = Negative pixel
        
        # Foreground binary mask
        binary_mask = (is_foreground * 255).astype(np.uint8)
    else:
        _, binary_mask = cv2.threshold(seg_array, seg_thresh, 255, cv2.THRESH_BINARY)
        posneg_vis = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    
    # Connected components analysis
    num_labels, labels = cv2.connectedComponents(binary_mask)
    
    # Assign colors to connected components
    np.random.seed(42)  # Fixed seed for consistent colors
    colors = np.random.randint(50, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black
    components_vis = colors[labels]
    
    # Final classification visualization
    final_vis = np.zeros((*seg_array.shape[:2], 3), dtype=np.uint8)
    for cell in cells_info:
        coords = cell['coords']
        color = [255, 80, 80] if cell['is_positive'] else [80, 80, 255]  # Red=Pos, Blue=Neg
        final_vis[coords[:, 0], coords[:, 1]] = color
    
    # Marker image
    if marker_array.ndim == 3:
        marker_gray = cv2.cvtColor(marker_array, cv2.COLOR_RGB2GRAY)
    else:
        marker_gray = marker_array.copy()
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # 1. Original Seg Image
    axes[0, 0].imshow(seg_array)
    axes[0, 0].set_title('1. Original Seg Image', fontsize=12)
    axes[0, 0].axis('off')
    
    # 2. Positive/Negative Pixel Mask
    axes[0, 1].imshow(posneg_vis)
    axes[0, 1].set_title('2. Pixel Class (Red=Pos, Blue=Neg)', fontsize=12)
    axes[0, 1].axis('off')
    
    # 3. Connected Components
    axes[0, 2].imshow(components_vis)
    axes[0, 2].set_title(f'3. Connected Components ({num_labels-1} regions)', fontsize=12)
    axes[0, 2].axis('off')
    
    # 4. Marker Image
    axes[1, 0].imshow(marker_gray, cmap='gray')
    axes[1, 0].set_title('4. Marker Image', fontsize=12)
    axes[1, 0].axis('off')
    
    # 5. Final Classification
    axes[1, 1].imshow(final_vis)
    if show_labels:
        for cell in cells_info:
            cy, cx = cell['center']
            color = 'yellow'
            axes[1, 1].text(cx, cy, str(cell['id']), 
                           fontsize=8, color=color, ha='center', va='center',
                           fontweight='bold')
    
    # Statistics
    pos_count = sum(1 for c in cells_info if c['is_positive'])
    neg_count = len(cells_info) - pos_count
    axes[1, 1].set_title(f'5. Final Classification (Pos:{pos_count}, Neg:{neg_count})', fontsize=12)
    axes[1, 1].axis('off')
    
    # Legend
    legend_elements = [
        Patch(facecolor='red', label=f'Positive ({pos_count})'),
        Patch(facecolor='blue', label=f'Negative ({neg_count})')
    ]
    axes[1, 1].legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # 6. Cell Statistics Table
    axes[1, 2].axis('off')
    
    # Create statistics table
    if len(cells_info) > 0:
        # Show only first 10 cells
        display_cells = cells_info[:min(10, len(cells_info))]
        table_data = []
        for c in display_cells:
            is_pos = 'Y' if c['is_positive'] else 'N'
            pos_pix = c.get('count_positive_pixels', '-')
            neg_pix = c.get('count_negative_pixels', '-')
            table_data.append([
                c['id'], 
                c['pixel_count'], 
                f"{pos_pix}/{neg_pix}",
                f"{c['marker_max']:.0f}",
                is_pos
            ])
        
        columns = ['ID', 'Pixels', 'Pos/Neg Pix', 'MaxMarker', 'Pos?']
        table = axes[1, 2].table(
            cellText=table_data,
            colLabels=columns,
            loc='center',
            cellLoc='center',
            colColours=['lightgray'] * 5
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        if len(cells_info) > 10:
            axes[1, 2].text(0.5, 0.05, f'... {len(cells_info) - 10} more cells hidden',
                           ha='center', transform=axes[1, 2].transAxes, fontsize=10)
    
    axes[1, 2].set_title('6. Cell Statistics', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    if show_plot:
        plt.show()
    
    plt.close(fig)  # Close the figure to free memory
