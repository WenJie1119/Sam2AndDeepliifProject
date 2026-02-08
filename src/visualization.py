#!/usr/bin/env python3
"""
visualization.py — 可视化模块

包含：
- 图像面板生成
- 图表绘制
- SAM2 结果面板
- 对比图生成与保存
"""

import os
import math
import numpy as np
import cv2

from .mask_utils import generate_distinct_colors, merge_overlapping_boxes


def img_to_panel(img_array: np.ndarray, h: int, w: int) -> np.ndarray:
    """Convert image array to BGR panel for display."""
    if img_array is None:
        return np.ones((h, w, 3), dtype=np.uint8) * 128
    if len(img_array.shape) == 2:
        return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    else:
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


def draw_single_chart(chart_w: int, chart_h: int, ids: list, scores: list, 
                      title: str, color: tuple, filtered_list: list = None) -> np.ndarray:
    """Draw a single line chart for one mode. Filtered points shown in blue."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    chart = np.ones((chart_h, chart_w, 3), dtype=np.uint8) * 255
    margin_left, margin_right, margin_top, margin_bottom = 60, 20, 60, 50
    plot_w = chart_w - margin_left - margin_right
    plot_h = chart_h - margin_top - margin_bottom
    
    # Combine regular and filtered ids to get proper x-axis range
    all_ids = list(ids) + [item[0] for item in (filtered_list or [])]
    max_id = max(all_ids) if all_ids else 1
    if max_id == 0: max_id = 1
    
    # Title at top
    cv2.putText(chart, title, (chart_w // 2 - 80, 25), font, 0.55, (0,0,0), 2)
    
    # Draw axes
    cv2.line(chart, (margin_left, margin_top), (margin_left, chart_h - margin_bottom), (0, 0, 0), 2)
    cv2.line(chart, (margin_left, chart_h - margin_bottom), (chart_w - margin_right, chart_h - margin_bottom), (0, 0, 0), 2)

    # Y-axis labels
    for i in range(6):
        y_val = i * 0.2
        y_pos = int(chart_h - margin_bottom - y_val * plot_h)
        cv2.line(chart, (margin_left - 5, y_pos), (margin_left, y_pos), (0, 0, 0), 1)
        cv2.putText(chart, f"{y_val:.1f}", (margin_left - 40, y_pos + 5), font, 0.4, (0, 0, 0), 1)
    
    # X-axis labels
    for inst_id in range(1, max_id + 1):
        x = int(margin_left + (inst_id / max_id) * plot_w)
        cv2.line(chart, (x, chart_h - margin_bottom), (x, chart_h - margin_bottom + 5), (0, 0, 0), 1)
        cv2.putText(chart, str(inst_id), (x - 5, chart_h - margin_bottom + 18), font, 0.35, (0, 0, 0), 1)
    
    # Draw points and lines for regular instances
    if len(ids) > 1:
        points = [(int(margin_left + (inst_id / max_id) * plot_w), int(chart_h - margin_bottom - score * plot_h)) 
                  for inst_id, score in zip(ids, scores)]
        for j in range(len(points) - 1):
            cv2.line(chart, points[j], points[j + 1], color, 2)
    
    for inst_id, score in zip(ids, scores):
        x = int(margin_left + (inst_id / max_id) * plot_w)
        y = int(chart_h - margin_bottom - score * plot_h)
        cv2.circle(chart, (x, y), 5, color, -1)
    
    # Draw BLUE points for FILTERED instances
    if filtered_list:
        for inst_id, score in filtered_list:
            x = int(margin_left + (inst_id / max_id) * plot_w)
            y = int(chart_h - margin_bottom - score * plot_h)
            cv2.circle(chart, (x, y), 6, (255, 0, 0), -1)
            cv2.putText(chart, "F", (x - 4, y - 10), font, 0.35, (255, 0, 0), 1)
    
    cv2.putText(chart, "Instance ID", (chart_w // 2 - 40, chart_h - 10), font, 0.5, (0, 0, 0), 1)
    return chart


def draw_rgb_histogram(img_array: np.ndarray, panel_w: int, panel_h: int, 
                       title: str = "") -> np.ndarray:
    """Draw Red channel histogram, ignoring the massive background peak at 0."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    hist_panel = np.ones((panel_h, panel_w, 3), dtype=np.uint8) * 255
    
    if img_array is None or len(img_array.shape) < 3:
        cv2.putText(hist_panel, "No RGB data", (panel_w//2 - 50, panel_h//2), font, 0.6, (128, 128, 128), 2)
        return hist_panel
    
    margin_left, margin_right, margin_top, margin_bottom = 60, 30, 70, 60
    plot_w = panel_w - margin_left - margin_right
    plot_h = panel_h - margin_top - margin_bottom
    
    # Title
    cv2.putText(hist_panel, title, (panel_w // 2 - 80, 25), font, 0.55, (0,0,0), 2)
    
    # Plot background
    cv2.rectangle(hist_panel, (margin_left, margin_top), 
                 (panel_w - margin_right, panel_h - margin_bottom), (245, 245, 245), -1)
    
    # Grid lines
    for val in [0, 50, 100, 150, 200, 255]:
        x = int(margin_left + (val / 255) * plot_w)
        cv2.line(hist_panel, (x, margin_top), (x, panel_h - margin_bottom), (220, 220, 220), 1)
    
    # Calculate histogram for Red channel
    color_display = (0, 0, 255)
    hist = cv2.calcHist([img_array], [0], None, [256], [0, 256])
    
    # Ignore peak at 0 for scaling
    if len(hist) > 1:
        max_val = hist[1:].max()
    else:
        max_val = hist.max()
    if max_val <= 0:
        max_val = 1.0
        
    # Y-axis grid
    y_ticks = 5
    for i in range(y_ticks + 1):
        y_ratio = i / y_ticks
        y_pos = int(panel_h - margin_bottom - y_ratio * plot_h * 0.95)
        cv2.line(hist_panel, (margin_left, y_pos), (panel_w - margin_right, y_pos), (220, 220, 220), 1)
    
    # Axes
    cv2.line(hist_panel, (margin_left, margin_top), (margin_left, panel_h - margin_bottom), (0, 0, 0), 2)
    cv2.line(hist_panel, (margin_left, panel_h - margin_bottom), (panel_w - margin_right, panel_h - margin_bottom), (0, 0, 0), 2)
    
    # X-axis ticks
    for val in [0, 50, 100, 150, 200, 255]:
        x = int(margin_left + (val / 255) * plot_w)
        cv2.line(hist_panel, (x, panel_h - margin_bottom), (x, panel_h - margin_bottom + 5), (0, 0, 0), 2)
        cv2.putText(hist_panel, str(val), (x - 10, panel_h - margin_bottom + 20), font, 0.4, (0, 0, 0), 1)
    
    # Y-axis ticks
    for i in range(y_ticks + 1):
        y_ratio = i / y_ticks
        y_pos = int(panel_h - margin_bottom - y_ratio * plot_h * 0.95)
        y_val = int(max_val * y_ratio * 0.95)
        cv2.line(hist_panel, (margin_left - 5, y_pos), (margin_left, y_pos), (0, 0, 0), 2)
        if y_val >= 1000:
            label = f"{y_val//1000}K"
        else:
            label = str(y_val)
        cv2.putText(hist_panel, label, (5, y_pos + 5), font, 0.35, (0, 0, 0), 1)
    
    # Draw histogram
    points = []
    for x_idx in range(256):
        x = int(margin_left + (x_idx / 255) * plot_w)
        raw_height = (hist[x_idx][0] / max_val) * plot_h * 0.95
        y = int(panel_h - margin_bottom - min(raw_height, plot_h))
        points.append((x, y))
    
    for j in range(len(points) - 1):
        cv2.line(hist_panel, points[j], points[j + 1], color_display, 2)
    
    cv2.putText(hist_panel, "Pixel Value (Red)", (panel_w // 2 - 60, panel_h - 15), font, 0.5, (0, 0, 0), 1)
    
    return hist_panel


def draw_sam_panel(original: np.ndarray, mask: np.ndarray, 
                   positive_cells_info: list, scores_dict: dict, 
                   colors: list, filtered_list: list = None) -> np.ndarray:
    """Draw SAM2 result panel with masks and labels. Filtered cells shown with blue outline."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    panel = original.copy()
    h, w = original.shape[:2]
    
    num_cells = len(positive_cells_info) if positive_cells_info else 0
    filtered_ids = set(inst_id for inst_id, _ in (filtered_list or []))
    
    # Draw all masks
    if num_cells > 0:
        for idx in range(num_cells):
            inst_id = idx + 1
            inst_mask = mask == inst_id
            mask_area = np.sum(inst_mask)
            if mask_area >= 10:
                panel[inst_mask] = colors[idx]
    
    # Draw labels
    for idx, cell in enumerate(positive_cells_info or []):
        inst_id = idx + 1
        inst_mask = mask == inst_id
        mask_area = np.sum(inst_mask)
        
        if mask_area >= 10:
            center_y, center_x = cell['center']
            label = f"{inst_id}"
            cv2.putText(panel, label, (center_x-5, center_y+5), font, 0.45, (255, 255, 255), 2)
            cv2.putText(panel, label, (center_x-5, center_y+5), font, 0.45, (0, 0, 0), 1)
    
    # Draw BLUE outlines for FILTERED cells
    for idx, cell in enumerate(positive_cells_info or []):
        inst_id = idx + 1
        if inst_id in filtered_ids:
            coords = cell['coords']
            center_y, center_x = cell['center']
            
            cell_mask = np.zeros((h, w), dtype=np.uint8)
            cell_mask[coords[:, 0], coords[:, 1]] = 255
            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            cv2.drawContours(panel, contours, -1, (0, 0, 255), 2)
            
            label = f"F{inst_id}"
            cv2.putText(panel, label, (center_x - 12, center_y + 5), font, 0.4, (0, 0, 0), 3)
            cv2.putText(panel, label, (center_x - 12, center_y + 5), font, 0.4, (0, 0, 255), 2)
            
    return cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)


def add_title_bar(panel: np.ndarray, title: str, bar_height: int = 30) -> np.ndarray:
    """Add a white title bar on top of the panel."""
    h, w = panel.shape[:2]
    title_bar = np.ones((bar_height, w, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(title, font, 0.5, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (bar_height + text_size[1]) // 2
    cv2.putText(title_bar, title, (text_x, text_y), font, 0.5, (0, 0, 0), 2)
    return np.concatenate([title_bar, panel], axis=0)


def create_positive_marker_refined(original_array: np.ndarray, 
                                    cells_info: list[dict]) -> np.ndarray:
    """
    Create SegRefined-style image showing only positive cells with colored boundaries.
    """
    h, w = original_array.shape[:2]
    refined = np.zeros((h, w, 3), dtype=np.uint8)
    
    colors = generate_distinct_colors(len(cells_info))
    
    for idx, cell in enumerate(cells_info):
        coords = cell['coords']
        color = colors[idx] if idx < len(colors) else (255, 0, 0)
        refined[coords[:, 0], coords[:, 1]] = color
    
    # Draw green boundaries
    cell_mask = np.zeros((h, w), dtype=np.uint8)
    for cell in cells_info:
        coords = cell['coords']
        cell_mask.fill(0)
        cell_mask[coords[:, 0], coords[:, 1]] = 255
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(refined, contours, -1, (0, 255, 0), 2)
    
    return refined


def create_positive_marker_overlaid(original_array: np.ndarray, 
                                     cells_info: list[dict]) -> np.ndarray:
    """
    Create SegOverlaid-style image with positive cells overlaid on original.
    """
    overlaid = original_array.copy()
    h, w = original_array.shape[:2]
    
    cell_mask = np.zeros((h, w), dtype=np.uint8)
    
    for cell in cells_info:
        coords = cell['coords']
        cell_mask.fill(0)
        cell_mask[coords[:, 0], coords[:, 1]] = 255
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlaid, contours, -1, (255, 0, 0), 2)
    
    return overlaid


def save_comparison(output_dir: str, base_name: str, original: np.ndarray, 
                    seg: np.ndarray, seg_overlaid: np.ndarray, seg_refined: np.ndarray, 
                    red_mask: np.ndarray, sam_mask_box_mask: np.ndarray, 
                    sam_mask_only: np.ndarray, clusters: list = None,
                    scores_box_mask: list = None, scores_mask_only: list = None,
                    deepliif_params: dict = None, sam_params: dict = None,
                    marker: np.ndarray = None, positive_cells_info: list = None,
                    filtered_box_mask: list = None, filtered_mask_only: list = None,
                    merge_info: list = None,
                    save_comparison_image: bool = True, save_combined_image: bool = True):
    """
    Save enhanced visual comparison with 17 panels (auto-grid layout).
    """
    if not save_comparison_image and not save_combined_image:
        return
    
    if save_comparison_image:
        os.makedirs(f"{output_dir}/comparison", exist_ok=True)
    
    h, w = original.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    line_height = 28
    
    num_cells = len(positive_cells_info) if positive_cells_info else 0
    colors = generate_distinct_colors(num_cells)
    
    scores_bm_dict = {inst_id: score for inst_id, score in (scores_box_mask or [])}
    scores_mo_dict = {inst_id: score for inst_id, score in (scores_mask_only or [])}
    
    # === Row 1: Input and DeepLIIF Processing ===
    panel1 = cv2.cvtColor(original.copy(), cv2.COLOR_RGB2BGR)
    
    panel2 = np.ones((h, w, 3), dtype=np.uint8) * 255
    if deepliif_params:
        y_offset = 60
        for key, val in deepliif_params.items():
            text = f"{key}: {val}"
            cv2.putText(panel2, text, (10, y_offset), font, font_scale, (0, 0, 0), 2)
            y_offset += line_height
            
    panel3 = img_to_panel(seg, h, w)
    
    # === Row 2: Marker Analysis ===
    if marker is not None:
        panel4 = img_to_panel(marker, h, w)
    else:
        panel4 = np.ones((h, w, 3), dtype=np.uint8) * 128
        cv2.putText(panel4, "No Marker", (w//2-50, h//2), font, 0.6, (255, 255, 255), 2)
        
    panel5 = img_to_panel(red_mask, h, w)
    
    # Panel 6: Annotated Positive Cells
    panel6 = np.zeros((h, w, 3), dtype=np.uint8)
    if positive_cells_info and len(positive_cells_info) > 0:
        for idx, cell in enumerate(positive_cells_info):
            coords = cell['coords']
            panel6[coords[:, 0], coords[:, 1]] = (255, 255, 255)
            
            cy, cx = cell['center']
            display_id = idx + 1
            annotation_text = f"{display_id},{cell['pixel_count']},{cell['marker_max']}"
            
            text_size = cv2.getTextSize(annotation_text, font, 0.3, 1)[0]
            text_x = max(5, min(cx - text_size[0] // 2, w - text_size[0] - 5))
            text_y = max(text_size[1] + 5, min(cy + text_size[1] // 2, h - 5))
            
            cv2.putText(panel6, annotation_text, (text_x, text_y), font, 0.3, (255, 255, 255), 2)
            cv2.putText(panel6, annotation_text, (text_x, text_y), font, 0.3, (0, 0, 0), 1)
    else:
        cv2.putText(panel6, "No Positive Cells", (w//2-80, h//2), font, 0.6, (255, 255, 255), 2)
        
    # === Row 3: SAM2 Prompts & Results ===
    panel7 = np.ones((h, w, 3), dtype=np.uint8) * 255
    if sam_params:
        y_offset = 60
        for key, val in sam_params.items():
            text = f"{key}: {val}"
            cv2.putText(panel7, text, (10, y_offset), font, font_scale, (0, 0, 0), 2)
            y_offset += line_height
            
    panel8 = draw_sam_panel(original, sam_mask_only, positive_cells_info, scores_mo_dict, colors, filtered_mask_only)
    
    mo_ids = [item[0] for item in (scores_mask_only or [])]
    mo_scores = [item[1] for item in (scores_mask_only or [])]
    panel9 = draw_single_chart(w, h, mo_ids, mo_scores, "Mask-Only Scores", (0, 0, 255), filtered_mask_only)
    

    
    # Panel 10: Summary
    panel10 = np.ones((h, w, 3), dtype=np.uint8) * 255
    y_offset = 40
    cv2.putText(panel10, "Summary:", (10, y_offset), font, 0.6, (0, 0, 0), 2)
    y_offset += 30
    if mo_scores:
        avg_mo = sum(mo_scores) / len(mo_scores)
        cv2.putText(panel10, f"Mask-Only Avg: {avg_mo:.3f}", (10, y_offset), font, 0.5, (0, 0, 255), 2)
        y_offset += 25
    
    # === Row 4: Seg Histogram ===
    panel11 = draw_rgb_histogram(seg, w, h, "Seg Histogram")
    
    # === Panel 12: Legend ===
    panel12 = np.ones((h, w, 3), dtype=np.uint8) * 255
    legend_y = 40
    legend_line_h = 35
    cv2.putText(panel12, "Category Legend:", (10, legend_y), font, 0.7, (0, 0, 0), 2)
    legend_y += legend_line_h + 10
    
    legend_items = [
        ("A", "Input / Config", (100, 100, 100)),
        ("B", "DeepLIIF Direct Output", (0, 128, 0)),
        ("C", "Post-Processing / Extract", (255, 128, 0)),
        ("D", "SAM2 Result", (0, 0, 200)),
        ("E", "Summary", (128, 0, 128)),
    ]
    for code, desc, color in legend_items:
        cv2.rectangle(panel12, (10, legend_y - 18), (35, legend_y + 2), color, -1)
        cv2.rectangle(panel12, (10, legend_y - 18), (35, legend_y + 2), (0, 0, 0), 1)
        cv2.putText(panel12, f"{code} = {desc}", (45, legend_y), font, 0.55, (0, 0, 0), 2)
        legend_y += legend_line_h
    
    # === Add title bars ===
    titles = [
        "A1. Input: Original Image", 
        "A2. Config: DeepLIIF Params", 
        "B1. DeepLIIF Output: Seg",
        "B2. DeepLIIF Output: Marker", 
        "C1. Extract: Positive Mask", 
        "C2. Extract: Cells(id,px,marker)",
        "A3. Config: SAM2 Params", 
        "D1. SAM2 Result: Mask-Only", 
        "D2. SAM2 Score: Mask-Only",
        "E1. Summary", 
        "C3. Analyze: Seg Histogram",
        "Legend"
    ]
    
    panels = [panel1, panel2, panel3, panel4, panel5, panel6, panel7, panel8, panel9,
              panel10, panel11, panel12]
    
    panels_with_titles = [add_title_bar(p, t) for p, t in zip(panels, titles)]
    
    # Auto-calculate grid layout
    n_panels = len(panels_with_titles)
    n_cols = math.ceil(math.sqrt(n_panels))
    n_rows = math.ceil(n_panels / n_cols)
    
    empty_panel = np.ones_like(panels_with_titles[0]) * 255
    total_slots = n_rows * n_cols
    padded_panels = panels_with_titles + [empty_panel] * (total_slots - n_panels)
    
    rows = []
    for row_idx in range(n_rows):
        start_idx = row_idx * n_cols
        end_idx = start_idx + n_cols
        row = np.concatenate(padded_panels[start_idx:end_idx], axis=1)
        rows.append(row)
    
    comparison = np.concatenate(rows, axis=0)
    
    if save_comparison_image:
        cv2.imwrite(f"{output_dir}/comparison/{base_name}_compare.png", comparison)
    
    # === Save A1_C1_D1 combined image ===
    if save_combined_image:
        panel_a1 = panels_with_titles[0]
        panel_c1 = panels_with_titles[4]
        
        # 使用合并后的 merged mask (sam_mask_box_mask 参数位置传入的是 merged)
        panel_d1_clean = cv2.cvtColor(original.copy(), cv2.COLOR_RGB2BGR)
        merged_max_id = int(np.max(sam_mask_box_mask)) if sam_mask_box_mask is not None else 0
        if merged_max_id > 0:
            for inst_id in range(1, merged_max_id + 1):
                inst_mask = sam_mask_box_mask == inst_id
                mask_area = np.sum(inst_mask)
                if mask_area >= 10:
                    panel_d1_clean[inst_mask] = (0, 0, 255)  # 统一使用红色
        panel_d1_clean = add_title_bar(panel_d1_clean, "D1. SAM2 Result: Merged")
        
        a1_c1_d1_combined = np.concatenate([panel_a1, panel_c1, panel_d1_clean], axis=1)
        os.makedirs(f"{output_dir}/combined", exist_ok=True)
        cv2.imwrite(f"{output_dir}/combined/{base_name}_A1_C1_D1.png", a1_c1_d1_combined)


def save_original_sam_comparison(output_dir: str, base_name: str, 
                                   original: np.ndarray, sam_mask: np.ndarray,
                                   add_titles: bool = True) -> str:
    """
    保存原图与 SAM2 结果的左右拼接对比图。
    
    Args:
        output_dir: 输出目录
        base_name: 图像基础名
        original: 原始图像 (RGB)
        sam_mask: SAM2 实例分割掩码
        add_titles: 是否添加标题栏
        
    Returns:
        保存的文件路径
    """
    h, w = original.shape[:2]
    
    # 左侧：原图
    left_panel = cv2.cvtColor(original.copy(), cv2.COLOR_RGB2BGR)
    
    # 右侧：原图 + SAM2 掩码叠加
    right_panel = cv2.cvtColor(original.copy(), cv2.COLOR_RGB2BGR)
    
    if sam_mask is not None:
        max_id = int(np.max(sam_mask))
        if max_id > 0:
            # 生成不同颜色用于区分不同实例
            colors = generate_distinct_colors(max_id)
            
            for inst_id in range(1, max_id + 1):
                inst_mask = sam_mask == inst_id
                mask_area = np.sum(inst_mask)
                if mask_area >= 10:
                    color = colors[inst_id - 1] if inst_id - 1 < len(colors) else (255, 0, 0)
                    # BGR 格式
                    bgr_color = (color[2], color[1], color[0])
                    
                    # 半透明叠加
                    alpha = 0.5
                    right_panel[inst_mask] = (
                        np.array(right_panel[inst_mask]) * (1 - alpha) + 
                        np.array(bgr_color) * alpha
                    ).astype(np.uint8)
                    
                    # 绘制轮廓
                    mask_binary = inst_mask.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(right_panel, contours, -1, bgr_color, 2)
    
    # 添加标题栏
    if add_titles:
        left_panel = add_title_bar(left_panel, "Original Image")
        right_panel = add_title_bar(right_panel, "SAM2 Segmentation Result")
    
    # 左右拼接
    comparison = np.concatenate([left_panel, right_panel], axis=1)
    
    # 保存
    save_dir = os.path.join(output_dir, "original_sam_comparison")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{base_name}_original_vs_sam2.png")
    cv2.imwrite(save_path, comparison)
    
    return save_path
