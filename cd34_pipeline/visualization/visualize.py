#!/usr/bin/env python3
"""
visualization.py — 可视化模块

包含：
- SAM2 结果对比图
- 分组可视化
"""

import os
import numpy as np
import cv2

from cd34_pipeline.cell.mask_utils import generate_distinct_colors


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


def save_grouping_visualization(original_image: np.ndarray, cell_groups: list,
                                cells_info: list, distance_threshold: float,
                                save_path: str):
    """
    保存分组过程的可视化图片：不同组用不同颜色，组内成员连线。
    """
    h, w = original_image.shape[:2]
    viz = original_image.copy()

    np.random.seed(42)
    group_colors = np.random.randint(100, 255, size=(len(cell_groups) + 1, 3))

    for group in cell_groups:
        group_id = group['group_id']
        color = tuple(int(c) for c in group_colors[group_id])
        member_cells = group['member_cells']

        for cell in member_cells:
            coords = cell['coords']
            rows = coords[:, 0].astype(np.intp)
            cols = coords[:, 1].astype(np.intp)
            viz[rows, cols] = (np.array(viz[rows, cols]) * 0.5 + np.array(color) * 0.5).astype(np.uint8)

        if len(member_cells) > 1:
            for i in range(len(member_cells)):
                for j in range(i + 1, len(member_cells)):
                    c1 = member_cells[i]['center']
                    c2 = member_cells[j]['center']
                    cv2.line(viz, (c1[1], c1[0]), (c2[1], c2[0]), color, 2)

        center_y, center_x = group['center']
        label = f"G{group_id}({len(member_cells)})"
        cv2.putText(viz, label, (center_x - 20, center_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(viz, label, (center_x - 20, center_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.putText(viz, f"Groups: {len(cell_groups)}, Threshold: {distance_threshold}px",
               (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(viz, f"Groups: {len(cell_groups)}, Threshold: {distance_threshold}px",
               (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imwrite(save_path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
