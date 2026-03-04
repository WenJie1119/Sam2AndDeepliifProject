#!/usr/bin/env python3
"""
pipeline_visualization.py — 详细的流水线可视化模块

生成 step-by-step 的可视化图像，展示细胞提取和分类的完整过程。

主流程：
    1. Step 1: 前景提取 - 识别所有细胞区域
    2. Step 2: 像素分类 - 基于 Seg 的 R/B 通道分类每个像素
    3. Step 3: Marker 分析 - 可视化 Marker 强度分布
    4. Step 4: 细胞分类 - 最终的阳性/阴性细胞判定
    5. Step 5: SAM2 输入 - 为阳性细胞生成 mask prompts
    6. 生成汇总 CSV 和 Summary 图像
"""

import os
import numpy as np
import cv2
from typing import Optional
import matplotlib.pyplot as plt

# 导入共享辅助函数，确保和 extract_cells_from_seg 使用相同逻辑
from src.cell_extraction import compute_posneg_mask, compute_marker_threshold


# ============================================================================
# Step 1: 前景提取
# ============================================================================
def save_step1_foreground(is_foreground: np.ndarray, vis_dir: str) -> np.ndarray:
    """保存前景掩码可视化"""
    foreground_mask = (is_foreground * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(vis_dir, "step1_seg_foreground.png"), foreground_mask)
    return foreground_mask


# ============================================================================
# Step 2: 像素分类 (R-B 差值)
# ============================================================================
def save_step2_posneg_pixels(is_foreground: np.ndarray, 
                              is_pos_pixel: np.ndarray, 
                              rb_diff: np.ndarray, 
                              vis_dir: str) -> np.ndarray:
    """保存 R-B 差值热力图和阳性/阴性像素可视化"""
    h, w = is_foreground.shape
    
    # 2a: R-B 差值热力图
    fig, ax = plt.subplots(figsize=(10, 10))
    rb_masked = np.where(is_foreground, rb_diff, np.nan)
    im = ax.imshow(rb_masked, cmap='RdBu_r', vmin=-100, vmax=100)
    ax.set_title('R-B Difference Heatmap (Red=Positive, Blue=Negative)', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.8, label='R - B')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "step2_rb_difference_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2b: 阳性/阴性像素可视化
    posneg_vis = np.zeros((h, w, 3), dtype=np.uint8)
    posneg_vis[is_foreground & is_pos_pixel] = [255, 0, 0]   # Red = Positive
    posneg_vis[is_foreground & ~is_pos_pixel] = [0, 0, 255]  # Blue = Negative
    cv2.imwrite(os.path.join(vis_dir, "step2_seg_pos_neg_pixels.png"), 
                cv2.cvtColor(posneg_vis, cv2.COLOR_RGB2BGR))
    
    return posneg_vis


# ============================================================================
# Step 3: Marker 分析
# ============================================================================
def save_step3_marker_analysis(marker_gray: np.ndarray, 
                                marker_thresh: int,
                                original_image: Optional[np.ndarray],
                                seg_array: np.ndarray,
                                vis_dir: str) -> None:
    """保存 Marker 热力图、叠加图和阈值图"""
    h, w = marker_gray.shape
    
    # 3a: Marker 热力图
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(marker_gray, cmap='hot')
    ax.set_title('Marker Intensity Heatmap', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Intensity')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "step3_marker_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3b: Marker 叠加在原图上
    base_img = original_image if original_image is not None else seg_array
    if base_img.ndim == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
    overlay = base_img.copy().astype(np.float32)
    marker_normalized = (marker_gray / 255.0)[:,:,np.newaxis]
    overlay_color = np.array([255, 255, 0], dtype=np.float32)  # Yellow
    overlay = overlay * (1 - marker_normalized * 0.5) + overlay_color * marker_normalized * 0.5
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(vis_dir, "step3_marker_overlay.png"), 
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # 3c: Marker 超过阈值的区域
    marker_above = np.zeros((h, w, 3), dtype=np.uint8)
    marker_above[marker_gray > marker_thresh] = [255, 255, 255]
    cv2.imwrite(os.path.join(vis_dir, "step3_marker_above_thresh.png"), marker_above)


# ============================================================================
# Step 4: 细胞分类
# ============================================================================
def save_step4_cell_classification(cells_info: list[dict], 
                                    h: int, w: int,
                                    vis_dir: str) -> tuple:
    """保存所有细胞、阳性细胞、阴性细胞的可视化"""
    np.random.seed(42)
    num_cells = len(cells_info)
    cell_colors = np.random.randint(50, 255, size=(num_cells + 1, 3), dtype=np.uint8)
    
    all_cells_vis = np.zeros((h, w, 3), dtype=np.uint8)
    pos_cells_vis = np.zeros((h, w, 3), dtype=np.uint8)
    neg_cells_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    for idx, cell in enumerate(cells_info):
        coords = cell['coords']
        color = tuple(int(c) for c in cell_colors[idx])
        all_cells_vis[coords[:, 0], coords[:, 1]] = color
        
        if cell.get('is_positive', False):
            pos_cells_vis[coords[:, 0], coords[:, 1]] = [255, 100, 100]  # Red
        else:
            neg_cells_vis[coords[:, 0], coords[:, 1]] = [100, 100, 255]  # Blue
    
    cv2.imwrite(os.path.join(vis_dir, "step4_all_cells.png"), 
                cv2.cvtColor(all_cells_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(vis_dir, "step4_positive_cells.png"), 
                cv2.cvtColor(pos_cells_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(vis_dir, "step4_negative_cells.png"), 
                cv2.cvtColor(neg_cells_vis, cv2.COLOR_RGB2BGR))
    
    return all_cells_vis, pos_cells_vis, neg_cells_vis


# ============================================================================
# Step 5: SAM2 输入 (Mask Prompts)
# ============================================================================
def save_step5_mask_prompts(cells_info: list[dict], 
                             h: int, w: int,
                             vis_dir: str) -> np.ndarray:
    """保存 SAM2 的 mask prompts (全分辨率和 256x256)"""
    positive_cells = [c for c in cells_info if c.get('is_positive', False)]
    
    # 5a: 全分辨率 prompts
    all_prompts_fullres = np.zeros((h, w), dtype=np.uint8)
    for cell in positive_cells:
        coords = cell['coords']
        all_prompts_fullres[coords[:, 0], coords[:, 1]] = 255
    cv2.imwrite(os.path.join(vis_dir, "step5_all_prompts_fullres.png"), all_prompts_fullres)
    
    # 5b: 256x256 版本
    all_prompts_256 = cv2.resize(all_prompts_fullres, (256, 256), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(vis_dir, "step5_all_prompts_256x256.png"), all_prompts_256)
    
    # 5c: 256x256 热力图
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(all_prompts_256, cmap='hot')
    ax.set_title('SAM2 Mask Prompts (256x256)', fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "step5_all_prompts_256x256_heatmap.png"), dpi=100, bbox_inches='tight')
    plt.close()
    
    # 5d: 单独每个细胞的 prompt
    for cell in positive_cells:
        cell_id = cell['id']
        coords = cell['coords']
        cell_mask = np.zeros((h, w), dtype=np.uint8)
        cell_mask[coords[:, 0], coords[:, 1]] = 255
        cell_mask_256 = cv2.resize(cell_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(vis_dir, f"step5_mask_prompt_cell_{cell_id}.png"), cell_mask_256)
    
    print(f"    Saved {len(positive_cells)} individual cell prompts")
    return all_prompts_fullres


# ============================================================================
# Step 6: 过滤后的 SAM2 输入 (经过 filter_positive_cells)
# ============================================================================
def save_step6_filtered_prompts(cells_info: list[dict],
                                 filtered_cells_info: list[dict],
                                 h: int, w: int,
                                 filter_params: dict,
                                 vis_dir: str) -> np.ndarray:
    """
    保存经过 filter_positive_cells 过滤后的 SAM2 mask prompts。
    
    对比原始阳性细胞和过滤后的细胞，明确显示哪些细胞被过滤掉了。
    """
    # 获取原始阳性细胞和过滤后的细胞
    original_positive_cells = [c for c in cells_info if c.get('is_positive', False)]
    filtered_ids = set(c['id'] for c in filtered_cells_info)
    
    # 6a: 过滤后的 prompts (全分辨率)
    filtered_prompts = np.zeros((h, w), dtype=np.uint8)
    for cell in filtered_cells_info:
        coords = cell['coords']
        filtered_prompts[coords[:, 0], coords[:, 1]] = 255
    cv2.imwrite(os.path.join(vis_dir, "step6_filtered_prompts_fullres.png"), filtered_prompts)
    
    # 6b: 256x256 版本
    filtered_prompts_256 = cv2.resize(filtered_prompts, (256, 256), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(vis_dir, "step6_filtered_prompts_256x256.png"), filtered_prompts_256)
    
    # 6c: 对比可视化 - 绿色=保留, 红色=被过滤掉
    comparison_vis = np.zeros((h, w, 3), dtype=np.uint8)
    for cell in original_positive_cells:
        coords = cell['coords']
        if cell['id'] in filtered_ids:
            # 保留的细胞：绿色
            comparison_vis[coords[:, 0], coords[:, 1]] = [0, 255, 0]
        else:
            # 被过滤掉的细胞：红色
            comparison_vis[coords[:, 0], coords[:, 1]] = [255, 0, 0]
    cv2.imwrite(os.path.join(vis_dir, "step6_filter_comparison.png"), 
                cv2.cvtColor(comparison_vis, cv2.COLOR_RGB2BGR))
    
    # 6d: 256x256 热力图
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(filtered_prompts_256, cmap='hot')
    params_text = f"marker_sum>{filter_params.get('marker_sum_thresh', '?')}, " \
                  f"marker_max>={filter_params.get('marker_max_thresh', '?')}, " \
                  f"pixels>={filter_params.get('min_pixel_count', '?')}"
    ax.set_title(f'Filtered SAM2 Prompts (256x256)\n{params_text}', fontsize=9)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "step6_filtered_prompts_256x256_heatmap.png"), dpi=100, bbox_inches='tight')
    plt.close()
    
    # 打印过滤统计
    num_original = len(original_positive_cells)
    num_filtered = len(filtered_cells_info)
    num_removed = num_original - num_filtered
    print(f"    Step 6: {num_original} -> {num_filtered} cells (removed {num_removed} by filter_positive_cells)")
    
    return filtered_prompts


# ============================================================================
# CSV: 分类详情
# ============================================================================
def save_classification_csv(cells_info: list[dict], 
                             marker_thresh: int,
                             vis_dir: str) -> None:
    """保存每个细胞的详细分类信息 CSV"""
    csv_path = os.path.join(vis_dir, "cells_classification_details.csv")
    
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("cell_id,pixel_count,pos_pixels,neg_pixels,marker_max,marker_mean,"
                "initial_positive,is_positive,center_y,center_x,decision_reason\n")
        
        for cell in cells_info:
            cell_id = cell['id']
            pixel_count = cell['pixel_count']
            pos_pix = cell.get('count_positive_pixels', 0)
            neg_pix = cell.get('count_negative_pixels', 0)
            marker_max = cell['marker_max']
            marker_mean = cell['marker_mean']
            
            initial_positive = pos_pix >= neg_pix
            is_positive = cell.get('is_positive', False)
            center_y, center_x = cell['center']
            
            # 判定原因
            if is_positive:
                if initial_positive:
                    reason = "Seg(R>=B majority)"
                else:
                    reason = f"Marker(max={marker_max}>{marker_thresh})"
            else:
                reason = "Negative"
            
            f.write(f"{cell_id},{pixel_count},{pos_pix},{neg_pix},{marker_max},"
                    f"{marker_mean:.2f},{initial_positive},{is_positive},"
                    f"{center_y},{center_x},{reason}\n")
    
    print(f"    Saved classification CSV with {len(cells_info)} cells")


# ============================================================================
# Summary: 汇总图
# ============================================================================
def save_summary_image(seg_array: np.ndarray,
                        foreground_mask: np.ndarray,
                        posneg_vis: np.ndarray,
                        marker_gray: np.ndarray,
                        all_cells_vis: np.ndarray,
                        pos_cells_vis: np.ndarray,
                        neg_cells_vis: np.ndarray,
                        all_prompts: np.ndarray,
                        cells_info: list[dict],
                        base_name: str,
                        marker_thresh: int,
                        seg_thresh: int,
                        vis_dir: str) -> None:
    """保存 2x4 汇总图"""
    positive_cells = [c for c in cells_info if c.get('is_positive', False)]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1
    axes[0, 0].imshow(seg_array)
    axes[0, 0].set_title('1. Original Seg')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(foreground_mask, cmap='gray')
    axes[0, 1].set_title('2. Foreground Mask')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(posneg_vis)
    axes[0, 2].set_title('3. Pos/Neg Pixels')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(marker_gray, cmap='hot')
    axes[0, 3].set_title('4. Marker Heatmap')
    axes[0, 3].axis('off')
    
    # Row 2
    axes[1, 0].imshow(all_cells_vis)
    axes[1, 0].set_title(f'5. All Cells ({len(cells_info)})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pos_cells_vis)
    axes[1, 1].set_title(f'6. Positive ({len(positive_cells)})')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(neg_cells_vis)
    axes[1, 2].set_title(f'7. Negative ({len(cells_info) - len(positive_cells)})')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(all_prompts, cmap='gray')
    axes[1, 3].set_title('8. SAM2 Prompts')
    axes[1, 3].axis('off')
    
    plt.suptitle(f'{base_name} - Pipeline Summary\n'
                 f'marker_thresh={marker_thresh}, seg_thresh={seg_thresh}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "pipeline_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 主流程入口
# ============================================================================
def save_pipeline_visualization(seg_array: np.ndarray,
                                 marker_array: np.ndarray,
                                 cells_info: list[dict],
                                 output_dir: str,
                                 base_name: str,
                                 seg_thresh: int = 120,
                                 marker_thresh: Optional[int] = None,
                                 original_image: Optional[np.ndarray] = None,
                                 filtered_cells_info: Optional[list[dict]] = None,
                                 filter_params: Optional[dict] = None) -> str:
    """
    主流程：生成详细的 pipeline 可视化图像和 CSV 数据。
    
    流程步骤：
        Step 1: 前景提取
        Step 2: 像素分类 (R-B 差值)
        Step 3: Marker 分析
        Step 4: 细胞分类
        Step 5: SAM2 Mask Prompts (所有 is_positive 细胞)
        Step 6: 过滤后的 Prompts (经过 filter_positive_cells)
        + CSV 详情 + Summary 汇总图
    """
    # ========== 准备工作 ==========
    vis_dir = os.path.join(output_dir, "pipeline_visualization")
    os.makedirs(vis_dir, exist_ok=True)
    print(f"  > Saving pipeline visualization to {vis_dir}/")
    
    # 转换 Marker 为灰度
    if marker_array.ndim == 3:
        marker_gray = cv2.cvtColor(marker_array, cv2.COLOR_RGB2GRAY)
    else:
        marker_gray = marker_array.copy()
    
    h, w = seg_array.shape[:2]
    
    # 使用共享辅助函数 (与 extract_cells_from_seg 一致)
    posneg_mask, is_foreground, rb_diff = compute_posneg_mask(seg_array, seg_thresh)
    is_pos_pixel = (posneg_mask == 2)
    
    if marker_thresh is None:
        marker_thresh = compute_marker_threshold(marker_gray)
    
    # ========== Step 1: 前景提取 ==========
    foreground_mask = save_step1_foreground(is_foreground, vis_dir)
    
    # ========== Step 2: 像素分类 ==========
    posneg_vis = save_step2_posneg_pixels(is_foreground, is_pos_pixel, rb_diff, vis_dir)
    
    # ========== Step 3: Marker 分析 ==========
    save_step3_marker_analysis(marker_gray, marker_thresh, original_image, seg_array, vis_dir)
    
    # ========== Step 4: 细胞分类 ==========
    all_cells_vis, pos_cells_vis, neg_cells_vis = save_step4_cell_classification(
        cells_info, h, w, vis_dir)
    
    # ========== Step 5: SAM2 Mask Prompts ==========
    all_prompts = save_step5_mask_prompts(cells_info, h, w, vis_dir)
    
    # ========== Step 6: 过滤后的 Prompts (可选) ==========
    filtered_prompts = None
    if filtered_cells_info is not None:
        if filter_params is None:
            filter_params = {}
        filtered_prompts = save_step6_filtered_prompts(
            cells_info, filtered_cells_info, h, w, filter_params, vis_dir)
    
    # ========== CSV 详情 ==========
    save_classification_csv(cells_info, marker_thresh, vis_dir)
    
    # ========== Summary 汇总图 ==========
    save_summary_image(
        seg_array, foreground_mask, posneg_vis, marker_gray,
        all_cells_vis, pos_cells_vis, neg_cells_vis, 
        filtered_prompts if filtered_prompts is not None else all_prompts,
        cells_info, base_name, marker_thresh, seg_thresh, vis_dir)
    
    return vis_dir
