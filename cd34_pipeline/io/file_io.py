#!/usr/bin/env python3
"""
file_io.py — 文件输入输出模块

包含：
- 图像文件读写
- CSV 导出
- 结果保存辅助函数
"""

import os
import csv
import json
import shutil
from typing import Optional

import numpy as np
import cv2
from PIL import Image


def get_image_files(input_path: str) -> tuple[str, list[str]]:
    """
    获取输入路径下的所有图像文件。
    
    Args:
        input_path: 输入目录路径或单个文件路径
        
    Returns:
        tuple: (input_dir, image_files) 
               - input_dir: 图像所在的目录
               - image_files: 图像文件名列表
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    if os.path.isfile(input_path):
        # Single file mode
        input_dir = os.path.dirname(input_path)
        image_files = [os.path.basename(input_path)]
    else:
        # Directory mode
        input_dir = input_path
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        image_files = sorted([
            f for f in os.listdir(input_path) 
            if f.lower().endswith(valid_extensions)
        ])
    
    return input_dir, image_files


def read_image(image_path: str) -> np.ndarray:
    """
    读取图像并转换为 RGB numpy 数组。
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        RGB numpy 数组 (H, W, 3)
    """
    pil_image = Image.open(image_path).convert('RGB')
    return np.array(pil_image)


def save_positive_cells_csv(output_path: str, cells_info: list[dict]):
    """
    Save cells information to CSV file.
    
    Args:
        output_path: Path to save CSV file
        cells_info: List of cell info dicts
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Cell_ID', 'Is_Positive', 'Pixel_Count', 'Marker_Sum', 'Marker_Mean', 
                        'Marker_Max', 'Marker_Min', 'Center_Y', 'Center_X'])
        for cell in cells_info:
            writer.writerow([
                cell['id'],
                'Yes' if cell.get('is_positive', True) else 'No',
                cell['pixel_count'],
                cell['marker_sum'],
                f"{cell['marker_mean']:.2f}",
                cell['marker_max'],
                cell['marker_min'],
                cell['center'][0],
                cell['center'][1]
            ])


def save_deepliif_outputs(results: dict, save_dir: str, prefix: str = "step2_deepliif",
                          save_all: bool = False):
    """
    保存 DeepLIIF 推理结果到指定目录。

    Args:
        results: DeepLIIF 推理返回的字典 (key -> PIL.Image)
        save_dir: 保存目录
        prefix: 文件名前缀，生成 {prefix}_{key}.png
        save_all: False 只保存 Seg 和 Marker，True 保存全部
    """
    os.makedirs(save_dir, exist_ok=True)

    for key, val_img in results.items():
        if isinstance(val_img, Image.Image):
            if not save_all and key not in ('Seg', 'Marker'):
                continue
            val_img.save(os.path.join(save_dir, f"{prefix}_{key}.png"))


def save_sam2_mask_visualization(mask_data: np.ndarray, 
                                  output_path: str,
                                  cells_info: list[dict] = None,
                                  filtered_ids: set = None,
                                  colors: list = None):
    """
    保存 SAM2 掩码可视化结果。
    
    Args:
        mask_data: 实例分割掩码 (H, W)
        output_path: 输出文件路径
        cells_info: 细胞信息列表
        filtered_ids: 被过滤的实例 ID 集合
        colors: 颜色列表
    """
    from cd34_pipeline.cell.mask_utils import generate_distinct_colors
    
    h, w = mask_data.shape
    labeled_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 获取 mask 中实际存在的 ID
    unique_ids = np.unique(mask_data)
    unique_ids = unique_ids[unique_ids > 0]  # 排除背景
    
    num_instances = len(unique_ids)
    if colors is None:
        colors = generate_distinct_colors(num_instances) if num_instances > 0 else []
    
    if filtered_ids is None:
        filtered_ids = set()
    
    # Draw all instances with distinct colors
    for color_idx, inst_id in enumerate(sorted(unique_ids)):
        inst_mask = mask_data == inst_id
        mask_area = np.sum(inst_mask)
        if mask_area > 0:
            color = colors[color_idx] if color_idx < len(colors) else (255, 255, 255)
            labeled_img[inst_mask] = color
    
    # Draw labels - 计算每个区域的中心并标注
    for color_idx, inst_id in enumerate(sorted(unique_ids)):
        inst_mask = mask_data == inst_id
        final_area = np.sum(inst_mask)
        
        if final_area > 0:
            # 计算区域的质心
            ys, xs = np.where(inst_mask)
            center_y = int(np.mean(ys))
            center_x = int(np.mean(xs))
            
            label = str(int(inst_id))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(labeled_img, label, (center_x - 8, center_y + 5), 
                       font, 0.6, (0, 0, 0), 3)
            cv2.putText(labeled_img, label, (center_x - 8, center_y + 5), 
                       font, 0.6, (255, 255, 255), 2)
    
    # Draw BLUE outlines for FILTERED cells (仅当 cells_info 存在时)
    if cells_info:
        for idx, cell in enumerate(cells_info):
            inst_id = idx + 1
            if inst_id in filtered_ids:
                coords = cell['coords']
                center_y, center_x = cell['center']
                
                cell_mask = np.zeros((h, w), dtype=np.uint8)
                cell_mask[coords[:, 0], coords[:, 1]] = 255
                contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                cv2.drawContours(labeled_img, contours, -1, (0, 0, 255), 2)
                
                label = f"F{inst_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(labeled_img, label, (center_x - 12, center_y + 5), 
                           font, 0.5, (0, 0, 0), 3)
                cv2.putText(labeled_img, label, (center_x - 12, center_y + 5), 
                           font, 0.5, (0, 0, 255), 2)
    
    cv2.imwrite(output_path, cv2.cvtColor(labeled_img, cv2.COLOR_RGB2BGR))


def save_mask_npy(mask: np.ndarray, output_path: str) -> str:
    """
    保存实例分割 mask 为 npy 格式。

    Args:
        mask: 实例分割掩码 (H, W)，每个像素值为实例ID (0=背景)
        output_path: 输出文件路径 (.npy)

    Returns:
        str: 保存的 npy 文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 根据实例数量选择合适的数据类型
    max_id = int(np.max(mask))
    if max_id <= 255:
        save_mask = mask.astype(np.uint8)
    elif max_id <= 65535:
        save_mask = mask.astype(np.uint16)
    else:
        save_mask = mask.astype(np.uint32)

    # 保存 npy 文件
    np.save(output_path, save_mask)

    print(f"    Saved mask npy: {output_path} (dtype={save_mask.dtype}, max_id={max_id})")
    return output_path


def load_mask_npy(npy_path: str) -> np.ndarray:
    """
    加载 npy 格式的实例分割 mask。

    Args:
        npy_path: npy 文件路径

    Returns:
        mask 数组
    """
    return np.load(npy_path)


def save_seg_probability_npy(seg_image, output_path: str) -> str:
    """
    保存 DeepLIIF Seg 概率图为 npy 格式。
    
    Seg 图像的颜色通道含义:
    - 红色通道 (R, channel 0): 阳性细胞概率 (0-255)
    - 绿色通道 (G, channel 1): 背景概率
    - 蓝色通道 (B, channel 2): 阴性细胞概率 (0-255)
    
    判断逻辑 (来自 DeepLIIF postprocessing.py):
    - 当 R + B > 阈值 且 G <= 80 时，该像素被识别为细胞
    - 如果 R >= B，则该细胞为阳性(positive)
    - 如果 R < B，则该细胞为阴性(negative)
    
    Args:
        seg_image: PIL Image 或 numpy array (RGB)，DeepLIIF 的 Seg 输出
        output_path: 输出文件路径 (.npy)
        metadata: 可选的元数据字典
        
    Returns:
        str: 保存的 npy 文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换为 numpy array
    if isinstance(seg_image, Image.Image):
        seg_array = np.array(seg_image)
    else:
        seg_array = seg_image
    
    # 确保是 RGB 格式 (H, W, 3)
    if len(seg_array.shape) == 2:
        # 灰度图，扩展为3通道
        seg_array = np.stack([seg_array, seg_array, seg_array], axis=-1)
    
    # 保存为 uint8 npy
    np.save(output_path, seg_array.astype(np.uint8))

    print(f"    Saved Seg probability npy: {output_path} (shape={seg_array.shape})")
    return output_path


def handle_no_positive_cells(img_path: str, img_name: str, output_dir: str, base_name: str):
    """
    无阳性细胞时：将原图及相关中间文件移到 no_positive_cells 目录。
    """
    no_positive_dir = os.path.join(output_dir, "no_positive_cells")
    os.makedirs(no_positive_dir, exist_ok=True)

    # 移动原始图像
    if os.path.exists(img_path):
        dest_path = os.path.join(no_positive_dir, img_name)
        shutil.move(img_path, dest_path)
        print(f"    Moved image with no positive cells to: {dest_path}")

    # 移动 background 图像 (如果存在)
    bg_file = os.path.join(output_dir, "background", f"{base_name}.png")
    if os.path.exists(bg_file):
        dest_bg = os.path.join(no_positive_dir, f"{base_name}_background.png")
        shutil.move(bg_file, dest_bg)
        print(f"    Moved background file to: {dest_bg}")


def save_cell_groups_csv(cell_groups: list, output_dir: str, base_name: str,
                         distance_threshold: float) -> str:
    """
    保存细胞分组信息 CSV（组ID、成员、距离等）。
    """
    group_csv_dir = os.path.join(output_dir, "cell_groups")
    os.makedirs(group_csv_dir, exist_ok=True)
    group_csv_path = os.path.join(group_csv_dir, f"{base_name}_cell_groups.csv")

    with open(group_csv_path, 'w') as f:
        f.write("group_id,num_cells,member_ids,total_pixels,center_y,center_x,distance_threshold,member_distances\n")
        for group in cell_groups:
            member_ids_str = ';'.join(map(str, group['member_ids']))
            center_y, center_x = group['center']

            member_cells = group['member_cells']
            n_members = len(member_cells)
            if n_members > 1:
                centers = np.array([c['center'] for c in member_cells])  # (n, 2)
                cell_ids = [c['id'] for c in member_cells]
                # 向量化两两距离计算
                diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (n, n, 2)
                dists = np.sqrt(np.sum(diffs ** 2, axis=2))  # (n, n)
                pi, pj = np.triu_indices(n_members, k=1)
                member_distances = [f"{cell_ids[i]}-{cell_ids[j]}:{dists[i, j]:.1f}"
                                    for i, j in zip(pi, pj)]
            else:
                member_distances = []

            distances_str = ';'.join(member_distances) if member_distances else 'single_cell'
            f.write(f"{group['group_id']},{len(group['member_ids'])},\"{member_ids_str}\",{group['total_pixels']},{center_y},{center_x},{distance_threshold},\"{distances_str}\"\n")

    print(f"    Saved grouping info to: {group_csv_path}")
    return group_csv_path


def save_merged_regions_csv(sam_mask_merged: np.ndarray, scores_merged: list,
                            output_dir: str, base_name: str):
    """
    保存合并区域统计 CSV（区域ID、像素数、分数）。
    """
    if sam_mask_merged is None or np.max(sam_mask_merged) == 0:
        return

    merged_regions_csv_path = os.path.join(output_dir, "merged_regions", f"{base_name}_merged_regions.csv")
    os.makedirs(os.path.dirname(merged_regions_csv_path), exist_ok=True)

    unique_ids = np.unique(sam_mask_merged)
    unique_ids = unique_ids[unique_ids > 0]

    with open(merged_regions_csv_path, 'w') as f:
        f.write("region_id,pixel_count,avg_score,member_instances\n")
        for region_id in sorted(unique_ids):
            pixel_count = int(np.sum(sam_mask_merged == region_id))
            score_info = next((s for s in scores_merged if s[0] == region_id), None)
            if score_info:
                avg_score = score_info[1]
                member_ids = score_info[2] if len(score_info) > 2 else []
            else:
                avg_score = 0.0
                member_ids = []
            f.write(f"{region_id},{pixel_count},{avg_score:.4f},\"{member_ids}\"\n")

    print(f"  > Saved merged regions CSV: {merged_regions_csv_path}")
    print(f"    Total {len(unique_ids)} regions")


def compute_geojson_statistics(geojson_path: str, output_dir: str = None) -> dict:
    """
    读取 GeoJSON 文件，计算每个区域的面积、重心，以及全局统计（数量、面积均值、面积标准差）。

    使用 Shoelace 公式计算多边形面积和重心，不依赖 shapely。

    Args:
        geojson_path: GeoJSON 文件路径（list of Features，或 FeatureCollection）
        output_dir: 输出目录，若提供则保存 CSV 统计文件

    Returns:
        dict: {
            'count': int,
            'area_mean': float,
            'area_std': float,
            'regions': list[dict]  # 每个 dict: {id, area, centroid_x, centroid_y}
        }
    """
    import math

    with open(geojson_path, 'r') as f:
        data = json.load(f)

    # 兼容 FeatureCollection 和裸 list
    if isinstance(data, dict) and 'features' in data:
        features = data['features']
    elif isinstance(data, list):
        features = data
    else:
        print(f"  WARNING: Unrecognized GeoJSON format in {geojson_path}")
        return {'count': 0, 'area_mean': 0, 'area_std': 0, 'regions': []}

    def _polygon_area_and_centroid(ring):
        """Shoelace 公式计算多边形面积和重心。"""
        n = len(ring)
        if n > 1 and ring[0] == ring[-1]:
            n -= 1
        if n < 3:
            return 0.0, 0.0, 0.0

        signed_area = 0.0
        cx = 0.0
        cy = 0.0
        for i in range(n):
            j = (i + 1) % n
            cross = ring[i][0] * ring[j][1] - ring[j][0] * ring[i][1]
            signed_area += cross
            cx += (ring[i][0] + ring[j][0]) * cross
            cy += (ring[i][1] + ring[j][1]) * cross

        area = abs(signed_area) / 2.0
        if area > 0:
            cx = abs(cx) / (6.0 * area)
            cy = abs(cy) / (6.0 * area)
        else:
            cx = sum(p[0] for p in ring[:n]) / n
            cy = sum(p[1] for p in ring[:n]) / n
        return area, cx, cy

    regions = []
    for idx, feat in enumerate(features):
        geom = feat.get('geometry', {})
        coords = geom.get('coordinates', [])
        if geom.get('type') == 'Polygon' and len(coords) > 0:
            ring = coords[0]
            area, cx, cy = _polygon_area_and_centroid(ring)
        else:
            area, cx, cy = 0.0, 0.0, 0.0

        regions.append({
            'id': idx + 1,
            'area': area,
            'centroid_x': cx,
            'centroid_y': cy,
        })

    count = len(regions)
    areas = [r['area'] for r in regions]

    if count > 0:
        area_mean = sum(areas) / count
        area_std = math.sqrt(sum((a - area_mean) ** 2 for a in areas) / count)
    else:
        area_mean = 0.0
        area_std = 0.0

    # 保存 CSV
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(geojson_path))[0]
        csv_path = os.path.join(output_dir, f"{stem}_statistics.csv")

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['region_id', 'area_px2', 'centroid_x', 'centroid_y'])
            for r in regions:
                writer.writerow([r['id'], f"{r['area']:.2f}",
                                 f"{r['centroid_x']:.2f}", f"{r['centroid_y']:.2f}"])

        # 写入汇总行
        summary_path = os.path.join(output_dir, f"{stem}_summary.csv")
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['count', count])
            writer.writerow(['area_mean', f"{area_mean:.2f}"])
            writer.writerow(['area_std', f"{area_std:.2f}"])
            writer.writerow(['area_min', f"{min(areas):.2f}" if areas else '0'])
            writer.writerow(['area_max', f"{max(areas):.2f}" if areas else '0'])

        print(f"  Statistics CSV: {csv_path}")
        print(f"  Summary CSV:    {summary_path}")

        # 面积分布直方图
        if count > 0:
            hist_path = os.path.join(output_dir, f"{stem}_area_histogram.png")
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                areas_np = np.array(areas)
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # 左图：线性尺度
                axes[0].hist(areas_np, bins=100, color='steelblue', edgecolor='white')
                axes[0].set_xlabel('Area (px\u00b2)')
                axes[0].set_ylabel('Count')
                axes[0].set_title(f'Area Distribution (n={count})')
                axes[0].axvline(np.median(areas_np), color='red', linestyle='--',
                                label=f'median={np.median(areas_np):.0f}')
                axes[0].axvline(area_mean, color='orange', linestyle='-.',
                                label=f'mean={area_mean:.0f}')
                axes[0].legend()

                # 右图：log 尺度
                log_min = np.log10(max(1, areas_np.min()))
                log_max = np.log10(max(2, areas_np.max()))
                axes[1].hist(areas_np, bins=np.logspace(log_min, log_max, 80),
                             color='steelblue', edgecolor='white')
                axes[1].set_xscale('log')
                axes[1].set_xlabel('Area (px\u00b2, log scale)')
                axes[1].set_ylabel('Count')
                axes[1].set_title('Log-scale Distribution')
                axes[1].axvline(np.median(areas_np), color='red', linestyle='--',
                                label=f'median={np.median(areas_np):.0f}')
                axes[1].legend()

                plt.tight_layout()
                plt.savefig(hist_path, dpi=150)
                plt.close()
                print(f"  Histogram:      {hist_path}")
            except ImportError:
                print("  (matplotlib not available, skipping histogram)")

    return {
        'count': count,
        'area_mean': area_mean,
        'area_std': area_std,
        'regions': regions,
    }


def save_sam2_outputs(sam_out_dir: str, original_np: np.ndarray,
                      positive_cells_info: list,
                      sam_mask_only: np.ndarray, sam_mask_only_merged: np.ndarray,
                      filtered_mask_only: list):
    """保存 SAM2 Mask-Only 输出结果（掩码可视化和 prompt 信息）。"""
    from cd34_pipeline.cell.mask_utils import generate_mask_from_cluster, generate_distinct_colors

    # 保存 mask prompts
    mask_prompt_dir = os.path.join(sam_out_dir, "mask_prompts")
    os.makedirs(mask_prompt_dir, exist_ok=True)

    if positive_cells_info and len(positive_cells_info) > 0:
        h, w = original_np.shape[:2]
        colors = generate_distinct_colors(len(positive_cells_info))

        # Combined visualization
        combined_mask_viz = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, cell in enumerate(positive_cells_info):
            coords = cell['coords']
            color = colors[idx] if idx < len(colors) else (255, 0, 0)
            combined_mask_viz[coords[:, 0], coords[:, 1]] = color

            center_y, center_x = cell['center']
            cv2.putText(combined_mask_viz, str(idx + 1), (center_x - 8, center_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(combined_mask_viz, str(idx + 1), (center_x - 8, center_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imwrite(f"{mask_prompt_dir}/mask_prompts_combined.png",
                   cv2.cvtColor(combined_mask_viz, cv2.COLOR_RGB2BGR))

        # Low-res version
        low_res_combined = np.zeros((256, 256), dtype=np.float32)
        for idx, cell in enumerate(positive_cells_info):
            mask_input = generate_mask_from_cluster(cell['coords'], original_np.shape)
            low_res_combined = np.maximum(low_res_combined, mask_input[0])

        low_res_viz = ((low_res_combined + 10) / 20 * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(f"{mask_prompt_dir}/mask_prompts_256x256.png", low_res_viz)

        print(f"  > Saved mask prompts to {mask_prompt_dir}/")

    # 保存 mask-only 结果
    filtered_ids = set(inst_id for inst_id, _ in filtered_mask_only)

    save_sam2_mask_visualization(
        sam_mask_only,
        f"{sam_out_dir}/sam_mask_only.png",
        positive_cells_info,
        filtered_ids
    )

    save_sam2_mask_visualization(
        sam_mask_only_merged,
        f"{sam_out_dir}/sam_mask_only_merged.png",
        positive_cells_info,
        set()
    )
