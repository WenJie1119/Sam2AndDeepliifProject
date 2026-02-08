#!/usr/bin/env python3
"""
file_io.py — 文件输入输出模块

包含：
- 图像文件读写
- CSV 导出
- LabelMe JSON 导出
- 结果保存辅助函数
"""

import os
import csv
import json
import base64
from pathlib import Path
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


def mask_to_labelme_json(instance_mask: np.ndarray, original_image_path: str, 
                          image_array: np.ndarray, cells_info: list = None,
                          include_image_data: bool = False, 
                          simplify_contour: bool = True,
                          epsilon_ratio: float = 0.002) -> dict:
    """
    将实例分割掩码转换为 LabelMe JSON 格式。
    
    通过提取每个实例的轮廓并转换为多边形点序列，生成可被 LabelMe 工具打开的 JSON 文件。
    
    Args:
        instance_mask: 实例掩码数组 (H, W)，每个像素值为实例ID (0=背景, 1,2,3...=实例)
        original_image_path: 原始图像路径 (用于 imagePath 字段)
        image_array: 原始图像数组 RGB (用于获取尺寸和可选的 base64 编码)
        cells_info: 可选的细胞信息列表，用于生成更丰富的标签
        include_image_data: 是否在 JSON 中嵌入 base64 编码的图像数据
        simplify_contour: 是否简化轮廓以减少点数 (使用 Douglas-Peucker 算法)
        epsilon_ratio: 轮廓简化的容差比例 (相对于轮廓周长)
    
    Returns:
        dict: LabelMe 格式的字典，可直接 json.dump 保存
    """
    h, w = instance_mask.shape[:2]
    shapes = []
    
    # 获取所有唯一的实例ID（排除背景0）
    instance_ids = np.unique(instance_mask)
    instance_ids = instance_ids[instance_ids > 0]
    
    # 用于生成连续的 group_id
    continuous_group_id = 0
    
    for inst_id in instance_ids:
        # 创建单实例二值掩码
        binary_mask = (instance_mask == inst_id).astype(np.uint8) * 255
        
        # 提取轮廓 (RETR_EXTERNAL 只获取外轮廓，忽略孔洞)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) < 3:  # 多边形至少需要3个点
                continue
            
            # 可选：使用 Douglas-Peucker 算法简化轮廓
            if simplify_contour:
                epsilon = epsilon_ratio * cv2.arcLength(contour, True)
                contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # 转换为 LabelMe 格式的点列表 [[x1, y1], [x2, y2], ...]
            points = [[float(pt[0][0]), float(pt[0][1])] for pt in contour]
            
            # 生成连续的 group_id
            continuous_group_id += 1
            
            # 生成标签
            label = "cell"
            group_id = continuous_group_id  # 使用连续编号
            
            # 如果有细胞信息，添加更丰富的标签
            if cells_info and int(inst_id) <= len(cells_info):
                cell = cells_info[int(inst_id) - 1]  # 1-based index
                is_positive = cell.get('is_positive', True)
                
                # 标签格式: positive_cell 或 negative_cell (统一类名，不带编号)
                label = "positive_cell" if is_positive else "negative_cell"
            
            shapes.append({
                "label": label,
                "points": points,
                "group_id": group_id,
                "shape_type": "polygon",
                "flags": {},
                "description": ""
            })
    
    # 构建 LabelMe JSON 结构
    labelme_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": Path(original_image_path).name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }
    
    # 可选：嵌入 base64 图像数据
    if include_image_data and image_array is not None:
        import io
        img_pil = Image.fromarray(image_array)
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        labelme_data["imageData"] = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return labelme_data


def export_labelme_annotation(output_dir: str, base_name: str, 
                               instance_mask: np.ndarray, 
                               original_image_array: np.ndarray,
                               cells_info: list = None,
                               include_image_data: bool = False,
                               original_image_path: str = None) -> tuple[str, int]:
    """
    导出 SAM2 分割结果为 LabelMe JSON 格式。
    
    Args:
        output_dir: 输出目录 (会在其中创建 labelme 子目录)
        base_name: 基础文件名 (不含扩展名)
        instance_mask: 实例分割掩码 (H, W)
        original_image_array: 原始图像数组 RGB
        cells_info: 可选的细胞信息列表
        include_image_data: 是否在 JSON 中嵌入 base64 图像
        original_image_path: 原始图像的绝对路径 (用于 imagePath 字段)
    
    Returns:
        tuple: (JSON文件路径, 多边形数量)
    """
    # 创建 labelme 输出目录
    labelme_dir = os.path.join(output_dir, "labelme")
    os.makedirs(labelme_dir, exist_ok=True)
    
    # 定义输出文件路径
    json_filename = f"{base_name}.json"
    json_path = os.path.join(labelme_dir, json_filename)
    
    # imagePath 使用原始图像的绝对路径（成功后原图会被移动到 labelme/）
    if original_image_path:
        # 使用文件名（因为原图会被移动到同目录）
        image_path_for_json = os.path.basename(original_image_path)
    else:
        image_path_for_json = f"{base_name}.png"
    
    # 生成 LabelMe JSON
    labelme_data = mask_to_labelme_json(
        instance_mask=instance_mask,
        original_image_path=image_path_for_json,
        image_array=original_image_array,
        cells_info=cells_info,
        include_image_data=include_image_data,
        simplify_contour=True,
        epsilon_ratio=0.01
    )
    
    # 保存 JSON 文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_data, f, indent=2, ensure_ascii=False)
    
    num_shapes = len(labelme_data['shapes'])
    print(f"    LabelMe export: {num_shapes} polygons -> {json_path}")
    
    return json_path, num_shapes


def save_deepliif_outputs(results: dict, output_dir: str, base_name: str):
    """
    保存 DeepLIIF 推理结果到指定目录。
    
    Args:
        results: DeepLIIF 推理返回的字典 (key -> PIL.Image)
        output_dir: 输出根目录
        base_name: 图像基础名称
    """
    deepliif_out_dir = os.path.join(output_dir, "deepliif_outputs", base_name)
    os.makedirs(deepliif_out_dir, exist_ok=True)
    
    for key, val_img in results.items():
        if isinstance(val_img, Image.Image):
            val_img.save(f"{deepliif_out_dir}/{key}.png")


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
    from .mask_utils import generate_distinct_colors
    
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


def save_mask_npy(mask: np.ndarray, output_path: str, 
                  metadata: dict = None) -> str:
    """
    保存实例分割 mask 为 npy 格式。
    
    Args:
        mask: 实例分割掩码 (H, W)，每个像素值为实例ID (0=背景)
        output_path: 输出文件路径 (.npy)
        metadata: 可选的元数据字典，将保存为同名 .json 文件
        
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
    
    # 如果有元数据，保存为同名 json 文件
    if metadata:
        json_path = output_path.replace('.npy', '_meta.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"    Saved mask npy: {output_path} (dtype={save_mask.dtype}, max_id={max_id})")
    return output_path


def load_mask_npy(npy_path: str) -> tuple[np.ndarray, dict]:
    """
    加载 npy 格式的实例分割 mask。
    
    Args:
        npy_path: npy 文件路径
        
    Returns:
        tuple: (mask数组, 元数据字典或None)
    """
    mask = np.load(npy_path)
    
    # 尝试加载元数据
    json_path = npy_path.replace('.npy', '_meta.json')
    metadata = None
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    return mask, metadata


def save_seg_probability_npy(seg_image, output_path: str, 
                              metadata: dict = None) -> str:
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
    
    # 构建默认元数据
    default_metadata = {
        'format': 'DeepLIIF_Seg_Probability',
        'channels': {
            'R (channel 0)': 'positive_probability',
            'G (channel 1)': 'background_probability', 
            'B (channel 2)': 'negative_probability'
        },
        'dtype': 'uint8',
        'shape': list(seg_array.shape),
        'usage': 'R + B > threshold && G <= 80 -> cell; R >= B -> positive, R < B -> negative'
    }
    
    # 合并用户提供的元数据
    if metadata:
        default_metadata.update(metadata)
    
    # 保存元数据
    json_path = output_path.replace('.npy', '_meta.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(default_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"    Saved Seg probability npy: {output_path} (shape={seg_array.shape})")
    return output_path
