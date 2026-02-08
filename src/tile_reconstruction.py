#!/usr/bin/env python3
"""
tile_reconstruction.py — 瓦片重建模块

使用基于 npy mask 的连通组件分析方法，从多个 tile 的推理结果中重建完整图像的实例分割。

核心算法：
1. 解析 tile 文件名获取位置信息
2. 分块加载 tile masks（内存可控）
3. cv2.connectedComponents 做连通域分析
4. 重新分配全局唯一 ID（方案 B）
5. 输出 npy + LabelMe JSON

Author: Gemini Assistant
"""

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

from .file_io import load_mask_npy, save_mask_npy, mask_to_labelme_json


def parse_tile_filename(filename: str) -> Optional[tuple[int, int, int, int]]:
    """
    解析 tile 文件名获取位置信息。
    
    支持的文件名格式：
    - tile_100_10_50688_4608.npy  -> (100, 10, 50688, 4608) = (row, col, x_offset, y_offset)
    - tile_r100_c10.npy           -> (100, 10, None, None)
    - 100_10.npy                  -> (100, 10, None, None)
    
    Args:
        filename: tile 文件名
        
    Returns:
        tuple: (row, col, x_offset, y_offset) 或 None 如果无法解析
    """
    basename = os.path.splitext(filename)[0]
    
    # 格式1: tile_100_10_50688_4608
    match = re.match(r'tile_(\d+)_(\d+)_(\d+)_(\d+)', basename)
    if match:
        row, col, x_off, y_off = map(int, match.groups())
        return (row, col, x_off, y_off)
    
    # 格式2: tile_r100_c10
    match = re.match(r'tile_r(\d+)_c(\d+)', basename)
    if match:
        row, col = map(int, match.groups())
        return (row, col, None, None)
    
    # 格式3: 100_10
    match = re.match(r'(\d+)_(\d+)', basename)
    if match:
        row, col = map(int, match.groups())
        return (row, col, None, None)
    
    return None


def parse_tile_positions(tile_dir: str, pattern: str = "*.npy") -> dict:
    """
    解析目录中所有 tile 文件的位置信息。
    
    Args:
        tile_dir: 包含 tile npy 文件的目录
        pattern: 文件匹配模式
        
    Returns:
        dict: {(row, col): {'path': npy_path, 'x_offset': x, 'y_offset': y}}
    """
    tiles = {}
    tile_dir = Path(tile_dir)
    
    for npy_path in sorted(tile_dir.glob(pattern)):
        parsed = parse_tile_filename(npy_path.name)
        if parsed:
            row, col, x_off, y_off = parsed
            tiles[(row, col)] = {
                'path': str(npy_path),
                'x_offset': x_off,
                'y_offset': y_off
            }
    
    return tiles


def get_grid_dimensions(tiles: dict, tile_size: int = 512) -> tuple[int, int, int, int]:
    """
    从 tiles 信息计算网格尺寸和原图尺寸。
    
    Args:
        tiles: parse_tile_positions 返回的字典
        tile_size: 每个 tile 的尺寸
        
    Returns:
        tuple: (max_row, max_col, full_height, full_width)
    """
    max_row = max(pos[0] for pos in tiles.keys()) + 1
    max_col = max(pos[1] for pos in tiles.keys()) + 1
    
    # 尝试从 offset 计算精确尺寸
    max_x = 0
    max_y = 0
    has_offset = False
    
    for (row, col), info in tiles.items():
        if info['x_offset'] is not None and info['y_offset'] is not None:
            has_offset = True
            max_x = max(max_x, info['x_offset'] + tile_size)
            max_y = max(max_y, info['y_offset'] + tile_size)
    
    if has_offset:
        full_height = max_y
        full_width = max_x
    else:
        full_height = max_row * tile_size
        full_width = max_col * tile_size
    
    return max_row, max_col, full_height, full_width


def load_tile_block(tiles: dict, block_row: int, block_col: int, 
                    block_size: int = 2, tile_size: int = 512) -> tuple[np.ndarray, tuple]:
    """
    加载指定块区域的 tile masks。
    
    Args:
        tiles: parse_tile_positions 返回的字典
        block_row: 块的起始行（以 tile 为单位）
        block_col: 块的起始列（以 tile 为单位）
        block_size: 块大小（NxN tiles）
        tile_size: 每个 tile 的尺寸
        
    Returns:
        tuple: (block_mask, (y_offset, x_offset))
    """
    block_h = block_size * tile_size
    block_w = block_size * tile_size
    block_mask = np.zeros((block_h, block_w), dtype=np.uint32)
    
    for dr in range(block_size):
        for dc in range(block_size):
            row = block_row + dr
            col = block_col + dc
            
            if (row, col) in tiles:
                tile_path = tiles[(row, col)]['path']
                tile_mask, _ = load_mask_npy(tile_path)
                
                y_start = dr * tile_size
                x_start = dc * tile_size
                
                # 处理 tile 尺寸不一致的情况
                h, w = tile_mask.shape
                h = min(h, block_h - y_start)
                w = min(w, block_w - x_start)
                
                block_mask[y_start:y_start+h, x_start:x_start+w] = tile_mask[:h, :w]
    
    y_offset = block_row * tile_size
    x_offset = block_col * tile_size
    
    return block_mask, (y_offset, x_offset)


def merge_block_to_global(global_mask: np.ndarray, 
                          block_mask: np.ndarray, 
                          block_offset: tuple,
                          current_max_id: int) -> tuple[np.ndarray, int]:
    """
    将块内的连通组件合并到全局 mask，重新分配 ID。
    
    Args:
        global_mask: 全局实例 mask (会被修改)
        block_mask: 块内的实例 mask
        block_offset: 块在全局 mask 中的偏移 (y, x)
        current_max_id: 当前全局 mask 中最大的实例 ID
        
    Returns:
        tuple: (更新后的 global_mask, 新的 max_id)
    """
    y_off, x_off = block_offset
    bh, bw = block_mask.shape
    
    # 确保不越界
    gh, gw = global_mask.shape
    bh = min(bh, gh - y_off)
    bw = min(bw, gw - x_off)
    
    if bh <= 0 or bw <= 0:
        return global_mask, current_max_id
    
    # 创建块的二值 mask
    block_binary = (block_mask[:bh, :bw] > 0).astype(np.uint8) * 255
    
    # 连通组件分析
    num_labels, labels = cv2.connectedComponents(block_binary)
    
    # 重新分配 ID 并写入全局 mask
    new_max_id = current_max_id
    for comp_id in range(1, num_labels):
        comp_mask = labels == comp_id
        
        # 检查这个组件是否与全局 mask 中已有实例重叠
        global_region = global_mask[y_off:y_off+bh, x_off:x_off+bw]
        overlap_ids = np.unique(global_region[comp_mask])
        overlap_ids = overlap_ids[overlap_ids > 0]
        
        if len(overlap_ids) > 0:
            # 与已有实例重叠，使用最小的已有 ID（合并）
            use_id = int(np.min(overlap_ids))
        else:
            # 新实例
            new_max_id += 1
            use_id = new_max_id
        
        global_mask[y_off:y_off+bh, x_off:x_off+bw][comp_mask] = use_id
    
    return global_mask, new_max_id


def final_connected_component_pass(global_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """
    最终的连通组件分析，确保所有连通的像素有相同的 ID。
    
    Args:
        global_mask: 全局实例 mask
        
    Returns:
        tuple: (重新编号的 mask, 总实例数)
    """
    binary = (global_mask > 0).astype(np.uint8) * 255
    num_labels, labels = cv2.connectedComponents(binary)
    
    # labels 已经是连续编号 (1, 2, 3, ...)
    final_mask = labels.astype(np.uint32)
    num_instances = num_labels - 1  # 减去背景
    
    return final_mask, num_instances


def reconstruct_tiles(tile_dir: str, 
                      output_dir: str,
                      tile_size: int = 512,
                      block_size: int = 2,
                      original_image_path: str = None,
                      save_npy: bool = True,
                      export_labelme: bool = True,
                      output_name: str = "reconstructed") -> np.ndarray:
    """
    从多个 tile 的 npy mask 重建完整图像的实例分割。
    
    核心流程：
    1. 解析 tile 文件名获取位置
    2. 分块加载（每次 block_size × block_size tiles）
    3. 连通组件分析与 ID 重分配
    4. 最终全局连通组件检查
    5. 输出 npy + LabelMe JSON
    
    Args:
        tile_dir: 包含 tile npy 文件的目录
        output_dir: 输出目录
        tile_size: 每个 tile 的尺寸
        block_size: 分块大小（NxN tiles per block）
        original_image_path: 原始大图路径（用于 LabelMe JSON）
        save_npy: 是否保存 npy 格式 mask
        export_labelme: 是否导出 LabelMe JSON
        output_name: 输出文件名（不含扩展名）
        
    Returns:
        np.ndarray: 重建后的完整实例 mask
    """
    print(f"\n{'='*60}")
    print("TILE RECONSTRUCTION STARTED")
    print(f"Input: {tile_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # 1. 解析 tile 位置
    tiles = parse_tile_positions(tile_dir)
    print(f"Found {len(tiles)} tile files.")
    
    if len(tiles) == 0:
        print("ERROR: No valid tile files found!")
        return None
    
    # 2. 计算网格尺寸
    max_row, max_col, full_h, full_w = get_grid_dimensions(tiles, tile_size)
    print(f"Grid: {max_row} rows x {max_col} cols")
    print(f"Full image size: {full_h} x {full_w}")
    
    # 3. 初始化全局 mask
    global_mask = np.zeros((full_h, full_w), dtype=np.uint32)
    current_max_id = 0
    
    # 4. 分块处理
    num_blocks_y = (max_row + block_size - 1) // block_size
    num_blocks_x = (max_col + block_size - 1) // block_size
    total_blocks = num_blocks_y * num_blocks_x
    
    print(f"\nProcessing {total_blocks} blocks ({block_size}x{block_size} tiles each)...")
    
    block_idx = 0
    for by in range(0, max_row, block_size):
        for bx in range(0, max_col, block_size):
            block_idx += 1
            print(f"  Block {block_idx}/{total_blocks}: rows {by}-{by+block_size-1}, cols {bx}-{bx+block_size-1}")
            
            # 加载块
            block_mask, block_offset = load_tile_block(tiles, by, bx, block_size, tile_size)
            
            # 合并到全局 mask
            global_mask, current_max_id = merge_block_to_global(
                global_mask, block_mask, block_offset, current_max_id
            )
            
            print(f"    Current max instance ID: {current_max_id}")
    
    # 5. 最终连通组件分析（确保跨块边界的实例正确合并）
    print("\nFinal connected component analysis...")
    final_mask, num_instances = final_connected_component_pass(global_mask)
    print(f"Total instances after reconstruction: {num_instances}")
    
    # 6. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    if save_npy:
        npy_path = os.path.join(output_dir, f"{output_name}.npy")
        metadata = {
            'source_tile_dir': tile_dir,
            'tile_size': tile_size,
            'grid_size': [max_row, max_col],
            'image_size': [full_h, full_w],
            'num_instances': num_instances
        }
        save_mask_npy(final_mask, npy_path, metadata)
    
    if export_labelme:
        print("Exporting to LabelMe JSON...")
        json_path = os.path.join(output_dir, f"{output_name}.json")
        
        # 如果有原图，用于 LabelMe
        image_array = None
        if original_image_path and os.path.exists(original_image_path):
            from PIL import Image
            image_array = np.array(Image.open(original_image_path).convert('RGB'))
        else:
            # 创建占位图像
            image_array = np.zeros((full_h, full_w, 3), dtype=np.uint8)
        
        labelme_data = mask_to_labelme_json(
            instance_mask=final_mask,
            original_image_path=original_image_path or f"{output_name}.png",
            image_array=image_array,
            cells_info=None,
            include_image_data=False,
            simplify_contour=True
        )
        
        import json
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, indent=2, ensure_ascii=False)
        
        print(f"  LabelMe JSON saved: {json_path}")
        print(f"  Total polygons: {len(labelme_data['shapes'])}")
    
    print(f"\n{'='*60}")
    print("TILE RECONSTRUCTION COMPLETED")
    print(f"{'='*60}")
    
    return final_mask
