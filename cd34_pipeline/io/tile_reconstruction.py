#!/usr/bin/env python3
"""
tile_reconstruction.py — tile 位置解析与 GeoJSON 导出工具

提供 tile 文件名解析、tile 位置扫描，以及基于 tile overlap 的
GeoJSON 导出能力。
"""

import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

from cd34_pipeline.io.file_io import load_mask_npy


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



# ── GeoJSON 导出 ────────────────────────────────────────────────


def _poly_to_geojson_feature(poly, classification_name="CD34+",
                              classification_color=None):
    """将一个 Shapely Polygon 转为 QuPath GeoJSON Feature。

    Returns:
        feature dict 或 None（无效时）。
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.geometry.polygon import orient as shapely_orient
    from shapely.validation import make_valid

    if classification_color is None:
        classification_color = [200, 50, 50]

    try:
        p = shapely_orient(poly, sign=1.0)

        raw_ring = list(p.exterior.coords)
        ring = []
        for x, y in raw_ring:
            pt = [int(round(x)), int(round(y))]
            if not ring or pt != ring[-1]:
                ring.append(pt)

        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])

        if len(ring) < 4:
            return None

        final_poly = ShapelyPolygon(ring)
        if not final_poly.is_valid:
            final_poly = make_valid(final_poly)
        if final_poly.is_empty:
            return None
        if final_poly.geom_type == 'GeometryCollection':
            polys = [g for g in final_poly.geoms if g.geom_type == 'Polygon']
            if not polys:
                return None
            final_poly = max(polys, key=lambda g: g.area)
        elif final_poly.geom_type == 'MultiPolygon':
            final_poly = max(final_poly.geoms, key=lambda g: g.area)
        elif final_poly.geom_type != 'Polygon':
            return None
        ring = [[int(round(x)), int(round(y))] for x, y in final_poly.exterior.coords]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        if len(ring) < 4:
            return None

        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring],
            },
            "properties": {
                "objectType": "detection",
                "classification": {
                    "name": classification_name,
                    "color": classification_color,
                },
            },
        }
    except Exception:
        return None


class _UnionFind:
    """Union-Find (并查集) 用于合并跨 tile 的实例。"""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # 路径压缩
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # 小编号做 root，保持稳定
            if ra > rb:
                ra, rb = rb, ra
            self.parent[rb] = ra


def _extract_tile_polygons(mask: np.ndarray, x_off: int, y_off: int):
    """从单个 tile 的实例 mask 提取多边形（全局坐标），保留原始精度。

    Args:
        mask: (H, W) uint8 实例 mask，0=背景，>0=实例 ID
        x_off: tile 左上角在全局坐标系中的 x 偏移
        y_off: tile 左上角在全局坐标系中的 y 偏移

    Returns:
        list of (inst_id_in_tile, shapely.Polygon)
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.validation import make_valid

    results = []
    inst_ids = np.unique(mask)

    for inst_id in inst_ids:
        if inst_id == 0:
            continue

        binary = (mask == inst_id).astype(np.uint8) * 255
        # 闭运算填补对角相连的 1px 缝隙，消除轮廓自交叉
        _close_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _close_kern)
        # CHAIN_APPROX_SIMPLE 压缩水平/垂直/对角线段的中间点
        # 精度完全相同（轮廓覆盖同样的像素），但点数大幅减少
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) < 3:
                continue

            coords = [(int(pt[0][0]) + x_off, int(pt[0][1]) + y_off)
                       for pt in contour]

            try:
                poly = ShapelyPolygon(coords)
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.is_empty:
                    continue
                # make_valid 可能返回非 Polygon
                if poly.geom_type == 'MultiPolygon':
                    for p in poly.geoms:
                        if p.area > 0:
                            results.append((int(inst_id), p))
                elif poly.geom_type == 'GeometryCollection':
                    for g in poly.geoms:
                        if g.geom_type == 'Polygon' and g.area > 0:
                            results.append((int(inst_id), g))
                        elif g.geom_type == 'MultiPolygon':
                            for p in g.geoms:
                                if p.area > 0:
                                    results.append((int(inst_id), p))
                elif poly.geom_type == 'Polygon' and poly.area > 0:
                    results.append((int(inst_id), poly))
            except Exception:
                continue

    return results


def export_geojson(tile_dir: str,
                   output_path: str,
                   tile_size: int = 512,
                   stride: int = 384,
                   simplify: float = 0.0,
                   contour_tolerance: float = 0.5,
                   min_area: float = 50) -> str:
    """
    从 npy_masks 目录导出 QuPath 兼容的 GeoJSON 标注文件。

    基于 tile 级别工作，不构建全局 mask，内存友好。
    利用 overlap 区域的像素重合关系，通过 Union-Find 合并跨 tile 的同一实例，
    最终用 shapely.ops.unary_union 合并多边形，保留原始轮廓精度。

    算法流程:
      Pass 1: 逐 tile 提取每个实例的轮廓多边形（全局坐标）
      Pass 2: 扫描相邻 tile 的 overlap 区域，用像素匹配做 Union-Find 合并
      Pass 3: 按合并后的组做 unary_union → 导出 GeoJSON

    Args:
        tile_dir: 包含 tile npy 文件的目录
        output_path: 输出 .geojson 文件路径
        tile_size: tile 尺寸 (默认 512)
        stride: tile 步长 (默认 384, overlap = tile_size - stride = 128)
        simplify: 轮廓简化比例 (0=不简化，按面积比例简化)
        contour_tolerance: 固定像素容差 Douglas-Peucker 简化
                           (默认 0.5，去除偏差 <0.5px 的冗余点，像素级精度不变)
        min_area: 最小轮廓面积（像素²）

    Returns:
        输出文件路径
    """
    import json as json_mod
    from shapely.ops import unary_union

    print(f"\n{'='*60}")
    print("GEOJSON EXPORT (tile-based merge)")
    print(f"Input:    {tile_dir}")
    print(f"Output:   {output_path}")
    print(f"Tile:     {tile_size}, stride: {stride}, overlap: {tile_size - stride}")
    print(f"Simplify: {simplify}, contour_tolerance: {contour_tolerance}, min_area: {min_area}")
    print(f"{'='*60}\n")

    tiles = parse_tile_positions(tile_dir)
    print(f"Found {len(tiles)} tile files.")

    if not tiles:
        print("  No tiles found!")
        return output_path

    t0 = time.time()

    # ── Pass 1: 逐 tile 提取多边形 ──────────────────────────────
    print(f"\n[Pass 1] Extracting polygons from each tile...")

    # global_id: 全局唯一实例标识 = (row, col, tile_inst_id)
    # poly_map: global_id → list[Polygon]
    poly_map = {}
    tile_keys = sorted(tiles.keys())
    total = len(tile_keys)

    for idx, (row, col) in enumerate(tile_keys):
        info = tiles[(row, col)]
        npy_path = info['path']
        mask = load_mask_npy(npy_path)

        if info['x_offset'] is not None:
            x_off, y_off = info['x_offset'], info['y_offset']
        else:
            x_off = col * stride
            y_off = row * stride

        tile_polys = _extract_tile_polygons(mask, x_off, y_off)

        for inst_id, poly in tile_polys:
            gid = (row, col, inst_id)
            if gid not in poly_map:
                poly_map[gid] = []
            poly_map[gid].append(poly)

        if (idx + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"    {idx + 1}/{total} tiles, {len(poly_map)} instances, {elapsed:.0f}s")

    t1 = time.time()
    print(f"  Pass 1 done: {len(poly_map)} tile-level instances ({t1-t0:.0f}s)")

    # ── Pass 2: overlap 区域像素匹配 → Union-Find 合并 ──────────
    print(f"\n[Pass 2] Matching instances across tile overlaps...")

    uf = _UnionFind()
    # 将所有 gid 注册到 uf
    for gid in poly_map:
        uf.find(gid)

    overlap = tile_size - stride  # 128
    merge_count = 0
    pair_count = 0

    # 构建 (row,col) → info 的快速查找
    tile_set = set(tiles.keys())

    for idx, (row, col) in enumerate(tile_keys):
        info = tiles[(row, col)]
        npy_path = info['path']
        mask_a = load_mask_npy(npy_path)

        if info['x_offset'] is not None:
            x_off_a, y_off_a = info['x_offset'], info['y_offset']
        else:
            x_off_a = col * stride
            y_off_a = row * stride

        # 检查右邻居 (row, col+1)
        if (row, col + 1) in tile_set:
            info_b = tiles[(row, col + 1)]
            mask_b = load_mask_npy(info_b['path'])

            # overlap 区域在 tile_a 中的位置: 右侧 overlap 列
            # tile_a 的右边 overlap 列: x 范围 [stride, tile_size)
            # tile_b 的左边 overlap 列: x 范围 [0, overlap)
            region_a = mask_a[:, stride:tile_size]  # (512, 128)
            region_b = mask_b[:, :overlap]           # (512, 128)

            # 找到两个区域中同时有前景的像素
            both_fg = (region_a > 0) & (region_b > 0)
            if np.any(both_fg):
                pairs_a = region_a[both_fg]
                pairs_b = region_b[both_fg]
                # 提取唯一配对
                unique_pairs = set(zip(pairs_a.tolist(), pairs_b.tolist()))
                for ia, ib in unique_pairs:
                    gid_a = (row, col, ia)
                    gid_b = (row, col + 1, ib)
                    if gid_a in poly_map and gid_b in poly_map:
                        uf.union(gid_a, gid_b)
                        merge_count += 1
                pair_count += len(unique_pairs)

        # 检查下邻居 (row+1, col)
        if (row + 1, col) in tile_set:
            info_b = tiles[(row + 1, col)]
            mask_b = load_mask_npy(info_b['path'])

            # tile_a 的下边 overlap 行: y 范围 [stride, tile_size)
            # tile_b 的上边 overlap 行: y 范围 [0, overlap)
            region_a = mask_a[stride:tile_size, :]  # (128, 512)
            region_b = mask_b[:overlap, :]           # (128, 512)

            both_fg = (region_a > 0) & (region_b > 0)
            if np.any(both_fg):
                pairs_a = region_a[both_fg]
                pairs_b = region_b[both_fg]
                unique_pairs = set(zip(pairs_a.tolist(), pairs_b.tolist()))
                for ia, ib in unique_pairs:
                    gid_a = (row, col, ia)
                    gid_b = (row + 1, col, ib)
                    if gid_a in poly_map and gid_b in poly_map:
                        uf.union(gid_a, gid_b)
                        merge_count += 1
                pair_count += len(unique_pairs)

        # 检查右下邻居 (row+1, col+1) — 对角 overlap 区域
        if (row + 1, col + 1) in tile_set:
            info_b = tiles[(row + 1, col + 1)]
            mask_b = load_mask_npy(info_b['path'])

            # tile_a 右下角: [stride:, stride:]
            # tile_b 左上角: [:overlap, :overlap]
            region_a = mask_a[stride:tile_size, stride:tile_size]
            region_b = mask_b[:overlap, :overlap]

            both_fg = (region_a > 0) & (region_b > 0)
            if np.any(both_fg):
                pairs_a = region_a[both_fg]
                pairs_b = region_b[both_fg]
                unique_pairs = set(zip(pairs_a.tolist(), pairs_b.tolist()))
                for ia, ib in unique_pairs:
                    gid_a = (row, col, ia)
                    gid_b = (row + 1, col + 1, ib)
                    if gid_a in poly_map and gid_b in poly_map:
                        uf.union(gid_a, gid_b)
                        merge_count += 1
                pair_count += len(unique_pairs)

        if (idx + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"    {idx + 1}/{total} tiles scanned, {merge_count} merges, {elapsed:.0f}s")

    t2 = time.time()
    print(f"  Pass 2 done: {pair_count} overlap pairs, {merge_count} merges ({t2-t1:.0f}s)")

    # ── Pass 3: 按组合并多边形 → GeoJSON ────────────────────────
    print(f"\n[Pass 3] Merging polygons and building GeoJSON...")

    # 按 Union-Find root 分组
    groups = {}
    for gid, polys in poly_map.items():
        root = uf.find(gid)
        if root not in groups:
            groups[root] = []
        groups[root].extend(polys)

    # 释放 poly_map
    del poly_map

    print(f"  {len(groups)} merged groups (from tile-level instances)")

    features = []
    skipped = 0

    for grp_idx, (root, polys) in enumerate(groups.items()):
        try:
            if len(polys) == 1:
                merged = polys[0]
            else:
                merged = unary_union(polys)

            # unary_union 可能返回多种类型
            if merged.is_empty:
                skipped += 1
                continue

            if merged.geom_type == 'Polygon':
                final_polys = [merged]
            elif merged.geom_type == 'MultiPolygon':
                final_polys = list(merged.geoms)
            elif merged.geom_type == 'GeometryCollection':
                final_polys = [g for g in merged.geoms
                               if g.geom_type == 'Polygon']
            else:
                skipped += 1
                continue

            for p in final_polys:
                if p.area < min_area:
                    skipped += 1
                    continue

                if simplify > 0:
                    p = p.simplify(simplify * np.sqrt(p.area),
                                   preserve_topology=True)

                # 固定容差简化：tolerance=0.5 去除偏差 <0.5px 的点
                # 整数像素坐标下不改变实际多边形形状
                if contour_tolerance > 0:
                    p = p.simplify(contour_tolerance,
                                   preserve_topology=True)

                feat = _poly_to_geojson_feature(p)
                if feat is not None:
                    features.append(feat)
                else:
                    skipped += 1

        except Exception:
            skipped += 1
            continue

        if (grp_idx + 1) % 10000 == 0:
            elapsed = time.time() - t0
            print(f"    {grp_idx + 1}/{len(groups)} groups, "
                  f"{len(features)} features, {elapsed:.0f}s")

    t3 = time.time()
    print(f"  Pass 3 done: {len(features)} features, "
          f"{skipped} skipped ({t3-t2:.0f}s)")

    # ── 写入 GeoJSON ──
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    print(f"\nWriting GeoJSON: {output_path}")
    with open(output_path, 'w') as f:
        json_mod.dump(features, f, separators=(',', ':'))

    file_mb = os.path.getsize(output_path) / (1024 ** 2)
    elapsed = time.time() - t0
    print(f"  Size: {file_mb:.1f} MB, {len(features)} features")
    print(f"  Total time: {elapsed:.0f}s")

    print(f"\n{'='*60}")
    print("GEOJSON EXPORT COMPLETED")
    print(f"  QuPath: File → Open → .ndpi, then Import objects → .geojson")
    print(f"{'='*60}")

    return output_path
