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
        def _as_polygon(geom):
            if not geom.is_valid:
                geom = make_valid(geom)
            if geom.is_empty:
                return None
            if geom.geom_type == 'Polygon':
                return geom
            if geom.geom_type == 'MultiPolygon':
                polys = [g for g in geom.geoms if g.area > 0]
                return max(polys, key=lambda g: g.area) if polys else None
            if geom.geom_type == 'GeometryCollection':
                polys = []
                for g in geom.geoms:
                    if g.geom_type == 'Polygon' and g.area > 0:
                        polys.append(g)
                    elif g.geom_type == 'MultiPolygon':
                        polys.extend([p for p in g.geoms if p.area > 0])
                return max(polys, key=lambda g: g.area) if polys else None
            return None

        def _ring_to_int_coords(ring):
            coords = []
            for x, y in ring.coords:
                pt = [int(round(x)), int(round(y))]
                if not coords or pt != coords[-1]:
                    coords.append(pt)
            if coords and coords[0] != coords[-1]:
                coords.append(coords[0])
            return coords if len(coords) >= 4 else None

        final_poly = _as_polygon(poly)
        if final_poly is None:
            return None

        final_poly = shapely_orient(final_poly, sign=1.0)
        exterior = _ring_to_int_coords(final_poly.exterior)
        if exterior is None:
            return None
        holes = [
            hole for hole in
            (_ring_to_int_coords(r) for r in final_poly.interiors)
            if hole is not None
        ]

        snapped = _as_polygon(ShapelyPolygon(exterior, holes))
        if snapped is None:
            return None
        snapped = shapely_orient(snapped, sign=1.0)

        exterior = _ring_to_int_coords(snapped.exterior)
        if exterior is None:
            return None
        holes = [
            hole for hole in
            (_ring_to_int_coords(r) for r in snapped.interiors)
            if hole is not None
        ]

        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [exterior] + holes,
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
            return True
        return False


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

    def _contour_coords(contour):
        if len(contour) < 3:
            return None
        coords = []
        for pt in contour:
            xy = (int(pt[0][0]) + x_off, int(pt[0][1]) + y_off)
            if not coords or xy != coords[-1]:
                coords.append(xy)
        if len(coords) >= 2 and coords[0] == coords[-1]:
            coords.pop()
        return coords if len(coords) >= 3 else None

    def _append_valid_polygon(inst_id: int, poly) -> None:
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty:
            return
        if poly.geom_type == 'MultiPolygon':
            for p in poly.geoms:
                if p.area > 0:
                    results.append((inst_id, p))
        elif poly.geom_type == 'GeometryCollection':
            for g in poly.geoms:
                if g.geom_type == 'Polygon' and g.area > 0:
                    results.append((inst_id, g))
                elif g.geom_type == 'MultiPolygon':
                    for p in g.geoms:
                        if p.area > 0:
                            results.append((inst_id, p))
        elif poly.geom_type == 'Polygon' and poly.area > 0:
            results.append((inst_id, poly))

    for inst_id in inst_ids:
        if inst_id == 0:
            continue

        binary = (mask == inst_id).astype(np.uint8) * 255
        # 闭运算填补对角相连的 1px 缝隙，消除轮廓自交叉
        _close_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _close_kern)
        # CHAIN_APPROX_SIMPLE 压缩水平/垂直/对角线段的中间点
        # 精度完全相同（轮廓覆盖同样的像素），但点数大幅减少
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]

        for idx, contour in enumerate(contours):
            # Parent == -1 means this is an exterior ring. Child contours are
            # holes and must be preserved, otherwise GeoJSON fills lumens.
            if hierarchy[idx][3] != -1:
                continue
            shell = _contour_coords(contour)
            if shell is None:
                continue

            holes = []
            child = hierarchy[idx][2]
            while child != -1:
                hole = _contour_coords(contours[child])
                if hole is not None:
                    holes.append(hole)
                child = hierarchy[child][0]

            try:
                _append_valid_polygon(int(inst_id), ShapelyPolygon(shell, holes))
            except Exception:
                continue

    return results


def export_geojson(tile_dir: Optional[str],
                   output_path: str,
                   tile_size: int = 512,
                   stride: int = 384,
                   simplify: float = 0.0,
                   contour_tolerance: float = 0.5,
                   min_area: float = 50,
                   poly_map: Optional[dict] = None,
                   masks: Optional[dict] = None,
                   level_downsample: float = 1.0,
                   crop_origin: Optional[tuple[int, int]] = None,
                   debug_dir: Optional[str] = None,
                   merge_mode: str = "overlap-merge") -> str:
    """
    导出 QuPath 兼容的 GeoJSON 标注文件。

    支持两种模式:
      - 磁盘模式 (tile_dir 非 None): 从 npy_masks 目录读取 (兼容独立脚本)
      - 内存模式 (poly_map + masks 非 None): 直接使用内存数据，跳过 Pass 1

    算法流程:
      Pass 1: 逐 tile 提取每个实例的轮廓多边形（全局坐标）[磁盘模式]
      Pass 2: 扫描相邻 tile 的 overlap 区域，用像素匹配做 Union-Find 合并
      Pass 3: 按合并后的组做 unary_union → 导出 GeoJSON

    Args:
        tile_dir: 包含 tile npy 文件的目录 (磁盘模式), None for 内存模式
        output_path: 输出 .geojson 文件路径
        tile_size: tile 尺寸 (默认 512)
        stride: tile 步长 (默认 384, overlap = tile_size - stride = 128)
        simplify: 轮廓简化比例 (0=不简化，按面积比例简化)
        contour_tolerance: 固定像素容差 Douglas-Peucker 简化
        min_area: 最小轮廓面积（像素²）
        poly_map: 内存模式 — {(row, col, inst_id): [Polygon, ...]}
        masks: 内存模式 — {(row, col): ndarray}
        level_downsample: 从 target level 到 level-0 的缩放因子 (默认 1.0)
        crop_origin: (x, y) crop_region 左上角在 level-0 全图中的像素坐标。
                     若非 None，导出时会将多边形坐标从 crop-relative target-level
                     转换为 level-0 绝对坐标: coord * level_downsample + crop_origin
        debug_dir: 若提供，写出跨 tile stitching 的候选/拒绝/合并诊断。
        merge_mode:
            - "overlap-merge": 旧行为，扫描 overlap 并合并相邻 tile 实例。
            - "center-valid": polygon 使用 owner/中心有效区；如果提供未裁剪
              masks，仍扫描 overlap 作为跨 tile 身份匹配证据。
            - "center-valid-raw": polygon 使用 owner/中心有效区，并跳过跨
              tile 合并，用于查看不拼接的原始 tile 输出。

    Returns:
        输出文件路径
    """
    import json as json_mod
    from shapely.ops import unary_union

    in_memory = poly_map is not None and masks is not None
    merge_mode = merge_mode.replace("_", "-")
    if merge_mode not in {
            "overlap-merge", "center-valid", "center-valid-raw"}:
        raise ValueError(
            "merge_mode must be 'overlap-merge', 'center-valid', "
            "or 'center-valid-raw'")

    print(f"\n{'='*60}")
    print(f"GEOJSON EXPORT ({'in-memory' if in_memory else 'tile-based'} merge)")
    if not in_memory:
        print(f"Input:    {tile_dir}")
    print(f"Output:   {output_path}")
    print(f"Tile:     {tile_size}, stride: {stride}, overlap: {tile_size - stride}")
    print(f"Mode:     {merge_mode}")
    print(f"Simplify: {simplify}, contour_tolerance: {contour_tolerance}, min_area: {min_area}")
    need_transform = level_downsample != 1.0 or crop_origin is not None
    if need_transform:
        ox = crop_origin[0] if crop_origin else 0
        oy = crop_origin[1] if crop_origin else 0
        print(f"Transform: scale={level_downsample}, crop_origin=({ox}, {oy})")
    print(f"{'='*60}\n")

    t0 = time.time()

    if in_memory:
        # ── 内存模式: Pass 1 已完成，直接使用 poly_map ──
        tile_set = set(masks.keys())
        tile_keys = sorted(tile_set)
        total = len(tile_keys)
        print(f"In-memory: {total} tiles, {len(poly_map)} tile-level instances")
        t1 = time.time()
    else:
        # ── 磁盘模式: Pass 1 从 npy 文件提取多边形 ──
        tiles = parse_tile_positions(tile_dir)
        print(f"Found {len(tiles)} tile files.")

        if not tiles:
            print("  No tiles found!")
            return output_path

        print(f"\n[Pass 1] Extracting polygons from each tile...")

        poly_map = {}
        tile_keys = sorted(tiles.keys())
        tile_set = set(tile_keys)
        total = len(tile_keys)

        for idx, (row, col) in enumerate(tile_keys):
            info = tiles[(row, col)]
            mask = load_mask_npy(info['path'])

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
    for gid in poly_map:
        uf.find(gid)

    overlap = tile_size - stride
    merge_count = 0
    pair_count = 0
    eligible_pair_count = 0

    # Conservative object-level stitching thresholds.  Overlap pixels now
    # create candidate matches; only strong mutual-best matches are merged.
    MERGE_MIN_INTERSECTION_PIXELS = 20
    MERGE_MIN_DICE = 0.20
    MERGE_MIN_OVERLAP_RATIO = 0.30
    MERGE_MIN_AREA_RATIO = 0.25
    MERGE_MIN_CENTROID_DISTANCE = 32.0
    MERGE_CENTROID_DIAMETER_FACTOR = 2.0
    MERGE_STRONG_MIN_INTERSECTION_PIXELS = 100
    MERGE_STRONG_MIN_DICE = 0.60
    MERGE_STRONG_MIN_OVERLAP_RATIO = 0.60
    MERGE_ENABLE_DIAGONAL = False
    debug_matches = [] if debug_dir else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    instance_areas = [
        sum(float(poly.area) for poly in polys)
        for polys in poly_map.values()
    ]
    instance_areas = [area for area in instance_areas if area > 0]
    if instance_areas:
        median_instance_area = float(np.median(instance_areas))
    else:
        median_instance_area = float(max(min_area, 1.0))
    median_diameter = float(np.sqrt(4.0 * median_instance_area / np.pi))
    max_centroid_distance = max(
        MERGE_MIN_CENTROID_DISTANCE,
        MERGE_CENTROID_DIAMETER_FACTOR * median_diameter,
    )
    max_centroid_distance = min(max_centroid_distance, tile_size * 0.5)

    print("  Stitching policy: mutual-best object matching")
    print(f"    min_intersection={MERGE_MIN_INTERSECTION_PIXELS}, "
          f"min_dice={MERGE_MIN_DICE:.2f}, "
          f"min_overlap_ratio={MERGE_MIN_OVERLAP_RATIO:.2f}, "
          f"min_area_ratio={MERGE_MIN_AREA_RATIO:.2f}")
    print(f"    strong_overlap bypass: "
          f"intersection>={MERGE_STRONG_MIN_INTERSECTION_PIXELS}, "
          f"dice>={MERGE_STRONG_MIN_DICE:.2f}, "
          f"overlap_ratio>={MERGE_STRONG_MIN_OVERLAP_RATIO:.2f}")
    print(f"    max_centroid_distance={max_centroid_distance:.1f}px, "
          f"diagonal={'ON' if MERGE_ENABLE_DIAGONAL else 'OFF'}")

    def _load_mask(row, col):
        """Load mask from memory or disk depending on mode."""
        if in_memory:
            return masks.get((row, col))
        info = tiles.get((row, col))
        if info is None:
            return None
        return load_mask_npy(info['path'])

    stats_cache = {}
    coord_cache = {}

    def _tile_offset(row, col):
        if not in_memory:
            info = tiles.get((row, col))
            if info and info['x_offset'] is not None:
                return info['x_offset'], info['y_offset']
        return col * stride, row * stride

    def _ensure_tile_stats(row, col, mask):
        """Compute per-instance area and global centroid for one tile."""
        key = (row, col)
        if key in stats_cache:
            return stats_cache[key]

        if mask is None or mask.size == 0:
            stats_cache[key] = {}
            return stats_cache[key]

        labels = mask.ravel().astype(np.int64, copy=False)
        max_label = int(labels.max()) if labels.size else 0
        if max_label <= 0:
            stats_cache[key] = {}
            return stats_cache[key]

        h, w = mask.shape[:2]
        coord_key = (h, w)
        if coord_key not in coord_cache:
            yy, xx = np.indices((h, w))
            coord_cache[coord_key] = (yy.ravel(), xx.ravel())
        ys, xs = coord_cache[coord_key]

        counts = np.bincount(labels, minlength=max_label + 1)
        y_sums = np.bincount(labels, weights=ys, minlength=max_label + 1)
        x_sums = np.bincount(labels, weights=xs, minlength=max_label + 1)
        x_off, y_off = _tile_offset(row, col)

        tile_stats = {}
        for inst_id in np.flatnonzero(counts):
            if inst_id == 0:
                continue
            gid = (row, col, int(inst_id))
            if gid not in poly_map:
                continue
            area = int(counts[inst_id])
            if area <= 0:
                continue
            tile_stats[gid] = {
                'area': area,
                'cx': x_off + float(x_sums[inst_id]) / area,
                'cy': y_off + float(y_sums[inst_id]) / area,
            }

        stats_cache[key] = tile_stats
        return tile_stats

    def _gid_list(gid):
        return [int(gid[0]), int(gid[1]), int(gid[2])]

    def _append_match_record(record):
        if debug_matches is not None:
            debug_matches.append(record)

    def _accept_neighbor_matches(row_a, col_a, row_b, col_b,
                                 mask_a, mask_b, region_a, region_b,
                                 direction, diagonal=False):
        """Find strict mutual-best matches for one neighboring tile pair."""
        if region_a.size == 0 or region_b.size == 0:
            return 0, 0, 0

        both_fg = (region_a > 0) & (region_b > 0)
        if not np.any(both_fg):
            return 0, 0, 0

        labels_a = region_a.ravel().astype(np.int64, copy=False)
        labels_b = region_b.ravel().astype(np.int64, copy=False)
        max_a = int(labels_a.max()) if labels_a.size else 0
        max_b = int(labels_b.max()) if labels_b.size else 0
        counts_a = np.bincount(labels_a, minlength=max_a + 1)
        counts_b = np.bincount(labels_b, minlength=max_b + 1)

        pair_pixels = np.column_stack((region_a[both_fg], region_b[both_fg]))
        pair_labels, intersections = np.unique(
            pair_pixels.astype(np.int64, copy=False),
            axis=0,
            return_counts=True,
        )

        raw_pairs = len(intersections)
        stats_a = _ensure_tile_stats(row_a, col_a, mask_a)
        stats_b = _ensure_tile_stats(row_b, col_b, mask_b)
        candidates = []

        multiplier = 2.0 if diagonal else 1.0
        min_intersection = int(MERGE_MIN_INTERSECTION_PIXELS * multiplier)
        min_dice = MERGE_MIN_DICE * multiplier
        min_overlap_ratio = min(MERGE_MIN_OVERLAP_RATIO * multiplier, 0.95)

        for (ia, ib), intersection in zip(pair_labels, intersections):
            ia = int(ia)
            ib = int(ib)
            gid_a = (row_a, col_a, ia)
            gid_b = (row_b, col_b, ib)
            base_record = None
            if debug_matches is not None:
                base_record = {
                    'direction': direction,
                    'tile_a': [int(row_a), int(col_a)],
                    'tile_b': [int(row_b), int(col_b)],
                    'gid_a': _gid_list(gid_a),
                    'gid_b': _gid_list(gid_b),
                    'intersection': int(intersection),
                    'accepted': False,
                    'unioned': False,
                    'rejected_reason': None,
                }
            if gid_a not in poly_map or gid_b not in poly_map:
                if base_record is not None:
                    base_record['rejected_reason'] = 'missing_polygon'
                    _append_match_record(base_record)
                continue

            area_a_overlap = int(counts_a[ia]) if ia < len(counts_a) else 0
            area_b_overlap = int(counts_b[ib]) if ib < len(counts_b) else 0
            if area_a_overlap <= 0 or area_b_overlap <= 0:
                if base_record is not None:
                    base_record['area_a_overlap'] = area_a_overlap
                    base_record['area_b_overlap'] = area_b_overlap
                    base_record['rejected_reason'] = 'empty_instance_overlap'
                    _append_match_record(base_record)
                continue

            intersection = int(intersection)
            dice = (2.0 * intersection /
                    (area_a_overlap + area_b_overlap))
            overlap_ratio = (intersection /
                             min(area_a_overlap, area_b_overlap))

            stat_a = stats_a.get(gid_a)
            stat_b = stats_b.get(gid_b)
            if stat_a is None or stat_b is None:
                if base_record is not None:
                    base_record.update({
                        'area_a_overlap': area_a_overlap,
                        'area_b_overlap': area_b_overlap,
                        'dice': dice,
                        'overlap_ratio': overlap_ratio,
                        'rejected_reason': 'missing_instance_stats',
                    })
                    _append_match_record(base_record)
                continue

            full_area_a = stat_a['area']
            full_area_b = stat_b['area']
            area_ratio = min(full_area_a, full_area_b) / max(full_area_a, full_area_b)
            centroid_distance = float(np.hypot(
                stat_a['cx'] - stat_b['cx'],
                stat_a['cy'] - stat_b['cy'],
            ))
            strong_overlap = (
                intersection >= MERGE_STRONG_MIN_INTERSECTION_PIXELS * multiplier
                and dice >= MERGE_STRONG_MIN_DICE
                and overlap_ratio >= MERGE_STRONG_MIN_OVERLAP_RATIO
            )

            record = None
            if base_record is not None:
                record = base_record
                record.update({
                    'area_a_overlap': area_a_overlap,
                    'area_b_overlap': area_b_overlap,
                    'dice': float(dice),
                    'overlap_ratio': float(overlap_ratio),
                    'full_area_a': int(full_area_a),
                    'full_area_b': int(full_area_b),
                    'area_ratio': float(area_ratio),
                    'centroid_distance': float(centroid_distance),
                    'strong_overlap': bool(strong_overlap),
                    'centroid_a': [float(stat_a['cx']), float(stat_a['cy'])],
                    'centroid_b': [float(stat_b['cx']), float(stat_b['cy'])],
                })

            if intersection < min_intersection:
                if record is not None:
                    record['rejected_reason'] = 'min_intersection'
                    _append_match_record(record)
                continue
            if dice < min_dice:
                if record is not None:
                    record['rejected_reason'] = 'min_dice'
                    _append_match_record(record)
                continue
            if overlap_ratio < min_overlap_ratio:
                if record is not None:
                    record['rejected_reason'] = 'min_overlap_ratio'
                    _append_match_record(record)
                continue
            if area_ratio < MERGE_MIN_AREA_RATIO and not strong_overlap:
                if record is not None:
                    record['rejected_reason'] = 'min_area_ratio'
                    _append_match_record(record)
                continue
            if centroid_distance > max_centroid_distance and not strong_overlap:
                if record is not None:
                    record['rejected_reason'] = 'max_centroid_distance'
                    _append_match_record(record)
                continue

            centroid_penalty = centroid_distance / max_centroid_distance
            score = dice + overlap_ratio + 0.2 * area_ratio - 0.25 * centroid_penalty
            if record is not None:
                record['score'] = float(score)
                record['rejected_reason'] = 'candidate'
                _append_match_record(record)
            candidates.append({
                'gid_a': gid_a,
                'gid_b': gid_b,
                'score': score,
                'record': record,
            })

        if not candidates:
            return raw_pairs, 0, 0

        best_for_a = {}
        best_for_b = {}
        for candidate in candidates:
            gid_a = candidate['gid_a']
            gid_b = candidate['gid_b']
            if (gid_a not in best_for_a or
                    candidate['score'] > best_for_a[gid_a]['score']):
                best_for_a[gid_a] = candidate
            if (gid_b not in best_for_b or
                    candidate['score'] > best_for_b[gid_b]['score']):
                best_for_b[gid_b] = candidate

        accepted = 0
        for candidate in candidates:
            gid_a = candidate['gid_a']
            gid_b = candidate['gid_b']
            record = candidate.get('record')
            if best_for_a[gid_a] is candidate and best_for_b[gid_b] is candidate:
                unioned = uf.union(gid_a, gid_b)
                if record is not None:
                    record['accepted'] = True
                    record['unioned'] = bool(unioned)
                    record['rejected_reason'] = (
                        None if unioned else 'already_connected')
                if unioned:
                    accepted += 1
            elif record is not None:
                record['rejected_reason'] = 'not_mutual_best'

        return raw_pairs, len(candidates), accepted

    if merge_mode == "center-valid-raw":
        print("  Center-valid raw mode: skipping cross-tile matching.")
        t2 = time.time()
        print(f"  Pass 2 done: 0 overlap pairs, 0 eligible, 0 merges "
              f"({t2-t1:.0f}s)")
    elif overlap <= 0:
        print("  No tile overlap configured; skipping cross-tile matching.")
        t2 = time.time()
        print(f"  Pass 2 done: 0 overlap pairs, 0 eligible, 0 merges ({t2-t1:.0f}s)")
    else:
        if merge_mode == "center-valid":
            print("  Center-valid mode: matching with overlap masks while "
                  "exporting owner-cropped polygons.")
        for idx, (row, col) in enumerate(tile_keys):
            mask_a = _load_mask(row, col)
            if mask_a is None:
                continue

            # 检查右邻居 (row, col+1)
            if (row, col + 1) in tile_set:
                mask_b = _load_mask(row, col + 1)
                if mask_b is not None:
                    region_a = mask_a[:, stride:tile_size]
                    region_b = mask_b[:, :overlap]
                    raw, eligible, accepted = _accept_neighbor_matches(
                        row, col, row, col + 1,
                        mask_a, mask_b, region_a, region_b,
                        direction='right')
                    pair_count += raw
                    eligible_pair_count += eligible
                    merge_count += accepted

            # 检查下邻居 (row+1, col)
            if (row + 1, col) in tile_set:
                mask_b = _load_mask(row + 1, col)
                if mask_b is not None:
                    region_a = mask_a[stride:tile_size, :]
                    region_b = mask_b[:overlap, :]
                    raw, eligible, accepted = _accept_neighbor_matches(
                        row, col, row + 1, col,
                        mask_a, mask_b, region_a, region_b,
                        direction='down')
                    pair_count += raw
                    eligible_pair_count += eligible
                    merge_count += accepted

            # 对角合并默认关闭；如需打开，会使用更严格阈值。
            if MERGE_ENABLE_DIAGONAL and (row + 1, col + 1) in tile_set:
                mask_b = _load_mask(row + 1, col + 1)
                if mask_b is not None:
                    region_a = mask_a[stride:tile_size, stride:tile_size]
                    region_b = mask_b[:overlap, :overlap]
                    raw, eligible, accepted = _accept_neighbor_matches(
                        row, col, row + 1, col + 1,
                        mask_a, mask_b, region_a, region_b,
                        direction='diagonal',
                        diagonal=True)
                    pair_count += raw
                    eligible_pair_count += eligible
                    merge_count += accepted

            if (idx + 1) % 2000 == 0:
                elapsed = time.time() - t0
                print(f"    {idx + 1}/{total} tiles scanned, "
                      f"{merge_count} merges, {elapsed:.0f}s")

        t2 = time.time()
        print(f"  Pass 2 done: {pair_count} overlap pairs, "
              f"{eligible_pair_count} eligible, {merge_count} merges "
              f"({t2-t1:.0f}s)")

    if debug_matches is not None:
        import csv

        matches_json = os.path.join(debug_dir, "stitch_matches.json")
        with open(matches_json, "w", encoding="utf-8") as f:
            json_mod.dump(debug_matches, f, indent=2)

        matches_csv = os.path.join(debug_dir, "stitch_matches.csv")
        preferred_fields = [
            "direction", "tile_a", "tile_b", "gid_a", "gid_b",
            "intersection", "area_a_overlap", "area_b_overlap",
            "dice", "overlap_ratio", "full_area_a", "full_area_b",
            "area_ratio", "centroid_distance", "strong_overlap", "score",
            "accepted", "unioned", "rejected_reason",
            "centroid_a", "centroid_b",
        ]
        all_fields = set()
        for record in debug_matches:
            all_fields.update(record.keys())
        fieldnames = preferred_fields + sorted(all_fields - set(preferred_fields))
        with open(matches_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in debug_matches:
                row_out = {}
                for key in fieldnames:
                    value = record.get(key)
                    if isinstance(value, (list, dict)):
                        row_out[key] = json_mod.dumps(value, separators=(",", ":"))
                    else:
                        row_out[key] = value
                writer.writerow(row_out)
        print(f"  Stitch debug: {matches_json}")

    # ── Pass 3: 按组合并多边形 → GeoJSON ────────────────────────
    print(f"\n[Pass 3] Merging polygons and building GeoJSON...")

    groups = {}
    for gid, polys in poly_map.items():
        root = uf.find(gid)
        if root not in groups:
            groups[root] = []
        groups[root].extend(polys)

    if debug_dir:
        group_members = {}
        for gid in poly_map:
            root = uf.find(gid)
            group_members.setdefault(root, []).append(gid)
        group_debug = []
        for root, members in group_members.items():
            polys = groups.get(root, [])
            group_debug.append({
                'root': [int(root[0]), int(root[1]), int(root[2])],
                'member_count': len(members),
                'members': [
                    [int(g[0]), int(g[1]), int(g[2])]
                    for g in sorted(members)
                ],
                'polygon_count': len(polys),
                'area': float(sum(float(p.area) for p in polys)),
            })
        group_debug.sort(
            key=lambda g: (g['member_count'], g['area']),
            reverse=True,
        )
        groups_path = os.path.join(debug_dir, "stitch_groups.json")
        with open(groups_path, "w", encoding="utf-8") as f:
            json_mod.dump(group_debug, f, indent=2)
        print(f"  Stitch groups: {groups_path}")

    del poly_map

    print(f"  {len(groups)} merged groups (from tile-level instances)")

    features = []
    skipped = 0
    center_valid_seam_buffer = 0.5

    def _union_group_polygons(polys):
        if len(polys) == 1:
            return polys[0]

        merged = unary_union(polys)
        if (merge_mode == "center-valid" and
                center_valid_seam_buffer > 0 and
                not merged.is_empty):
            try:
                buffered = [
                    p.buffer(center_valid_seam_buffer, join_style=2)
                    for p in polys
                    if not p.is_empty
                ]
                if buffered:
                    closed = unary_union(buffered).buffer(
                        -center_valid_seam_buffer,
                        join_style=2,
                    )
                    if not closed.is_empty:
                        return closed
            except Exception:
                pass
        return merged

    for grp_idx, (root, polys) in enumerate(groups.items()):
        try:
            merged = _union_group_polygons(polys)

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

                if contour_tolerance > 0:
                    p = p.simplify(contour_tolerance,
                                   preserve_topology=True)

                # 坐标变换: crop-relative target-level → level-0 绝对坐标
                if need_transform:
                    from shapely.affinity import affine_transform as shapely_affine
                    p = shapely_affine(p, [level_downsample, 0, 0,
                                          level_downsample, ox, oy])

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
