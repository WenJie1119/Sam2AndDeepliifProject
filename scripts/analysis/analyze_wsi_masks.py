#!/usr/bin/env python3
"""
analyze_wsi_masks.py — WSI SAM2 mask 分析与可视化

五个子命令:
  stats              统计实例数量和面积（Union-Find 跨 tile 边界合并）
  overlay            mask 叠加到 WSI 原图上查看（单 tile / tile 范围）
  export-geojson     导出 QuPath 兼容 GeoJSON（跨 tile 合并）
  optimize-geojson   优化已有 GeoJSON（Douglas-Peucker 简化，流式低内存）
  locate             根据 µm 坐标定位 tile，返回对应的图像文件名

Usage:
    # 统计
    python scripts/analysis/analyze_wsi_masks.py stats \
        --npy-dir /path/to/npy_masks --output stats.csv

    # 叠加 — 单 tile
    python scripts/analysis/analyze_wsi_masks.py overlay \
        --npy-dir /path/to/npy_masks \
        --wsi /path/to/slide.ndpi \
        --tile 58,130 --output overlay.png

    # 叠加 — tile 范围
    python scripts/analysis/analyze_wsi_masks.py overlay \
        --npy-dir /path/to/npy_masks \
        --wsi /path/to/slide.ndpi \
        --tile-range 56,128,61,133 --output overlay_region.png

    # 定位 tile（µm 坐标）
    python scripts/analysis/analyze_wsi_masks.py locate \
        --x 11000.5 --y 2300.0 --mpp 0.2264 \
        --npy-dir /path/to/npy_masks
"""

import argparse
import colorsys
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# 确保项目根目录在 Python path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cd34_pipeline.io.tile_reconstruction import parse_tile_filename


# ── 公共工具 ─────────────────────────────────────────────────

def id_to_color(instance_id: int) -> tuple:
    """将实例 ID 映射为鲜艳的 RGB 颜色（黄金角散列）。"""
    if instance_id == 0:
        return (0, 0, 0)
    hue = (instance_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.9)
    return (int(r * 255), int(g * 255), int(b * 255))


def scan_tile_dir(npy_dir: str) -> dict:
    """
    扫描 npy_masks 目录，返回 {(row, col): npy_path} 映射。
    """
    tiles = {}
    npy_dir = Path(npy_dir)
    for npy_path in sorted(npy_dir.glob("*.npy")):
        parsed = parse_tile_filename(npy_path.name)
        if parsed:
            row, col, x_off, y_off = parsed
            tiles[(row, col)] = str(npy_path)
    return tiles


def load_tile_mask(npy_path: str) -> np.ndarray:
    """加载单个 tile 的 mask（纯数组，不带 metadata）。"""
    return np.load(npy_path)


# ── Union-Find ───────────────────────────────────────────────

class UnionFind:
    """路径压缩 + 按秩合并的 Union-Find。"""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        # 路径压缩
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ── stats 子命令 ──────────────────────────────────────────────

def cmd_stats(args):
    """统计实例数量和面积，跨 tile 边界用 Union-Find 合并。"""
    t0 = time.time()

    print(f"Scanning tiles: {args.npy_dir}")
    tiles = scan_tile_dir(args.npy_dir)
    print(f"  Found {len(tiles)} tiles with masks")

    if not tiles:
        print("  No tiles found!")
        return

    # 计算网格范围
    max_row = max(rc[0] for rc in tiles) + 1
    max_col = max(rc[1] for rc in tiles) + 1
    print(f"  Grid range: {max_row} rows x {max_col} cols")

    # Pass 1: 为每个 tile 的每个局部实例分配全局 ID，并累计面积
    MAX_LOCAL = 10000  # 假设单 tile 内最多 10000 个实例
    uf = UnionFind()
    area = defaultdict(int)  # global_id → pixel count

    print("\n[Pass 1] Loading tiles and recording instance areas...")
    for idx, ((row, col), npy_path) in enumerate(sorted(tiles.items())):
        mask = load_tile_mask(npy_path)
        local_ids = np.unique(mask)
        local_ids = local_ids[local_ids > 0]

        base = row * max_col * MAX_LOCAL + col * MAX_LOCAL
        for lid in local_ids:
            gid = base + int(lid)
            pixel_count = int(np.count_nonzero(mask == lid))
            area[gid] += pixel_count
            uf.find(gid)  # 初始化

        if (idx + 1) % 2000 == 0:
            print(f"    {idx + 1}/{len(tiles)} tiles loaded")

    print(f"  Total local instances (before merge): {len(area)}")

    # Pass 2: 扫描边界，合并跨 tile 实例
    print("\n[Pass 2] Scanning tile boundaries for cross-tile merging...")
    merge_count = 0
    tile_size = args.tile_size

    for (row, col), npy_path in sorted(tiles.items()):
        mask_a = load_tile_mask(npy_path)
        base_a = row * max_col * MAX_LOCAL + col * MAX_LOCAL

        # 右邻 tile
        if (row, col + 1) in tiles:
            mask_b = load_tile_mask(tiles[(row, col + 1)])
            base_b = row * max_col * MAX_LOCAL + (col + 1) * MAX_LOCAL

            # A 的右边缘列 vs B 的左边缘列
            col_a = mask_a[:, -1]
            col_b = mask_b[:, 0]
            both_fg = (col_a > 0) & (col_b > 0)
            for y_pos in np.where(both_fg)[0]:
                gid_a = base_a + int(col_a[y_pos])
                gid_b = base_b + int(col_b[y_pos])
                if uf.find(gid_a) != uf.find(gid_b):
                    uf.union(gid_a, gid_b)
                    merge_count += 1

        # 下邻 tile
        if (row + 1, col) in tiles:
            mask_b = load_tile_mask(tiles[(row + 1, col)])
            base_b = (row + 1) * max_col * MAX_LOCAL + col * MAX_LOCAL

            row_a = mask_a[-1, :]
            row_b = mask_b[0, :]
            both_fg = (row_a > 0) & (row_b > 0)
            for x_pos in np.where(both_fg)[0]:
                gid_a = base_a + int(row_a[x_pos])
                gid_b = base_b + int(row_b[x_pos])
                if uf.find(gid_a) != uf.find(gid_b):
                    uf.union(gid_a, gid_b)
                    merge_count += 1

    print(f"  Boundary merges: {merge_count}")

    # Pass 3: 聚合合并后的面积
    print("\n[Pass 3] Aggregating merged instance areas...")
    merged_area = defaultdict(int)
    for gid, pix in area.items():
        root = uf.find(gid)
        merged_area[root] += pix

    areas = sorted(merged_area.values(), reverse=True)
    num_instances = len(areas)
    areas_np = np.array(areas)

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"  Total instances (after merge): {num_instances}")
    print(f"  Area (pixels):")
    print(f"    min:    {areas_np.min()}")
    print(f"    max:    {areas_np.max()}")
    print(f"    mean:   {areas_np.mean():.1f}")
    print(f"    median: {np.median(areas_np):.1f}")
    print(f"    total:  {areas_np.sum()}")

    # 输出 CSV
    output_csv = args.output or os.path.join(args.npy_dir, "..", "instance_stats.csv")
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['instance_id', 'area_pixels'])
        for i, a in enumerate(areas, 1):
            writer.writerow([i, a])
    print(f"\n  CSV saved: {output_csv}")

    # 面积分布直方图
    hist_path = output_csv.replace('.csv', '_histogram.png')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 线性尺度
        axes[0].hist(areas_np, bins=100, color='steelblue', edgecolor='white')
        axes[0].set_xlabel('Area (pixels)')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Instance Area Distribution (n={num_instances})')
        axes[0].axvline(np.median(areas_np), color='red', linestyle='--',
                        label=f'median={np.median(areas_np):.0f}')
        axes[0].legend()

        # log 尺度
        axes[1].hist(areas_np, bins=np.logspace(
            np.log10(max(1, areas_np.min())), np.log10(areas_np.max()), 80),
            color='steelblue', edgecolor='white')
        axes[1].set_xscale('log')
        axes[1].set_xlabel('Area (pixels, log scale)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Log-scale Distribution')

        plt.tight_layout()
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f"  Histogram saved: {hist_path}")
    except ImportError:
        print("  (matplotlib not available, skipping histogram)")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")


# ── overlay 子命令 ────────────────────────────────────────────

def cmd_overlay(args):
    """将 SAM2 mask 叠加到 WSI 原图上。"""
    tile_size = args.tile_size

    # 解析 tile 范围
    if args.tile:
        parts = list(map(int, args.tile.split(',')))
        r1, c1 = parts
        r2, c2 = r1 + 1, c1 + 1
    elif args.tile_range:
        parts = list(map(int, args.tile_range.split(',')))
        r1, c1, r2, c2 = parts
    else:
        print("Error: must specify --tile or --tile-range")
        sys.exit(1)

    num_rows = r2 - r1
    num_cols = c2 - c1
    print(f"ROI: tile rows [{r1}, {r2}), cols [{c1}, {c2}) = {num_rows}x{num_cols} tiles")

    # 扫描 tile masks
    tiles = scan_tile_dir(args.npy_dir)
    print(f"  Total tiles with masks: {len(tiles)}")

    # 加载 ROI 内的 masks 并拼接
    roi_h = num_rows * tile_size
    roi_w = num_cols * tile_size
    roi_mask = np.zeros((roi_h, roi_w), dtype=np.uint32)

    # 为不同 tile 的实例分配不同 ID 范围，避免冲突
    id_offset = 0
    tiles_loaded = 0
    for dr in range(num_rows):
        for dc in range(num_cols):
            row, col = r1 + dr, c1 + dc
            if (row, col) not in tiles:
                continue
            mask = load_tile_mask(tiles[(row, col)])
            h, w = mask.shape
            h = min(h, roi_h - dr * tile_size)
            w = min(w, roi_w - dc * tile_size)
            region = mask[:h, :w].astype(np.uint32)
            # 偏移局部 ID
            region[region > 0] += id_offset
            y0 = dr * tile_size
            x0 = dc * tile_size
            roi_mask[y0:y0 + h, x0:x0 + w] = region
            id_offset += int(mask.max())
            tiles_loaded += 1

    max_id = int(roi_mask.max())
    print(f"  Loaded {tiles_loaded} tile masks, max instance ID: {max_id}")

    if max_id == 0:
        print("  No instances in this ROI!")
        return

    # 读取 WSI 原图对应区域 — 自动匹配 npy 使用的倍率
    from cd34_pipeline.io.wsi_reader import WSIReader
    from openslide import OpenSlide
    print(f"\nOpening WSI: {args.wsi}")

    # 从 npy 文件名坐标推断 pipeline 使用的图像尺寸
    max_tile_x = max(
        int(parse_tile_filename(os.path.basename(p))[2] or 0)
        for p in tiles.values()
    )
    max_tile_y = max(
        int(parse_tile_filename(os.path.basename(p))[3] or 0)
        for p in tiles.values()
    )
    npy_extent_w = max_tile_x + tile_size
    npy_extent_h = max_tile_y + tile_size

    slide = OpenSlide(args.wsi)
    base_mag = float(slide.properties.get('openslide.objective-power', 40))
    best_level = 0
    best_diff = float('inf')
    for lvl in range(slide.level_count):
        w, h = slide.level_dimensions[lvl]
        diff = abs(w - npy_extent_w) + abs(h - npy_extent_h)
        if diff < best_diff:
            best_diff = diff
            best_level = lvl
    matched_mag = base_mag / slide.level_downsamples[best_level]
    slide.close()
    print(f"  Auto-detected magnification: {matched_mag:.1f}x (level {best_level})")

    reader = WSIReader(args.wsi, tile_size=tile_size,
                       target_magnification=matched_mag)

    # 计算像素坐标
    px_x = c1 * tile_size
    px_y = r1 * tile_size

    # 直接用 OpenSlide 读取区域
    x_level0 = int(px_x * reader.level_downsample)
    y_level0 = int(px_y * reader.level_downsample)
    region = reader.slide.read_region(
        (x_level0, y_level0), reader.level, (roi_w, roi_h))
    original = np.array(region.convert('RGB'))
    reader.close()
    print(f"  Original image region: {original.shape}")

    # 渲染叠加
    print("  Rendering overlay...")
    overlay = original.copy()

    # 构建 LUT
    lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for i in range(1, max_id + 1):
        lut[i] = id_to_color(i)

    # 半透明填充
    fg = roi_mask > 0
    colors = lut[roi_mask[fg]]
    overlay[fg] = (overlay[fg].astype(np.float32) * 0.5 +
                   colors.astype(np.float32) * 0.5).astype(np.uint8)

    # 轮廓
    for inst_id in range(1, max_id + 1):
        inst_mask = (roi_mask == inst_id).astype(np.uint8) * 255
        if inst_mask.sum() == 0:
            continue
        contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

    # Side-by-side
    comparison = np.concatenate([original, overlay], axis=1)

    output = args.output or os.path.join(args.npy_dir, "..", "overlay.png")
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    cv2.imwrite(output, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {output} ({comparison.shape[1]}x{comparison.shape[0]})")



# ── export-geojson 子命令 ─────────────────────────────────────

def _poly_to_geojson_feature(poly, classification_name="CD34+",
                              classification_color=None):
    """将一个 Shapely Polygon 转为 QuPath GeoJSON Feature，带验证。

    返回 feature dict 或 None（无效时）。
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.geometry.polygon import orient as shapely_orient
    from shapely.validation import make_valid

    if classification_color is None:
        classification_color = [200, 50, 50]

    try:
        p = shapely_orient(poly, sign=1.0)

        # 提取外环坐标（整数），去除连续重复点
        raw_ring = list(p.exterior.coords)
        ring = []
        for x, y in raw_ring:
            pt = [int(round(x)), int(round(y))]
            if not ring or pt != ring[-1]:
                ring.append(pt)

        # 确保环闭合
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])

        # 至少需要 4 个点（3 个不同点 + 闭合点）
        if len(ring) < 4:
            return None

        # 用整数坐标重建验证
        final_poly = ShapelyPolygon(ring)
        if not final_poly.is_valid:
            final_poly = make_valid(final_poly)
        if final_poly.is_empty or final_poly.geom_type not in ('Polygon', 'MultiPolygon'):
            return None
        if final_poly.geom_type == 'MultiPolygon':
            final_poly = max(final_poly.geoms, key=lambda g: g.area)
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


def cmd_export_geojson(args):
    """
    从 npy_masks 导出 QuPath 兼容的 GeoJSON 标注文件。

    四步流程:
      Pass 1: 逐 tile 提取轮廓 → Shapely Polygon，按全局实例 ID 存储
      Pass 2: Union-Find 扫描 tile 边界，合并跨 tile 实例
      Pass 3: 按合并后 root ID 聚合多边形，用 unary_union 合并几何
      Pass 4: 验证 + 导出 GeoJSON
    """
    import json as json_mod
    from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    tile_size = args.tile_size
    epsilon_ratio = args.simplify
    min_area = args.min_area

    print(f"Scanning tiles: {args.npy_dir}")
    tiles = scan_tile_dir(args.npy_dir)
    print(f"  Found {len(tiles)} tiles with masks")

    if not tiles:
        print("  No tiles found!")
        return

    t0 = time.time()

    max_row = max(rc[0] for rc in tiles) + 1
    max_col = max(rc[1] for rc in tiles) + 1
    MAX_LOCAL = 10000
    print(f"  Grid range: {max_row} rows x {max_col} cols")

    # ── Pass 1: 提取轮廓，按全局 ID 存储 Shapely Polygon ──────
    print(f"\n[Pass 1] Extracting contours (min_area={min_area}, simplify={epsilon_ratio})...")
    uf = UnionFind()
    instance_polys = defaultdict(list)  # gid → [ShapelyPolygon, ...]
    skipped = 0
    total_local = 0

    for idx, ((row, col), npy_path) in enumerate(sorted(tiles.items())):
        mask = load_tile_mask(npy_path)

        parsed = parse_tile_filename(os.path.basename(npy_path))
        if parsed and parsed[2] is not None:
            x_off, y_off = parsed[2], parsed[3]
        else:
            x_off = col * tile_size
            y_off = row * tile_size

        base = row * max_col * MAX_LOCAL + col * MAX_LOCAL
        local_ids = np.unique(mask)
        local_ids = local_ids[local_ids > 0]

        for lid in local_ids:
            gid = base + int(lid)
            uf.find(gid)  # 初始化

            inst_mask = (mask == int(lid)).astype(np.uint8) * 255
            contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    skipped += 1
                    continue

                if epsilon_ratio > 0:
                    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
                    contour = cv2.approxPolyDP(contour, epsilon, True)

                if len(contour) < 3:
                    skipped += 1
                    continue

                coords = [(int(pt[0][0]) + x_off, int(pt[0][1]) + y_off)
                           for pt in contour]

                try:
                    poly = ShapelyPolygon(coords)
                    if not poly.is_valid:
                        poly = make_valid(poly)
                    if poly.is_empty:
                        skipped += 1
                        continue
                    # 收集有效多边形
                    if poly.geom_type == 'MultiPolygon':
                        for g in poly.geoms:
                            if g.area >= min_area:
                                instance_polys[gid].append(g)
                                total_local += 1
                    elif poly.geom_type == 'Polygon':
                        if poly.area >= min_area:
                            instance_polys[gid].append(poly)
                            total_local += 1
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1
                    continue

        if (idx + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"    {idx + 1}/{len(tiles)} tiles, {total_local} polygons, {elapsed:.0f}s")

    print(f"  Local instances: {len(instance_polys)}, polygons: {total_local} ({skipped} skipped)")

    # ── Pass 2: Union-Find 扫描边界，合并跨 tile 实例 ──────────
    print("\n[Pass 2] Scanning tile boundaries for cross-tile merging...")
    merge_count = 0

    for (row, col), npy_path in sorted(tiles.items()):
        mask_a = load_tile_mask(npy_path)
        base_a = row * max_col * MAX_LOCAL + col * MAX_LOCAL

        # 右邻 tile
        if (row, col + 1) in tiles:
            mask_b = load_tile_mask(tiles[(row, col + 1)])
            base_b = row * max_col * MAX_LOCAL + (col + 1) * MAX_LOCAL
            col_a = mask_a[:, -1]
            col_b = mask_b[:, 0]
            both_fg = (col_a > 0) & (col_b > 0)
            for y_pos in np.where(both_fg)[0]:
                gid_a = base_a + int(col_a[y_pos])
                gid_b = base_b + int(col_b[y_pos])
                if uf.find(gid_a) != uf.find(gid_b):
                    uf.union(gid_a, gid_b)
                    merge_count += 1

        # 下邻 tile
        if (row + 1, col) in tiles:
            mask_b = load_tile_mask(tiles[(row + 1, col)])
            base_b = (row + 1) * max_col * MAX_LOCAL + col * MAX_LOCAL
            row_a = mask_a[-1, :]
            row_b = mask_b[0, :]
            both_fg = (row_a > 0) & (row_b > 0)
            for x_pos in np.where(both_fg)[0]:
                gid_a = base_a + int(row_a[x_pos])
                gid_b = base_b + int(row_b[x_pos])
                if uf.find(gid_a) != uf.find(gid_b):
                    uf.union(gid_a, gid_b)
                    merge_count += 1

    print(f"  Boundary merges: {merge_count}")

    # ── Pass 3: 按合并后的 root ID 聚合 + unary_union ──────��──
    print("\n[Pass 3] Aggregating merged instances...")
    merged_polys = defaultdict(list)
    for gid, poly_list in instance_polys.items():
        root = uf.find(gid)
        merged_polys[root].extend(poly_list)

    print(f"  Merged instances: {len(merged_polys)} (from {len(instance_polys)} local)")

    # ── Pass 4: 合并几何 + 验证 + 导出 ────────────────────────
    print("\n[Pass 4] Building GeoJSON features...")
    features = []
    total_polygons = 0
    merge_skipped = 0

    for root_id, poly_list in merged_polys.items():
        # 合并同一实例的所有多边形
        try:
            if len(poly_list) == 1:
                merged = poly_list[0]
            else:
                merged = unary_union(poly_list)
                # 膨胀再收缩，消除 tile 边界缝隙
                merged = merged.buffer(1).buffer(-1)
            if not merged.is_valid:
                merged = make_valid(merged)
            if merged.is_empty:
                merge_skipped += 1
                continue
        except Exception:
            merge_skipped += 1
            continue

        # 提取最终的多边形列表（可能是 MultiPolygon）
        if merged.geom_type == 'Polygon':
            final_polys = [merged]
        elif merged.geom_type == 'MultiPolygon':
            final_polys = list(merged.geoms)
        elif merged.geom_type == 'GeometryCollection':
            final_polys = [g for g in merged.geoms
                           if g.geom_type == 'Polygon' and g.area >= min_area]
        else:
            merge_skipped += 1
            continue

        for p in final_polys:
            if p.area < min_area:
                merge_skipped += 1
                continue
            feat = _poly_to_geojson_feature(p)
            if feat is not None:
                features.append(feat)
                total_polygons += 1
            else:
                merge_skipped += 1

    print(f"  Total features: {total_polygons} ({merge_skipped} skipped in merge)")

    # 写入 GeoJSON
    geojson = features  # QuPath expects a JSON array of features

    output = args.output or os.path.join(args.npy_dir, "..", "annotations.geojson")
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

    print(f"\nWriting GeoJSON: {output}")
    with open(output, 'w') as f:
        json_mod.dump(geojson, f)

    file_mb = os.path.getsize(output) / (1024 ** 2)
    elapsed = time.time() - t0
    print(f"  Size: {file_mb:.1f} MB")
    print(f"  Completed in {elapsed:.0f}s")
    print(f"\n  QuPath 使用方法:")
    print(f"  1. File → Open → 选择原图 .ndpi")
    print(f"  2. File → Object → Import objects → 选择此 .geojson")


# ── optimize-geojson 子命令 ────────────────────────────────────

def _dp_simplify(coords, tolerance):
    """Douglas-Peucker 线简化，纯 Python 实现（无需 shapely）。"""
    if len(coords) <= 2:
        return coords
    start, end = coords[0], coords[-1]
    max_dist = 0
    max_idx = 0
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    line_len_sq = dx * dx + dy * dy
    for i in range(1, len(coords) - 1):
        if line_len_sq == 0:
            dist = ((coords[i][0] - start[0]) ** 2 + (coords[i][1] - start[1]) ** 2) ** 0.5
        else:
            t = ((coords[i][0] - start[0]) * dx + (coords[i][1] - start[1]) * dy) / line_len_sq
            t = max(0, min(1, t))
            proj_x = start[0] + t * dx
            proj_y = start[1] + t * dy
            dist = ((coords[i][0] - proj_x) ** 2 + (coords[i][1] - proj_y) ** 2) ** 0.5
        if dist > max_dist:
            max_dist = dist
            max_idx = i
    if max_dist > tolerance:
        left = _dp_simplify(coords[:max_idx + 1], tolerance)
        right = _dp_simplify(coords[max_idx:], tolerance)
        return left[:-1] + right
    else:
        return [coords[0], coords[-1]]


def _simplify_feature(feature, tolerance):
    """简化单个 Feature 的几何坐标（Polygon / MultiPolygon）。"""
    geom = feature.get("geometry", {})
    geom_type = geom.get("type", "")

    def _simplify_rings(rings):
        out = []
        for ring in rings:
            s = _dp_simplify(ring, tolerance)
            out.append(s if len(s) >= 4 else ring if len(ring) >= 4 else s)
        return out

    if geom_type == "Polygon":
        geom["coordinates"] = _simplify_rings(geom["coordinates"])
    elif geom_type == "MultiPolygon":
        geom["coordinates"] = [_simplify_rings(pc) for pc in geom["coordinates"]]
    return feature


def _count_coords(feature):
    """统计 feature 中的坐标点总数。"""
    geom = feature.get("geometry", {})
    geom_type = geom.get("type", "")
    total = 0
    if geom_type == "Polygon":
        for ring in geom.get("coordinates", []):
            total += len(ring)
    elif geom_type == "MultiPolygon":
        for poly in geom.get("coordinates", []):
            for ring in poly:
                total += len(ring)
    return total


def _stream_features(filepath):
    """
    流式解析 GeoJSON Feature 数组（纯 Python，无第三方依赖）。
    支持格式: [Feature, Feature, ...] 或 {"type":"FeatureCollection","features":[...]}
    按大括号深度切分，每找到一个完整的顶层 {} 就解析一个 Feature。
    """
    import json as json_mod
    with open(filepath, 'r', encoding='utf-8') as f:
        while True:
            ch = f.read(1)
            if not ch:
                return
            if ch == '{':
                break
        buf = ['{']
        depth = 1
        in_string = False
        escape = False
        while True:
            ch = f.read(1)
            if not ch:
                break
            buf.append(ch)
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    yield json_mod.loads(''.join(buf))
                    buf = []
                    while True:
                        ch = f.read(1)
                        if not ch:
                            return
                        if ch == '{':
                            buf = ['{']
                            depth = 1
                            break


def cmd_optimize_geojson(args):
    """
    对已有 GeoJSON 文件做 Douglas-Peucker 简化，流式处理（低内存）。

    tolerance=0.5 时，去除偏差 <0.5px 的冗余点，像素级精度不变，
    文件大小通常减少 ~70%。
    """
    import json as json_mod

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        return

    tolerance = args.tolerance
    out_path = Path(args.output) if args.output else \
        input_path.parent / f"{input_path.stem}_optimized.geojson"

    input_size = input_path.stat().st_size
    print(f"Input:     {input_path}")
    print(f"Size:      {input_size / 1024 / 1024:.1f} MB")
    print(f"Output:    {out_path}")
    print(f"Tolerance: {tolerance} px (<=0.5 preserves pixel-level shape)")
    print()

    total_features = 0
    total_before = 0
    total_after = 0
    t0 = time.time()

    with open(out_path, 'w') as out_f:
        out_f.write('[')
        first = True
        for feature in _stream_features(str(input_path)):
            cb = _count_coords(feature)
            feature = _simplify_feature(feature, tolerance)
            ca = _count_coords(feature)
            total_before += cb
            total_after += ca
            total_features += 1
            if not first:
                out_f.write(',')
            first = False
            json_mod.dump(feature, out_f, separators=(',', ':'))
            if total_features % 2000 == 0:
                elapsed = time.time() - t0
                pct = total_after / max(total_before, 1) * 100
                print(f"\r  Processed {total_features:,} features, "
                      f"coords kept {pct:.1f}%, {elapsed:.1f}s",
                      end="", flush=True)
        out_f.write(']')

    elapsed = time.time() - t0
    out_size = out_path.stat().st_size
    reduction = (1 - total_after / max(total_before, 1)) * 100

    print(f"\n\n{'='*60}")
    print(f"Optimization complete ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  Features:          {total_features:,}")
    print(f"  Coords before:     {total_before:,}")
    print(f"  Coords after:      {total_after:,}")
    print(f"  Coords reduced:    {reduction:.1f}%")
    print(f"  File before:       {input_size / 1024 / 1024:.1f} MB")
    print(f"  File after:        {out_size / 1024 / 1024:.1f} MB")
    print(f"  File reduced:      {(1 - out_size / input_size) * 100:.1f}%")
    print(f"  Output:            {out_path}")
    print(f"\nImport in QuPath: File → Import Objects")
    print(f"Tip: set QuPath heap to 8-16GB (Edit → Preferences → Java heap size)")



# ── locate 子命令 ─────────────────────────────────────────────

def cmd_locate(args):
    """根据 µm 坐标定位 tile，返回对应的图像文件名。"""
    tile_size = args.tile_size
    x_um, y_um = args.x, args.y

    # µm → pixel 换算
    mpp = args.mpp
    if mpp is None or mpp <= 0:
        # 尝试从 WSI 自动获取 mpp
        if args.wsi:
            try:
                import openslide
                slide = openslide.OpenSlide(args.wsi)
                mpp = float(slide.properties.get('openslide.mpp-x', 0))
                slide.close()
                if mpp <= 0:
                    print("Error: WSI 中未找到 mpp 信息，请用 --mpp 手动指定")
                    sys.exit(1)
                print(f"mpp={mpp:.4f} (from WSI)")
            except Exception as e:
                print(f"Error: 无法从 WSI 读取 mpp: {e}")
                sys.exit(1)
        else:
            print("Error: 必须指定 --mpp 或 --wsi 以进行 µm → pixel 换算")
            sys.exit(1)

    px = int(round(x_um / mpp))
    py = int(round(y_um / mpp))
    print(f"Input (µm): ({x_um}, {y_um}),  mpp={mpp}")
    print(f"  → pixel: ({px}, {py})")

    # 扫描所有 npy 文件，用文件名中的实际像素偏移量匹配
    npy_dir = Path(args.npy_dir)
    matches = []
    for npy_path in npy_dir.glob("*.npy"):
        parsed = parse_tile_filename(npy_path.name)
        if not parsed:
            continue
        row, col, x_off, y_off = parsed
        if x_off is None or y_off is None:
            continue
        if x_off <= px < x_off + tile_size and y_off <= py < y_off + tile_size:
            matches.append((row, col, npy_path.name))

    if matches:
        for row, col, filename in matches:
            print(f"  → tile (row={row}, col={col})")
            print(f"  → 文件名: {filename}")
    else:
        print(f"  → 未找到对应的 tile 文件")


# ── main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize WSI SAM2 segmentation masks")
    sub = parser.add_subparsers(dest='command')

    # stats
    p_stats = sub.add_parser('stats',
        help='Statistics: instance count and area with cross-tile merging')
    p_stats.add_argument('--npy-dir', required=True,
        help='Directory containing tile npy masks')
    p_stats.add_argument('--output', '-o', help='Output CSV path')
    p_stats.add_argument('--tile-size', type=int, default=512)

    # overlay
    p_over = sub.add_parser('overlay',
        help='Overlay SAM2 mask on WSI original image')
    p_over.add_argument('--npy-dir', required=True,
        help='Directory containing tile npy masks')
    p_over.add_argument('--wsi', required=True, help='Path to WSI file (.ndpi/.svs)')
    p_over.add_argument('--tile', help='row,col (single tile)')
    p_over.add_argument('--tile-range', help='r1,c1,r2,c2 (tile range)')
    p_over.add_argument('--output', '-o', help='Output PNG path')
    p_over.add_argument('--tile-size', type=int, default=512)
    p_over.add_argument('--magnification', type=float, default=20.0,
        help='Target magnification (default: 20.0)')

    # locate
    p_loc = sub.add_parser('locate',
        help='Locate tile by µm coordinate, return image filename')
    p_loc.add_argument('--x', type=float, required=True,
        help='X coordinate in µm')
    p_loc.add_argument('--y', type=float, required=True,
        help='Y coordinate in µm')
    p_loc.add_argument('--mpp', type=float, default=None,
        help='Microns per pixel (e.g. 0.2264). Auto-read from WSI if --wsi is given')
    p_loc.add_argument('--wsi', type=str, default=None,
        help='Path to WSI file (.ndpi/.svs) to auto-read mpp')
    p_loc.add_argument('--npy-dir', required=True,
        help='npy_masks directory for locating tile files')
    p_loc.add_argument('--tile-size', type=int, default=512)

    # export-geojson
    p_geo = sub.add_parser('export-geojson',
        help='Export masks as GeoJSON for QuPath overlay')
    p_geo.add_argument('--npy-dir', required=True,
        help='Directory containing tile npy masks')
    p_geo.add_argument('--output', '-o', help='Output .geojson path')
    p_geo.add_argument('--tile-size', type=int, default=512)
    p_geo.add_argument('--simplify', type=float, default=0.002,
        help='Contour simplification ratio (0=none, default: 0.002)')
    p_geo.add_argument('--min-area', type=float, default=50,
        help='Minimum contour area in pixels (default: 50)')

    # optimize-geojson
    p_opt = sub.add_parser('optimize-geojson',
        help='Optimize existing GeoJSON file (Douglas-Peucker simplify, streaming)')
    p_opt.add_argument('input', help='Input .geojson file path')
    p_opt.add_argument('--output', '-o', help='Output .geojson path (default: <name>_optimized.geojson)')
    p_opt.add_argument('--tolerance', type=float, default=0.5,
        help='Douglas-Peucker tolerance in pixels (default: 0.5, pixel-level lossless)')

    args = parser.parse_args()

    if args.command == 'stats':
        cmd_stats(args)
    elif args.command == 'overlay':
        cmd_overlay(args)
    elif args.command == 'export-geojson':
        cmd_export_geojson(args)
    elif args.command == 'optimize-geojson':
        cmd_optimize_geojson(args)
    elif args.command == 'locate':
        cmd_locate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
