#!/usr/bin/env python3
"""
visualize_reconstructed.py — 可视化 WSI 重建的实例分割 mask

支持两种模式:
  thumbnail  缩略图概览（逐行流式读取，低内存）
  crop       ROI 区域裁剪放大（支持像素坐标 / tile 坐标 / tile 范围）

兼容截断的 npy 文件（文件大小 < header 声明的数组大小时，只读取可用行数）。

Usage:
    # 缩略图
    python scripts/visualize_reconstructed.py thumbnail \
        --npy /path/to/reconstructed.npy \
        --output thumb.png --scale 0.01

    # ROI: tile 坐标
    python scripts/visualize_reconstructed.py crop \
        --npy /path/to/reconstructed.npy \
        --tile 100,100 --output tile.png

    # ROI: tile 范围
    python scripts/visualize_reconstructed.py crop \
        --npy /path/to/reconstructed.npy \
        --tile-range 98,98,103,103 --output region.png --contour

    # ROI: 像素坐标
    python scripts/visualize_reconstructed.py crop \
        --npy /path/to/reconstructed.npy \
        --roi 50000,50000,2048,2048 --output roi.png
"""

import argparse
import colorsys
import os
import sys
import time

import cv2
import numpy as np


# ── npy 读取工具 ──────────────────────────────────────────────

def _read_npy_header(path):
    """读取 npy header，返回 (shape, dtype, header_offset, available_rows)。"""
    with open(path, 'rb') as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        header_offset = f.tell()
        f.seek(0, 2)
        file_size = f.tell()

    h, w = shape
    row_bytes = w * dtype.itemsize
    data_bytes = file_size - header_offset
    available_rows = data_bytes // row_bytes

    if available_rows < h:
        print(f"  WARNING: npy truncated — {available_rows}/{h} rows available "
              f"(file {file_size/1024**3:.1f}GB, expected {h*row_bytes/1024**3:.1f}GB)")

    return shape, dtype, header_offset, min(available_rows, h)


def _read_rows(path, dtype, header_offset, width, row_start, row_end):
    """从 npy 文件中按行读取指定范围的数据。"""
    row_bytes = width * dtype.itemsize
    offset = header_offset + row_start * row_bytes
    n_rows = row_end - row_start

    with open(path, 'rb') as f:
        f.seek(offset)
        raw = f.read(n_rows * row_bytes)

    actual_rows = len(raw) // row_bytes
    if actual_rows < n_rows:
        raw = raw[:actual_rows * row_bytes]

    return np.frombuffer(raw, dtype=dtype).reshape(actual_rows, width)


# ── 颜色映射 ─────────────────────────────────────────────────

def id_to_color(instance_id: int) -> tuple:
    """将实例 ID 映射为鲜艳的 RGB 颜色（黄金角散列）。"""
    if instance_id == 0:
        return (0, 0, 0)
    hue = (instance_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.9)
    return (int(r * 255), int(g * 255), int(b * 255))


def render_mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """用 LUT 将 instance mask 渲染为 RGB。"""
    max_id = int(mask.max())
    if max_id == 0:
        return np.zeros((*mask.shape, 3), dtype=np.uint8)

    lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for i in range(1, max_id + 1):
        lut[i] = id_to_color(i)

    return lut[mask]


# ── thumbnail ─────────────────────────────────────────────────

def cmd_thumbnail(args):
    """流式生成缩略图：逐行采样，不需要将全图加载到内存。"""
    print(f"Reading header: {args.npy}")
    t0 = time.time()

    (h, w), dtype, header_offset, avail_rows = _read_npy_header(args.npy)
    print(f"  Shape: {h} x {w}, dtype: {dtype}, available rows: {avail_rows}")

    step = max(1, int(1.0 / args.scale))
    thumb_h = (avail_rows + step - 1) // step
    thumb_w = (w + step - 1) // step
    print(f"  Thumbnail: {thumb_h} x {thumb_w} (step={step})")

    # 逐行采样
    thumb = np.zeros((thumb_h, thumb_w), dtype=np.uint32)
    row_bytes = w * dtype.itemsize

    with open(args.npy, 'rb') as f:
        for ti, row_idx in enumerate(range(0, avail_rows, step)):
            f.seek(header_offset + row_idx * row_bytes)
            raw = f.read(row_bytes)
            if len(raw) < row_bytes:
                break
            row = np.frombuffer(raw, dtype=dtype)
            thumb[ti] = row[::step][:thumb_w].astype(np.uint32)

            if (ti + 1) % 200 == 0:
                print(f"    Row {ti+1}/{thumb_h}")

    print("  Rendering colors...")
    rgb = render_mask_to_rgb(thumb)

    unique_count = len(np.unique(thumb)) - (1 if 0 in thumb else 0)
    fg_pixels = int(np.count_nonzero(thumb))
    print(f"  Instances visible: {unique_count}, foreground pixels: {fg_pixels}")

    output = args.output or os.path.join(os.path.dirname(args.npy), "thumbnail.png")
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    cv2.imwrite(output, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    elapsed = time.time() - t0
    print(f"  Saved: {output} ({elapsed:.1f}s)")


# ── crop ──────────────────────────────────────────────────────

def cmd_crop(args):
    """ROI 裁剪可视化。"""
    tile_size = args.tile_size

    if args.roi:
        parts = list(map(int, args.roi.split(',')))
        y0, x0, rh, rw = parts
    elif args.tile:
        parts = list(map(int, args.tile.split(',')))
        row, col = parts
        y0 = row * tile_size
        x0 = col * tile_size
        rh = tile_size
        rw = tile_size
    elif args.tile_range:
        parts = list(map(int, args.tile_range.split(',')))
        r1, c1, r2, c2 = parts
        y0 = r1 * tile_size
        x0 = c1 * tile_size
        rh = (r2 - r1) * tile_size
        rw = (c2 - c1) * tile_size
    else:
        print("Error: must specify --roi, --tile, or --tile-range")
        sys.exit(1)

    print(f"Reading header: {args.npy}")
    t0 = time.time()
    (h, w), dtype, header_offset, avail_rows = _read_npy_header(args.npy)
    print(f"  Full shape: {h} x {w}, available rows: {avail_rows}")

    # 裁剪边界
    y1 = min(y0 + rh, avail_rows)
    x1 = min(x0 + rw, w)
    if y0 >= avail_rows:
        print(f"  Error: ROI y_start ({y0}) beyond available rows ({avail_rows})")
        sys.exit(1)
    print(f"  ROI: y=[{y0}:{y1}], x=[{x0}:{x1}] ({y1-y0} x {x1-x0})")

    # 读取 ROI 行，裁剪列
    full_rows = _read_rows(args.npy, dtype, header_offset, w, y0, y1)
    roi = full_rows[:, x0:x1].astype(np.uint32)
    del full_rows

    max_id = int(roi.max())
    unique_count = len(np.unique(roi)) - (1 if 0 in roi else 0)
    print(f"  Max ID: {max_id}, instances in ROI: {unique_count}")

    print("  Rendering colors...")
    rgb = render_mask_to_rgb(roi)

    if args.contour:
        print("  Drawing contours...")
        binary = (roi > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, (255, 255, 255), 1)

    output = args.output or os.path.join(os.path.dirname(args.npy), "crop.png")
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    cv2.imwrite(output, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    elapsed = time.time() - t0
    print(f"  Saved: {output} ({elapsed:.1f}s)")


# ── main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize reconstructed WSI instance mask")
    sub = parser.add_subparsers(dest='command')

    # thumbnail
    p_thumb = sub.add_parser('thumbnail', help='Generate downsampled thumbnail')
    p_thumb.add_argument('--npy', required=True, help='Path to reconstructed.npy')
    p_thumb.add_argument('--output', '-o', help='Output PNG path')
    p_thumb.add_argument('--scale', type=float, default=0.01,
                         help='Downsample scale (default: 0.01 = 1/100)')

    # crop
    p_crop = sub.add_parser('crop', help='Crop and visualize a ROI')
    p_crop.add_argument('--npy', required=True, help='Path to reconstructed.npy')
    p_crop.add_argument('--output', '-o', help='Output PNG path')
    p_crop.add_argument('--roi', help='y_start,x_start,height,width (pixels)')
    p_crop.add_argument('--tile', help='row,col (single tile)')
    p_crop.add_argument('--tile-range', help='r1,c1,r2,c2 (tile range)')
    p_crop.add_argument('--tile-size', type=int, default=512)
    p_crop.add_argument('--contour', action='store_true',
                         help='Draw white contour outlines')

    args = parser.parse_args()

    if args.command == 'thumbnail':
        cmd_thumbnail(args)
    elif args.command == 'crop':
        cmd_crop(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
