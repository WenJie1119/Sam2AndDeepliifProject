#!/usr/bin/env python3
"""
wsi_reader.py — WSI (Whole Slide Image) 读取模块

基于 OpenSlide 的全切片图像读取器，支持从 .ndpi/.svs 等 WSI 格式
按需读取指定区域的 tile，无需预先切割保存到磁盘。

参考: Pathological-image-slide-processing/src/image_slicer.py
"""

import sys
import numpy as np

try:
    from openslide import OpenSlide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
except OSError:
    OPENSLIDE_AVAILABLE = False

from PIL import Image


class WSIReader:
    """
    全切片图像读取器。

    从 WSI 文件中按需读取 tile_size x tile_size 的区域，
    tile 之间以 stride = tile_size - overlap 为步长滑动。
    overlap > 0 时相邻 tile 有重叠，用于消除跨 tile 边界的分割断裂。
    """

    def __init__(self, wsi_path: str, tile_size: int = 512,
                 target_magnification: float = 20.0, overlap: int = 0):
        """
        Args:
            wsi_path: WSI 文件路径 (.ndpi, .svs, .tif 等)
            tile_size: tile 边长（像素），默认 512
            target_magnification: 目标放大倍数，默认 20.0x
            overlap: tile 间重叠像素数，默认 0（无重叠）。
                     stride = tile_size - overlap。
        """
        if not OPENSLIDE_AVAILABLE:
            print("Error: openslide-python not available.")
            print("Install with: pip install openslide-python")
            sys.exit(1)

        self.wsi_path = wsi_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.target_magnification = target_magnification

        if self.stride <= 0:
            raise ValueError(f"overlap ({overlap}) must be less than tile_size ({tile_size})")

        self.slide = OpenSlide(wsi_path)
        self.level, self.actual_magnification = self.get_level_for_magnification()
        self.level_dimensions = self.slide.level_dimensions[self.level]  # (width, height)
        self.level_downsample = self.slide.level_downsamples[self.level]
        self.mpp = float(self.slide.properties.get('openslide.mpp-x', 0))

        print(f"  WSI opened: {wsi_path}")
        print(f"  Level {self.level}: {self.level_dimensions[0]}x{self.level_dimensions[1]} "
              f"(magnification={self.actual_magnification:.1f}x, downsample={self.level_downsample:.2f})")
        if overlap > 0:
            print(f"  Overlap: {overlap}px, stride: {self.stride}px")

    def get_level_for_magnification(self) -> tuple[int, float]:
        """
        根据目标放大倍数找最佳金字塔层级。

        Returns:
            (level_index, actual_magnification)
        """
        # 获取基础放大倍数
        try:
            if 'openslide.objective-power' in self.slide.properties:
                base_mag = float(self.slide.properties['openslide.objective-power'])
            elif 'aperio.AppMag' in self.slide.properties:
                base_mag = float(self.slide.properties['aperio.AppMag'])
            else:
                print("  Warning: Cannot determine base magnification, assuming 40x")
                base_mag = 40.0
        except (ValueError, KeyError):
            base_mag = 40.0

        best_level = 0
        best_diff = float('inf')

        for level in range(self.slide.level_count):
            downsample = self.slide.level_downsamples[level]
            level_mag = base_mag / downsample
            diff = abs(level_mag - self.target_magnification)
            if diff < best_diff:
                best_diff = diff
                best_level = level

        actual_mag = base_mag / self.slide.level_downsamples[best_level]
        return best_level, actual_mag

    def get_slide_info(self) -> dict:
        """返回 WSI 元数据信息。"""
        props = self.slide.properties
        return {
            'wsi_path': self.wsi_path,
            'level_count': self.slide.level_count,
            'level_dimensions': [self.slide.level_dimensions[i]
                                 for i in range(self.slide.level_count)],
            'level_downsamples': [self.slide.level_downsamples[i]
                                  for i in range(self.slide.level_count)],
            'selected_level': self.level,
            'selected_dimensions': self.level_dimensions,
            'actual_magnification': self.actual_magnification,
            'target_magnification': self.target_magnification,
            'tile_size': self.tile_size,
            'vendor': props.get('openslide.vendor', 'unknown'),
            'objective_power': props.get('openslide.objective-power', 'unknown'),
        }

    def enumerate_tiles(self) -> list[dict]:
        """
        生成所有 tile 坐标列表。

        stride = tile_size - overlap。

        Returns:
            list of dict: [{row, col, x, y, x_level0, y_level0, actual_w, actual_h}, ...]
            其中 (x, y) 是该 tile 在当前 level 下的像素坐标，
            (x_level0, y_level0) 是在 level 0 下的坐标（用于 read_region）。
        """
        width, height = self.level_dimensions
        tile_size = self.tile_size
        stride = self.stride
        downsample = self.level_downsample

        tiles = []
        row = 0
        for y in range(0, height, stride):
            col = 0
            for x in range(0, width, stride):
                actual_w = min(tile_size, width - x)
                actual_h = min(tile_size, height - y)

                # read_region 需要 level 0 坐标
                x_level0 = int(x * downsample)
                y_level0 = int(y * downsample)

                tiles.append({
                    'row': row,
                    'col': col,
                    'x': x,
                    'y': y,
                    'x_level0': x_level0,
                    'y_level0': y_level0,
                    'actual_w': actual_w,
                    'actual_h': actual_h,
                })
                col += 1
            row += 1

        return tiles

    def read_tile(self, tile_info: dict) -> Image.Image:
        """
        从 WSI 读取单个 tile，返回 PIL Image (RGB)。

        边缘 tile 不足 tile_size 时用白色 (255,255,255) 填充。

        Args:
            tile_info: enumerate_tiles() 返回的单个 tile dict

        Returns:
            PIL.Image.Image: RGB 图像，尺寸固定为 tile_size x tile_size
        """
        actual_w = tile_info['actual_w']
        actual_h = tile_info['actual_h']

        x_level0 = tile_info['x_level0']
        y_level0 = tile_info['y_level0']

        # OpenSlide read_region: 坐标是 level 0 空间，size 是目标 level 下的像素数
        region = self.slide.read_region(
            (x_level0, y_level0), self.level, (actual_w, actual_h)
        )

        # read_region 返回 RGBA，转 RGB
        region_rgb = region.convert('RGB')

        # 如果不足 tile_size，用白色填充
        if actual_w < self.tile_size or actual_h < self.tile_size:
            padded = Image.new('RGB', (self.tile_size, self.tile_size), (255, 255, 255))
            padded.paste(region_rgb, (0, 0))
            return padded

        return region_rgb

    def read_tile_np(self, tile_info: dict) -> np.ndarray:
        """
        读取 tile 并返回 numpy RGB 数组。

        Args:
            tile_info: enumerate_tiles() 返回的单个 tile dict

        Returns:
            np.ndarray: RGB 数组 (H, W, 3)，dtype=uint8
        """
        return np.array(self.read_tile(tile_info))

    def get_tile_filename(self, tile_info: dict) -> str:
        """
        生成 tile 文件名（不含扩展名），匹配 tile_reconstruction.py 的解析格式。

        格式: tile_{row}_{col}_{x}_{y}

        Args:
            tile_info: tile dict

        Returns:
            str: 如 "tile_5_12_5632_2048"
        """
        return f"tile_{tile_info['row']}_{tile_info['col']}_{tile_info['x']}_{tile_info['y']}"

    def get_tile_vis_filename(self, tile_info: dict) -> str:
        """
        生成 tile 可视化文件名（不含扩展名），使用 um 坐标。

        格式: tile_{row}_{col}_{x_um}um_{y_um}um

        Args:
            tile_info: tile dict (需包含 x_level0, y_level0)

        Returns:
            str: 如 "tile_5_12_24192um_4128um"
        """
        if self.mpp > 0:
            x_um = tile_info['x_level0'] * self.mpp
            y_um = tile_info['y_level0'] * self.mpp
            return f"tile_{tile_info['row']}_{tile_info['col']}_{x_um:.0f}um_{y_um:.0f}um"
        return self.get_tile_filename(tile_info)

    def close(self):
        """关闭 OpenSlide 句柄。"""
        if hasattr(self, 'slide') and self.slide is not None:
            self.slide.close()
            self.slide = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()
