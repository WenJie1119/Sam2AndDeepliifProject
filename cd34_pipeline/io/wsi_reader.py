#!/usr/bin/env python3
"""
wsi_reader.py — WSI (Whole Slide Image) 读取模块

基于 OpenSlide 的全切片图像读取器，支持从 .ndpi/.svs 等 WSI 格式
按需读取指定区域的 tile，无需预先切割保存到磁盘。

参考: Pathological-image-slide-processing/src/image_slicer.py
"""

import sys
import time
import numpy as np
from typing import Optional

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

    支持 preload 模式：一次性将整个 level 读入内存（numpy 数组），
    后续 read_tile 变成纯内存切片操作，全程零 I/O。
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

        # preload 数据：None 表示未预载
        self._preloaded_image: Optional[np.ndarray] = None

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

    @property
    def is_preloaded(self) -> bool:
        """是否已预载全图到内存。"""
        return self._preloaded_image is not None

    def preload(self, max_mem_gb: float = 100.0):
        """
        将整个 level 读入内存为 numpy RGB 数组。

        采用分块读取策略：按行条带（strip）逐块读取并直接写入
        预分配的 RGB 数组，避免产生巨大的 RGBA 中间对象。

        峰值额外内存 ≈ strip_height × width × 4 bytes（单条 RGBA），
        远小于整图 RGBA 的内存开销。

        Args:
            max_mem_gb: 允许的最大 RGB 数组内存（GB）。超过此值时
                        打印警告并跳过预载，走按需读取。默认 100 GB。
        """
        if self._preloaded_image is not None:
            print("  Already preloaded, skipping.")
            return

        width, height = self.level_dimensions
        mem_gb = width * height * 3 / (1024 ** 3)

        if mem_gb > max_mem_gb:
            print(f"  WARNING: Preload would require {mem_gb:.1f} GB "
                  f"(limit={max_mem_gb:.1f} GB). Skipping preload.")
            print(f"  Tip: Use a lower --target-magnification (e.g. 20) "
                  f"or increase limit with --preload-max-gb.")
            return

        print(f"  Preloading level {self.level} into memory "
              f"({width}x{height}, estimated {mem_gb:.1f} GB)...")

        t0 = time.time()

        # 预分配目标 RGB 数组（仅 3 通道）
        self._preloaded_image = np.empty((height, width, 3), dtype=np.uint8)

        # 按行条带分块读取，每条最多 strip_height 行
        # 控制每条 RGBA 临时对象 ≈ strip_height × width × 4 bytes
        # strip_height=4096 时，200k 宽度下每条 RGBA ≈ 3.2 GB，很安全
        strip_height = 4096
        downsample = self.level_downsample
        strips_done = 0
        total_strips = (height + strip_height - 1) // strip_height

        for y_start in range(0, height, strip_height):
            h = min(strip_height, height - y_start)

            # read_region 需要 level 0 坐标
            x0_level0 = 0
            y0_level0 = int(y_start * downsample)

            region = self.slide.read_region(
                (x0_level0, y0_level0), self.level, (width, h)
            )

            # RGBA → RGB，直接写入预分配数组，然后释放 region
            strip_rgb = np.array(region)[:, :, :3]
            self._preloaded_image[y_start:y_start + h, :, :] = strip_rgb
            del region, strip_rgb

            strips_done += 1
            print(f"    Preloading: strip {strips_done}/{total_strips} "
                  f"(rows {y_start}-{y_start + h - 1})", end='\r')

        elapsed = time.time() - t0
        actual_gb = self._preloaded_image.nbytes / (1024 ** 3)
        print(f"\n  Preload complete: {actual_gb:.1f} GB in {elapsed:.1f}s "
              f"(shape={self._preloaded_image.shape})")

    def unload(self):
        """释放预载的全图内存。"""
        if self._preloaded_image is not None:
            shape = self._preloaded_image.shape
            gb = self._preloaded_image.nbytes / (1024 ** 3)
            del self._preloaded_image
            self._preloaded_image = None
            print(f"  Unloaded preloaded image ({shape}, {gb:.1f} GB freed)")

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

        如果已 preload，直接从内存数组切片（零 I/O）。
        边缘 tile 不足 tile_size 时用白色 (255,255,255) 填充。

        Args:
            tile_info: enumerate_tiles() 返回的单个 tile dict

        Returns:
            PIL.Image.Image: RGB 图像，尺寸固定为 tile_size x tile_size
        """
        x = tile_info['x']
        y = tile_info['y']
        actual_w = tile_info['actual_w']
        actual_h = tile_info['actual_h']

        if self._preloaded_image is not None:
            # 从预载的 numpy 数组中切片 (H, W, 3)
            tile_np = self._preloaded_image[y:y + actual_h, x:x + actual_w].copy()

            if actual_w < self.tile_size or actual_h < self.tile_size:
                padded = np.full((self.tile_size, self.tile_size, 3), 255, dtype=np.uint8)
                padded[:actual_h, :actual_w] = tile_np
                return Image.fromarray(padded)
            return Image.fromarray(tile_np)

        # 未 preload：走 OpenSlide read_region
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

        如果已 preload，直接从内存数组切片，跳过 PIL 转换。

        Args:
            tile_info: enumerate_tiles() 返回的单个 tile dict

        Returns:
            np.ndarray: RGB 数组 (H, W, 3)，dtype=uint8
        """
        if self._preloaded_image is not None:
            x = tile_info['x']
            y = tile_info['y']
            actual_w = tile_info['actual_w']
            actual_h = tile_info['actual_h']

            tile_np = self._preloaded_image[y:y + actual_h, x:x + actual_w].copy()

            if actual_w < self.tile_size or actual_h < self.tile_size:
                padded = np.full((self.tile_size, self.tile_size, 3), 255, dtype=np.uint8)
                padded[:actual_h, :actual_w] = tile_np
                return padded
            return tile_np

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
        """关闭 OpenSlide 句柄并释放预载内存。"""
        self.unload()
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
