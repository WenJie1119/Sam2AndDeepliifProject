#!/usr/bin/env python3
"""
tile_classifier.py — YOLO tile 分类模块

使用 YOLO 分类模型对 WSI tile 进行 background/target 分类。
支持批量推理、CSV 持久化，以及从已有 CSV 加载分类结果。
"""

import csv
import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread
from typing import Optional

import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class TileClassifier:
    """
    YOLO tile 分类器。

    使用 YOLO 分类模型判断每个 tile 是 background 还是 target，
    支持批量推理以提高效率。
    """

    def __init__(self, model_path: str, device: str = 'cuda',
                 batch_size: int = 64, imgsz: int = 512):
        """
        Args:
            model_path: YOLO .pt 模型路径
            device: 推理设备 ('cuda' 或 'cpu')
            batch_size: 批量推理大小
            imgsz: 输入图像尺寸
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")

        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.imgsz = imgsz

        print(f"  Loading YOLO classifier: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names  # {0: 'background', 1: 'target'} etc.
        print(f"  YOLO classes: {self.class_names}")

    def classify_tiles_from_wsi(self, wsi_reader, tile_list: list[dict],
                                progress_callback=None,
                                num_workers: int = 4) -> list[dict]:
        """
        批量从 WSI 读取 tile 并用 YOLO 分类。

        使用多线程预读取 tile（I/O 密集），与 GPU 推理形成流水线，
        避免 GPU 空等磁盘读取。

        Args:
            wsi_reader: WSIReader 实例
            tile_list: enumerate_tiles() 返回的 tile 列表
            progress_callback: 可选回调 fn(current, total)
            num_workers: 预读取线程数 (default: 4)

        Returns:
            list[dict]: 每个 tile dict 增加 'classification' 字段
        """
        total = len(tile_list)
        classified = 0

        # 预读队列：最多缓存 2 个 batch 的 images，防止内存爆炸
        max_queued_batches = 2
        batch_queue: Queue = Queue(maxsize=max_queued_batches)

        def _read_batch(batch_tiles):
            """在线程池中并行读取一个 batch 的 tile images。"""
            images = [None] * len(batch_tiles)
            def _read_one(idx, tile_info):
                images[idx] = wsi_reader.read_tile(tile_info)
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = []
                for i, tile_info in enumerate(batch_tiles):
                    futures.append(pool.submit(_read_one, i, tile_info))
                for f in futures:
                    f.result()  # 等待全部完成
            return images

        def _producer():
            """后台线程：按 batch 预读 tile 并放入队列。"""
            for batch_start in range(0, total, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total)
                batch_tiles = tile_list[batch_start:batch_end]
                batch_images = _read_batch(batch_tiles)
                batch_queue.put((batch_tiles, batch_images))
            batch_queue.put(None)  # 哨兵：通知消费者结束

        # 启动预读生产者线程
        producer_thread = Thread(target=_producer, daemon=True)
        producer_thread.start()

        # 消费者：从队列取出已读取的 batch，直接送入 GPU 推理
        while True:
            item = batch_queue.get()
            if item is None:
                break
            batch_tiles, batch_images = item

            # YOLO 批量推理（images 已经在内存中，无需等待 I/O）
            results = self.model.predict(
                batch_images,
                verbose=False,
                batch=self.batch_size,
                half=True,
                device=self.device,
                imgsz=self.imgsz
            )

            # 解析结果
            for i, result in enumerate(results):
                probs = result.probs
                top_class_idx = probs.top1
                class_name = self.class_names[top_class_idx]
                batch_tiles[i]['classification'] = class_name

            classified += len(batch_tiles)

            # 及时释放 images 内存
            del batch_images

            if progress_callback:
                progress_callback(classified, total)
            else:
                print(f"    YOLO classification: {classified}/{total} tiles", end='\r')

        producer_thread.join()
        print(f"    YOLO classification: {total}/{total} tiles - done")
        return tile_list

    def classify_tiles_streaming(self, wsi_reader, tile_list: list[dict],
                                   on_target_tile,
                                   num_workers: int = 4) -> list[dict]:
        """
        流式分类：每个 batch 分类完毕后，立即将 target tile 通过回调发出，
        而不是等全部分类完成。

        Args:
            wsi_reader: WSIReader 实例
            tile_list: enumerate_tiles() 返回的 tile 列表
            on_target_tile: callable(tile_info) — tile 被分类为 target 时调用
            num_workers: 预读取线程数

        Returns:
            list[dict]: 完整的 tile 列表（含 classification 字段），用于保存 CSV
        """
        total = len(tile_list)
        classified = 0

        max_queued_batches = 2
        batch_queue: Queue = Queue(maxsize=max_queued_batches)

        def _read_batch(batch_tiles):
            images = [None] * len(batch_tiles)
            def _read_one(idx, tile_info):
                images[idx] = wsi_reader.read_tile(tile_info)
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = []
                for i, tile_info in enumerate(batch_tiles):
                    futures.append(pool.submit(_read_one, i, tile_info))
                for f in futures:
                    f.result()
            return images

        def _producer():
            for batch_start in range(0, total, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total)
                batch_tiles = tile_list[batch_start:batch_end]
                batch_images = _read_batch(batch_tiles)
                batch_queue.put((batch_tiles, batch_images))
            batch_queue.put(None)

        producer_thread = Thread(target=_producer, daemon=True)
        producer_thread.start()

        while True:
            item = batch_queue.get()
            if item is None:
                break
            batch_tiles, batch_images = item

            results = self.model.predict(
                batch_images,
                verbose=False,
                batch=self.batch_size,
                half=True,
                device=self.device,
                imgsz=self.imgsz
            )

            for i, result in enumerate(results):
                probs = result.probs
                top_class_idx = probs.top1
                class_name = self.class_names[top_class_idx]
                batch_tiles[i]['classification'] = class_name

                # 流式：target tile 立即通过回调发出
                if class_name.lower() == 'target':
                    on_target_tile(batch_tiles[i])

            classified += len(batch_tiles)
            del batch_images
            print(f"    YOLO classification: {classified}/{total} tiles", end='\r')

        producer_thread.join()
        print(f"    YOLO classification: {total}/{total} tiles - done")
        return tile_list

    @staticmethod
    def save_tile_map(tile_list: list[dict], output_path: str):
        """
        保存分类结果为 CSV。

        CSV 格式: row,col,x,y,classification

        Args:
            tile_list: 带有 'classification' 字段的 tile 列表
            output_path: 输出 CSV 路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['row', 'col', 'x', 'y', 'classification'])
            for tile in tile_list:
                writer.writerow([
                    tile['row'], tile['col'],
                    tile['x'], tile['y'],
                    tile.get('classification', 'unknown')
                ])
        print(f"    Saved tile map: {output_path} ({len(tile_list)} tiles)")

    @staticmethod
    def load_tile_map(csv_path: str) -> list[dict]:
        """
        加载已有的 CSV tile map。

        Args:
            csv_path: CSV 文件路径

        Returns:
            list[dict]: tile 信息列表，每个 dict 包含 row, col, x, y, classification
        """
        tiles = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tiles.append({
                    'row': int(row['row']),
                    'col': int(row['col']),
                    'x': int(row['x']),
                    'y': int(row['y']),
                    'classification': row['classification'],
                })
        print(f"    Loaded tile map: {csv_path} ({len(tiles)} tiles)")
        return tiles

    @staticmethod
    def get_target_tiles(tile_list: list[dict]) -> list[dict]:
        """
        过滤返回 "target" 类别的 tile 列表。

        Args:
            tile_list: 带有 'classification' 字段的 tile 列表

        Returns:
            list[dict]: 仅包含 classification == 'target' 的 tile
        """
        targets = [t for t in tile_list if t.get('classification', '').lower() == 'target']
        return targets

    @staticmethod
    def summarize_tile_map(tile_list: list[dict]):
        """打印 tile map 分类统计。"""
        from collections import Counter
        counts = Counter(t.get('classification', 'unknown') for t in tile_list)
        total = len(tile_list)
        print(f"    Tile classification summary ({total} total):")
        for cls, count in sorted(counts.items()):
            pct = count / total * 100 if total > 0 else 0
            print(f"      {cls}: {count} ({pct:.1f}%)")
