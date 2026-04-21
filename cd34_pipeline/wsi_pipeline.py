#!/usr/bin/env python3
"""
wsi_pipeline.py — WSI (Whole Slide Image) Pipeline 主流程

从 .ndpi/.svs 原图直接读取 512x512 tile，使用 YOLO 模型过滤背景，
只对目标 tile 执行 DeepLIIF + cell extraction + SAM2 完整 pipeline，
最终拼接成全切片实例分割结果。

支持流水线模式：YOLO 分类完一批 target tile 后立即送入 DeepLIIF+SAM2 处理，
无需等待全部分类完成。

Usage:
    python scripts/run_wsi_pipeline.py \
        --wsi-path /path/to/slide.ndpi \
        --output-dir /path/to/output \
        --use-connected-regions --save-npy
"""

import gc
import os
import sys
import time
import queue
import multiprocessing as mp
from threading import Thread

import numpy as np
import torch

from cd34_pipeline.config import (
    parse_arguments,
    validate_config,
    parse_size_thresh,
    parse_large_noise_thresh,
)
from cd34_pipeline.sam2_wrapper.model_loader import load_deepliif, load_sam2
from cd34_pipeline.cell.extraction import (
    get_clusters_from_cells,
    create_binary_mask_from_cells,
)
from cd34_pipeline.cell.mask_utils import get_clusters_from_mask_image
from cd34_pipeline.sam2_wrapper.inference import (
    run_sam2_segmentation,
    merge_connected_masks,
)
from cd34_pipeline.io.file_io import (
    save_deepliif_outputs,
    save_mask_npy,
    save_merged_regions_csv,
    compute_geojson_statistics,
)
from cd34_pipeline.io.wsi_reader import WSIReader
from cd34_pipeline.io.tile_classifier import TileClassifier
from cd34_pipeline.io.tile_reconstruction import export_geojson
from cd34_pipeline.io.gpu_utils import detect_available_gpus, build_worker_assignments

from PIL import Image


class PipelineTimer:
    """Pipeline 各阶段计时器，记录每个阶段的耗时并输出汇总表。"""

    def __init__(self):
        self._stages = []          # [(name, elapsed_seconds)]
        self._current_name = None
        self._current_start = None
        self._total_start = time.time()

    def start(self, stage_name: str):
        """开始计时一个新阶段。如果上一个阶段未结束，自动结束它。"""
        if self._current_name is not None:
            self.stop()
        self._current_name = stage_name
        self._current_start = time.time()

    def stop(self):
        """结束当前阶段的计时。"""
        if self._current_name is None:
            return
        elapsed = time.time() - self._current_start
        self._stages.append((self._current_name, elapsed))
        self._current_name = None
        self._current_start = None

    def summary(self) -> str:
        """生成各阶段耗时汇总表（文本格式）。"""
        if self._current_name is not None:
            self.stop()

        total_elapsed = time.time() - self._total_start

        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("PIPELINE TIMING SUMMARY")
        lines.append("=" * 60)
        lines.append(f"{'Stage':<40} {'Time':>8} {'Pct':>6}")
        lines.append("-" * 60)

        for name, elapsed in self._stages:
            pct = elapsed / total_elapsed * 100 if total_elapsed > 0 else 0
            lines.append(f"{name:<40} {elapsed:>7.1f}s {pct:>5.1f}%")

        lines.append("-" * 60)
        lines.append(f"{'TOTAL':<40} {total_elapsed:>7.1f}s {100.0:>5.1f}%")
        lines.append("=" * 60)

        return "\n".join(lines)

    def save_csv(self, output_path: str):
        """将计时结果保存为 CSV 文件。"""
        import csv as csv_mod
        if self._current_name is not None:
            self.stop()

        total_elapsed = time.time() - self._total_start
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv_mod.writer(f)
            writer.writerow(['stage', 'seconds', 'percent'])
            for name, elapsed in self._stages:
                pct = elapsed / total_elapsed * 100 if total_elapsed > 0 else 0
                writer.writerow([name, f"{elapsed:.2f}", f"{pct:.1f}"])
            writer.writerow(['TOTAL', f"{total_elapsed:.2f}", '100.0'])


def summarize_tile_timings(all_tile_timings, output_dir=None, wsi_stem=None):
    """
    汇总所有 tile 的细粒度计时，输出各阶段的平均时长和标准差。

    Args:
        all_tile_timings: list[dict]，每个 dict 是 process_single_tile 返回的 timing
        output_dir: 可选，保存 CSV 的目录
        wsi_stem: 可选，文件名前缀

    Returns:
        str: 汇总表文本
    """
    import math
    import csv as csv_mod

    if not all_tile_timings:
        return ""

    # 定义阶段顺序和中文标签
    stage_order = [
        ('tile_read',    'Tile 读取'),
        ('deepliif',     'DeepLIIF 推理'),
        ('cell_extract', 'DeepLIIF→SAM2 提示词'),
        ('sam2',         'SAM2 推理'),
        ('mask_merge',   'Mask 合并'),
        ('npy_save',     'NPY 保存'),
        ('vis_save',     '可视化保存'),
    ]

    # 收集每个阶段的值
    stage_values = {key: [] for key, _ in stage_order}
    for t in all_tile_timings:
        for key, _ in stage_order:
            if key in t:
                stage_values[key].append(t[key])

    # 计算 per-tile 总时间
    tile_totals = []
    for t in all_tile_timings:
        tile_totals.append(sum(t.values()))

    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("PER-TILE TIMING STATISTICS")
    lines.append(f"  Total tiles timed: {len(all_tile_timings)}")
    lines.append("=" * 72)
    lines.append(f"{'Stage':<30} {'Count':>6} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
    lines.append("-" * 72)

    csv_rows = []  # 用于保存 CSV

    for key, label in stage_order:
        vals = stage_values[key]
        if not vals:
            continue
        n = len(vals)
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
        mn = min(vals)
        mx = max(vals)
        lines.append(f"{label:<30} {n:>6} {mean:>8.3f}s {std:>8.3f}s {mn:>8.3f}s {mx:>8.3f}s")
        csv_rows.append([key, label, n, f"{mean:.4f}", f"{std:.4f}", f"{mn:.4f}", f"{mx:.4f}"])

    # Tile 总计
    if tile_totals:
        n = len(tile_totals)
        mean = sum(tile_totals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in tile_totals) / n) if n > 1 else 0.0
        mn = min(tile_totals)
        mx = max(tile_totals)
        lines.append("-" * 72)
        lines.append(f"{'Per-tile TOTAL':<30} {n:>6} {mean:>8.3f}s {std:>8.3f}s {mn:>8.3f}s {mx:>8.3f}s")
        csv_rows.append(['tile_total', 'Per-tile TOTAL', n, f"{mean:.4f}", f"{std:.4f}", f"{mn:.4f}", f"{mx:.4f}"])

    lines.append("=" * 72)

    # 保存 CSV
    if output_dir and wsi_stem:
        csv_path = os.path.join(output_dir, f"{wsi_stem}_tile_timing_stats.csv")
        os.makedirs(output_dir, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv_mod.writer(f)
            writer.writerow(['stage_key', 'stage_name', 'count', 'mean_s', 'std_s', 'min_s', 'max_s'])
            for row in csv_rows:
                writer.writerow(row)
        lines.append(f"  Tile timing CSV: {csv_path}")

    return "\n".join(lines)


def load_crop_region(crop_csv_path, wsi_path):
    """
    从 crop CSV 中查找与当前 WSI 匹配的裁剪区域。

    CSV 格式: filename,x,y,width,height（level 0 坐标）

    Args:
        crop_csv_path: CSV 文件路径
        wsi_path: 当前 WSI 文件路径

    Returns:
        dict: {'x': int, 'y': int, 'width': int, 'height': int} 或 None
    """
    import csv

    wsi_basename = os.path.basename(wsi_path)
    wsi_stem = os.path.splitext(wsi_basename)[0]

    with open(crop_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_filename = row['filename'].strip()
            csv_stem = os.path.splitext(csv_filename)[0]
            if csv_stem == wsi_stem or csv_filename == wsi_basename:
                return {
                    'x': int(row['x']),
                    'y': int(row['y']),
                    'width': int(row['width']),
                    'height': int(row['height']),
                }

    return None


def filter_tiles_by_crop_region(all_tiles, crop_region, level_downsample):
    """
    过滤 tile 列表，只保留与裁剪区域有重叠的 tile。

    Args:
        all_tiles: enumerate_tiles() 返回的完整 tile 列表
        crop_region: load_crop_region() 返回的裁剪区域 dict（level 0 坐标）
        level_downsample: 当前 level 相对于 level 0 的缩放因子

    Returns:
        list: 过滤后的 tile 列表
    """
    # 裁剪区域在 level 0 坐标下的边界
    roi_x0 = crop_region['x']
    roi_y0 = crop_region['y']
    roi_x1 = roi_x0 + crop_region['width']
    roi_y1 = roi_y0 + crop_region['height']

    filtered = []
    for tile in all_tiles:
        # tile 在 level 0 坐标下的范围
        tx0 = tile['x_level0']
        ty0 = tile['y_level0']
        tx1 = tx0 + int(tile['actual_w'] * level_downsample)
        ty1 = ty0 + int(tile['actual_h'] * level_downsample)

        # 检查是否与 ROI 有重叠（矩形相交判定）
        if tx0 < roi_x1 and tx1 > roi_x0 and ty0 < roi_y1 and ty1 > roi_y0:
            filtered.append(tile)

    return filtered


def process_single_tile(tile_info, wsi_reader, deepliif_engine, sam2_predictor,
                        args, output_dir, size_thresh, large_noise_thresh):
    """
    处理单个 tile 的完整 pipeline: DeepLIIF → cell extraction → SAM2 → save。

    Args:
        tile_info: tile 坐标信息 dict
        wsi_reader: WSIReader 实例
        deepliif_engine: DeepLIIF 推理引擎
        sam2_predictor: SAM2 predictor
        args: 命令行参数
        output_dir: 输出根目录
        size_thresh: DeepLIIF size threshold
        large_noise_thresh: DeepLIIF large noise threshold

    Returns:
        tuple: (bool, dict) — 是否成功产出 mask, 各阶段耗时字典(秒)
               timing keys: tile_read, deepliif, cell_extract, sam2, mask_merge,
                             npy_save, vis_save
    """
    import cv2

    timing = {}

    tile_name = wsi_reader.get_tile_filename(tile_info)
    row, col = tile_info['row'], tile_info['col']
    debug_vis = getattr(args, 'debug_vis', False)

    # debug 输出目录
    debug_dir = None
    if debug_vis:
        debug_dir = os.path.join(output_dir, "debug_vis", tile_name)
        os.makedirs(debug_dir, exist_ok=True)
        print(f"      Debug vis → {debug_dir}")

    # 1. 读取 tile
    _t0 = time.time()
    tile_pil = wsi_reader.read_tile(tile_info)
    tile_np = np.array(tile_pil)
    timing['tile_read'] = time.time() - _t0

    if debug_vis:
        Image.fromarray(tile_np).save(os.path.join(debug_dir, "step1_original.png"))

    # 2. DeepLIIF 推理
    _t0 = time.time()
    deepliif_results = deepliif_engine.inference(
        tile_pil,
        tile_size=args.tile_size,
        seg_weights=args.seg_weights,
        resolution=args.resolution,
        do_postprocessing=args.enable_postprocessing,
        seg_thresh=args.seg_thresh,
        size_thresh=size_thresh,
        marker_thresh=args.marker_thresh,
        size_thresh_upper=getattr(args, 'size_thresh_upper', None),
        noise_thresh=args.noise_thresh,
        large_noise_thresh=large_noise_thresh,
        color_dapi=getattr(args, 'color_dapi', False),
        color_marker=getattr(args, 'color_marker', False),
    )
    timing['deepliif'] = time.time() - _t0

    if debug_vis:
        save_deepliif_outputs(deepliif_results, debug_dir,
                              save_all=getattr(args, 'save_all_deepliif', False))

    # 检查 Seg 输出
    if deepliif_results.get('Seg') is None:
        print(f"      No Seg output for tile ({row},{col}), skipping.")
        return False, timing

    # 3. 细胞提取与分类（= DeepLIIF 结果 → SAM2 提示词）
    _t0 = time.time()
    seg_img = deepliif_results.get('Seg')
    marker_img = deepliif_results.get('Marker')
    seg_np = np.array(seg_img)

    if marker_img is None:
        print(f"      No Marker output for tile ({row},{col}), skipping.")
        timing['cell_extract'] = time.time() - _t0
        return False, timing

    marker_np = np.array(marker_img)

    from cd34_pipeline.cell.extraction import extract_connected_positive_regions
    positive_cells_info = extract_connected_positive_regions(
        seg_np, marker_np,
        seg_thresh=args.seg_thresh,
        marker_thresh=args.marker_thresh,
        morphology_kernel=args.morphology_kernel,
        min_area=args.min_mask_area,
        debug_dir=debug_dir if debug_vis else None,
    )

    if debug_vis:
        # 在原图上绘制提取到的正性细胞区域
        vis_cells = tile_np.copy()
        for ci in positive_cells_info:
            coords = ci.get('coords')
            if coords is not None and len(coords) > 0:
                rows, cols = coords[:, 0], coords[:, 1]
                vis_cells[rows, cols] = (vis_cells[rows, cols] * 0.5 +
                                         np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        Image.fromarray(vis_cells).save(
            os.path.join(debug_dir, f"step3_positive_cells_{len(positive_cells_info)}.png"))

    if len(positive_cells_info) == 0:
        if debug_vis:
            print(f"      Debug: 0 positive cells found")
        timing['cell_extract'] = time.time() - _t0
        return False, timing

    clusters = get_clusters_from_cells(positive_cells_info)
    if len(clusters) == 0:
        timing['cell_extract'] = time.time() - _t0
        return False, timing
    timing['cell_extract'] = time.time() - _t0

    # 4. SAM2 推理 — 每个 tile 必须重新 set_image
    _t0 = time.time()
    sam_mask, scores, filtered = run_sam2_segmentation(
        sam2_predictor, tile_np, clusters,
        min_area=args.min_mask_area,
        set_image=True,
        score_threshold=0.1,
        debug_dir=debug_dir if debug_vis else None,
    )
    timing['sam2'] = time.time() - _t0

    if debug_vis:
        # SAM2 原始输出 — 每个实例不同颜色
        vis_sam = tile_np.copy()
        for inst_id in range(1, int(np.max(sam_mask)) + 1):
            inst_pixels = sam_mask == inst_id
            if not np.any(inst_pixels):
                continue
            color = np.array([(inst_id * 67) % 256, (inst_id * 137) % 256,
                              (inst_id * 221) % 256])
            vis_sam[inst_pixels] = (vis_sam[inst_pixels] * 0.4 + color * 0.6).astype(np.uint8)
            # 画轮廓
            contours, _ = cv2.findContours(
                inst_pixels.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_sam, contours, -1, color.tolist(), 1)
        Image.fromarray(vis_sam).save(
            os.path.join(debug_dir, f"step4_sam2_raw_{int(np.max(sam_mask))}inst.png"))

    # 5. 合并连通掩码
    _t0 = time.time()
    sam_mask_merged, scores_merged, merge_mapping, merged_cells_info = merge_connected_masks(
        sam_mask, scores, positive_cells_info,
        min_area=200,
    )
    timing['mask_merge'] = time.time() - _t0

    if debug_vis:
        # 合并后的最终 mask
        vis_merged = tile_np.copy()
        for inst_id in range(1, int(np.max(sam_mask_merged)) + 1):
            inst_pixels = sam_mask_merged == inst_id
            if not np.any(inst_pixels):
                continue
            color = np.array([(inst_id * 67) % 256, (inst_id * 137) % 256,
                              (inst_id * 221) % 256])
            vis_merged[inst_pixels] = (vis_merged[inst_pixels] * 0.4 +
                                       color * 0.6).astype(np.uint8)
            contours, _ = cv2.findContours(
                inst_pixels.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_merged, contours, -1, color.tolist(), 2)
        Image.fromarray(vis_merged).save(
            os.path.join(debug_dir, f"step5_merged_{int(np.max(sam_mask_merged))}inst.png"))

    # 6. 检查是否有有效结果
    if np.max(sam_mask_merged) == 0:
        return False, timing

    # 7. 保存 npy mask
    _t0 = time.time()
    npy_dir = os.path.join(output_dir, "npy_masks")
    npy_path = os.path.join(npy_dir, f"{tile_name}.npy")
    save_mask_npy(sam_mask_merged, npy_path)
    timing['npy_save'] = time.time() - _t0

    # 8. 保存 tile 可视化 PNG（--save-tile-vis）
    _t0 = time.time()
    if getattr(args, 'save_tile_vis', False):
        vis_dir = os.path.join(output_dir, "tile_vis")
        os.makedirs(vis_dir, exist_ok=True)
        vis_img = tile_np.copy()
        for inst_id in range(1, int(np.max(sam_mask_merged)) + 1):
            inst_pixels = sam_mask_merged == inst_id
            if not np.any(inst_pixels):
                continue
            color = np.array([(inst_id * 67) % 256, (inst_id * 137) % 256,
                              (inst_id * 221) % 256])
            vis_img[inst_pixels] = (vis_img[inst_pixels] * 0.4 +
                                    color * 0.6).astype(np.uint8)
            contours, _ = cv2.findContours(
                inst_pixels.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours, -1, color.tolist(), 2)
        vis_name = wsi_reader.get_tile_vis_filename(tile_info)
        Image.fromarray(vis_img).save(os.path.join(vis_dir, f"{vis_name}.png"))

    if debug_vis:
        print(f"      Debug vis saved: 5 steps → {debug_dir}")

    timing['vis_save'] = time.time() - _t0

    # 释放内存
    del tile_pil, tile_np, deepliif_results, seg_np, marker_np
    del sam_mask, sam_mask_merged
    gc.collect()

    return True, timing


def _worker_process_tiles(gpu_id, worker_id, tile_list, args, output_dir,
                          result_dict):
    """
    Worker 进程：在指定 GPU 上加载模型并处理一组 tiles。

    Args:
        gpu_id: CUDA 设备编号
        worker_id: worker 编号（用于日志）
        tile_list: 该 worker 负责处理的 tile 列表
        args: 命令行参数（Namespace）
        output_dir: 输出目录
        result_dict: multiprocessing.Manager().dict()，用于返回结果
    """
    device = f'cuda:{gpu_id}'
    tag = f"[Worker {worker_id} | GPU {gpu_id}]"

    try:
        torch.autograd.set_grad_enabled(False)

        # 1. 每个 worker 独立打开 WSI（不 preload，避免内存翻倍）
        wsi_reader = WSIReader(
            args.wsi_path,
            tile_size=args.tile_size,
            target_magnification=args.target_magnification,
            overlap=args.overlap,
        )

        # 2. 在指定 GPU 上加载模型
        print(f"  {tag} Loading DeepLIIF + SAM2 on {device}...")
        deepliif_engine = load_deepliif(args.deepliif_model_dir, device)
        sam2_predictor = load_sam2(args.sam_config, args.sam_checkpoint, device)

        size_thresh = parse_size_thresh(args.size_thresh)
        large_noise_thresh = parse_large_noise_thresh(args.large_noise_thresh)

        # 3. 处理分配的 tiles
        success_count = 0
        skip_count = 0
        all_tile_timings = []

        for idx, tile_info in enumerate(tile_list):
            row, col = tile_info['row'], tile_info['col']
            print(f"  {tag} Tile [{idx+1}/{len(tile_list)}] ({row},{col})")

            success, tile_timing = process_single_tile(
                tile_info, wsi_reader, deepliif_engine, sam2_predictor,
                args, output_dir, size_thresh, large_noise_thresh,
            )
            all_tile_timings.append(tile_timing)
            if success:
                success_count += 1
            else:
                skip_count += 1

        wsi_reader.close()

        # 通过共享 dict 返回结果
        result_dict[worker_id] = {
            'success': success_count,
            'skip': skip_count,
            'tile_timings': all_tile_timings,
        }
        print(f"  {tag} Done: {success_count} masks, {skip_count} skipped")

    except Exception as e:
        import traceback
        print(f"  {tag} ERROR: {e}")
        traceback.print_exc()
        result_dict[worker_id] = {'success': 0, 'skip': 0, 'error': str(e), 'tile_timings': []}


def _worker_process_tiles_from_queue(gpu_id, worker_id, tile_queue, args,
                                     output_dir, result_dict):
    """
    Worker 进程（流水线模式）：从队列中持续获取 tile 并处理，直到收到 sentinel。

    Args:
        gpu_id: CUDA 设备编号
        worker_id: worker 编号
        tile_queue: multiprocessing.Queue，生产者放入 target tile，None 为结束信号
        args: 命令行参数
        output_dir: 输出目录
        result_dict: multiprocessing.Manager().dict()
    """
    device = f'cuda:{gpu_id}'
    tag = f"[Worker {worker_id} | GPU {gpu_id}]"

    try:
        torch.autograd.set_grad_enabled(False)

        wsi_reader = WSIReader(
            args.wsi_path,
            tile_size=args.tile_size,
            target_magnification=args.target_magnification,
            overlap=args.overlap,
        )

        print(f"  {tag} Loading DeepLIIF + SAM2 on {device}...")
        deepliif_engine = load_deepliif(args.deepliif_model_dir, device)
        sam2_predictor = load_sam2(args.sam_config, args.sam_checkpoint, device)

        size_thresh = parse_size_thresh(args.size_thresh)
        large_noise_thresh = parse_large_noise_thresh(args.large_noise_thresh)

        success_count = 0
        skip_count = 0
        processed = 0
        all_tile_timings = []

        while True:
            tile_info = tile_queue.get()
            if tile_info is None:
                break

            processed += 1
            row, col = tile_info['row'], tile_info['col']
            print(f"  {tag} Tile #{processed} ({row},{col})")

            success, tile_timing = process_single_tile(
                tile_info, wsi_reader, deepliif_engine, sam2_predictor,
                args, output_dir, size_thresh, large_noise_thresh,
            )
            all_tile_timings.append(tile_timing)
            if success:
                success_count += 1
            else:
                skip_count += 1

        wsi_reader.close()

        result_dict[worker_id] = {
            'success': success_count,
            'skip': skip_count,
            'tile_timings': all_tile_timings,
        }
        print(f"  {tag} Done: {success_count} masks, {skip_count} skipped")

    except Exception as e:
        import traceback
        print(f"  {tag} ERROR: {e}")
        traceback.print_exc()
        result_dict[worker_id] = {'success': 0, 'skip': 0, 'error': str(e), 'tile_timings': []}


def _pipeline_single_device(wsi_reader, all_tiles, args, output_dir,
                            device, size_thresh, large_noise_thresh):
    """
    单设备流水线：YOLO 在后台线程分类，主线程用 DeepLIIF+SAM2 处理 target tile。

    YOLO 分类一个 batch 后，target tile 立即入队，主线程取出并处理。
    由于都在同一 GPU 上，YOLO batch 推理很快完成后 GPU 空闲给 DeepLIIF+SAM2。

    Returns:
        (success_count, skip_count, classified_tiles)
    """
    # 加载 YOLO + DeepLIIF + SAM2
    print(f"  Loading YOLO classifier...")
    classifier = TileClassifier(
        model_path=args.yolo_model_path,
        device=device,
        batch_size=args.yolo_batch_size,
        imgsz=args.tile_size,
    )

    print(f"  Loading DeepLIIF + SAM2 on {device}...")
    deepliif_engine = load_deepliif(args.deepliif_model_dir, device)
    sam2_predictor = load_sam2(args.sam_config, args.sam_checkpoint, device)

    # 流水线队列：YOLO → DeepLIIF+SAM2
    tile_queue = queue.Queue(maxsize=64)

    def _yolo_producer():
        """后台线程：YOLO 流式分类，target tile 入队。"""
        def on_target(tile_info):
            tile_queue.put(tile_info)

        classifier.classify_tiles_streaming(
            wsi_reader, all_tiles,
            on_target_tile=on_target,
            num_workers=args.yolo_prefetch_workers,
        )
        tile_queue.put(None)  # sentinel: 分类完毕

    print(f"\n[Pipeline] Starting YOLO classification + DeepLIIF/SAM2 processing...")
    producer = Thread(target=_yolo_producer, daemon=True)
    producer.start()

    # 主线程：从队列取 target tile 并处理
    success_count = 0
    skip_count = 0
    processed = 0
    all_tile_timings = []

    while True:
        tile_info = tile_queue.get()
        if tile_info is None:
            break

        processed += 1
        row, col = tile_info['row'], tile_info['col']
        print(f"\n  --- Target tile #{processed} ({row},{col}) "
              f"at ({tile_info['x']},{tile_info['y']}) ---")

        success, tile_timing = process_single_tile(
            tile_info, wsi_reader, deepliif_engine, sam2_predictor,
            args, output_dir, size_thresh, large_noise_thresh,
        )
        all_tile_timings.append(tile_timing)
        if success:
            success_count += 1
        else:
            skip_count += 1
            print(f"      No positive cells in tile ({row},{col})")

    producer.join()

    # 分类完成后，all_tiles 已被 classify_tiles_streaming 原地更新了 classification 字段
    classified_tiles = all_tiles
    return success_count, skip_count, classified_tiles, all_tile_timings


def _pipeline_multi_gpu(wsi_reader, all_tiles, args, output_dir,
                        gpu_info, size_thresh, large_noise_thresh):
    """
    多 GPU 流水线：YOLO 在后台线程中分类，target tile 通过 mp.Queue
    分发给各 worker 进程（各自在不同 GPU 上运行 DeepLIIF+SAM2）。

    YOLO 很轻量（~1.5GB），放到空闲最少但够用的 GPU 上。
    DeepLIIF+SAM2 较重（~8GB），优先分配到空闲最多的 GPU。

    Returns:
        (success_count, skip_count, classified_tiles)
    """
    YOLO_MEM_MB = 1500  # YOLO 分类模型 + 推理开销
    WORKER_MEM_MB = 8000  # DeepLIIF + SAM2 每个 worker

    # gpu_info 已按 free_mb 降序排列
    # YOLO 放到空闲最少但够放 YOLO 的 GPU（把大显存留给 worker）
    # 从末尾往前找第一个够放 YOLO 的 GPU
    yolo_gpu_idx = len(gpu_info) - 1  # 默认最后一个（空闲最少）
    for i in range(len(gpu_info) - 1, -1, -1):
        if gpu_info[i]['free_mb'] >= YOLO_MEM_MB:
            yolo_gpu_idx = i
            break

    yolo_gpu = gpu_info[yolo_gpu_idx]['gpu_id']
    yolo_device = f'cuda:{yolo_gpu}'

    # worker GPU: 所有 free_mb 足够放 DeepLIIF+SAM2 的 GPU（排除 YOLO 独占的卡）
    worker_gpu_info = []
    for g in gpu_info:
        if g['gpu_id'] == yolo_gpu:
            # YOLO 所在 GPU: 扣除 YOLO 占用后，如果还够放 worker 就也用
            remaining = g['free_mb'] - YOLO_MEM_MB
            if remaining >= WORKER_MEM_MB:
                worker_gpu_info.append({**g, 'free_mb': remaining})
        else:
            if g['free_mb'] >= WORKER_MEM_MB:
                worker_gpu_info.append(g)

    total_workers = sum(g['workers'] for g in worker_gpu_info)
    if total_workers == 0:
        # 没有够放 worker 的 GPU — fallback: 与 YOLO 共享
        total_workers = 1
        worker_gpu_info = [gpu_info[0]]  # 用空闲最多的

    # 使用 spawn 方式启动子进程
    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    result_dict = manager.dict()
    tile_queue = ctx.Queue(maxsize=128)

    # 启动 worker 进程（从 queue 消费 tile）
    print(f"\n[Pipeline] YOLO on GPU {yolo_gpu}, "
          f"{total_workers} workers on {len(worker_gpu_info)} other GPUs")
    workers = []
    worker_id = 0
    for gi in worker_gpu_info:
        for _ in range(gi['workers']):
            p = ctx.Process(
                target=_worker_process_tiles_from_queue,
                args=(gi['gpu_id'], worker_id, tile_queue,
                      args, output_dir, result_dict),
            )
            workers.append(p)
            print(f"  Worker {worker_id}: GPU {gi['gpu_id']}")
            worker_id += 1

    for p in workers:
        p.start()

    # 后台线程：YOLO 分类，target tile 入队
    print(f"  Starting YOLO classification on {yolo_device} (streaming to workers)...")
    classified_result = [None]  # 用列表传递结果给主线程

    def _yolo_thread():
        classifier = TileClassifier(
            model_path=args.yolo_model_path,
            device=yolo_device,
            batch_size=args.yolo_batch_size,
            imgsz=args.tile_size,
        )

        def on_target(tile_info):
            tile_queue.put(tile_info)

        result = classifier.classify_tiles_streaming(
            wsi_reader, all_tiles,
            on_target_tile=on_target,
            num_workers=args.yolo_prefetch_workers,
        )
        classified_result[0] = result

        # 分类完毕，发送 sentinel 给每个 worker
        for _ in workers:
            tile_queue.put(None)

    yolo_thread = Thread(target=_yolo_thread, daemon=True)
    yolo_thread.start()

    # 等待所有 worker 完成（worker 收到 sentinel 后会自动退出）
    for p in workers:
        p.join()

    yolo_thread.join()
    classified_tiles = classified_result[0]

    # 汇总结果
    success_count = 0
    skip_count = 0
    errors = []
    all_tile_timings = []
    for wid, result in result_dict.items():
        success_count += result.get('success', 0)
        skip_count += result.get('skip', 0)
        all_tile_timings.extend(result.get('tile_timings', []))
        if 'error' in result:
            errors.append(f"Worker {wid}: {result['error']}")

    if errors:
        print(f"\n  WARNING: {len(errors)} worker(s) had errors:")
        for err in errors:
            print(f"    {err}")

    return success_count, skip_count, classified_tiles, all_tile_timings


def _process_tiles_batch(target_tiles, wsi_reader, args, output_dir,
                         gpu_info, total_workers, size_thresh, large_noise_thresh):
    """
    批量处理模式（用于 --tile-map 场景，tile 列表已知）。

    Returns:
        (success_count, skip_count, all_tile_timings)
    """
    all_tile_timings = []

    if total_workers == 1 and not gpu_info:
        # 回退：单设备串行
        print(f"\n[Step 4] Loading models on {args.device}...")
        deepliif_engine = load_deepliif(args.deepliif_model_dir, args.device)
        sam2_predictor = load_sam2(args.sam_config, args.sam_checkpoint, args.device)

        print(f"\n[Step 4] Processing {len(target_tiles)} target tiles (single device)...")
        success_count = 0
        skip_count = 0

        for idx, tile_info in enumerate(target_tiles):
            row, col = tile_info['row'], tile_info['col']
            print(f"\n  --- Tile [{idx+1}/{len(target_tiles)}] ({row},{col}) "
                  f"at ({tile_info['x']},{tile_info['y']}) ---")

            success, tile_timing = process_single_tile(
                tile_info, wsi_reader, deepliif_engine, sam2_predictor,
                args, output_dir, size_thresh, large_noise_thresh,
            )
            all_tile_timings.append(tile_timing)
            if success:
                success_count += 1
            else:
                skip_count += 1
                print(f"      No positive cells in tile ({row},{col})")

    elif total_workers == 1:
        gpu_id = gpu_info[0]['gpu_id']
        device = f'cuda:{gpu_id}'
        print(f"\n[Step 4] Loading models on {device}...")
        deepliif_engine = load_deepliif(args.deepliif_model_dir, device)
        sam2_predictor = load_sam2(args.sam_config, args.sam_checkpoint, device)

        print(f"\n[Step 4] Processing {len(target_tiles)} target tiles (single GPU: {device})...")
        success_count = 0
        skip_count = 0

        for idx, tile_info in enumerate(target_tiles):
            row, col = tile_info['row'], tile_info['col']
            print(f"\n  --- Tile [{idx+1}/{len(target_tiles)}] ({row},{col}) "
                  f"at ({tile_info['x']},{tile_info['y']}) ---")

            success, tile_timing = process_single_tile(
                tile_info, wsi_reader, deepliif_engine, sam2_predictor,
                args, output_dir, size_thresh, large_noise_thresh,
            )
            all_tile_timings.append(tile_timing)
            if success:
                success_count += 1
            else:
                skip_count += 1
                print(f"      No positive cells in tile ({row},{col})")

    else:
        # 多 GPU 并行
        if wsi_reader.is_preloaded:
            print("  Unloading preloaded WSI before forking workers...")
            wsi_reader.unload()

        assignments = build_worker_assignments(gpu_info, len(target_tiles))
        print(f"\n[Step 4] Processing {len(target_tiles)} target tiles "
              f"({total_workers} workers across {len(gpu_info)} GPUs)...")
        for a in assignments:
            count = a['tile_end'] - a['tile_start']
            print(f"  Worker {a['worker_id']}: GPU {a['gpu_id']}, "
                  f"{count} tiles [{a['tile_start']}:{a['tile_end']}]")

        ctx = mp.get_context('spawn')
        manager = ctx.Manager()
        result_dict = manager.dict()

        processes = []
        for a in assignments:
            tile_subset = target_tiles[a['tile_start']:a['tile_end']]
            p = ctx.Process(
                target=_worker_process_tiles,
                args=(a['gpu_id'], a['worker_id'], tile_subset,
                      args, output_dir, result_dict),
            )
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        success_count = 0
        skip_count = 0
        errors = []
        for wid, result in result_dict.items():
            success_count += result.get('success', 0)
            skip_count += result.get('skip', 0)
            all_tile_timings.extend(result.get('tile_timings', []))
            if 'error' in result:
                errors.append(f"Worker {wid}: {result['error']}")

        if errors:
            print(f"\n  WARNING: {len(errors)} worker(s) had errors:")
            for err in errors:
                print(f"    {err}")

    return success_count, skip_count, all_tile_timings


def main_wsi():
    """WSI Pipeline 主流程。"""

    # ========== 1. 配置解析与验证 ==========
    args = parse_arguments()
    args = validate_config(args)

    torch.autograd.set_grad_enabled(False)

    wsi_stem = os.path.splitext(os.path.basename(args.wsi_path))[0]
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("WSI PIPELINE STARTED")
    print(f"WSI: {args.wsi_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")

    start_time = time.time()
    timer = PipelineTimer()

    try:
        # ========== 2. 打开 WSI，枚举 tile ==========
        timer.start("WSI open & tile enumeration")
        print("[Step 1] Opening WSI and enumerating tiles...")
        wsi_reader = WSIReader(
            args.wsi_path,
            tile_size=args.tile_size,
            target_magnification=args.target_magnification,
            overlap=args.overlap,
        )
        slide_info = wsi_reader.get_slide_info()
        all_tiles = wsi_reader.enumerate_tiles()
        print(f"  Total tiles: {len(all_tiles)}")

        # ===== 2.0.1 裁剪区域过滤（--crop-csv）=====
        crop_region = None
        if getattr(args, 'crop_csv', None):
            crop_region = load_crop_region(args.crop_csv, args.wsi_path)
            if crop_region is None:
                print(f"  WARNING: No matching crop region found for "
                      f"{os.path.basename(args.wsi_path)} in {args.crop_csv}")
                print(f"  Processing all tiles without crop filtering.")
            else:
                before_count = len(all_tiles)
                all_tiles = filter_tiles_by_crop_region(
                    all_tiles, crop_region, wsi_reader.level_downsample)
                print(f"  Crop region (level 0): x={crop_region['x']}, y={crop_region['y']}, "
                      f"w={crop_region['width']}, h={crop_region['height']}")
                print(f"  Tiles after crop filtering: {len(all_tiles)}/{before_count}")

        # 预载全图到内存（可选）
        if getattr(args, 'preload_wsi', False):
            wsi_reader.preload(max_mem_gb=getattr(args, 'preload_max_gb', 100.0))

        timer.stop()

        # ========== 2.1 --tile-index 模式：只处理单个 tile ==========
        if args.tile_index is not None:
            parts = args.tile_index.split(',')
            if len(parts) != 2:
                print(f"Error: --tile-index must be ROW,COL format, got '{args.tile_index}'")
                sys.exit(1)
            target_row, target_col = int(parts[0]), int(parts[1])

            # 找到对应的 tile
            target_tile = None
            for t in all_tiles:
                if t['row'] == target_row and t['col'] == target_col:
                    target_tile = t
                    break

            if target_tile is None:
                print(f"Error: Tile ({target_row},{target_col}) not found in the WSI grid.")
                sys.exit(1)

            print(f"\n[Single Tile Mode] Processing tile ({target_row},{target_col})...")

            # 自动检测空闲 GPU
            gpu_info = detect_available_gpus(
                min_free_mb=args.gpu_min_free_mb,
                max_gpus=1,
                max_workers_per_gpu=1,
            )
            device = f'cuda:{gpu_info[0]["gpu_id"]}' if gpu_info else args.device

            # 加载模型
            print(f"[Step 2] Loading models (DeepLIIF + SAM2) on {device}...")
            deepliif_engine = load_deepliif(args.deepliif_model_dir, device)
            sam2_predictor = load_sam2(args.sam_config, args.sam_checkpoint, device)

            size_thresh = parse_size_thresh(args.size_thresh)
            large_noise_thresh = parse_large_noise_thresh(args.large_noise_thresh)

            success, tile_timing = process_single_tile(
                target_tile, wsi_reader, deepliif_engine, sam2_predictor,
                args, output_dir, size_thresh, large_noise_thresh,
            )
            print(f"  Result: {'Success (mask saved)' if success else 'No positive cells found'}")
            if tile_timing:
                print(f"\n  Per-step timing:")
                for step_name, step_sec in tile_timing.items():
                    print(f"    {step_name:<20} {step_sec:.3f}s")

            wsi_reader.close()
            elapsed = time.time() - start_time
            print(f"\nSingle tile mode completed in {elapsed:.1f}s")
            return

        # ========== 3. 分类 + 处理 ==========
        tile_map_path = os.path.join(output_dir, f"{wsi_stem}_tile_map.csv")
        target_tile_count = 0  # 用于最终统计
        all_tile_timings = []  # 收集所有 tile 的细粒度计时

        if args.tile_map is not None:
            # --tile-map 模式：加载已有 tile map → 批量处理
            timer.start("Load tile map")
            print("\n[Step 2] Loading existing tile map...")
            print(f"  Loading: {args.tile_map}")
            classified_tiles = TileClassifier.load_tile_map(args.tile_map)
            TileClassifier.summarize_tile_map(classified_tiles)

            target_tiles = TileClassifier.get_target_tiles(classified_tiles)
            print(f"\n  Target tiles to process: {len(target_tiles)}")
            target_tile_count = len(target_tiles)

            if len(target_tiles) == 0:
                print("  No target tiles found. Nothing to process.")
                wsi_reader.close()
                return

            # 补全 tile_info 字段
            tiles_by_pos = {(t['row'], t['col']): t for t in all_tiles}
            enriched_targets = []
            for t in target_tiles:
                key = (t['row'], t['col'])
                if key in tiles_by_pos:
                    full_tile = tiles_by_pos[key].copy()
                    full_tile['classification'] = t['classification']
                    enriched_targets.append(full_tile)
                else:
                    print(f"  Warning: Target tile ({t['row']},{t['col']}) not found in WSI grid, skipping.")
            target_tiles = enriched_targets

            # 检测 GPU 并处理
            print("\n[Step 3] Detecting GPU resources...")
            gpu_info = detect_available_gpus(
                min_free_mb=args.gpu_min_free_mb,
                max_gpus=args.num_gpus,
                max_workers_per_gpu=args.workers_per_gpu,
            )
            total_workers = sum(g['workers'] for g in gpu_info)
            if total_workers == 0:
                print(f"  No GPUs meet resource requirements, falling back to {args.device}")
                total_workers = 1
                gpu_info = []

            size_thresh = parse_size_thresh(args.size_thresh)
            large_noise_thresh = parse_large_noise_thresh(args.large_noise_thresh)

            timer.start("DeepLIIF + SAM2 tile processing")
            success_count, skip_count, batch_tile_timings = _process_tiles_batch(
                target_tiles, wsi_reader, args, output_dir,
                gpu_info, total_workers, size_thresh, large_noise_thresh,
            )
            all_tile_timings.extend(batch_tile_timings)

            timer.stop()

        elif args.classify_only:
            # --classify-only 模式：只分类不处理
            timer.start("YOLO classification (classify-only)")
            print("\n[Step 2] Tile classification (classify-only mode)...")
            classifier = TileClassifier(
                model_path=args.yolo_model_path,
                device=args.device,
                batch_size=args.yolo_batch_size,
                imgsz=args.tile_size,
            )
            classified_tiles = classifier.classify_tiles_from_wsi(
                wsi_reader, all_tiles,
                num_workers=args.yolo_prefetch_workers,
            )
            TileClassifier.summarize_tile_map(classified_tiles)
            TileClassifier.save_tile_map(classified_tiles, tile_map_path)
            timer.stop()
            print(f"\n[Classify Only] Tile map saved to: {tile_map_path}")
            wsi_reader.close()
            elapsed = time.time() - start_time
            print(timer.summary())
            print(f"Classification completed in {elapsed:.1f}s")
            return

        else:
            # ===== 流水线模式：YOLO 分类与 DeepLIIF+SAM2 并行 =====
            print("\n[Step 2] Detecting GPU resources...")
            gpu_info = detect_available_gpus(
                min_free_mb=args.gpu_min_free_mb,
                max_gpus=args.num_gpus,
                max_workers_per_gpu=args.workers_per_gpu,
            )
            total_workers = sum(g['workers'] for g in gpu_info)
            if total_workers == 0:
                print(f"  No GPUs meet resource requirements, falling back to {args.device}")
                total_workers = 1
                gpu_info = []

            size_thresh = parse_size_thresh(args.size_thresh)
            large_noise_thresh = parse_large_noise_thresh(args.large_noise_thresh)

            timer.start("YOLO + DeepLIIF + SAM2 pipeline")
            if total_workers <= 1:
                # 单设备/单 GPU 流水线
                device = args.device if not gpu_info else f'cuda:{gpu_info[0]["gpu_id"]}'
                success_count, skip_count, classified_tiles, pipe_tile_timings = _pipeline_single_device(
                    wsi_reader, all_tiles, args, output_dir,
                    device, size_thresh, large_noise_thresh,
                )
            else:
                # 多 GPU 流水线
                if wsi_reader.is_preloaded:
                    print("  Unloading preloaded WSI before forking workers...")
                    wsi_reader.unload()

                success_count, skip_count, classified_tiles, pipe_tile_timings = _pipeline_multi_gpu(
                    wsi_reader, all_tiles, args, output_dir,
                    gpu_info, size_thresh, large_noise_thresh,
                )

            all_tile_timings.extend(pipe_tile_timings)

            target_tile_count = success_count + skip_count
            timer.stop()

            # 保存 tile map CSV
            TileClassifier.summarize_tile_map(classified_tiles)
            TileClassifier.save_tile_map(classified_tiles, tile_map_path)

        print(f"\n  Tile processing complete: {success_count} tiles with masks, {skip_count} skipped")

        # ========== 4. 导出 GeoJSON ==========
        if not getattr(args, 'skip_reconstruction', False) and success_count > 0:
            timer.start("GeoJSON export")
            print(f"\n[Step 5] Exporting GeoJSON for QuPath...")
            npy_dir = os.path.join(output_dir, "npy_masks")
            geojson_path = os.path.join(output_dir, f"{wsi_stem}.geojson")

            export_geojson(
                tile_dir=npy_dir,
                output_path=geojson_path,
                tile_size=args.tile_size,
                stride=args.tile_size - args.overlap,
                simplify=getattr(args, 'geojson_simplify', 0),
                contour_tolerance=getattr(args, 'contour_tolerance', 0.5),
                min_area=args.min_mask_area,
            )

            timer.start("GeoJSON statistics")
            # ========== 5.1 GeoJSON 统计 ==========
            print(f"\n[Step 6] Computing GeoJSON statistics...")
            stats = compute_geojson_statistics(geojson_path, output_dir)
            print(f"  Region count:  {stats['count']}")
            print(f"  Area mean:     {stats['area_mean']:.2f} px^2")
            print(f"  Area std:      {stats['area_std']:.2f} px^2")
            timer.stop()
        elif success_count == 0:
            print("\n  No masks produced, skipping reconstruction.")
        else:
            print("\n  Skipping reconstruction (--skip-reconstruction).")

        # ========== 5. 清理 ==========
        wsi_reader.close()

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start_time

    # 输出计时汇总
    print(timer.summary())
    timer.save_csv(os.path.join(output_dir, f"{wsi_stem}_timing.csv"))
    print(f"  Timing CSV: {os.path.join(output_dir, f'{wsi_stem}_timing.csv')}")

    # 输出 per-tile 细粒度计时统计
    if all_tile_timings:
        tile_timing_text = summarize_tile_timings(
            all_tile_timings, output_dir=output_dir, wsi_stem=wsi_stem)
        print(tile_timing_text)

    print(f"\n  Total tiles: {len(all_tiles)}")
    print(f"\n{'='*60}")
    print(f"WSI PIPELINE COMPLETED ({elapsed:.1f}s)")
    print(f"  Tiles processed: {success_count}/{target_tile_count}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main_wsi()
