#!/usr/bin/env python3
"""
compare_score_thresholds.py — SAM2 score_threshold 对比实验

只跑一次完整 pipeline (DeepLIIF + SAM2)，以 score_threshold=0 保存所有
原始预测。然后对同一批原始结果，分别用 0.1 / 0.01 / 0.001 过滤，生成
不同的 NPY mask → GeoJSON，最终对比区域数量、面积分布等指标。

原理:
  score_threshold 只影响 SAM2 predict 之后的过滤，不影响预测本身。
  从 threshold=0 的结果中，零化低分实例等价于用更高 threshold 重跑。

输出目录结构:
  output_dir/
  ├── raw_data/               # Phase 1: SAM2 原始输出 (threshold=0)
  │   ├── tile_R_C_X_Y.npy    # 每个 tile 的实例 mask
  │   └── tile_R_C_X_Y_scores.json
  ├── thresh_0.1/             # Phase 2: 按阈值过滤后
  │   ├── npy_masks/          #   merge_connected_masks 后的 NPY
  │   ├── slide.geojson       #   GeoJSON 导出
  │   └── ...statistics.csv
  ├── thresh_0.01/
  ├── thresh_0.001/
  └── comparison_report.txt   # 对比汇总表

Usage:
    # 全自动模式（YOLO 自动分类 → 处理 target tiles）
    python scripts/analysis/compare_score_thresholds.py \
        --wsi-path /path/to/slide.ndpi \
        --output-dir ./output/thresh_compare

    # 单 tile 调试
    python scripts/analysis/compare_score_thresholds.py \
        --wsi-path /path/to/slide.ndpi \
        --output-dir ./output/thresh_compare \
        --tile-index 5,12

    # 使用已有 tile map + crop 区域
    python scripts/analysis/compare_score_thresholds.py \
        --wsi-path /path/to/slide.ndpi \
        --output-dir ./output/thresh_compare \
        --tile-map /path/to/tile_map.csv \
        --crop-csv /path/to/crop.csv

    # 跳过 Phase 1（已有 raw_data），只重新生成 GeoJSON
    python scripts/analysis/compare_score_thresholds.py \
        --wsi-path /path/to/slide.ndpi \
        --output-dir ./output/thresh_compare \
        --skip-phase1

    # GPU 等待模式：提前配好参数，有空闲 GPU 时自动开跑
    python scripts/analysis/compare_score_thresholds.py \
        --wsi-path /path/to/slide.ndpi \
        --output-dir ./output/thresh_compare \
        --tile-map /path/to/tile_map.csv \
        --wait-for-gpu --check-interval 30 --gpu-min-free-mb 9000

    # 自定义阈值
    python scripts/analysis/compare_score_thresholds.py \
        --wsi-path /path/to/slide.ndpi \
        --output-dir ./output/thresh_compare \
        --tile-index 5,12 \
        --thresholds 0.2 0.1 0.05 0.01 0.001
"""

import gc
import os
import sys
import time
import json
import csv

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch
from cd34_pipeline.config import parse_size_thresh, parse_large_noise_thresh
from cd34_pipeline.sam2_wrapper.model_loader import load_deepliif, load_sam2
from cd34_pipeline.cell.extraction import (
    get_clusters_from_cells,
    extract_connected_positive_regions,
)
from cd34_pipeline.cell.mask_utils import generate_mask_from_cluster
from cd34_pipeline.sam2_wrapper.inference import merge_connected_masks
from cd34_pipeline.io.file_io import (
    save_mask_npy,
    load_mask_npy,
    compute_geojson_statistics,
)
from cd34_pipeline.io.wsi_reader import WSIReader
from cd34_pipeline.io.tile_classifier import TileClassifier
from cd34_pipeline.io.tile_reconstruction import export_geojson
from cd34_pipeline.io.gpu_utils import detect_available_gpus
from cd34_pipeline.wsi_pipeline import load_crop_region, filter_tiles_by_crop_region


# ═══════════════════════════════════════════════════════════════
# Phase 1: 完整 pipeline 跑一次，保存原始 SAM2 结果
# ═══════════════════════════════════════════════════════════════

def run_sam2_save_raw(predictor, image, clusters, min_area=10):
    """
    对每个 cluster 执行 SAM2 predict，**不做 score 过滤**，
    返回带置信度优先合并的 mask 以及每个实例的分数。

    与 run_sam2_segmentation(threshold=0) 行为一致，但额外返回
    全量分数信息用于后续按不同阈值过滤。

    Returns:
        combined_mask: (H, W) uint8, 实例 ID (0=背景)
        all_scores: list of dict, 每项:
            {inst_id, best_score, mask_area, cluster_size, all_3_scores}
    """
    predictor.set_image(image)
    h, w = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.float32)
    all_scores = []

    for idx, cluster in enumerate(clusters):
        if len(cluster) < min_area:
            continue
        try:
            mask_input = generate_mask_from_cluster(cluster, image.shape)
            masks, scores, _ = predictor.predict(
                mask_input=mask_input, multimask_output=True)

            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            best_mask = masks[best_idx].astype(bool)
            mask_area = int(np.sum(best_mask))

            inst_id = idx + 1

            # 置信度优先合并（与 run_sam2_segmentation 逻辑一致）
            overwrite = best_mask & (best_score > score_map)
            combined_mask[overwrite] = inst_id
            score_map[overwrite] = best_score

            all_scores.append({
                'inst_id': inst_id,
                'best_score': best_score,
                'mask_area': mask_area,
                'cluster_size': len(cluster),
                'all_3_scores': [float(s) for s in scores],
            })

            final_area = int(np.sum(combined_mask == inst_id))
            print(f"      Instance {inst_id}: score={best_score:.4f}, "
                  f"area={mask_area} [final: {final_area}]")

        except Exception as e:
            import traceback
            print(f"    SAM2 Error on cluster {idx}: {e}")
            traceback.print_exc()

    return combined_mask, all_scores


def process_tile_phase1(tile_info, wsi_reader, deepliif_engine,
                        sam2_predictor, args, raw_data_dir):
    """
    Phase 1: 处理单个 tile 的完整 pipeline (DeepLIIF → Cell → SAM2)，
    以 score_threshold=0 保存原始 mask + scores。

    Returns:
        (success: bool, tile_name: str)
    """
    tile_name = wsi_reader.get_tile_filename(tile_info)
    row, col = tile_info['row'], tile_info['col']

    # 1. 读取 tile
    t0 = time.time()
    tile_pil = wsi_reader.read_tile(tile_info)
    tile_np = np.array(tile_pil)
    t_read = time.time() - t0

    # 2. DeepLIIF 推理
    t0 = time.time()
    size_thresh = parse_size_thresh(args.size_thresh)
    large_noise_thresh = parse_large_noise_thresh(args.large_noise_thresh)
    deepliif_results = deepliif_engine.inference(
        tile_pil,
        tile_size=args.tile_size,
        seg_weights=args.seg_weights,
        resolution=args.resolution,
        do_postprocessing=args.enable_postprocessing,
        seg_thresh=args.seg_thresh,
        size_thresh=size_thresh,
        marker_thresh=args.marker_thresh,
        size_thresh_upper=args.size_thresh_upper,
        noise_thresh=args.noise_thresh,
        large_noise_thresh=large_noise_thresh,
        color_dapi=getattr(args, 'color_dapi', False),
        color_marker=getattr(args, 'color_marker', False),
    )
    t_deepliif = time.time() - t0

    seg_img = deepliif_results.get('Seg')
    marker_img = deepliif_results.get('Marker')
    if seg_img is None or marker_img is None:
        print(f"      No Seg/Marker output for tile ({row},{col}), skip.")
        return False, tile_name

    # 3. 细胞提取
    t0 = time.time()
    seg_np = np.array(seg_img)
    marker_np = np.array(marker_img)

    positive_cells_info = extract_connected_positive_regions(
        seg_np, marker_np,
        seg_thresh=args.seg_thresh,
        marker_thresh=args.marker_thresh,
        morphology_kernel=args.morphology_kernel,
        min_area=args.min_mask_area,
    )
    clusters = get_clusters_from_cells(positive_cells_info)
    t_cell = time.time() - t0

    if len(clusters) == 0:
        print(f"      Tile ({row},{col}): 0 clusters, skip.")
        return False, tile_name

    # 4. SAM2 推理 (threshold=0, 全部保留)
    t0 = time.time()
    raw_mask, all_scores = run_sam2_save_raw(
        sam2_predictor, tile_np, clusters,
        min_area=args.min_mask_area)
    t_sam2 = time.time() - t0

    if np.max(raw_mask) == 0:
        print(f"      Tile ({row},{col}): 0 SAM2 instances, skip.")
        return False, tile_name

    # 5. 保存原始结果
    npy_path = os.path.join(raw_data_dir, f"{tile_name}.npy")
    save_mask_npy(raw_mask, npy_path)

    scores_path = os.path.join(raw_data_dir, f"{tile_name}_scores.json")
    with open(scores_path, 'w') as f:
        json.dump(all_scores, f, indent=2)

    print(f"    Tile ({row},{col}): {len(all_scores)} instances saved  "
          f"[read={t_read:.2f}s deepliif={t_deepliif:.2f}s "
          f"cell={t_cell:.2f}s sam2={t_sam2:.2f}s]")

    # 释放内存
    del tile_pil, tile_np, deepliif_results, seg_np, marker_np, raw_mask
    gc.collect()

    return True, tile_name


# ═══════════════════════════════════════════════════════════════
# Phase 2: 对原始结果按不同阈值过滤 → NPY mask
# ═══════════════════════════════════════════════════════════════

def filter_raw_mask_by_threshold(raw_mask, scores_info, threshold,
                                 merge_min_area=200):
    """
    从 Phase 1 保存的原始 mask + scores 中，按 threshold 过滤实例，
    然后跑 merge_connected_masks 合并连通区域。

    Args:
        raw_mask: (H, W) 原始实例 mask (threshold=0)
        scores_info: list of dict from _scores.json
        threshold: score_threshold
        merge_min_area: merge_connected_masks 的 min_area

    Returns:
        merged_mask, kept_count, filtered_count, merged_instance_count
    """
    # 1. 过滤: 低于阈值的实例从 mask 中移除
    filtered_mask = raw_mask.copy()
    scores_list = []
    filtered_list = []

    for s in scores_info:
        inst_id = s['inst_id']
        score = s['best_score']
        if threshold > 0 and score < threshold:
            # 清除该实例的所有像素
            filtered_mask[filtered_mask == inst_id] = 0
            filtered_list.append((inst_id, score))
        else:
            scores_list.append((inst_id, score))

    # 2. merge_connected_masks (合并连通区域 + 面积过滤)
    merged_mask, _, _, _ = merge_connected_masks(
        filtered_mask, scores_list,
        positive_cells_info=[],
        min_area=merge_min_area,
    )

    merged_count = int(np.max(merged_mask))
    return merged_mask, len(scores_list), len(filtered_list), merged_count


def postprocess_all_tiles_for_threshold(raw_data_dir, npy_output_dir,
                                        threshold, merge_min_area=200):
    """
    Phase 2: 遍历 raw_data 目录中的所有 tile，按指定阈值过滤并保存
    merge 后的 NPY mask 到 npy_output_dir。

    Returns:
        dict: 汇总统计
    """
    os.makedirs(npy_output_dir, exist_ok=True)

    # 扫描所有 raw mask 文件
    npy_files = sorted([f for f in os.listdir(raw_data_dir)
                        if f.endswith('.npy') and not f.endswith('_scores.npy')])

    total_kept = 0
    total_filtered = 0
    total_merged = 0
    tiles_with_mask = 0

    for npy_file in npy_files:
        npy_path = os.path.join(raw_data_dir, npy_file)
        scores_path = os.path.join(
            raw_data_dir, npy_file.replace('.npy', '_scores.json'))

        raw_mask, _ = load_mask_npy(npy_path)

        if not os.path.exists(scores_path):
            print(f"    WARNING: scores file not found for {npy_file}, skip.")
            continue

        with open(scores_path, 'r') as f:
            scores_info = json.load(f)

        merged_mask, kept, filtered, merged_count = filter_raw_mask_by_threshold(
            raw_mask, scores_info, threshold, merge_min_area)

        total_kept += kept
        total_filtered += filtered
        total_merged += merged_count

        if merged_count > 0:
            tiles_with_mask += 1
            out_path = os.path.join(npy_output_dir, npy_file)
            save_mask_npy(merged_mask, out_path)

    return {
        'threshold': threshold,
        'tiles_processed': len(npy_files),
        'tiles_with_mask': tiles_with_mask,
        'total_kept': total_kept,
        'total_filtered': total_filtered,
        'total_merged_instances': total_merged,
    }


# ═══════════════════════════════════════════════════════════════
# Phase 3: GeoJSON 导出 + 对比报告
# ═══════════════════════════════════════════════════════════════

def generate_comparison_report(all_stats, all_geojson_stats, output_path):
    """生成各阈值的对比报告。"""
    lines = []
    lines.append("=" * 80)
    lines.append("SCORE THRESHOLD COMPARISON REPORT")
    lines.append("=" * 80)

    # Phase 2 汇总表: SAM2 实例级
    lines.append(f"\n{'Threshold':<12} {'Tiles':>6} {'Kept':>8} {'Filtered':>10} "
                 f"{'Merged':>8}")
    lines.append("-" * 55)
    for s in all_stats:
        lines.append(f"{s['threshold']:<12.4f} {s['tiles_with_mask']:>6} "
                     f"{s['total_kept']:>8} {s['total_filtered']:>10} "
                     f"{s['total_merged_instances']:>8}")

    # Phase 3 汇总表: GeoJSON 级 (含跨 tile 合并)
    if all_geojson_stats:
        lines.append(f"\n{'Threshold':<12} {'GeoJSON Regions':>16} "
                     f"{'Area Mean':>12} {'Area Std':>12}")
        lines.append("-" * 55)
        for thresh, gs in all_geojson_stats.items():
            lines.append(f"{thresh:<12.4f} {gs['count']:>16} "
                         f"{gs['area_mean']:>12.2f} {gs['area_std']:>12.2f}")

    lines.append("\n" + "=" * 80)

    # 注解
    lines.append("\nNote:")
    lines.append("  'Kept'     = SAM2 instances with score >= threshold (pre-merge)")
    lines.append("  'Filtered' = SAM2 instances with score <  threshold (removed)")
    lines.append("  'Merged'   = instances after merge_connected_masks (tile-level)")
    lines.append("  'GeoJSON Regions' = final regions after cross-tile merge (slide-level)")
    lines.append("")

    report = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(report)

    return report


# ═══════════════════════════════════════════════════════════════
# 命令行参数
# ═══════════════════════════════════════════════════════════════

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="SAM2 score_threshold 对比实验: "
                    "跑一次 pipeline, 对比多个阈值的 GeoJSON 结果")

    # 必需参数
    parser.add_argument('--wsi-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)

    # Tile 选择 (三选一)
    tile_group = parser.add_mutually_exclusive_group()
    tile_group.add_argument('--tile-index', type=str, default=None,
        help='ROW,COL 格式，多个用分号分隔: "5,12;6,13"')
    tile_group.add_argument('--tile-map', type=str, default=None,
        help='已有 tile map CSV (跳过 YOLO)')

    # 实验参数
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[0.1, 0.01, 0.001],
                        help='要对比的 score_threshold 列表 (default: 0.1 0.01 0.001)')
    parser.add_argument('--skip-phase1', action='store_true',
        help='跳过 Phase 1 (已有 raw_data/)，只重新跑 Phase 2+3')
    parser.add_argument('--merge-min-area', type=int, default=200,
        help='merge_connected_masks 的 min_area (default: 200)')

    # Crop 区域过滤
    parser.add_argument('--crop-csv', type=str, default=None,
        help='裁剪区域 CSV 文件路径 (列: filename,x,y,width,height, level 0 坐标)。'
             '仅处理与 WSI 匹配且在裁剪区域内的 tile。')

    # GPU 等待模式
    parser.add_argument('--wait-for-gpu', action='store_true',
        help='启用 GPU 等待模式: 每隔 --check-interval 分钟扫描 GPU，'
             '有空闲资源时自动开始任务')
    parser.add_argument('--check-interval', type=int, default=30,
        help='GPU 扫描间隔 (分钟, default: 30)')
    parser.add_argument('--gpu-min-free-mb', type=int, default=9000,
        help='最小空闲显存要求 (MB, default: 9000)')

    # YOLO 分类参数
    parser.add_argument('--yolo-model-path', type=str,
        default='./data/models/yolo/yolo11n_cls_cd34_bg_target_20601.pt',
        help='Path to YOLO classification model for tile background/target filtering')
    parser.add_argument('--yolo-batch-size', type=int, default=64,
        help='Batch size for YOLO tile classification (default: 64)')
    parser.add_argument('--yolo-prefetch-workers', type=int, default=4,
        help='Number of threads for prefetching WSI tiles during YOLO classification (default: 4)')

    # 模型参数
    parser.add_argument('--deepliif-model-dir', type=str,
                        default='./data/models/deepliif/')
    parser.add_argument('--sam-checkpoint', type=str,
                        default='./data/models/sam2/sam2.1_hiera_large.pt')
    parser.add_argument('--sam-config', type=str,
                        default='configs/sam2.1/sam2.1_hiera_l.yaml')
    parser.add_argument('--tile-size', type=int, default=512)
    parser.add_argument('--resolution', type=str, default='40x')
    parser.add_argument('--target-magnification', type=float, default=40.0)
    parser.add_argument('--overlap', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')

    # DeepLIIF 参数
    parser.add_argument('--seg-weights', type=float, nargs=5, default=None)
    parser.add_argument('--seg-thresh', type=int, default=120)
    parser.add_argument('--size-thresh', type=str, default='default')
    parser.add_argument('--size-thresh-upper', type=int, default=None)
    parser.add_argument('--marker-thresh', type=int, default=None)
    parser.add_argument('--noise-thresh', type=int, default=4)
    parser.add_argument('--large-noise-thresh', type=str, default='default')
    parser.add_argument('--enable-postprocessing', action='store_true')
    parser.add_argument('--color-dapi', action='store_true')
    parser.add_argument('--color-marker', action='store_true')

    # SAM2 / 细胞提取
    parser.add_argument('--min-mask-area', type=int, default=50)
    parser.add_argument('--morphology-kernel', type=int, default=11)

    # GeoJSON 导出参数
    parser.add_argument('--geojson-simplify', type=float, default=0)
    parser.add_argument('--contour-tolerance', type=float, default=0.5)

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    thresholds = sorted(args.thresholds, reverse=True)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    wsi_stem = os.path.splitext(os.path.basename(args.wsi_path))[0]
    raw_data_dir = os.path.join(output_dir, "raw_data")
    stride = args.tile_size - args.overlap

    print(f"\n{'='*60}")
    print("SCORE THRESHOLD COMPARISON EXPERIMENT")
    print(f"  WSI:        {args.wsi_path}")
    print(f"  Thresholds: {thresholds}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}")

    # ── 打开 WSI, 枚举 tile, 可选 crop 过滤 ──
    wsi_reader = WSIReader(
        args.wsi_path,
        tile_size=args.tile_size,
        target_magnification=args.target_magnification,
        overlap=args.overlap,
    )
    all_tiles = wsi_reader.enumerate_tiles()
    print(f"  Total tiles: {len(all_tiles)}")

    # --crop-csv: 裁剪区域过滤
    if args.crop_csv:
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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Phase 1: 跑一次完整 pipeline, 保存原始 SAM2 结果
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    if not args.skip_phase1:
        print(f"\n{'='*60}")
        print("[Phase 1] Running pipeline with score_threshold=0 (keep all)")
        print(f"{'='*60}")

        os.makedirs(raw_data_dir, exist_ok=True)

        torch.autograd.set_grad_enabled(False)

        # 确定要处理的 tiles（YOLO 分类在此执行，如果需要）
        target_tiles = _resolve_target_tiles(
            args, all_tiles, wsi_reader=wsi_reader, output_dir=output_dir)

        if not target_tiles:
            print("  No target tiles to process. Exiting.")
            wsi_reader.close()
            return

        # 加载模型（含 GPU 等待 + OOM 重试）
        deepliif_engine, sam2_predictor, _ = _load_models_with_retry(
            args, wait_for_gpu=args.wait_for_gpu)

        print(f"\n  Processing {len(target_tiles)} tiles...")

        t0_phase1 = time.time()
        success_count = 0

        for idx, tile_info in enumerate(target_tiles):
            row, col = tile_info['row'], tile_info['col']
            print(f"\n  --- Tile [{idx+1}/{len(target_tiles)}] ({row},{col}) ---")

            success, _ = process_tile_phase1(
                tile_info, wsi_reader, deepliif_engine, sam2_predictor,
                args, raw_data_dir)
            if success:
                success_count += 1

        t_phase1 = time.time() - t0_phase1
        print(f"\n  Phase 1 complete: {success_count}/{len(target_tiles)} tiles "
              f"saved to raw_data/ ({t_phase1:.1f}s)")

        # 释放 GPU 模型
        del deepliif_engine, sam2_predictor
        torch.cuda.empty_cache()
        gc.collect()

    else:
        print(f"\n  [Phase 1] Skipped (--skip-phase1). Using existing raw_data/")
        if not os.path.isdir(raw_data_dir):
            print(f"  ERROR: {raw_data_dir} does not exist!")
            wsi_reader.close()
            return

    # 检查 raw_data 是否有数据
    raw_npy_count = len([f for f in os.listdir(raw_data_dir)
                         if f.endswith('.npy')])
    if raw_npy_count == 0:
        print("  No raw masks found in raw_data/. Nothing to compare.")
        wsi_reader.close()
        return

    print(f"  raw_data/ contains {raw_npy_count} tile masks")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Phase 2: 对每个阈值过滤 → merge → 保存 NPY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print(f"\n{'='*60}")
    print("[Phase 2] Filtering raw masks at each threshold")
    print(f"{'='*60}")

    all_stats = []

    for thresh in thresholds:
        thresh_str = _thresh_to_str(thresh)
        thresh_dir = os.path.join(output_dir, f"thresh_{thresh_str}")
        npy_dir = os.path.join(thresh_dir, "npy_masks")

        print(f"\n  --- threshold = {thresh} ---")
        t0 = time.time()

        stats = postprocess_all_tiles_for_threshold(
            raw_data_dir, npy_dir, thresh, args.merge_min_area)
        stats['time_seconds'] = time.time() - t0

        all_stats.append(stats)
        print(f"    kept={stats['total_kept']}, filtered={stats['total_filtered']}, "
              f"merged={stats['total_merged_instances']} ({stats['time_seconds']:.1f}s)")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Phase 3: 每个阈值导出 GeoJSON + 统计
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print(f"\n{'='*60}")
    print("[Phase 3] Exporting GeoJSON for each threshold")
    print(f"{'='*60}")

    all_geojson_stats = {}

    for thresh in thresholds:
        thresh_str = _thresh_to_str(thresh)
        thresh_dir = os.path.join(output_dir, f"thresh_{thresh_str}")
        npy_dir = os.path.join(thresh_dir, "npy_masks")
        geojson_path = os.path.join(thresh_dir, f"{wsi_stem}.geojson")

        # 检查 npy_dir 是否有文件
        if not os.path.isdir(npy_dir) or not os.listdir(npy_dir):
            print(f"\n  threshold={thresh}: no NPY masks, skip GeoJSON export.")
            all_geojson_stats[thresh] = {
                'count': 0, 'area_mean': 0, 'area_std': 0}
            continue

        print(f"\n  --- threshold = {thresh} → GeoJSON ---")
        export_geojson(
            tile_dir=npy_dir,
            output_path=geojson_path,
            tile_size=args.tile_size,
            stride=stride,
            simplify=args.geojson_simplify,
            contour_tolerance=args.contour_tolerance,
            min_area=args.min_mask_area,
        )

        # 统计
        gs = compute_geojson_statistics(geojson_path, thresh_dir)
        all_geojson_stats[thresh] = gs
        print(f"    Regions: {gs['count']}, "
              f"Area mean: {gs['area_mean']:.2f}, std: {gs['area_std']:.2f}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 对比报告
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    report_path = os.path.join(output_dir, "comparison_report.txt")
    report = generate_comparison_report(all_stats, all_geojson_stats, report_path)
    print(report)

    # 保存对比 CSV
    csv_path = os.path.join(output_dir, "comparison_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['threshold', 'tiles_with_mask',
                         'sam2_kept', 'sam2_filtered', 'tile_merged_instances',
                         'geojson_regions', 'area_mean', 'area_std',
                         'phase2_seconds'])
        for stats in all_stats:
            thresh = stats['threshold']
            gs = all_geojson_stats.get(thresh, {})
            writer.writerow([
                thresh,
                stats['tiles_with_mask'],
                stats['total_kept'],
                stats['total_filtered'],
                stats['total_merged_instances'],
                gs.get('count', 0),
                f"{gs.get('area_mean', 0):.2f}",
                f"{gs.get('area_std', 0):.2f}",
                f"{stats.get('time_seconds', 0):.2f}",
            ])
    print(f"  Comparison CSV: {csv_path}")

    # 保存原始分数分布
    _save_score_distribution(raw_data_dir, thresholds, output_dir)

    wsi_reader.close()

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"  Output:    {output_dir}")
    for thresh in thresholds:
        ts = _thresh_to_str(thresh)
        print(f"  thresh={thresh}: {output_dir}/thresh_{ts}/")
    print(f"  Report:    {report_path}")
    print(f"  CSV:       {csv_path}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _thresh_to_str(thresh):
    """将阈值转为文件名安全字符串: 0.01 → '0.01', 0.001 → '0.001'"""
    return f"{thresh:g}"


def _wait_for_gpu(min_free_mb, check_interval_min):
    """
    循环扫描 GPU 资源，直到有满足要求的 GPU 空闲。

    Args:
        min_free_mb: 最小空闲显存 (MB)
        check_interval_min: 扫描间隔 (分钟)

    Returns:
        gpu_info: detect_available_gpus 返回的 GPU 列表
    """
    from datetime import datetime

    attempt = 0
    while True:
        attempt += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n  [{now}] GPU scan #{attempt} "
              f"(need >= {min_free_mb} MB free)...")

        try:
            gpu_info = detect_available_gpus(
                min_free_mb=min_free_mb, max_gpus=1, max_workers_per_gpu=1)
        except Exception as e:
            print(f"  GPU query error: {e}")
            gpu_info = []

        if gpu_info:
            gpu = gpu_info[0]
            print(f"  GPU {gpu['gpu_id']} available! "
                  f"Free: {gpu['free_mb']} MB")
            return gpu_info

        print(f"  No GPU available. "
              f"Retrying in {check_interval_min} minutes...")
        time.sleep(check_interval_min * 60)


def _load_models_with_retry(args, wait_for_gpu=False):
    """
    加载 DeepLIIF + SAM2 模型，加载失败时自动清理并重试等待。

    GPU 扫描时报告有空闲显存，但实际加载模型时可能已被其他进程占走，
    导致 CUDA OOM。此函数在 OOM 时释放已加载的部分模型，等待后重试。

    Returns:
        (deepliif_engine, sam2_predictor, device_str)
    """
    from datetime import datetime

    while True:
        # 1. 获取 GPU
        if wait_for_gpu:
            gpu_info = _wait_for_gpu(args.gpu_min_free_mb, args.check_interval)
        else:
            gpu_info = detect_available_gpus(
                min_free_mb=args.gpu_min_free_mb, max_gpus=1,
                max_workers_per_gpu=1)

        device = f'cuda:{gpu_info[0]["gpu_id"]}' if gpu_info else args.device

        # 2. 尝试加载模型
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n  [{now}] Loading DeepLIIF + SAM2 on {device}...")

        deepliif_engine = None
        sam2_predictor = None

        try:
            deepliif_engine = load_deepliif(args.deepliif_model_dir, device)
            sam2_predictor = load_sam2(args.sam_config, args.sam_checkpoint, device)
            print(f"  Models loaded successfully on {device}")
            return deepliif_engine, sam2_predictor, device

        except Exception as e:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n  [{now}] Model loading failed: {e}")

            # 清理已加载的部分模型
            del deepliif_engine, sam2_predictor
            torch.cuda.empty_cache()
            gc.collect()

            if not wait_for_gpu:
                # 非等待模式直接报错退出
                raise

            # 等待模式: 等一轮后重试
            print(f"  Will retry in {args.check_interval} minutes...")
            time.sleep(args.check_interval * 60)


def _resolve_target_tiles(args, all_tiles, wsi_reader=None, output_dir=None):
    """根据 --tile-index / --tile-map / YOLO 自动分类 确定要处理的 tiles。"""

    if args.tile_index is not None:
        # 单个/多个 tile
        target_tiles = []
        for part in args.tile_index.split(';'):
            r, c = part.strip().split(',')
            target_row, target_col = int(r), int(c)
            for t in all_tiles:
                if t['row'] == target_row and t['col'] == target_col:
                    target_tiles.append(t)
                    break
            else:
                print(f"  WARNING: Tile ({target_row},{target_col}) not found.")
        return target_tiles

    elif args.tile_map is not None:
        # 从已有 tile map 加载 target tiles
        classified_tiles = TileClassifier.load_tile_map(args.tile_map)
        TileClassifier.summarize_tile_map(classified_tiles)
        target_list = TileClassifier.get_target_tiles(classified_tiles)

        # 补全 tile_info 字段
        tiles_by_pos = {(t['row'], t['col']): t for t in all_tiles}
        target_tiles = []
        for t in target_list:
            key = (t['row'], t['col'])
            if key in tiles_by_pos:
                full_tile = tiles_by_pos[key].copy()
                full_tile['classification'] = t['classification']
                target_tiles.append(full_tile)
        return target_tiles

    else:
        # YOLO 自动分类（与主 pipeline 一致）
        print("\n  [YOLO] Running tile classification...")

        # 检测空闲 GPU 用于 YOLO
        gpu_info = detect_available_gpus(
            min_free_mb=1500, max_gpus=1, max_workers_per_gpu=1)
        yolo_device = f'cuda:{gpu_info[0]["gpu_id"]}' if gpu_info else args.device

        classifier = TileClassifier(
            model_path=args.yolo_model_path,
            device=yolo_device,
            batch_size=args.yolo_batch_size,
            imgsz=args.tile_size,
        )
        classified_tiles = classifier.classify_tiles_from_wsi(
            wsi_reader, all_tiles,
            num_workers=args.yolo_prefetch_workers,
        )
        TileClassifier.summarize_tile_map(classified_tiles)

        # 保存 tile map 供后续复用
        if output_dir:
            wsi_stem = os.path.splitext(
                os.path.basename(args.wsi_path))[0]
            tile_map_path = os.path.join(
                output_dir, f"{wsi_stem}_tile_map.csv")
            TileClassifier.save_tile_map(classified_tiles, tile_map_path)

        # 释放 YOLO 模型
        del classifier
        torch.cuda.empty_cache()
        gc.collect()

        target_tiles = TileClassifier.get_target_tiles(classified_tiles)
        return target_tiles


def _save_score_distribution(raw_data_dir, thresholds, output_dir):
    """汇总所有 tile 的 score，保存分布信息。"""
    all_scores = []

    score_files = sorted([f for f in os.listdir(raw_data_dir)
                          if f.endswith('_scores.json')])
    for sf in score_files:
        with open(os.path.join(raw_data_dir, sf), 'r') as f:
            scores_info = json.load(f)
        for s in scores_info:
            all_scores.append(s['best_score'])

    if not all_scores:
        return

    all_scores.sort()

    # 保存分布 JSON
    dist = {
        'total_instances': len(all_scores),
        'score_min': float(min(all_scores)),
        'score_max': float(max(all_scores)),
        'score_mean': float(np.mean(all_scores)),
        'score_std': float(np.std(all_scores)),
        'score_median': float(np.median(all_scores)),
        'percentiles': {
            'p5': float(np.percentile(all_scores, 5)),
            'p25': float(np.percentile(all_scores, 25)),
            'p50': float(np.percentile(all_scores, 50)),
            'p75': float(np.percentile(all_scores, 75)),
            'p95': float(np.percentile(all_scores, 95)),
        },
        'per_threshold': {},
    }

    for thresh in sorted(thresholds, reverse=True):
        kept = sum(1 for s in all_scores if s >= thresh)
        filtered = len(all_scores) - kept
        dist['per_threshold'][str(thresh)] = {
            'kept': kept,
            'filtered': filtered,
            'kept_pct': f"{kept/len(all_scores)*100:.1f}%",
        }

    dist_path = os.path.join(output_dir, "score_distribution.json")
    with open(dist_path, 'w') as f:
        json.dump(dist, f, indent=2)

    # 打印分布摘要
    print(f"\n  Score Distribution ({len(all_scores)} total instances):")
    print(f"    Range:  [{dist['score_min']:.6f}, {dist['score_max']:.6f}]")
    print(f"    Mean:   {dist['score_mean']:.6f}")
    print(f"    Median: {dist['score_median']:.6f}")
    for t_str, info in dist['per_threshold'].items():
        print(f"    >= {t_str}: {info['kept']} ({info['kept_pct']})")


if __name__ == "__main__":
    main()
