#!/usr/bin/env python3
"""
config.py — 配置与参数解析模块

包含：
- 命令行参数解析（WSI Pipeline 专用）
- 配置验证
- 参数预处理
"""

import argparse
import os
import sys
import torch


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数（WSI Pipeline）。

    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="CD34 微血管检测 WSI Pipeline (DeepLIIF + SAM2)")

    # DeepLIIF Arguments - Core
    group_deepliif = parser.add_argument_group('DeepLIIF Core Parameters')
    group_deepliif.add_argument('--deepliif-model-dir', type=str, default='./data/models/deepliif/',
        help='Path to DeepLIIF models directory (containing G1.pt, G2.pt...)')
    group_deepliif.add_argument('--tile-size', type=int, default=512,
        help='Tile size for DeepLIIF inference (default: 512)')
    group_deepliif.add_argument('--resolution', type=str, default='40x', choices=['10x', '20x', '40x'],
        help='Microscope resolution, affects cell size thresholds (default: 40x)')
    group_deepliif.add_argument('--seg-weights', type=float, nargs=5, default=None, metavar='W',
        help='Segmentation aggregation weights for G51 G52 G53 G54 G55 (default: equal weights)')

    # DeepLIIF Arguments - Post-processing
    group_deepliif_post = parser.add_argument_group('DeepLIIF Post-processing Parameters')
    group_deepliif_post.add_argument('--seg-thresh', type=int, default=120,
        help='Segmentation threshold for foreground detection (default: 120)')
    group_deepliif_post.add_argument('--size-thresh', type=str, default='default',
        help='Minimum cell size threshold. Use "default" for auto-calculation or an integer value')
    group_deepliif_post.add_argument('--size-thresh-upper', type=int, default=None,
        help='Maximum cell size threshold to filter out large objects (default: None)')
    group_deepliif_post.add_argument('--marker-thresh', type=int, default=None,
        help='Marker intensity threshold for positive/negative classification (default: auto)')
    group_deepliif_post.add_argument('--noise-thresh', type=int, default=4,
        help='Noise threshold for filtering small debris (default: 4)')
    group_deepliif_post.add_argument('--large-noise-thresh', type=str, default='default',
        help='Large noise threshold to filter out very large objects. Use "default" (auto by resolution), "none" (no upper limit), or an integer value')
    group_deepliif_post.add_argument('--enable-postprocessing', action='store_true',
        help='Enable DeepLIIF post-processing to generate SegRefined/SegOverlaid [default: OFF]')
    group_deepliif_post.add_argument('--color-dapi', action='store_true',
        help='Apply cyan/blue pseudo-coloring to DAPI output')
    group_deepliif_post.add_argument('--color-marker', action='store_true',
        help='Apply yellow/brown pseudo-coloring to Marker output')

    # SAM2 Arguments
    group_sam = parser.add_argument_group('SAM2 Parameters')
    group_sam.add_argument('--sam-checkpoint', type=str,
        default="./data/models/sam2/sam2.1_hiera_large.pt",
        help='Path to SAM2 checkpoint')
    group_sam.add_argument('--sam-config', type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help='Path to SAM2 config file')
    group_sam.add_argument('--min-mask-area', type=int, default=50,
        help='Minimum pixel area for a Connected Component to be considered a cell (default: 50)')

    # General Arguments
    parser.add_argument('--wsi-path', type=str, required=True,
        help='Path to a WSI file (.ndpi, .svs, etc.)')
    parser.add_argument('--output-dir', type=str, required=True,
        help='Directory to save all results')
    parser.add_argument('--device', type=str, default='cuda',
        help='Device to use (cuda or cpu)')

    # Output Control Arguments
    group_output = parser.add_argument_group('Output Control Options',
        description='Options to enable saving additional outputs. All OFF by default to reduce disk usage.')
    group_output.add_argument('--save-csv', action='store_true',
        help='Save CSV files with positive cells information [default: OFF]')
    group_output.add_argument('--save-npy', action='store_true',
        help='Save instance segmentation masks as .npy files [default: OFF]')
    group_output.add_argument('--save-tile-vis', action='store_true',
        help='Save per-tile SAM2 mask overlay PNG for visual inspection [default: OFF]')
    group_output.add_argument('--save-all-deepliif', action='store_true',
        help='Save all DeepLIIF outputs (DAPI, Hema, etc.) in debug_vis; default only saves Seg and Marker [default: OFF]')

    # 连通区域提取参数
    group_region_mode = parser.add_argument_group('Connected Region Extraction Parameters')
    group_region_mode.add_argument('--morphology-kernel', type=int, default=11,
        help='Morphology kernel size for connecting nearby positive pixels. (default: 11)')

    # WSI 模式参数
    group_wsi = parser.add_argument_group('WSI Processing Options',
        description='Options for whole slide image processing, YOLO filtering, and debugging.')
    group_wsi.add_argument('--target-magnification', type=float, default=40.0,
        help='Target magnification level for reading WSI tiles (default: 40.0)')
    group_wsi.add_argument('--overlap', type=int, default=128,
        help='Tile overlap in pixels (default: 128). stride = tile_size - overlap. '
             'Overlap regions are merged by union to ensure cross-tile masks are intact.')
    group_wsi.add_argument('--crop-csv', type=str, default=None,
        help='Path to a crop coordinates CSV file (columns: filename,x,y,width,height in level 0 coords). '
             'Only tiles within the crop region matching the WSI filename will be processed.')
    group_wsi.add_argument('--yolo-model-path', type=str,
        default='./data/models/yolo/yolo11n_cls_cd34_bg_target_20601.pt',
        help='Path to YOLO classification model for tile background/target filtering')
    group_wsi.add_argument('--tile-map', type=str, default=None,
        help='Path to an existing tile map CSV file. If provided, skips YOLO classification.')
    group_wsi.add_argument('--yolo-batch-size', type=int, default=64,
        help='Batch size for YOLO tile classification (default: 64)')
    group_wsi.add_argument('--yolo-prefetch-workers', type=int, default=4,
        help='Number of threads for prefetching WSI tiles during YOLO classification. (default: 4)')
    group_wsi.add_argument('--preload-wsi', action='store_true',
        help='Preload entire WSI level into memory before processing. (default: OFF)')
    group_wsi.add_argument('--preload-max-gb', type=float, default=100.0,
        help='Maximum memory (GB) allowed for WSI preload. (default: 100.0)')
    group_wsi.add_argument('--classify-only', action='store_true',
        help='Only run YOLO classification to generate tile map CSV, skip pipeline processing.')
    group_wsi.add_argument('--skip-reconstruction', action='store_true',
        help='Skip final GeoJSON export step.')
    group_wsi.add_argument('--geojson-simplify', type=float, default=0,
        help='Contour simplification ratio for GeoJSON export (0=none, default: 0)')
    group_wsi.add_argument('--tile-index', type=str, default=None,
        help='Process a single tile by ROW,COL index (e.g. "5,12"). '
             'Skips YOLO classification and reconstruction. Useful for debugging.')
    group_wsi.add_argument('--debug-vis', action='store_true',
        help='Save step-by-step visualization for each tile (only useful with --tile-index).')

    # 多 GPU 并行参数
    group_multigpu = parser.add_argument_group('Multi-GPU Parallel Processing',
        description='Use multiple GPUs to process tiles in parallel.')
    group_multigpu.add_argument('--num-gpus', type=int, default=0,
        help='Maximum number of GPUs to use. 0 = auto-detect all available. (default: 0)')
    group_multigpu.add_argument('--workers-per-gpu', type=int, default=1,
        help='Maximum workers (processes) per GPU. Each loads its own models (~8GB VRAM). (default: 1)')
    group_multigpu.add_argument('--gpu-min-free-mb', type=int, default=9000,
        help='Minimum free VRAM (MB) required to use a GPU. (default: 9000)')

    return parser.parse_args()


def validate_config(args: argparse.Namespace) -> argparse.Namespace:
    """
    验证并修正配置参数。
    """
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, switching to CPU.")
        args.device = 'cpu'

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Validate WSI file
    if not os.path.exists(args.wsi_path):
        print(f"Error: WSI file {args.wsi_path} does not exist.")
        sys.exit(1)

    return args


def parse_size_thresh(size_thresh_str: str) -> str | int:
    """解析 size_thresh 参数。"""
    if size_thresh_str == 'default':
        return 'default'
    try:
        return int(size_thresh_str)
    except ValueError:
        print(f"Warning: Invalid size-thresh '{size_thresh_str}', using 'default'")
        return 'default'


def parse_large_noise_thresh(large_noise_thresh_str: str) -> str | int | None:
    """解析 large_noise_thresh 参数。"""
    value = large_noise_thresh_str.lower()
    if value == 'none':
        return None
    if value == 'default':
        return 'default'
    try:
        return int(large_noise_thresh_str)
    except ValueError:
        print(f"Warning: Invalid large-noise-thresh '{large_noise_thresh_str}', using 'default'")
        return 'default'


def print_pipeline_header(args: argparse.Namespace):
    """打印流水线启动信息。"""
    print(f"\n{'='*60}")
    print(f"WSI PIPELINE STARTED")
    print(f"WSI: {args.wsi_path}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")


def print_pipeline_footer():
    """打印流水线完成信息。"""
    print(f"\n{'='*60}")
    print("Pipeline Completed Successfully.")
    print(f"{'='*60}")
