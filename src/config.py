#!/usr/bin/env python3
"""
config.py — 配置与参数解析模块

包含：
- 命令行参数解析
- 配置验证
- 参数预处理
"""

import argparse
import os
import sys
import torch


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数。
    
    保持与原 pipeline_full_inference.py 100% 兼容的命令行接口。
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="DeepLIIF + SAM2 Full Inference Pipeline")
    
    # DeepLIIF Arguments - Core
    group_deepliif = parser.add_argument_group('DeepLIIF Core Parameters')
    group_deepliif.add_argument('--deepliif-model-dir', type=str, default='./deepliif_models/', 
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
    group_deepliif_post.add_argument('--save-deepliif-outputs', action='store_true',
        help='Save DeepLIIF intermediate outputs (DAPI, Hema, Marker, Seg, etc.) [default: OFF]')
    
    # SAM2 Arguments
    group_sam = parser.add_argument_group('SAM2 Parameters')
    group_sam.add_argument('--sam-checkpoint', type=str, 
        default="/local1/yangwenjie/sam2/checkpoints/sam2.1_hiera_large.pt", 
        help='Path to SAM2 checkpoint')
    group_sam.add_argument('--sam-config', type=str, 
        default="configs/sam2.1/sam2.1_hiera_l.yaml", 
        help='Path to SAM2 config file')
    group_sam.add_argument('--min-mask-area', type=int, default=50, 
        help='Minimum pixel area for a Connected Component to be considered a cell (default: 50)')
    
    # General Arguments
    parser.add_argument('--input-dir', type=str, required=True, 
        help='Directory containing raw input images')
    parser.add_argument('--output-dir', type=str, required=True, 
        help='Directory to save all results')
    parser.add_argument('--device', type=str, default='cuda', 
        help='Device to use (cuda or cpu)')
    
    # LabelMe Export Arguments
    group_labelme = parser.add_argument_group('LabelMe Export Options', 
        description='Options for exporting SAM2 segmentation results to LabelMe JSON format for manual adjustment.')
    group_labelme.add_argument('--export-labelme', action='store_true',
        help='Export SAM2 mask-only results to LabelMe JSON format. '
             'Creates labelme/ subdirectory with JSON annotations and original images. '
             'Use: labelme output_dir/labelme/image_name.json to open and adjust.')
    group_labelme.add_argument('--labelme-include-imagedata', action='store_true',
        help='Embed base64-encoded image data in the JSON file (increases file size, '
             'but allows opening without the original image file)')

    
    # Output Control Arguments
    group_output = parser.add_argument_group('Output Control Options',
        description='Options to enable saving additional output directories. All OFF by default to reduce disk usage.')
    group_output.add_argument('--save-combined', action='store_true',
        help='Save combined/ directory (A1_C1_D1 stitched images) [default: OFF]')
    group_output.add_argument('--save-comparison', action='store_true',
        help='Save comparison/ directory (comparison grid images) [default: OFF]')
    group_output.add_argument('--save-sam-outputs', action='store_true',
        help='Save sam_outputs/ directory (SAM2 visualization results and mask prompts) [default: OFF]')
    group_output.add_argument('--save-csv', action='store_true',
        help='Save CSV files with positive cells information [default: OFF]')
    group_output.add_argument('--save-npy', action='store_true',
        help='Save instance segmentation masks as .npy files for later tile reconstruction [default: OFF]')
    group_output.add_argument('--save-seg-npy', action='store_true',
        help='Save DeepLIIF Seg probability map (RGB) as .npy file [default: OFF]. '
             'Red channel = positive probability, Blue = negative probability, Green = background.')
    group_output.add_argument('--save-sam-steps', action='store_true',
        help='Save SAM2 intermediate results for each instance (all 3 candidate masks, scores, and best mask) [default: OFF]')
    group_output.add_argument('--save-original-sam-comparison', action='store_true',
        help='Save side-by-side comparison of original image and final SAM2 result [default: OFF]')
    group_output.add_argument('--save-cell-extraction-vis', action='store_true',
        help='Save cell extraction visualization showing Seg analysis, pixel classification, and final cell labels [default: OFF]')
    
    # SAM2 Mode Control Arguments
    group_sam_mode = parser.add_argument_group('SAM2 Mode Control',
        description='Options to control SAM2 inference behavior.')
    group_sam_mode.add_argument('--skip-mask-only', action='store_true',
        help='Skip Mask-Only mode inference. NOTE: This also disables LabelMe export!')
    group_sam_mode.add_argument('--resume', action='store_true',
        help='Resume from last run: Skip images that already have output JSON files.')
    group_sam_mode.add_argument('--group-cells', action='store_true',
        help='Enable grouped cell mode: Group nearby cells and send merged mask prompts to SAM2. '
             'This can improve segmentation coherence for adjacent cells.')
    group_sam_mode.add_argument('--group-distance', type=float, default=50.0,
        help='Distance threshold (pixels) for grouping cells. Cells with center distance < this value '
             'will be grouped together. Only used when --group-cells is enabled. (default: 50.0)')
    
    return parser.parse_args()


def validate_config(args: argparse.Namespace) -> argparse.Namespace:
    """
    验证并修正配置参数。
    
    Args:
        args: 解析后的参数对象
        
    Returns:
        修正后的参数对象
    """
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, switching to CPU.")
        args.device = 'cpu'
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate input path
    if not os.path.exists(args.input_dir):
        print(f"Error: Input path {args.input_dir} does not exist.")
        sys.exit(1)
    
    return args


def parse_size_thresh(size_thresh_str: str) -> str | int:
    """
    解析 size_thresh 参数。
    
    Args:
        size_thresh_str: 原始字符串值
        
    Returns:
        'default' 或整数值
    """
    if size_thresh_str == 'default':
        return 'default'
    try:
        return int(size_thresh_str)
    except ValueError:
        print(f"Warning: Invalid size-thresh '{size_thresh_str}', using 'default'")
        return 'default'


def parse_large_noise_thresh(large_noise_thresh_str: str) -> str | int | None:
    """
    解析 large_noise_thresh 参数。
    
    Args:
        large_noise_thresh_str: 原始字符串值
        
    Returns:
        'default', None, 或整数值
    """
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
    """
    打印流水线启动信息。
    
    Args:
        args: 解析后的参数对象
    """
    print(f"\n{'='*60}")
    print(f"PIPELINE STARTED")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")


def print_pipeline_footer():
    """打印流水线完成信息。"""
    print(f"\n{'='*60}")
    print("Pipeline Completed Successfully.")
    print(f"{'='*60}")
