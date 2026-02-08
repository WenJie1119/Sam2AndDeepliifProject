#!/usr/bin/env python3
"""
reconstruct_tiles.py — 瓦片重建独立入口脚本

从多个 tile 的 npy mask 重建完整图像的实例分割。

Usage:
    python reconstruct_tiles.py --tile-dir /path/to/npy_masks --output-dir /path/to/output

Example:
    python reconstruct_tiles.py \\
        --tile-dir ./output/npy_masks \\
        --output-dir ./output/reconstructed \\
        --tile-size 512 \\
        --original-image ./original_large_image.png \\
        --export-labelme
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tile_reconstruction import reconstruct_tiles


def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Reconstruct full image segmentation from tile npy masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic reconstruction
  python reconstruct_tiles.py --tile-dir ./npy_masks --output-dir ./output

  # With original image for LabelMe
  python reconstruct_tiles.py --tile-dir ./npy_masks --output-dir ./output \\
      --original-image ./large_image.png --export-labelme

  # Custom tile size and block size
  python reconstruct_tiles.py --tile-dir ./npy_masks --output-dir ./output \\
      --tile-size 1024 --block-size 4
        """
    )
    
    # Required arguments
    parser.add_argument('--tile-dir', type=str, required=True,
        help='Directory containing tile npy mask files')
    parser.add_argument('--output-dir', type=str, required=True,
        help='Directory to save reconstructed results')
    
    # Tile parameters
    parser.add_argument('--tile-size', type=int, default=512,
        help='Size of each tile in pixels (default: 512)')
    parser.add_argument('--block-size', type=int, default=2,
        help='Number of tiles to process at once (NxN, default: 2, meaning 2x2=4 tiles)')
    
    # Image reference
    parser.add_argument('--original-image', type=str, default=None,
        help='Path to original large image (used for LabelMe JSON imagePath)')
    
    # Output control
    parser.add_argument('--output-name', type=str, default='reconstructed',
        help='Base name for output files (default: reconstructed)')
    
    output_group = parser.add_argument_group('Output Format Control')
    output_group.add_argument('--save-npy', action='store_true', default=True,
        help='Save reconstructed mask as npy file (default: True)')
    output_group.add_argument('--no-save-npy', action='store_true',
        help='Do not save npy file')
    output_group.add_argument('--export-labelme', action='store_true', default=True,
        help='Export to LabelMe JSON format (default: True)')
    output_group.add_argument('--no-export-labelme', action='store_true',
        help='Do not export LabelMe JSON')
    
    return parser.parse_args()


def main():
    """主函数。"""
    args = parse_arguments()
    
    # Validate input
    if not os.path.isdir(args.tile_dir):
        print(f"ERROR: Tile directory not found: {args.tile_dir}")
        sys.exit(1)
    
    # Handle output flags
    save_npy = args.save_npy and not args.no_save_npy
    export_labelme = args.export_labelme and not args.no_export_labelme
    
    if not save_npy and not export_labelme:
        print("WARNING: Both --no-save-npy and --no-export-labelme are set. No output will be saved!")
    
    # Run reconstruction
    result = reconstruct_tiles(
        tile_dir=args.tile_dir,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        block_size=args.block_size,
        original_image_path=args.original_image,
        save_npy=save_npy,
        export_labelme=export_labelme,
        output_name=args.output_name
    )
    
    if result is not None:
        print(f"\nReconstruction successful!")
        print(f"Output directory: {args.output_dir}")
    else:
        print("\nReconstruction failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
