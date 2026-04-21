#!/usr/bin/env python3
"""
CD34 微血管检测 Pipeline — WSI 模式入口脚本

从 .ndpi/.svs 全切片图像直接读取 tile 进行处理。

Usage:
    # 完整 pipeline
    python scripts/run_wsi_pipeline.py \
        --wsi-path /path/to/slide.ndpi \
        --output-dir ./output/wsi_test \
        --use-connected-regions --save-npy

    # 仅 YOLO 分类
    python scripts/run_wsi_pipeline.py \
        --wsi-path /path/to/slide.ndpi \
        --output-dir ./output/wsi_test \
        --classify-only

    # 单 tile 调试
    python scripts/run_wsi_pipeline.py \
        --wsi-path /path/to/slide.ndpi \
        --output-dir ./output/wsi_test \
        --tile-index 5,12
"""

import sys
import os

# 确保项目根目录在 Python path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cd34_pipeline.wsi_pipeline import main_wsi

if __name__ == "__main__":
    main_wsi()
