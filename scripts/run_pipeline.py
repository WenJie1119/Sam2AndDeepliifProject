#!/usr/bin/env python3
"""
CD34 微血管检测 Pipeline — 入口脚本

Usage:
    python scripts/run_pipeline.py --input-dir /path/to/images --output-dir /path/to/save
"""

import sys
import os

# 确保项目根目录在 Python path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cd34_pipeline.pipeline import main

if __name__ == "__main__":
    main()
