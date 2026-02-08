#!/usr/bin/env python3
"""
model_loader.py — 模型加载模块

包含：
- DeepLIIF 模型加载
- SAM2 模型加载
"""

import sys


def load_deepliif(model_dir: str, device: str):
    """
    加载 DeepLIIF 推理引擎。
    
    Args:
        model_dir: DeepLIIF 模型目录路径
        device: 设备 ('cuda' 或 'cpu')
        
    Returns:
        DeepLIIFInference 实例
    """
    try:
        from deepliif_inference import DeepLIIFInference
    except ImportError:
        print("Error: Could not import deepliif_inference.")
        print("Make sure deepliif_inference.py is in your path.")
        sys.exit(1)
    
    print("[Model] Initializing DeepLIIF Engine...")
    engine = DeepLIIFInference(model_dir=model_dir, device=device)
    print(f"  DeepLIIF loaded from: {model_dir}")
    return engine


def load_sam2(config_path: str, checkpoint_path: str, device: str):
    """
    加载 SAM2 模型并返回 predictor。
    
    Args:
        config_path: SAM2 配置文件路径
        checkpoint_path: SAM2 checkpoint 路径
        device: 设备 ('cuda' 或 'cpu')
        
    Returns:
        SAM2ImagePredictor 实例
    """
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        print("Error: Could not import sam2. Make sure you are in the sam2 environment.")
        sys.exit(1)
    
    print("[Model] Initializing SAM2 Model...")
    sam2_model = build_sam2(config_path, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print(f"  SAM2 loaded from: {checkpoint_path}")
    return predictor


def load_all_models(args):
    """
    加载所有需要的模型（便捷函数）。
    
    Args:
        args: 包含模型配置的 argparse.Namespace
        
    Returns:
        tuple: (deepliif_engine, sam2_predictor)
    """
    print("\n[Step 1/3] Loading Models...")
    
    deepliif_engine = load_deepliif(
        model_dir=args.deepliif_model_dir, 
        device=args.device
    )
    
    sam2_predictor = load_sam2(
        config_path=args.sam_config,
        checkpoint_path=args.sam_checkpoint,
        device=args.device
    )
    
    print("Models loaded successfully.\n")
    return deepliif_engine, sam2_predictor
