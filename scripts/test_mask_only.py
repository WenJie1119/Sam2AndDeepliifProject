#!/usr/bin/env python3
"""
测试 SAM2 mask-only 模式是否正常工作

用法:
    python test_mask_only.py
"""

import numpy as np
import torch
from PIL import Image

# 导入 SAM2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def test_mask_only():
    # 配置
    checkpoint = r"D:\GitupProject\sam2\checkpoints\sam2.1_hiera_small.pt"
    config = r"D:\GitupProject\sam2\sam2\configs\sam2.1\sam2.1_hiera_s.yaml"
    test_image = r"D:\GitupProject\DataImg\test\tile_38_198_18944_100864.png"
    
    # 加载模型
    print("Loading SAM2 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    sam2_model = build_sam2(config, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # 加载图像
    print(f"Loading image: {test_image}")
    image = np.array(Image.open(test_image).convert('RGB'))
    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}")
    
    predictor.set_image(image)
    
    # 创建一个简单的测试 mask (中心区域)
    # SAM2 期望 mask_input 尺寸为 256x256，值为 logits
    mask_input = np.full((1, 256, 256), -10.0, dtype=np.float32)
    
    # 在中心画一个 50x50 的方形作为前景
    center = 128
    size = 25
    mask_input[0, center-size:center+size, center-size:center+size] = 10.0
    
    print(f"\nTest 1: Mask-only (no point/box)")
    print(f"  mask_input shape: {mask_input.shape}")
    print(f"  mask foreground pixels: {np.sum(mask_input > 0)}")
    
    # 测试 1: 只使用 mask
    try:
        masks, scores, low_res_masks = predictor.predict(
            mask_input=mask_input,
            multimask_output=True
        )
        print(f"  Output masks shape: {masks.shape}")
        print(f"  Scores: {scores}")
        for i, (m, s) in enumerate(zip(masks, scores)):
            area = np.sum(m)
            print(f"    Mask {i}: score={s:.4f}, area={area} pixels ({area/(h*w)*100:.1f}%)")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # 测试 2: mask + 中心点
    print(f"\nTest 2: Mask + Center Point")
    center_y = h // 2
    center_x = w // 2
    point_coords = np.array([[center_x, center_y]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)
    
    try:
        masks, scores, low_res_masks = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            multimask_output=True
        )
        print(f"  Output masks shape: {masks.shape}")
        print(f"  Scores: {scores}")
        for i, (m, s) in enumerate(zip(masks, scores)):
            area = np.sum(m)
            print(f"    Mask {i}: score={s:.4f}, area={area} pixels ({area/(h*w)*100:.1f}%)")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # 测试 3: 只使用点（无 mask）
    print(f"\nTest 3: Point only (no mask)")
    try:
        masks, scores, low_res_masks = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        print(f"  Output masks shape: {masks.shape}")
        print(f"  Scores: {scores}")
        for i, (m, s) in enumerate(zip(masks, scores)):
            area = np.sum(m)
            print(f"    Mask {i}: score={s:.4f}, area={area} pixels ({area/(h*w)*100:.1f}%)")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print("\nDone!")


if __name__ == "__main__":
    test_mask_only()
