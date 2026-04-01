#!/usr/bin/env python3
"""
交互式 SAM2 分割测试脚本

功能：
    1. 读取一张图片
    2. 通过鼠标点击确定分割位置（左键添加正点，右键添加负点）
    3. 使用 SAM2 进行实时分割
    4. 按 'c' 清除所有点，按 'q' 退出

用法:
    python test_interactive_sam2.py --image <图片路径>
    python test_interactive_sam2.py  # 使用默认测试图片
"""

import argparse
import numpy as np
import cv2
import torch
from PIL import Image

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class InteractiveSAM2Segmentor:
    """交互式 SAM2 分割器"""
    
    def __init__(self, checkpoint: str, config: str, device: str = None):
        """
        初始化 SAM2 分割器
        
        Args:
            checkpoint: SAM2 模型权重路径
            config: SAM2 配置文件路径
            device: 运行设备 ('cuda' 或 'cpu')
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading SAM2 model on {device}...")
        self.device = device
        sam2_model = build_sam2(config, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        print("SAM2 model loaded successfully!")
        
        # 存储点击的点
        self.points = []  # [(x, y, label), ...] label: 1=前景, 0=背景
        self.image = None
        self.original_image = None
        self.mask_overlay = None
        self.current_mask = None
        
    def load_image(self, image_path: str):
        """加载图片并设置到 SAM2 predictor"""
        print(f"Loading image: {image_path}")
        self.original_image = np.array(Image.open(image_path).convert('RGB'))
        self.image = self.original_image.copy()
        h, w = self.image.shape[:2]
        print(f"Image size: {w}x{h}")
        
        print("Setting image to SAM2 predictor (computing embeddings)...")
        self.predictor.set_image(self.image)
        print("Image embeddings computed!")
        
        self.points = []
        self.current_mask = None
        
    def add_point(self, x: int, y: int, is_positive: bool = True):
        """
        添加一个点提示
        
        Args:
            x, y: 点的坐标
            is_positive: True=前景点(左键), False=背景点(右键)
        """
        label = 1 if is_positive else 0
        self.points.append((x, y, label))
        print(f"Added {'positive' if is_positive else 'negative'} point at ({x}, {y})")
        self._run_prediction()
        
    def clear_points(self):
        """清除所有点"""
        self.points = []
        self.current_mask = None
        print("Cleared all points")
        
    def _run_prediction(self):
        """运行 SAM2 预测"""
        if not self.points:
            self.current_mask = None
            return
            
        # 准备点坐标和标签
        point_coords = np.array([[p[0], p[1]] for p in self.points], dtype=np.float32)
        point_labels = np.array([p[2] for p in self.points], dtype=np.int32)
        
        print(f"Running SAM2 prediction with {len(self.points)} point(s)...")
        
        # 运行预测
        masks, scores, low_res_masks = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        
        # 选择最高分的 mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        # 确保 mask 是 numpy boolean 数组
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        self.current_mask = mask.astype(bool)
        
        print(f"Prediction complete! Best mask score: {scores[best_idx]:.4f}")
        print(f"  Mask area: {np.sum(self.current_mask)} pixels")
        
    def get_display_image(self):
        """获取用于显示的图像（带点和 mask 覆盖）"""
        display = self.original_image.copy()
        
        # 如果有 mask，绘制半透明覆盖
        if self.current_mask is not None:
            # 创建彩色 mask 覆盖（绿色）
            mask_color = np.zeros_like(display)
            mask_color[self.current_mask] = [0, 255, 0]  # 绿色
            
            # 混合原图和 mask
            display = cv2.addWeighted(display, 0.7, mask_color, 0.3, 0)
            
            # 绘制 mask 边界（青色）
            mask_uint8 = self.current_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (255, 255, 0), 2)  # 青色边界
        
        # 绘制点击的点
        for x, y, label in self.points:
            color = (0, 255, 0) if label == 1 else (0, 0, 255)  # 绿=正点，红=负点
            cv2.circle(display, (x, y), 8, color, -1)
            cv2.circle(display, (x, y), 8, (255, 255, 255), 2)  # 白色边框
            
        return display
    
    def save_mask(self, output_path: str):
        """保存当前 mask"""
        if self.current_mask is not None:
            mask_img = (self.current_mask * 255).astype(np.uint8)
            cv2.imwrite(output_path, mask_img)
            print(f"Mask saved to: {output_path}")
        else:
            print("No mask to save!")


def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数"""
    segmentor = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 左键点击 - 添加正点（前景）
        segmentor.add_point(x, y, is_positive=True)
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 右键点击 - 添加负点（背景）
        segmentor.add_point(x, y, is_positive=False)


def main():
    parser = argparse.ArgumentParser(description="Interactive SAM2 Segmentation")
    parser.add_argument(
        "--image", 
        type=str, 
        default=r"D:\GitupProject\DataImg\test\correction2\tile_38_235_18944_119808.png",
        help="输入图片路径"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r"D:\GitupProject\sam2\checkpoints\sam2.1_hiera_small.pt",
        help="SAM2 模型权重路径"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=r"D:\GitupProject\sam2\sam2\configs\sam2.1\sam2.1_hiera_s.yaml",
        help="SAM2 配置文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="interactive_mask.png",
        help="分割结果输出路径"
    )
    args = parser.parse_args()
    
    # 创建分割器
    segmentor = InteractiveSAM2Segmentor(
        checkpoint=args.checkpoint,
        config=args.config
    )
    
    # 加载图片
    segmentor.load_image(args.image)
    
    # 创建窗口
    window_name = "SAM2 Interactive Segmentation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback, segmentor)
    
    print("\n" + "="*60)
    print("SAM2 交互式分割工具")
    print("="*60)
    print("操作说明:")
    print("  - 左键点击: 添加前景点 (绿色，标记要分割的目标)")
    print("  - 右键点击: 添加背景点 (红色，标记不需要的区域)")
    print("  - 按 'c':   清除所有点")
    print("  - 按 's':   保存当前 mask")
    print("  - 按 'q':   退出程序")
    print("="*60 + "\n")
    
    while True:
        # 检测窗口是否被关闭（点击 × 按钮）
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed by user")
            break
            
        # 获取并显示当前图像
        display = segmentor.get_display_image()
        # OpenCV 使用 BGR 格式
        display_bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, display_bgr)
        
        # 处理键盘输入
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q'):
            # 退出
            print("Exiting...")
            break
            
        elif key == ord('c'):
            # 清除点
            segmentor.clear_points()
            
        elif key == ord('s'):
            # 保存 mask
            segmentor.save_mask(args.output)
    
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
