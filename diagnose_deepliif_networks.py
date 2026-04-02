"""
诊断脚本2：深入分析 DeepLIIF 5 个分割网络 (G51-G55) 的各自输出。
找出是哪些网络漏检了浅棕色，并测试不同 seg_weights 组合的效果。
"""

import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cd34_pipeline.deepliif.utils import (
    disable_batchnorm_tracking_stats, get_transform, tensor_to_pil,
    is_empty
)

# ============================================================================
# 配置
# ============================================================================
TILE_NAME = "tile_38_231_18944_117760"
BASE_DIR = Path(r"D:\GitupProject\sam2\test_results")
MODEL_DIR = Path(r"D:\GitupProject\sam2\data\models\deepliif")

original_path = BASE_DIR / "initial_img" / f"{TILE_NAME}.png"
output_dir = BASE_DIR / "diagnose_light_brown"
output_dir.mkdir(exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# 1. 加载模型并单独运行每个分割网络
# ============================================================================
transform = get_transform()
img = Image.open(str(original_path)).convert('RGB')
original_rgb = np.array(img)
h, w = original_rgb.shape[:2]
print(f"Image: {w}x{h}")

# 加载所有模型
nets = {}
modality_names = ['G1', 'G2', 'G3', 'G4']
seg_names = ['G51', 'G52', 'G53', 'G54', 'G55']

for name in modality_names + seg_names:
    model_path = MODEL_DIR / f'{name}.pt'
    if model_path.exists():
        net = torch.jit.load(str(model_path), map_location=device)
        net = disable_batchnorm_tracking_stats(net)
        net.eval()
        nets[name] = net
        print(f"  Loaded {name}")

# 推理
ts = transform(img).to(device)

# Modality 输出
with torch.no_grad():
    mod_results = {}
    for name in modality_names:
        if name in nets:
            mod_results[name] = nets[name](ts)

    # 各个分割网络的输出
    seg_outputs = {}    # tensor 格式
    seg_images = {}     # numpy RGB 格式

    # G51: 直接从 IHC 输入
    if 'G51' in nets:
        seg_outputs['G51'] = nets['G51'](ts)
        seg_images['G51 (IHC)'] = np.array(tensor_to_pil(seg_outputs['G51']))

    # G52-G55: 从对应 modality 输入
    mod_labels = {
        'G52': ('G1', 'Hema'),
        'G53': ('G2', 'DAPI'),
        'G54': ('G3', 'Lap2'),
        'G55': ('G4', 'Marker'),
    }
    for seg_name, (mod_name, mod_label) in mod_labels.items():
        if seg_name in nets and mod_name in mod_results:
            seg_outputs[seg_name] = nets[seg_name](mod_results[mod_name])
            seg_images[f'{seg_name} ({mod_label})'] = np.array(tensor_to_pil(seg_outputs[seg_name]))

print(f"\nLoaded {len(seg_images)} segmentation networks")

# ============================================================================
# 2. 分析每个网络对浅棕色区域的检测能力
# ============================================================================

# 用颜色检测原图棕色（同上一个脚本）
hsv = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV)
lab = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB)
brown_deep = cv2.inRange(hsv, (10, 60, 80), (25, 255, 220))
brown_light = cv2.inRange(hsv, (10, 30, 120), (30, 120, 255))
l_ch, a_ch, b_ch = lab[:,:,0], lab[:,:,1], lab[:,:,2]
lab_brown = ((b_ch > 135) & (a_ch > 128) & (l_ch > 60) & (l_ch < 220)).astype(np.uint8) * 255
color_brown = (brown_deep | brown_light | lab_brown).astype(np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
color_brown = cv2.morphologyEx(color_brown, cv2.MORPH_CLOSE, kernel)
color_brown = cv2.morphologyEx(color_brown, cv2.MORPH_OPEN, kernel)
brown_mask = color_brown > 0

print(f"\n{'='*80}")
print(f"各分割网络对棕色区域的检测分析 (棕色总像素: {np.sum(brown_mask)})")
print(f"{'='*80}")
print(f"{'Network':>20} | {'前景px':>10} | {'阳性px':>10} | {'覆盖棕色':>10} | {'棕色覆盖率':>10} | {'R+B mean':>10} | {'G mean':>10}")
print("-" * 100)

network_stats = {}
for label, seg_rgb in seg_images.items():
    r = seg_rgb[:,:,0].astype(np.int32)
    g = seg_rgb[:,:,1].astype(np.int32)
    b = seg_rgb[:,:,2].astype(np.int32)
    rb = r + b

    # 用较低阈值检测
    for thresh in [120, 80, 60]:
        fg = (rb > thresh) & (g <= 80)
        pos = fg & (r >= b)
        covered = pos & brown_mask

        if thresh == 120:
            network_stats[label] = {
                'fg': np.sum(fg), 'pos': np.sum(pos),
                'covered': np.sum(covered),
                'rb_on_brown': rb[brown_mask].mean() if np.any(brown_mask) else 0,
                'g_on_brown': g[brown_mask].mean() if np.any(brown_mask) else 0,
            }
            print(f"{label:>20} | {np.sum(fg):>10} | {np.sum(pos):>10} | "
                  f"{np.sum(covered):>10} | {np.sum(covered)/max(np.sum(brown_mask),1)*100:>9.1f}% | "
                  f"{rb[brown_mask].mean():>10.1f} | {g[brown_mask].mean():>10.1f}")

# ============================================================================
# 3. 测试不同 seg_weights 组合
# ============================================================================
print(f"\n{'='*80}")
print(f"不同权重组合的效果 (seg_thresh=120)")
print(f"{'='*80}")

seg_tensors = [seg_outputs.get(f'G5{i}') for i in range(1, 6)]
# 确保所有 tensor 都存在
if all(t is not None for t in seg_tensors):
    weight_configs = [
        ([0.25, 0.15, 0.25, 0.10, 0.25], "Default [.25,.15,.25,.10,.25]"),
        ([0.20, 0.20, 0.20, 0.20, 0.20], "Equal   [.20,.20,.20,.20,.20]"),
        ([0.35, 0.10, 0.20, 0.05, 0.30], "IHC+Marker heavy"),
        ([0.40, 0.10, 0.15, 0.05, 0.30], "IHC very heavy"),
        ([0.30, 0.10, 0.10, 0.05, 0.45], "Marker dominant"),
        ([0.15, 0.15, 0.35, 0.10, 0.25], "DAPI heavy"),
        ([0.50, 0.10, 0.10, 0.05, 0.25], "IHC dominant (0.5)"),
        ([0.30, 0.05, 0.25, 0.05, 0.35], "IHC+Marker, low Hema/Lap2"),
    ]

    print(f"{'Config':>35} | {'前景px':>10} | {'阳性px':>10} | {'覆盖棕色':>10} | {'覆盖率':>8} | {'漏检率':>8}")
    print("-" * 100)

    best_config = None
    best_coverage = 0
    weight_results = {}

    for weights, label in weight_configs:
        final_seg = torch.zeros_like(seg_tensors[0])
        for s, w_val in zip(seg_tensors, weights):
            final_seg += s * w_val

        seg_pil = tensor_to_pil(final_seg)
        seg_np = np.array(seg_pil)

        r = seg_np[:,:,0].astype(np.int32)
        g = seg_np[:,:,1].astype(np.int32)
        b = seg_np[:,:,2].astype(np.int32)
        rb = r + b

        fg = (rb > 120) & (g <= 80)
        pos = fg & (r >= b)
        covered = pos & brown_mask
        missed = brown_mask & ~pos

        coverage = np.sum(covered) / max(np.sum(brown_mask), 1) * 100
        miss_rate = np.sum(missed) / max(np.sum(brown_mask), 1) * 100

        print(f"{label:>35} | {np.sum(fg):>10} | {np.sum(pos):>10} | "
              f"{np.sum(covered):>10} | {coverage:>7.1f}% | {miss_rate:>7.1f}%")

        weight_results[label] = {
            'weights': weights, 'seg_np': seg_np,
            'pos': pos, 'covered': covered, 'coverage': coverage
        }

        if coverage > best_coverage:
            best_coverage = coverage
            best_config = label

    print(f"\n最佳权重: {best_config} (覆盖率: {best_coverage:.1f}%)")

    # 测试权重+降低阈值组合
    print(f"\n{'='*80}")
    print(f"最佳权重 + 降低阈值组合")
    print(f"{'='*80}")

    for label, info in weight_results.items():
        seg_np = info['seg_np']
        r = seg_np[:,:,0].astype(np.int32)
        g = seg_np[:,:,1].astype(np.int32)
        b = seg_np[:,:,2].astype(np.int32)
        rb = r + b

        for seg_t, g_t in [(120, 80), (80, 80), (60, 100), (60, 120)]:
            fg = (rb > seg_t) & (g <= g_t)
            pos = fg & (r >= b)
            covered = pos & brown_mask
            coverage = np.sum(covered) / max(np.sum(brown_mask), 1) * 100
            if coverage > best_coverage:
                best_coverage = coverage
                best_config = f"{label} + seg={seg_t},G<={g_t}"

        # 只打印当前 label 的最佳组合
        best_combo_cov = 0
        best_combo_str = ""
        for seg_t, g_t in [(120, 80), (80, 80), (80, 100), (60, 80), (60, 100), (60, 120)]:
            fg = (rb > seg_t) & (g <= g_t)
            pos = fg & (r >= b)
            covered = pos & brown_mask
            cov = np.sum(covered) / max(np.sum(brown_mask), 1) * 100
            if cov > best_combo_cov:
                best_combo_cov = cov
                best_combo_str = f"seg={seg_t},G<={g_t}"

        print(f"  {label:>35}: best={best_combo_str}, coverage={best_combo_cov:.1f}%")

    print(f"\n全局最佳: {best_config} (覆盖率: {best_coverage:.1f}%)")

# ============================================================================
# 4. 可视化：每个分割网络的输出对比
# ============================================================================
n_nets = len(seg_images)
fig, axes = plt.subplots(3, n_nets, figsize=(5 * n_nets, 15))

for idx, (label, seg_rgb) in enumerate(seg_images.items()):
    # Row 1: 网络原始 Seg 输出
    axes[0, idx].imshow(seg_rgb)
    axes[0, idx].set_title(f'{label}', fontsize=11)
    axes[0, idx].axis('off')

    # Row 2: 该网络检测到的阳性叠加在原图上
    r = seg_rgb[:,:,0].astype(np.int32)
    g = seg_rgb[:,:,1].astype(np.int32)
    b = seg_rgb[:,:,2].astype(np.int32)
    rb = r + b
    fg = (rb > 120) & (g <= 80)
    pos = fg & (r >= b)

    overlay = original_rgb.copy()
    overlay[pos] = [255, 0, 0]
    axes[1, idx].imshow(overlay)
    axes[1, idx].set_title(f'Positive (thresh=120)\n{np.sum(pos)} px', fontsize=10)
    axes[1, idx].axis('off')

    # Row 3: R+B heatmap
    im = axes[2, idx].imshow(np.clip(rb, 0, 300), cmap='hot', vmin=0, vmax=300)
    axes[2, idx].set_title(f'R+B heatmap', fontsize=10)
    axes[2, idx].axis('off')

plt.suptitle(f'Individual Segmentation Networks Analysis\n{TILE_NAME}', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(str(output_dir / f"{TILE_NAME}_network_analysis.png"), dpi=150, bbox_inches='tight')
print(f"\n网络分析图已保存: {output_dir / f'{TILE_NAME}_network_analysis.png'}")

# ============================================================================
# 5. 可视化：权重对比 (Top 4)
# ============================================================================
if weight_results:
    sorted_results = sorted(weight_results.items(), key=lambda x: x[1]['coverage'], reverse=True)
    top_n = min(6, len(sorted_results))

    fig2, axes2 = plt.subplots(2, 3, figsize=(20, 14))
    for idx in range(top_n):
        ax = axes2[idx // 3, idx % 3]
        label, info = sorted_results[idx]
        pos = info['pos']
        covered = info['covered']

        overlay = original_rgb.copy()
        red = overlay.copy()
        red[pos] = [255, 0, 0]
        overlay = cv2.addWeighted(overlay, 0.5, red, 0.5, 0)

        pos_u8 = pos.astype(np.uint8) * 255
        contours, _ = cv2.findContours(pos_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 1)

        ax.imshow(overlay)
        ax.set_title(f'{label}\nCoverage: {info["coverage"]:.1f}%, Pos: {np.sum(pos)} px',
                     fontsize=11)
        ax.axis('off')

    plt.suptitle('Weight Configurations Comparison (Top 6 by brown coverage)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(output_dir / f"{TILE_NAME}_weights_comparison.png"), dpi=150, bbox_inches='tight')
    print(f"权重对比图已保存: {output_dir / f'{TILE_NAME}_weights_comparison.png'}")

plt.show()
print("\nDone!")
