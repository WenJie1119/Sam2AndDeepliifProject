"""
诊断脚本：分析浅棕色区域在 DeepLIIF Seg/Marker 输出中的像素值。
找出为什么浅棕色细胞被漏检，并给出调参建议。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# 配置
# ============================================================================
TILE_NAME = "tile_38_231_18944_117760"
BASE_DIR = Path(r"D:\GitupProject\sam2\test_results")

original_path = BASE_DIR / "initial_img" / f"{TILE_NAME}.png"
seg_path = BASE_DIR / "refactor_test" / "deepliif_outputs" / TILE_NAME / "Seg.png"
marker_path = BASE_DIR / "refactor_test" / "deepliif_outputs" / TILE_NAME / "Marker.png"
hema_path = BASE_DIR / "refactor_test" / "deepliif_outputs" / TILE_NAME / "Hema.png"

output_dir = BASE_DIR / "diagnose_light_brown"
output_dir.mkdir(exist_ok=True)

# ============================================================================
# 加载图像
# ============================================================================
original = cv2.imread(str(original_path))
seg = cv2.imread(str(seg_path))
marker = cv2.imread(str(marker_path))
hema = cv2.imread(str(hema_path))

assert original is not None, f"Cannot load: {original_path}"
assert seg is not None, f"Cannot load: {seg_path}"
assert marker is not None, f"Cannot load: {marker_path}"

original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
seg_rgb = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
marker_gray = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
if hema is not None:
    hema_rgb = cv2.cvtColor(hema, cv2.COLOR_BGR2RGB)

h, w = original_rgb.shape[:2]
print(f"Image size: {w} x {h}")

# ============================================================================
# 1. 用颜色空间检测原图中的棕色区域（作为 ground truth）
# ============================================================================
# 转换到 HSV 和 LAB 空间检测棕色 (DAB staining)
hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)

# 棕色在 HSV 中的范围 (宽松范围，覆盖深棕到浅棕)
# H: 10-30 (棕色/橙色色调)
# S: 30+ (有一定饱和度)
# V: 50+ (不是纯黑)
brown_mask_deep = cv2.inRange(hsv, (10, 60, 80), (25, 255, 220))
brown_mask_light = cv2.inRange(hsv, (10, 30, 120), (30, 120, 255))
brown_mask_all = brown_mask_deep | brown_mask_light

# LAB 空间补充: b通道(黄蓝) > 135 且 a通道(红绿) > 128 偏暖色
l_ch, a_ch, b_ch = lab[:,:,0], lab[:,:,1], lab[:,:,2]
lab_brown = ((b_ch > 135) & (a_ch > 128) & (l_ch > 60) & (l_ch < 220)).astype(np.uint8) * 255

# 合并颜色检测结果
color_brown_mask = (brown_mask_all | lab_brown).astype(np.uint8)
# 轻微形态学清理
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
color_brown_mask = cv2.morphologyEx(color_brown_mask, cv2.MORPH_CLOSE, kernel)
color_brown_mask = cv2.morphologyEx(color_brown_mask, cv2.MORPH_OPEN, kernel)

print(f"\n=== 颜色空间检测到的棕色像素 ===")
print(f"  棕色像素数: {np.sum(color_brown_mask > 0)}")
print(f"  占比: {np.sum(color_brown_mask > 0) / (h * w) * 100:.1f}%")

# ============================================================================
# 2. 分析 Seg 输出在不同阈值下的前景检测
# ============================================================================
r_ch_seg = seg_rgb[:,:,0].astype(np.int32)
g_ch_seg = seg_rgb[:,:,1].astype(np.int32)
b_ch_seg = seg_rgb[:,:,2].astype(np.int32)
sum_rb = r_ch_seg + b_ch_seg

print(f"\n=== Seg 图像全局统计 ===")
print(f"  R channel: min={r_ch_seg.min()}, max={r_ch_seg.max()}, mean={r_ch_seg.mean():.1f}")
print(f"  G channel: min={g_ch_seg.min()}, max={g_ch_seg.max()}, mean={g_ch_seg.mean():.1f}")
print(f"  B channel: min={b_ch_seg.min()}, max={b_ch_seg.max()}, mean={b_ch_seg.mean():.1f}")
print(f"  R+B:       min={sum_rb.min()}, max={sum_rb.max()}, mean={sum_rb.mean():.1f}")

# 不同 seg_thresh 下的前景像素数量
thresholds = [20, 40, 60, 80, 100, 120, 150, 180]
g_limit = 80

print(f"\n=== 不同 seg_thresh 下的前景检测 (G<={g_limit}) ===")
print(f"{'seg_thresh':>12} | {'前景像素':>10} | {'占比':>8} | {'阳性(R>=B)':>12} | {'阴性(R<B)':>12}")
print("-" * 70)

for thresh in thresholds:
    fg = (sum_rb > thresh) & (g_ch_seg <= g_limit)
    pos = fg & (r_ch_seg >= b_ch_seg)
    neg = fg & (r_ch_seg < b_ch_seg)
    marker = " <-- 当前默认" if thresh == 120 else ""
    print(f"{thresh:>12} | {np.sum(fg):>10} | {np.sum(fg)/(h*w)*100:>7.1f}% | {np.sum(pos):>12} | {np.sum(neg):>12}{marker}")

# 也测试不同 G 通道阈值
print(f"\n=== 不同 G 通道阈值下的前景检测 (seg_thresh=120) ===")
for g_thresh in [60, 80, 100, 120, 150, 200]:
    fg = (sum_rb > 120) & (g_ch_seg <= g_thresh)
    marker_str = " <-- 当前默认" if g_thresh == 80 else ""
    print(f"  G <= {g_thresh:>3}: 前景 {np.sum(fg):>8} px ({np.sum(fg)/(h*w)*100:.1f}%){marker_str}")

# ============================================================================
# 3. 关键分析：颜色检测到的棕色 vs DeepLIIF Seg 检测到的
# ============================================================================
# DeepLIIF 当前检测的阳性 (默认 seg_thresh=120, G<=80)
deepliif_fg = (sum_rb > 120) & (g_ch_seg <= 80)
deepliif_pos = deepliif_fg & (r_ch_seg >= b_ch_seg)

# 颜色检测到但 DeepLIIF 没检测到的 = 漏检区域
color_detected = color_brown_mask > 0
missed = color_detected & ~deepliif_pos

print(f"\n=== 漏检分析 ===")
print(f"  颜色空间检测到的棕色: {np.sum(color_detected):>8} px")
print(f"  DeepLIIF Seg 阳性:    {np.sum(deepliif_pos):>8} px")
print(f"  被漏检的棕色像素:     {np.sum(missed):>8} px")
print(f"  漏检率: {np.sum(missed) / max(np.sum(color_detected), 1) * 100:.1f}%")

# 分析漏检区域在 Seg 图中的像素值分布
missed_rb = sum_rb[missed]
missed_g = g_ch_seg[missed]
missed_r = r_ch_seg[missed]
missed_b = b_ch_seg[missed]

if len(missed_rb) > 0:
    print(f"\n=== 漏检区域在 Seg 图中的像素值 ===")
    print(f"  R+B: min={missed_rb.min()}, max={missed_rb.max()}, "
          f"mean={missed_rb.mean():.1f}, median={np.median(missed_rb):.1f}")
    print(f"  G:   min={missed_g.min()}, max={missed_g.max()}, "
          f"mean={missed_g.mean():.1f}, median={np.median(missed_g):.1f}")
    print(f"  R:   min={missed_r.min()}, max={missed_r.max()}, "
          f"mean={missed_r.mean():.1f}, median={np.median(missed_r):.1f}")
    print(f"  B:   min={missed_b.min()}, max={missed_b.max()}, "
          f"mean={missed_b.mean():.1f}, median={np.median(missed_b):.1f}")

    # 分析漏检原因：到底是 R+B 太低还是 G 太高？
    missed_low_rb = missed & (sum_rb <= 120)
    missed_high_g = missed & (g_ch_seg > 80)
    missed_both = missed & (sum_rb <= 120) & (g_ch_seg > 80)
    missed_rb_only = missed & (sum_rb <= 120) & (g_ch_seg <= 80)
    missed_g_only = missed & (sum_rb > 120) & (g_ch_seg > 80)

    print(f"\n=== 漏检原因分解 ===")
    print(f"  仅因 R+B <= 120 (G OK):  {np.sum(missed_rb_only):>8} px ({np.sum(missed_rb_only)/max(np.sum(missed),1)*100:.1f}%)")
    print(f"  仅因 G > 80 (R+B OK):    {np.sum(missed_g_only):>8} px ({np.sum(missed_g_only)/max(np.sum(missed),1)*100:.1f}%)")
    print(f"  两者都不满足:             {np.sum(missed_both):>8} px ({np.sum(missed_both)/max(np.sum(missed),1)*100:.1f}%)")

    # 漏检区域在 Marker 图中的值
    missed_marker = marker_gray[missed]
    print(f"\n=== 漏检区域在 Marker 图中的值 ===")
    print(f"  Marker: min={missed_marker.min()}, max={missed_marker.max()}, "
          f"mean={missed_marker.mean():.1f}, median={np.median(missed_marker):.1f}")

    # 不同 seg_thresh 能挽救多少漏检
    print(f"\n=== 降低 seg_thresh 能挽救多少漏检像素 ===")
    for thresh in [100, 80, 60, 40, 20]:
        recoverable = missed & (sum_rb > thresh) & (g_ch_seg <= 80)
        print(f"  seg_thresh={thresh:>3}: 可挽救 {np.sum(recoverable):>8} px "
              f"({np.sum(recoverable)/max(np.sum(missed),1)*100:.1f}% of missed)")

    # 放宽 G 通道能挽救多少
    print(f"\n=== 放宽 G 通道阈值能挽救多少漏检像素 (seg_thresh=120) ===")
    for g_thresh in [100, 120, 150, 200]:
        recoverable = missed & (sum_rb > 120) & (g_ch_seg <= g_thresh)
        print(f"  G<={g_thresh:>3}: 可挽救 {np.sum(recoverable):>8} px "
              f"({np.sum(recoverable)/max(np.sum(missed),1)*100:.1f}% of missed)")

    # 组合调参
    print(f"\n=== 组合调参效果 ===")
    combos = [
        (80, 80, "seg=80, G<=80"),
        (80, 100, "seg=80, G<=100"),
        (60, 80, "seg=60, G<=80"),
        (60, 100, "seg=60, G<=100"),
        (60, 120, "seg=60, G<=120"),
        (40, 100, "seg=40, G<=100"),
    ]
    for seg_t, g_t, label in combos:
        recoverable = missed & (sum_rb > seg_t) & (g_ch_seg <= g_t)
        new_fg = (sum_rb > seg_t) & (g_ch_seg <= g_t)
        new_pos = new_fg & (r_ch_seg >= b_ch_seg)
        print(f"  {label:>20}: 挽救 {np.sum(recoverable):>8} px ({np.sum(recoverable)/max(np.sum(missed),1)*100:.1f}%), "
              f"新总阳性: {np.sum(new_pos):>8} px")

# ============================================================================
# 4. 可视化
# ============================================================================
fig, axes = plt.subplots(3, 3, figsize=(20, 20))

# Row 1: 原图、Seg、Marker
axes[0, 0].imshow(original_rgb)
axes[0, 0].set_title('Original', fontsize=14)

axes[0, 1].imshow(seg_rgb)
axes[0, 1].set_title('DeepLIIF Seg', fontsize=14)

axes[0, 2].imshow(marker_gray, cmap='hot')
axes[0, 2].set_title('DeepLIIF Marker (gray)', fontsize=14)
plt.colorbar(axes[0, 2].images[0], ax=axes[0, 2], shrink=0.8)

# Row 2: 检测对比
# 颜色空间检测到的棕色
overlay_color = original_rgb.copy()
overlay_color[color_brown_mask > 0] = [255, 255, 0]  # 黄色标记
axes[1, 0].imshow(overlay_color)
axes[1, 0].set_title('Color-space brown detection (yellow)', fontsize=14)

# DeepLIIF 检测到的阳性
overlay_deepliif = original_rgb.copy()
overlay_deepliif[deepliif_pos] = [255, 0, 0]  # 红色
axes[1, 1].imshow(overlay_deepliif)
axes[1, 1].set_title('DeepLIIF positive (red, thresh=120)', fontsize=14)

# 漏检区域 = 颜色检测到但 DeepLIIF 没有
overlay_missed = original_rgb.copy()
overlay_missed[deepliif_pos] = [0, 255, 0]  # 绿色 = 检测到的
overlay_missed[missed] = [255, 0, 0]        # 红色 = 漏检的
axes[1, 2].imshow(overlay_missed)
axes[1, 2].set_title('Green=detected, RED=MISSED', fontsize=14)

# Row 3: Seg 通道分析
# R+B heatmap
rb_display = np.clip(sum_rb, 0, 255).astype(np.uint8)
im_rb = axes[2, 0].imshow(rb_display, cmap='hot', vmin=0, vmax=300)
axes[2, 0].set_title('Seg R+B value (foreground signal)', fontsize=14)
plt.colorbar(im_rb, ax=axes[2, 0], shrink=0.8)

# G channel
im_g = axes[2, 1].imshow(g_ch_seg.astype(np.uint8), cmap='Greens', vmin=0, vmax=255)
axes[2, 1].set_title('Seg G channel (must be <=80)', fontsize=14)
plt.colorbar(im_g, ax=axes[2, 1], shrink=0.8)

# 漏检区域的 R+B 分布直方图
if len(missed_rb) > 0:
    axes[2, 2].hist(missed_rb, bins=50, color='red', alpha=0.7, label='Missed brown (R+B)')
    axes[2, 2].axvline(x=120, color='blue', linestyle='--', linewidth=2, label='Current thresh=120')
    axes[2, 2].axvline(x=80, color='green', linestyle='--', linewidth=2, label='Suggested thresh=80')
    axes[2, 2].axvline(x=60, color='orange', linestyle='--', linewidth=2, label='Aggressive thresh=60')
    axes[2, 2].set_xlabel('R+B value in Seg')
    axes[2, 2].set_ylabel('Pixel count')
    axes[2, 2].set_title('Missed brown pixels: R+B distribution', fontsize=14)
    axes[2, 2].legend()
else:
    axes[2, 2].text(0.5, 0.5, 'No missed pixels', ha='center', va='center', fontsize=16)

for ax in axes.flat:
    ax.axis('off') if ax != axes[2, 2] else None

plt.suptitle(f'Light Brown Diagnosis: {TILE_NAME}', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig(str(output_dir / f"{TILE_NAME}_diagnosis.png"), dpi=150, bbox_inches='tight')
print(f"\n可视化已保存: {output_dir / f'{TILE_NAME}_diagnosis.png'}")

# ============================================================================
# 5. 额外：生成推荐参数下的效果预览
# ============================================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(20, 14))

configs = [
    (120, 80, "Current: seg=120, G<=80"),
    (80, 80, "seg=80, G<=80"),
    (60, 80, "seg=60, G<=80"),
    (120, 120, "seg=120, G<=120"),
    (80, 100, "seg=80, G<=100"),
    (60, 100, "seg=60, G<=100"),
]

for idx, (seg_t, g_t, title) in enumerate(configs):
    ax = axes2[idx // 3, idx % 3]
    fg = (sum_rb > seg_t) & (g_ch_seg <= g_t)
    pos = fg & (r_ch_seg >= b_ch_seg)
    neg = fg & (r_ch_seg < b_ch_seg)

    overlay = original_rgb.copy()
    # 阳性半透明红色叠加
    red_overlay = overlay.copy()
    red_overlay[pos] = [255, 0, 0]
    overlay = cv2.addWeighted(overlay, 0.5, red_overlay, 0.5, 0)
    # 描边
    pos_uint8 = pos.astype(np.uint8) * 255
    contours, _ = cv2.findContours(pos_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 1)

    ax.imshow(overlay)
    ax.set_title(f'{title}\nPos: {np.sum(pos)} px', fontsize=12)
    ax.axis('off')

plt.suptitle('Parameter Comparison: Red = Positive regions detected', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(str(output_dir / f"{TILE_NAME}_param_comparison.png"), dpi=150, bbox_inches='tight')
print(f"参数对比已保存: {output_dir / f'{TILE_NAME}_param_comparison.png'}")

plt.show()
print("\nDone!")
