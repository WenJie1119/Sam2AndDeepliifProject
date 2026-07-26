# DeepLIIF 分割后处理参数详解

本文档详细解释 DeepLIIF 中影响分割图像输出的所有参数及其使用方法。

---

## 参数概览

| 参数名 | 类型 | 默认值 | 作用 |
|--------|------|--------|------|
| `seg_thresh` | int | 120 | 像素前景/背景阈值 |
| `size_thresh` | int/str | 'default' | 细胞尺寸下限 |
| `size_thresh_upper` | int/None | None | 细胞尺寸上限 |
| `weighted_marker_thresh` | int/None | auto_peak | weighted-points Marker 阳性阈值 |
| `noise_thresh` | int | 4 | 小噪声过滤阈值 |
| `large_noise_thresh` | str/int/None | 'default' | 大噪声过滤阈值 |
| `resolution` | str | '40x' | 显微镜放大倍率 |
| `sam_prompt_mode` | str | weighted-points | SAM2 prompt 流程 |

### 默认 SAM2 Prompt 流程

`--sam-prompt-mode weighted-points` 是当前默认验证流程：

默认会读取 `config/cell_main.json`。这个文件里的字段名使用
argparse 的下划线形式，例如命令行参数 `--weighted-dab-min-intensity`
在 JSON 中写成 `"weighted_dab_min_intensity"`。命令行传入的参数会覆盖
config 中的同名字段。

```text
DeepLIIF Seg/Marker
  -> -5..5 分级并逐像素取最大值
  -> 原图 HED-DAB 强度阈值过滤弱 DAB 泛染
  -> DAPI-dark 空腔 + Seg/Marker 包围结构保护浅 DAB 管壁并补入空腔
  -> 大块弱/中阳性伪影过滤
  -> 小块弱阳性碎片过滤（包含相连的 logit 0）
  -> 孤立小块过滤（包含小强阳性岛）
  -> close、填洞、灰色不确定边界
  -> 256x256 weighted mask
  +  logit 5 连通域中心强阳性点（默认最多 30 个）
  -> SAM2
```

关键参数：

- `--weighted-marker-thresh N`：不设置时逐 tile 使用非零 Marker 强度主峰作为阈值
- `--weighted-marker-max N`：不设置时逐 tile 使用 Marker 最大值
- `--weighted-dab-min-intensity 160`：原图 DAB 归一化强度低于该值的 prompt 像素会被删掉
- `--weighted-dab-strong-support`：原图强 DAB 区域会按强度分成 logit 1-5，并补进 SAM2 weighted mask prompt
- `--no-weighted-dab-filter`：关闭原图 DAB 强度过滤
- `--weighted-dapi-lumen-dark-max 15`：DAPI 灰度小于等于该值时作为无核空腔候选
- `--weighted-dapi-lumen-support-logit-min 1`：Seg/Marker 融合图中作为空腔壁支持的最低 logit
- `--weighted-dapi-lumen-wall-closing-kernel 5`：检测 DAPI 空腔包围关系前对 Seg/Marker 支持做闭合的 kernel
- `--weighted-artifact-min-area 700`
- `--weighted-small-fragment-max-area 100`
- `--weighted-isolated-fragment-max-area 200`
- `--weighted-isolated-fragment-gap 8`
- `--weighted-isolated-fragment-neighbor-min-area 700`
- `--weighted-point-min-area 20`
- `--weighted-max-positive-points 30`
- `--weighted-dab-lumen-macro-closing-kernel 31`：用大尺度 Seg/Marker wall close 识别人眼可见但局部 ring 不完整的空腔
- `--weighted-dab-lumen-macro-min-overlap 0.50`
- `--weighted-dab-lumen-macro-min-wall-ratio 0.30`
- `--weighted-dab-lumen-white-*`：旧近白 RGB 空腔参数保留为兼容字段；当前主流程使用 DAPI-dark 空腔候选
- `--weighted-dab-lumen-max-aspect-ratio 8.0`
- `--no-weighted-artifact-filter`：关闭大块伪影过滤
- `--no-weighted-small-fragment-filter`：关闭小碎片过滤
- `--no-weighted-isolated-fragment-filter`：关闭孤立小块过滤

主流程只保留 weighted-points prompt；逐连通域二值 mask prompt
不再作为可选运行路径。

---

## 详细参数说明

### 1. `seg_thresh` (分割阈值) - 最核心参数

**默认值**: `120`  
**性质**: 固定命令行参数，不会根据当前图像自动计算

**作用**:  
这是 DeepLIIF 后处理中最关键的参数。它决定了分割概率图中的像素如何被分类为前景（细胞）或背景。

程序默认直接使用 `120`；传入 `--seg-thresh N` 时直接使用 `N`。这里没有百分位数、均值或其他逐图像计算。由于判断对象是两个 8-bit 通道之和，`R+B` 的理论范围是 `0-510`。

使用 `--debug-region-um` 时，程序会自动输出阳性 R 强度曲线、非零 Marker 灰度强度曲线、DAPI 灰度强度曲线、原图 HED-DAB 强度曲线，以及对应的保留/移除 mask。阳性 R 曲线聚焦 `150-255`，并标出当前阳性规则下的最小 R；Marker、DAPI、DAB 曲线显示完整的 `0-255` 强度范围。Marker 曲线会标出实际采用的 `weighted_marker_thresh` 和最低保留强度，DAPI 曲线会标出 `weighted_dapi_lumen_dark_max`，DAB 曲线会标出 `weighted_dab_min_intensity`。

Seg 分支保留同时满足前景规则和 `R >= B` 的像素，不再额外按 R 绝对强度过滤。Marker 分支保留 `Marker > weighted_marker_thresh` 的像素，同时强制删除 `Marker < 20` 的弱响应；也就是自动或手动阈值低于 19 时，实际按 `Marker >= 20` 保留。Marker 不要求先通过 Seg 前景判断。最终 prompt 取两者逐像素最大 logit：

```text
pre_dab_raw = max(seg_logits, marker_logits)
```

Marker 因此是 Seg 的补充，并集结果继续进入形态学处理、连通域提取和 SAM2。

**工作原理**:
```
如果 (像素R通道 + 像素B通道) > seg_thresh 且 像素G通道 <= 80:
    如果 R通道 >= B通道:
        标记为阳性细胞 (Positive)
    否则:
        标记为阴性细胞 (Negative)
```

**使用建议**:
- 值越**低** → 检测到更多细胞（敏感度↑，可能包含更多噪声）
- 值越**高** → 检测到更少细胞（特异度↑，可能漏检部分细胞）
- 典型范围: `80-150`

**命令行示例**:
```bash
# 检测更多细胞（敏感）
python pipeline_full_inference.py --seg-thresh 80 ...

# 只保留高置信度细胞（保守）
python pipeline_full_inference.py --seg-thresh 150 ...
```

---

### 2. `size_thresh` (尺寸阈值 - 下限)

**默认值**: `'default'` (自动计算)  
**范围**: `0` 或更大整数，或字符串 `'default'`

**作用**:  
过滤掉**小于**此像素面积的区域，用于去除小噪点或碎片。

**工作原理**:
- 设为 `'default'` 时，系统根据检测到的细胞尺寸分布自动计算合适的阈值
- 自动计算基于 KDE (核密度估计) 找到尺寸分布的第一个谷值
- 根据 `resolution` 参数调整允许范围：
  - 40x: sqrt阈值范围 4-10 (对应面积 16-100)
  - 20x: sqrt阈值范围 3-6 (对应面积 9-36)  
  - 10x: sqrt阈值范围 2-3 (对应面积 4-9)

**使用建议**:
- 保持 `'default'` 通常效果最好
- 如果小碎片太多，手动设置较高值（如 `50`）
- 如果小细胞被误删，设置较低值（如 `10`）

**命令行示例**:
```bash
# 自动计算（推荐）
python pipeline_full_inference.py --size-thresh default ...

# 手动设置最小面积50像素
python pipeline_full_inference.py --size-thresh 50 ...
```

---

### 3. `size_thresh_upper` (尺寸阈值 - 上限)

**默认值**: `None` (无上限)  
**范围**: `None` 或正整数

**作用**:  
过滤掉**大于**此像素面积的区域，用于去除过大的伪影或组织块。

**使用建议**:
- 通常不需要设置，保持 `None`
- 如果图像中有大块伪影被误识别为细胞，可设置上限
- 典型阈值根据放大倍率：40x 约 2000-5000，20x 约 500-1000

**命令行示例**:
```bash
# 过滤掉面积超过3000像素的区域
python pipeline_full_inference.py --size-thresh-upper 3000 ...
```

---

### 4. `weighted_marker_thresh` (Marker 阈值)

**默认值**: 省略该参数时自动使用 two-stage Multi-Otsu，并强制最低保留强度为 `Marker >= 20`<br>
**范围**: `0-255` 整数；不传则自动

**作用**:  
决定 Marker 分支哪些像素进入 weighted-points prompt。超过阈值的 Marker
像素会按强度映射为 `1-5` logit，并和 Seg logit 逐像素取最大值。无论自动还是手动阈值，`Marker < 20` 都会被删除。

**工作原理**:
```
weighted_marker_thresh = two_stage_multi_otsu_threshold  # if omitted
effective_thresh = max(weighted_marker_thresh, 19)
marker_positive = Marker > effective_thresh  # equivalent to Marker >= 20 when floor applies
pre_dab_raw = max(seg_logits, marker_logits)
```

**使用建议**:
- 较**低**值 → 更多细胞被判定为阳性
- 较**高**值 → 更少细胞被判定为阳性

**命令行示例**:
```bash
# 手动设置阳性阈值
python -m cell.main --weighted-marker-thresh 180 ...
```

---

### 5. `noise_thresh` (噪声阈值)

**默认值**: `4`  
**范围**: `0` 或更大整数

**作用**:  
过滤掉像素数**小于等于**此值的微小区域。这是最基础的噪声过滤。

**与 size_thresh 的区别**:
- `noise_thresh`: 在连通组件分析阶段应用，过滤最基本的噪点 (默认 4 像素)
- `size_thresh`: 在后续细胞分类阶段应用，基于统计分析计算

**使用建议**:
- 保持默认值 `4` 通常足够
- 如果图像噪声严重，可适当增加

---

### 6. `large_noise_thresh` (大噪声阈值) ⭐ 新增

**默认值**: `'default'` (根据 resolution 自动设置)  
**范围**: `'default'`、`'none'` (无上限)、或正整数

**作用**:  
过滤掉像素数**大于等于**此值的超大区域。用于去除可能被错误检测的大块组织。

**自动值 (当设为 'default')**:
- 40x: `16000` 像素
- 20x: `4000` 像素
- 10x: `1000` 像素

**与 size_thresh_upper 的区别**:
- `large_noise_thresh`: 在连通组件阶段应用（更早）
- `size_thresh_upper`: 在细胞分类阶段应用（更晚）

**命令行示例**:
```bash
# 自动根据resolution设置
python pipeline_full_inference.py --large-noise-thresh default ...

# 不设上限
python pipeline_full_inference.py --large-noise-thresh none ...

# 手动设置
python pipeline_full_inference.py --large-noise-thresh 10000 ...
```

---

### 7. `resolution` (分辨率/放大倍率)

**默认值**: `'40x'`  
**可选值**: `'10x'`、`'20x'`、`'40x'`

**作用**:  
影响多个阈值的自动计算，因为不同放大倍率下细胞的像素尺寸不同。

**受影响的参数**:
- `size_thresh` 的自动计算范围
- `large_noise_thresh` 的自动值

---

## 参数处理流程图

```text
DeepLIIF Seg + Marker
       ↓
seg_thresh / auto-or-fixed weighted_marker_thresh → Seg logit + Marker logit
       ↓
pre_dab_raw = max(seg_logits, marker_logits)
       ↓
DAB 强度过滤 + DAPI-dark 空腔保护 + DAB 强支持补充
       ↓
artifact / small fragment / isolated fragment 过滤
       ↓
repair + lumen fill + positive points
       ↓
SAM2 weighted mask + points → instance mask
```

---

## 完整命令行示例

```bash
# 使用所有默认参数
python pipeline_full_inference.py \
    --input-dir /path/to/images \
    --output-dir /path/to/output

# 自定义所有后处理参数
python pipeline_full_inference.py \
    --input-dir /path/to/images \
    --output-dir /path/to/output \
    --resolution 40x \
    --seg-thresh 100 \
    --size-thresh default \
    --size-thresh-upper 5000 \
    --weighted-marker-thresh 180 \
    --noise-thresh 4 \
    --large-noise-thresh default

# 更敏感的检测（检测更多细胞）
python pipeline_full_inference.py \
    --input-dir /path/to/images \
    --output-dir /path/to/output \
    --seg-thresh 80 \
    --size-thresh 10 \
    --weighted-marker-thresh 150

# 更保守的检测（只保留高置信细胞）
python pipeline_full_inference.py \
    --input-dir /path/to/images \
    --output-dir /path/to/output \
    --seg-thresh 150 \
    --size-thresh 100 \
    --weighted-marker-thresh 200
```

---

## 参数调优建议

1. **先调 `seg_thresh`**: 这是影响最大的参数
2. **再调 `size_thresh`**: 处理小噪点问题
3. **最后调 `weighted_marker_thresh`**: 需要覆盖自动主峰阈值时，再手动微调 Marker 分支进入 prompt 的范围

> [!TIP]
> 使用 `--no-postprocessing` 可以跳过后处理，只获取原始 Seg 输出，用于调试或自定义后处理流程。
