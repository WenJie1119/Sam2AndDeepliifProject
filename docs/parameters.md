# DeepLIIF 分割后处理参数详解

本文档详细解释 DeepLIIF 中影响分割图像输出的所有参数及其使用方法。

---

## 参数概览

| 参数名 | 类型 | 默认值 | 作用 |
|--------|------|--------|------|
| `seg_thresh` | int | 120 | 像素前景/背景阈值 |
| `size_thresh` | int/str | 'default' | 细胞尺寸下限 |
| `size_thresh_upper` | int/None | None | 细胞尺寸上限 |
| `marker_thresh` | int/None | None (auto) | 标记物阳性阈值 |
| `noise_thresh` | int | 4 | 小噪声过滤阈值 |
| `large_noise_thresh` | str/int/None | 'default' | 大噪声过滤阈值 |
| `resolution` | str | '40x' | 显微镜放大倍率 |

---

## 详细参数说明

### 1. `seg_thresh` (分割阈值) - 最核心参数

**默认值**: `120`  
**范围**: `0-255`

**作用**:  
这是 DeepLIIF 后处理中最关键的参数。它决定了分割概率图中的像素如何被分类为前景（细胞）或背景。

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

### 4. `marker_thresh` (标记物阈值)

**默认值**: `None` (自动计算，约为标记物强度分布 99 百分位的 90%)  
**范围**: `None`、`'auto'` 或 `0-255` 整数

**作用**:  
决定细胞阳性/阴性分类。如果某细胞区域内的**最大标记物像素值**超过此阈值，该细胞将被重新分类为**阳性**。

**工作原理**:
```
最终分类 = 原始阳性判定 OR (细胞内最大Marker值 > marker_thresh)
```

**使用建议**:
- `None` 或不设置：使用自动计算的阈值（推荐）
- 较**低**值 → 更多细胞被判定为阳性
- 较**高**值 → 更少细胞被判定为阳性

**命令行示例**:
```bash
# 手动设置阳性阈值
python pipeline_full_inference.py --marker-thresh 180 ...
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

```
原始分割图 (Seg)
       ↓
   seg_thresh → 创建阳性/阴性掩码
       ↓
   mark_background → 填充背景
       ↓
   noise_thresh + large_noise_thresh → 连通组件分析，过滤极小和极大区域
       ↓
   size_thresh → 过滤小细胞
       ↓
   size_thresh_upper → 过滤大细胞
       ↓
   marker_thresh → 重新分类阳性/阴性
       ↓
输出: SegRefined + SegOverlaid + Scoring
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
    --marker-thresh 180 \
    --noise-thresh 4 \
    --large-noise-thresh default

# 更敏感的检测（检测更多细胞）
python pipeline_full_inference.py \
    --input-dir /path/to/images \
    --output-dir /path/to/output \
    --seg-thresh 80 \
    --size-thresh 10 \
    --marker-thresh 150

# 更保守的检测（只保留高置信细胞）
python pipeline_full_inference.py \
    --input-dir /path/to/images \
    --output-dir /path/to/output \
    --seg-thresh 150 \
    --size-thresh 100 \
    --marker-thresh 200
```

---

## 参数调优建议

1. **先调 `seg_thresh`**: 这是影响最大的参数
2. **再调 `size_thresh`**: 处理小噪点问题
3. **最后调 `marker_thresh`**: 微调阳性/阴性分类

> [!TIP]
> 使用 `--no-postprocessing` 可以跳过后处理，只获取原始 Seg 输出，用于调试或自定义后处理流程。
