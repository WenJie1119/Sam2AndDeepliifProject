# DeepLIIF Center-Valid Stitching Plan

本文档记录当前推荐方案：先稳定 DeepLIIF 产生的阳性 seed，再继续使用 SAM2 做边界细化。目标是解决相邻 tile 在 overlap 区域原图一致、但 DeepLIIF `Seg` 和 `step3_3_seg_positive.png` 不一致的问题。

## 1. 当前问题

当前 pipeline 的主要流程是：

```text
WSI 按 512x512 tile 读取，tile 间 overlap=128
    -> 每个 tile 独立跑 DeepLIIF
    -> 每个 tile 独立从 Seg/Marker 提取阳性 seed
    -> 每个 tile 独立喂给 SAM2
    -> tile mask 再拼接/跨 tile merge
```

问题出现在 SAM2 之前。

以这两个相邻 tile 为例：

```text
tile_39_12_4608_14976
tile_39_13_4992_14976
```

参数：

```text
tile_size = 512
overlap = 128
stride = 512 - 128 = 384
```

理论 overlap 区域是：

```text
tile_39_12 的 x = 384..512
tile_39_13 的 x = 0..128
```

已经验证过：

```text
原图 overlap: 65536 / 65536 像素完全一致
DeepLIIF Seg overlap: 约 94.7% 像素 RGB 不同
step3_3_seg_positive overlap: 二值阳性 IoU 约 0.28
```

结论：

```text
切片坐标没有错。
不一致来自 DeepLIIF 对每个 512 tile 独立推理时的边界上下文差异。
```

同一块物理区域在左 tile 里处于右边界，在右 tile 里处于左边界。模型看到的上下文不同，卷积 padding 和感受野不同，导致 `Seg` 输出不一致。后续 `compute_posneg_mask()` 再按 `R >= B` 和前景阈值提阳性，因此 `step3_3_seg_positive.png` 也不一致。

相关代码位置：

```text
DeepLIIF batch 推理:
cd34_pipeline/deepliif/batch_inference.py

阳性/阴性像素提取:
cd34_pipeline/cell/extraction.py

debug step3_3_seg_positive 可视化:
cell/debug_vis.py

WSI tile 读取:
cd34_pipeline/io/wsi_reader.py
```

## 2. 推荐方向

当前最应该先做：

```text
DeepLIIF 仍然用 512x512 输入，
但每个 tile 只保留中心 384x384 有效预测，
把这些中心有效区拼成 ROI/global 级 Seg 和 Marker，
再从拼好的 Seg/Marker 上提取阳性 seed。
```

不要优先微调 SAM2。原因是 SAM2 当前主要吃 DeepLIIF 产生的 prompt/seed。如果上游 prompt 已经不一致，SAM2 微调只能改善“给定 prompt 后的边界”，不能可靠解决“一个 tile 有 prompt、另一个 tile 没 prompt”的问题。

## 3. 核心概念

### 3.1 原始 overlap 不丢

这个方案不是把 overlap 区域全丢掉。

准确说：

```text
不要使用某个 tile 自己的边缘预测，
但物理位置上的 overlap 区域仍然会由邻居 tile 的中心有效区覆盖。
```

以水平相邻 tile 为例：

```text
tile A: x = 4608..5120
tile B: x = 4992..5504

原始 overlap:
x = 4992..5120，长度 128
```

如果设置：

```text
halo = overlap / 2 = 64
valid_size = 512 - 2 * 64 = 384
```

则：

```text
tile A 有效区: x = 4672..5056
tile B 有效区: x = 5056..5440
```

原始 overlap `4992..5120` 被分配为：

```text
4992..5056 由 tile A 的中心有效区负责
5056..5120 由 tile B 的中心有效区负责
```

所以 overlap 区没有丢，只是不再由两个 tile 重复预测和互相冲突。

### 3.2 最终每个全局像素只有一个来源

当前做法在 overlap 区有两个预测来源：

```text
同一全局像素
    -> tile A 的边缘预测
    -> tile B 的边缘预测
```

新方案改为：

```text
同一全局像素
    -> 只来自一个 tile 的中心有效区
```

这样不会再出现“同一 overlap 区两个阳性图不一致后再纠结用哪个”的问题。

### 3.3 tile 边缘只作为上下文

每个 512 输入仍然完整送进 DeepLIIF。被丢弃的是输出里的边缘预测，不是输入像素。

```text
512x512 输入:
[64 halo][384 valid][64 halo]

DeepLIIF 看完整 512x512。
最终只保留中间 384x384。
```

## 4. 目标流程

建议先只在 debug region 上实现和验证，不要一开始改全片主流程。

目标流程：

```text
1. 按当前 tile grid 读取 debug region 相关 tiles，额外带一圈邻居。
2. 每个 512 tile 跑 DeepLIIF，得到 Seg 和 Marker。
3. 从每个 Seg/Marker 裁出中心 384x384。
4. 按 tile 的全局 target-level 坐标，把中心有效区贴到 ROI/global canvas。
5. 得到 stitched Seg 和 stitched Marker。
6. 在 stitched Seg/Marker 上统一提取阳性 seed。
7. 把全局 seed 裁回每个标准 tile 的局部坐标。
8. SAM2 继续按 tile 跑，但 prompt 来自 stitched seed，而不是每个 tile 自己的 DeepLIIF 结果。
9. 最终 SAM2 mask 也建议只使用中心有效区或 owner rule，再做跨 tile merge/export。
```

## 5. 详细算法

### 5.1 输入

每个 tile 需要有：

```text
row
col
x
y
x_level0
y_level0
actual_w
actual_h
```

其中：

```text
x, y 是 target-level 坐标
x_level0, y_level0 是 level-0 坐标
```

当前 `WSIReader.get_tile_filename()` 生成的文件名形如：

```text
tile_{row}_{col}_{x}_{y}
```

### 5.2 有效区定义

先使用固定参数：

```text
tile_size = 512
overlap = 128
stride = 384
halo = 64
valid_size = 384
```

对非边界 tile：

```text
valid crop in local tile:
x0 = 64
y0 = 64
x1 = 448
y1 = 448
```

贴回全局坐标：

```text
global_x0 = tile.x + 64
global_y0 = tile.y + 64
global_x1 = tile.x + 448
global_y1 = tile.y + 448
```

### 5.3 ROI/debug region 边界处理

如果只处理一个 debug bbox，不能简单丢 debug bbox 最外圈的 64 px，否则会出现空洞。

推荐：

```text
debug bbox 选中的 core tiles 外再加一圈 neighbor tiles。
先用 core + neighbor 拼接出更大的 canvas。
最后再裁回用户真正关心的 debug bbox 或 core tile 范围。
```

当前 `enumerate_debug_region_tiles()` 已经支持 core tile 加 neighbor tile。可以复用这个思路。

如果是整张 WSI 的真实边界，没有外侧邻居，则可以：

```text
1. 保留最外边缘预测；或
2. 用 padding/mirror 方式补上下文后再裁回真实范围。
```

第一阶段建议先对 debug region 用 neighbor tiles，避免处理真实边界问题。

### 5.4 Canvas 构建

给定一组 tiles，先确定 canvas 范围：

```text
min_x = min(tile.x + valid_x0)
min_y = min(tile.y + valid_y0)
max_x = max(tile.x + valid_x1)
max_y = max(tile.y + valid_y1)
```

创建：

```text
seg_canvas    shape = (max_y - min_y, max_x - min_x, 3)
marker_canvas shape = (max_y - min_y, max_x - min_x, 3)
owner_canvas  shape = (max_y - min_y, max_x - min_x)
```

然后对每个 tile：

```text
seg_valid = seg_np[64:448, 64:448]
marker_valid = marker_np[64:448, 64:448]

dst_x0 = tile.x + 64 - min_x
dst_y0 = tile.y + 64 - min_y

seg_canvas[dst_y0:dst_y0+384, dst_x0:dst_x0+384] = seg_valid
marker_canvas[dst_y0:dst_y0+384, dst_x0:dst_x0+384] = marker_valid
owner_canvas[...] = tile_id
```

正常情况下，使用 `valid_size = stride = 384` 后，中心有效区之间应该无重叠、无空洞。

### 5.5 从 stitched Seg/Marker 提取阳性

先复用现有逻辑：

```text
cd34_pipeline.cell.extraction.extract_connected_positive_regions()
```

但输入从单 tile 的 `seg_np/marker_np` 改成：

```text
stitched_seg_np
stitched_marker_np
```

建议先保持参数不变：

```text
seg_thresh = 当前 CLI 参数
marker_percentile_factor = 当前 CLI 参数
morphology_kernel = 当前 CLI 参数
min_mask_area = 当前 CLI 参数
```

### 5.6 Marker 阈值改为 ROI 统一

当前 `marker_thresh=None` 时，每个 tile 会自己计算 marker threshold。这会产生另一个不一致来源。

新方案建议：

```text
如果处理 stitched ROI，则只在 stitched_marker 上计算一次 marker_thresh。
同一个 ROI 内所有阳性提取使用同一个 marker_thresh。
```

第一阶段可以直接让 `extract_connected_positive_regions()` 在 stitched marker 上自动计算一次。后续如果需要裁回 tile 后再局部处理，必须把这次算出的阈值传回 tile 级逻辑，不要重新逐 tile 计算。

### 5.7 全局 seed 裁回 tile

在 stitched canvas 上得到的 `regions_info` 坐标是 canvas 局部坐标。

需要转换为 target-level 全局坐标：

```text
global_row = canvas_row + min_y
global_col = canvas_col + min_x
```

对每个标准 tile，取与 tile box 相交的 region 像素：

```text
tile_box:
x = tile.x .. tile.x + tile_size
y = tile.y .. tile.y + tile_size

region global coords 落在 tile_box 内的部分:
local_row = global_row - tile.y
local_col = global_col - tile.x
```

生成该 tile 的 `positive_cells_info`：

```text
coords: tile-local coords
center: tile-local center
pixel_count
marker_sum / marker_max / marker_mean / marker_min
is_positive = True
global_id 或 original_global_id
```

注意：

```text
同一个 global region 可能跨多个 tile。
分配回 tile 时，可以在多个 tile 中都有该 region 的局部片段，
但最终跨 tile stitch/merge 要能用 global_id 或 overlap matching 合并。
```

第一阶段也可以更保守：

```text
只把一个 global region 分配给其中心所在的 owner tile。
```

这样重复更少，但如果 region 跨 tile 边界，SAM2 可能只能在一个 tile 里看到完整性不足。更稳妥的是允许跨 tile 分配，然后靠后续 stitch merge 合并。

## 6. SAM2 部分建议

第一阶段不建议微调 SAM2。

先做以下改动：

```text
SAM2 的 prompt 来源改为 stitched Seg/Marker 提取出的 global seed。
不要再使用每个 tile 独立 DeepLIIF 输出提 prompt。
```

之后可以尝试 prompt 增强：

```text
mask_input + box prompt
mask_input + positive center point
mask_input + negative points
```

代码中已有可参考入口：

```text
cd34_pipeline/sam2_wrapper/inference.py
run_sam2_merged_box_mask()
run_sam2_mask_with_point()
```

对 CD34 场景，negative points 很有价值。可从 seed 外围的非 DAB 棕色区域、背景、蓝色核区域采样，抑制 SAM2 外扩。

## 7. 推荐实施顺序

### Step 1: 做 debug-only stitched DeepLIIF 输出

新增或扩展 debug region 流程，输出：

```text
debug_region/08_stitched_deepliif_seg.png
debug_region/09_stitched_deepliif_marker.png
debug_region/10_stitched_seg_positive.png
debug_region/11_stitched_combined_positive.png
debug_region/12_stitched_positive_regions.png
```

这一阶段不改 SAM2，只验证 DeepLIIF seed 是否稳定。

### Step 2: 对比原 tile overlap 和 stitched 裁图 overlap

从 stitched Seg/Marker 裁回：

```text
tile_39_12 的 512 视图
tile_39_13 的 512 视图
```

再比较：

```text
tile_39_12 x=384..512
tile_39_13 x=0..128
```

验收目标：

```text
stitched step3_3_seg_positive overlap IoU 接近 1.0
stitched combined_positive overlap IoU 接近 1.0
```

如果为了避免空洞采用 owner rule，overlap 裁图理论上应来自同一张 stitched canvas，因此应该完全一致或接近完全一致。

### Step 3: 用 stitched seed 驱动 SAM2

将 producer 中的逻辑从：

```text
单 tile DeepLIIF -> 单 tile positive_cells_info
```

改为 debug mode 下：

```text
debug ROI stitched Seg/Marker -> global positive regions -> per-tile positive_cells_info
```

然后继续走现有：

```text
SAM2Processor.segment() 或 segment_batch()
PostProcessor.merge_and_process()
region_debug visualizations
```

### Step 4: SAM2 输出也做中心有效区或 owner rule

为了与 DeepLIIF 一致，SAM2 的最终 tile mask 也建议只使用中心有效区：

```text
tile internal area:
keep x=64..448, y=64..448
discard internal border output
```

debug region 最外圈仍由 neighbor tiles 保障覆盖，最终裁回 core/debug bbox。

### Step 5: 再考虑全片流程

debug region 验证通过后，再推广到全片 ROI。

全片不能一次性把所有 stitched Seg/Marker 放进内存时，可以按 super-tile/block 处理：

```text
每个 block 包含若干 tile，例如 8x8 或 16x16 tile。
block 外再加一圈 neighbor。
在 block 内做 center-valid DeepLIIF stitch。
只输出 block 的核心区域。
```

这等价于把 center-valid 方案提升到更大的分块层级。

## 8. 验收指标

最重要的指标：

```text
相邻 tile overlap 的 step3_3_seg_positive IoU
相邻 tile overlap 的 combined_positive IoU
```

当前问题样例：

```text
step3_3_seg_positive IoU 约 0.28
```

阶段目标：

```text
debug region 中同一 pair 的 IoU > 0.8
```

理想目标：

```text
如果两个 tile 都从同一 stitched canvas 裁出，overlap 应接近 1.0。
```

其他指标：

```text
SAM2 raw mask 是否减少 tile 边界断裂
region stitch_matches 中低质量 overlap match 是否减少
错误 union 是否减少
最终 GeoJSON 是否减少巨大异常 polygon
DAB filter 后区域数量是否更稳定
```

## 9. 风险和注意点

### 9.1 不要把大 ROI resize 成 512

DeepLIIF 模型输入可以是 512 patch，但不能把几千像素 ROI 直接 resize 到 512 再推理。这会改变组织尺度，影响细胞大小和染色结构。

正确做法：

```text
大 ROI 内部仍按 512 patch 推理。
拼接输出时只保留中心有效区。
```

### 9.2 debug bbox 必须加邻居

如果只处理 core tile，然后每个 tile 丢 64 px 边缘，debug bbox 外圈会缺预测。

必须：

```text
core tiles + neighbor ring
拼接后裁回 core/debug bbox
```

### 9.3 Global region ID 要保留

从 stitched mask 裁回 tile 后，建议保留：

```text
global_region_id
global_bbox
global_center
```

这样后续跨 tile merge 时可以更容易判断同一个阳性结构，而不是完全依赖 overlap 像素匹配。

### 9.4 Marker 阈值不要重复逐 tile 计算

如果 stitched 阶段已经算过 ROI-level marker threshold，裁回 tile 后不要再次用 tile-local marker 自动阈值，否则又会引入不一致。

### 9.5 SAM2 不是阳性分类器

SAM2 适合做边界 refinement，不适合作为“发现 CD34 阳性区域”的唯一模块。阳性区域的发现仍然要靠 DeepLIIF/DAB/专用 seed 模型。

## 10. 暂不优先做的方案

### 10.1 先微调 SAM2

暂不优先。当前根因在 SAM2 之前：

```text
DeepLIIF Seg overlap 已经不一致
阳性 seed overlap 已经不一致
```

微调 SAM2 只有在以下条件满足后才更有意义：

```text
1. prompt 来源已经稳定；
2. SAM2 看到合理 prompt 后仍然边界质量不够；
3. 有足够人工标注的 CD34 mask；
4. 训练时能模拟 DeepLIIF noisy seed。
```

如果以后要做 SAM2 微调，推荐目标是：

```text
给定 noisy DeepLIIF mask prompt，
输出更接近人工标注的 CD34 阳性区域。
```

训练上建议：

```text
冻结 image encoder，只训练 mask decoder 或 LoRA/adapter。
prompt 使用 DeepLIIF seed，并随机腐蚀、膨胀、断裂、平移、漏检。
loss 使用 Dice + BCE，可按小目标情况加 focal/Tversky。
验证必须包含 overlap consistency 指标。
```

## 11. 下一会话建议任务清单

建议按这个顺序开工：

```text
1. 新增 debug-only center-valid DeepLIIF stitching 工具函数。
2. 在 debug_region_um 流程里调用它，先只输出 stitched Seg/Marker/positive 可视化。
3. 写一个 overlap 对比脚本或函数，比较原始 tile 独立输出 vs stitched 裁回输出。
4. 验证 tile_39_12 和 tile_39_13 的 overlap IoU 是否明显提升。
5. 再把 stitched positive regions 转回 per-tile positive_cells_info。
6. 接入 SAM2Processor，跑同一个 debug region。
7. 对比 region_debug 的 mosaic、merge diff、stitch_matches。
```

第一阶段验收通过前，不建议直接改全片主流程。
