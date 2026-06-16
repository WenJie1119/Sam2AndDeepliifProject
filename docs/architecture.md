# `sample_batch_pipeline.py` 数据结构与代码架构说明

本文档从“面向对象协作”的角度解释 `sample_batch_pipeline.py` 的数据结构、核心类职责和五阶段处理流程。这个脚本实现的是一个 CD34 WSI 批处理管线，整体采用 Producer-Consumer 架构：

- `Producer` 线程负责前半段：YOLO 处理、DeepLIIF 处理、细胞提取。
- `Bucket` 作为线程间缓冲区，只保存 SAM2 阶段真正需要的数据。
- `Consumer` 线程负责后半段：SAM2 处理、后处理合并与保存。
- `main()` 负责参数解析、WSI 切片枚举、ROI 过滤、线程创建和最终统计。

## 1. 总体架构

脚本的主架构可以理解为几个对象之间的协作：

| 对象 | 类型 | 职责 |
| --- | --- | --- |
| `Bucket` | 队列控制对象 | 在生产者和消费者之间缓存待 SAM2 处理的数据，并通过水位线控制补货 |
| `BucketItem` | 数据载体对象 | 保存一个 tile 进入 SAM2 前所需的最小数据 |
| `Producer` | 生产者对象 | 读取 WSI tile，执行 YOLO、DeepLIIF、细胞提取，并把结果放入 `Bucket` |
| `Consumer` | 消费者对象 | 从 `Bucket` 取出 `BucketItem`，执行 SAM2 分割、mask 合并和结果保存 |
| `AsyncSaver` | 异步保存对象 | 在后台线程写入 `.npy` mask，避免阻塞 SAM2 推理 |
| `StickyProgress` | 进度显示对象 | 在终端底部展示 YOLO/SAM2/桶库存等进度信息 |
| `main()` | 编排入口 | 初始化参数、模型运行环境、WSI、tile 列表、线程和统计信息 |

整体数据流如下：

```text
WSIReader.enumerate_tiles()
        |
        v
tile_info 列表
        |
        v
Producer
  1. YOLO 过滤目标 tile
  2. DeepLIIF 生成 Seg / Marker
  3. 细胞提取生成 positive_cells_info / clusters
        |
        v
BucketItem
        |
        v
Bucket
        |
        v
Consumer
  4. SAM2 批量分割
  5. 后处理合并 mask 并异步保存
```

## 2. 核心数据结构

### 2.1 `args`

`args` 来自 `parse_args()`，本质上是一个运行配置对象。它控制输入输出路径、模型路径、设备、tile 参数、batch size、缓存开关和调试模式。

主要字段包括：

| 字段 | 作用 |
| --- | --- |
| `wsi_path` | 输入 WSI 文件路径 |
| `output_dir` | 输出目录 |
| `roi_csv` | ROI 过滤 CSV，格式为 `filename,x,y,width,height` |
| `deepliif_model_dir` | DeepLIIF 模型目录 |
| `sam_checkpoint` | SAM2 checkpoint |
| `sam_config` | SAM2 配置文件 |
| `yolo_model_path` | YOLO 分类模型路径 |
| `device` | 模型运行设备，例如 `cuda:0` |
| `tile_size` | tile 尺寸，默认 512 |
| `target_magnification` | 目标倍率，默认 40x |
| `overlap` | tile 重叠大小 |
| `yolo_batch_size` | YOLO 批大小 |
| `deepliif_batch_size` | DeepLIIF 批大小 |
| `sam2_batch_size` | SAM2 prompt 批大小 |
| `bucket_capacity` | 每轮补货处理的 tile 数 |
| `bucket_watermark` | 桶低水位比例 |
| `cache_deepliif` | 是否缓存 DeepLIIF 输出 |
| `cache_sam2` | 是否缓存 SAM2 原始 mask |

### 2.2 `tile_info`

`tile_info` 是 `WSIReader.enumerate_tiles()` 返回列表中的单个元素，类型是 `dict`。脚本中直接使用的字段包括：

| 字段 | 作用 |
| --- | --- |
| `row` | tile 所在行 |
| `col` | tile 所在列 |
| `x_level0` | tile 在 level-0 坐标系中的 x 坐标 |
| `y_level0` | tile 在 level-0 坐标系中的 y 坐标 |
| `actual_w` | tile 在目标 level 下的实际宽度 |
| `actual_h` | tile 在目标 level 下的实际高度 |

`tile_info` 在整个流程中承担“空间定位对象”的角色。它不仅用于读取 WSI tile，也用于生成 `tile_name`，最终输出的 mask 文件名也依赖它。

### 2.3 ROI 数据结构

`load_roi_csv()` 将 CSV 文件转换为 ROI 字典列表：

```python
{
    "x": int,
    "y": int,
    "w": int,
    "h": int,
}
```

这些坐标都在 level-0 坐标系下。`filter_tiles_by_roi()` 会把 `tile_info` 的范围换算到 level-0，与 ROI 矩形做重叠判断，只保留与 ROI 相交的 tile。

### 2.4 `Bucket`

`Bucket` 是生产者和消费者之间的固定容量队列。它封装了：

- 一个 `queue.Queue(maxsize=capacity)`。
- 一个 `Condition`，用于控制生产者睡眠和唤醒。
- 一个 `_need_refill` 标记，用于表示当前是否需要补货。
- 一个 `watermark_level`，当库存低于该值时唤醒生产者。

关键方法：

| 方法 | 调用方 | 作用 |
| --- | --- | --- |
| `wait_until_need_refill()` | `Producer` | 等待桶库存低于水位线 |
| `put(item)` | `Producer` | 放入一个 `BucketItem`，桶满时阻塞 |
| `mark_refill_done()` | `Producer` | 一轮补货完成后清除补货标记 |
| `get(timeout=0.5)` | `Consumer` | 取出一个 `BucketItem`，取完后检查是否要唤醒生产者 |
| `qsize()` / `empty()` / `full()` | 监控逻辑 | 查看桶状态 |

这个对象把线程同步逻辑集中起来，使 `Producer` 和 `Consumer` 不需要直接操作底层锁。

### 2.5 `BucketItem`

`BucketItem` 是一个 `dataclass`，也是前半段和后半段之间最关键的数据结构：

```python
@dataclass
class BucketItem:
    tile_np: np.ndarray
    clusters: list
    positive_cells_info: list
    tile_info: dict
    tile_name: str
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `tile_np` | RGB tile 图像，形状通常为 `(H, W, 3)` |
| `clusters` | 细胞提示点或点簇列表，每个元素通常是 `(N, 2)` 坐标数组 |
| `positive_cells_info` | 阳性细胞连通域信息，由细胞提取模块产生 |
| `tile_info` | 原始 tile 的空间位置信息 |
| `tile_name` | tile 文件名，例如 `tile_5_12_5632_2048` |

脚本刻意让 `BucketItem` 只保存 SAM2 需要的数据，不保存 DeepLIIF 的 `Seg` 和 `Marker` 图像。这样可以降低队列内存占用。

### 2.6 DeepLIIF 输出结构

DeepLIIF 的批处理结果是一个列表，每个元素是一个字典：

```python
{
    "Seg": PIL.Image,
    "Marker": PIL.Image,
}
```

脚本会将其转换为：

```python
seg_np = np.array(seg_img)
marker_np = np.array(marker_img)
```

随后这两个数组进入细胞提取阶段。如果启用 `--cache-deepliif`，则保存到：

```text
<output-dir>/cache/deepliif/<tile_name>.npy
```

### 2.7 SAM2 输出结构

`run_sam2_segmentation_batch()` 返回：

```python
sam_mask, scores, filtered = run_sam2_segmentation_batch(...)
```

其中：

| 字段 | 作用 |
| --- | --- |
| `sam_mask` | SAM2 生成的原始分割 mask |
| `scores` | 每个 mask 的置信度或评分 |
| `filtered` | 被过滤掉的候选结果 |

后处理阶段会调用：

```python
sam_mask_merged, _, _, _ = merge_connected_masks(...)
```

得到最终实例 mask。若 `np.max(sam_mask_merged) > 0`，则异步保存到：

```text
<output-dir>/npy_masks/<tile_name>.npy
```

## 3. 五阶段处理流程

## 第一步：YOLO 处理

YOLO 处理发生在 `Producer._run_impl()` 中，由 `TileClassifier` 负责。

### 参与对象

| 对象 | 职责 |
| --- | --- |
| `WSIReader` | 根据 `tile_info` 从 WSI 中读取 tile 图像 |
| `TileClassifier` | 对 tile 批量分类，判断是否为目标区域 |
| `Producer` | 组织 tile 预取、调用 YOLO、筛选目标 tile |

### 输入数据

YOLO 阶段的输入是：

- `chunk_tiles`：一批 `tile_info`。
- `prefetched_images`：与 `chunk_tiles` 对齐的一批 PIL tile 图像。

`Producer` 使用 `_prefetch_tiles()` 通过 `ThreadPoolExecutor` 并行读取 tile。读取完成后调用：

```python
classified = classifier.classify_tiles_with_images(
    chunk_tiles, prefetched_images
)
target_tiles = TileClassifier.get_target_tiles(classified)
```

### 输出数据

YOLO 阶段输出：

- `target_tiles`：被 YOLO 判定为目标区域的 `tile_info` 列表。

如果当前 chunk 没有任何目标 tile，`Producer` 会跳过 DeepLIIF 和细胞提取，直接进入下一轮补货。

### 架构意义

YOLO 在这里承担“粗筛对象”的角色。它先快速过滤掉背景 tile，避免 DeepLIIF 和 SAM2 对大量无效区域做昂贵推理。

## 第二步：DeepLIIF 处理

DeepLIIF 处理也发生在 `Producer._run_impl()` 中，由 `DeepLIIFBatchInference` 负责。

### 参与对象

| 对象 | 职责 |
| --- | --- |
| `DeepLIIFBatchInference` | 对目标 tile 执行批量 DeepLIIF 推理 |
| `Producer` | 按 `deepliif_batch_size` 分批组织输入和处理输出 |

### 输入数据

DeepLIIF 阶段的输入是 YOLO 输出的 `target_tiles`。脚本通过 `tile_image_map` 复用 YOLO 阶段已经读取过的 PIL 图像，避免重复读取 WSI：

```python
tile_image_map = {
    id(t): img
    for t, img in zip(chunk_tiles, prefetched_images)
}
```

每个 DeepLIIF batch 中：

```python
tile_pils = [tile_image_map[id(t)] for t in batch_tiles]
tile_nps = [np.array(p) for p in tile_pils]
```

随后调用：

```python
deepliif_results_list = deepliif_engine.inference_batch(
    tile_pils,
    batch_size=deepliif_batch_size,
    resolution=args.resolution,
)
```

### 输出数据

每个 tile 的 DeepLIIF 输出是：

- `Seg`：分割相关图像。
- `Marker`：阳性标记相关图像。

脚本将它们转换为 NumPy 数组：

```python
seg_np = np.array(seg_img)
marker_np = np.array(marker_img)
```

### 架构意义

DeepLIIF 在这里承担“语义增强对象”的角色。它不直接输出最终 mask，而是生成后续细胞提取所需的中间表示：`Seg` 和 `Marker`。

## 第三步：细胞提取

细胞提取仍然位于 `Producer._run_impl()` 中，由 `cd34_pipeline.cell.extraction` 模块提供函数。

### 参与对象

| 对象或函数 | 职责 |
| --- | --- |
| `extract_connected_positive_regions()` | 从 `Seg` 和 `Marker` 中提取阳性细胞连通域 |
| `get_clusters_from_cells()` | 将细胞信息转换为 SAM2 prompt 所需的点簇 |
| `BucketItem` | 将 tile 图像、细胞信息和 prompt 数据封装起来 |
| `Bucket` | 保存待 SAM2 消费的数据 |

### 输入数据

细胞提取阶段的输入是：

- `seg_np`
- `marker_np`
- 阈值和形态学参数，例如 `seg_thresh`、`marker_thresh`、`morphology_kernel`、`min_mask_area`

调用方式：

```python
positive_cells_info = extract_connected_positive_regions(
    seg_np,
    marker_np,
    seg_thresh=args.seg_thresh,
    marker_thresh=args.marker_thresh,
    morphology_kernel=args.morphology_kernel,
    min_area=args.min_mask_area,
)
```

### 中间数据

`positive_cells_info` 是阳性细胞连通域信息列表。具体字段由 `cd34_pipeline.cell.extraction` 定义；在本脚本中，它被当作一个细胞区域描述列表使用，后续传给：

- `get_clusters_from_cells()`
- `merge_connected_masks()`

随后生成：

```python
clusters = get_clusters_from_cells(positive_cells_info)
```

`clusters` 是 SAM2 的 prompt 数据。每个 cluster 通常是一组二维坐标点。

### 输出数据

如果 `positive_cells_info` 和 `clusters` 都非空，脚本会构造：

```python
item = BucketItem(
    tile_np=tile_np,
    clusters=clusters,
    positive_cells_info=positive_cells_info,
    tile_info=tile_info,
    tile_name=tile_name,
)
```

然后放入桶：

```python
self.bucket.put(item)
```

### 架构意义

细胞提取阶段是前后两半管线的接口转换层。它把 DeepLIIF 的图像结果转换成 SAM2 可以消费的 prompt 数据，并通过 `BucketItem` 形成稳定的数据契约。

## 第四步：SAM2 处理

SAM2 处理发生在 `Consumer._run_impl()` 中，由 `load_sam2()` 和 `run_sam2_segmentation_batch()` 负责。

### 参与对象

| 对象或函数 | 职责 |
| --- | --- |
| `Consumer` | 从 `Bucket` 中取出 `BucketItem` 并驱动 SAM2 推理 |
| `load_sam2()` | 加载 SAM2 predictor |
| `run_sam2_segmentation_batch()` | 对一个 tile 中的多个 prompt 批量执行 SAM2 |
| `BucketItem` | 提供 `tile_np`、`clusters`、`positive_cells_info` 和 `tile_name` |

### 输入数据

`Consumer` 从桶中取出：

```python
item: BucketItem = self.bucket.get(timeout=0.5)
```

SAM2 阶段使用 `BucketItem` 中的：

- `item.tile_np`
- `item.clusters`

调用方式：

```python
sam_mask, scores, _ = run_sam2_segmentation_batch(
    sam2_predictor,
    item.tile_np,
    item.clusters,
    min_area=args.min_mask_area,
    set_image=True,
    batch_size=args.sam2_batch_size,
    score_threshold=0.1,
)
```

### 输出数据

SAM2 阶段输出：

- `sam_mask`：原始分割 mask。
- `scores`：mask 对应评分。
- `_`：过滤结果，这里没有继续使用。

如果启用 `--cache-sam2`，原始 SAM2 输出会保存到：

```text
<output-dir>/cache/sam2/<tile_name>.npy
```

### 架构意义

SAM2 在这里承担“精分割对象”的角色。它接收细胞提取得到的 prompt，在原始 RGB tile 上生成更细粒度的 mask。

## 第五步：后处理

后处理发生在 `Consumer._run_impl()` 的 SAM2 推理之后，由 `merge_connected_masks()` 和 `AsyncSaver` 完成。

### 参与对象

| 对象或函数 | 职责 |
| --- | --- |
| `merge_connected_masks()` | 根据 SAM2 mask、score 和阳性细胞信息合并连通 mask |
| `AsyncSaver` | 异步写入最终 `.npy` mask |
| `save_mask_npy()` | 实际执行 mask 文件保存 |
| `Consumer` | 判断结果是否有效并提交保存任务 |

### 输入数据

后处理阶段使用：

- `sam_mask`
- `scores`
- `item.positive_cells_info`

调用方式：

```python
sam_mask_merged, _, _, _ = merge_connected_masks(
    sam_mask,
    scores,
    item.positive_cells_info,
    min_area=200,
)
```

### 输出数据

得到最终的实例 mask：

```python
sam_mask_merged
```

如果该 mask 中存在有效实例：

```python
if np.max(sam_mask_merged) > 0:
    npy_path = os.path.join(output_dir, "npy_masks", f"{item.tile_name}.npy")
    saver.submit(sam_mask_merged, npy_path)
```

`AsyncSaver` 后台线程最终调用 `save_mask_npy()` 保存文件。

### 架构意义

后处理阶段承担“结果整合与落盘对象”的角色。它把 SAM2 的多个候选 mask 合并为最终实例 mask，并将磁盘写入从主 GPU 推理流程中拆出去，减少 I/O 对推理吞吐的影响。

## 4. 线程与缓冲机制

这个脚本最重要的工程结构是 `Producer`、`Bucket`、`Consumer` 三者之间的协作。

### 4.1 生产者线程

`Producer` 的主循环按 chunk 工作：

1. 等待 `Bucket` 低于水位线。
2. 读取下一批 tile。
3. YOLO 分类。
4. 对目标 tile 执行 DeepLIIF。
5. 提取细胞并构造 `BucketItem`。
6. 将 `BucketItem` 放入 `Bucket`。
7. 标记本轮补货完成。

### 4.2 桶的水位线

`main()` 中计算实际队列大小：

```python
max_batch = max(args.yolo_batch_size, args.deepliif_batch_size)
queue_capacity = args.bucket_capacity * max_batch
```

`Bucket` 的水位线为：

```python
watermark_level = int(queue_capacity * args.bucket_watermark)
```

当消费者持续取数据导致库存低于水位线时，`Bucket.get()` 会唤醒生产者补货。

### 4.3 消费者线程

`Consumer` 的主循环持续从桶中取 `BucketItem`：

1. 如果桶为空但生产者尚未结束，则继续等待。
2. 如果桶为空且生产者已结束，则退出。
3. 对取出的 `BucketItem` 执行 SAM2。
4. 合并 mask。
5. 将有效结果提交给 `AsyncSaver`。

### 4.4 异步保存线程

`AsyncSaver` 内部维护一个队列和一个后台线程：

- `submit(mask, path)` 将保存任务放入队列。
- `_worker()` 持续从队列取任务并调用 `save_mask_npy()`。
- `shutdown()` 放入 sentinel 并等待所有保存任务完成。

这样 `Consumer` 不需要等待每个 `.npy` 文件写完，可以更快进入下一个 SAM2 推理。

## 5. 单 tile 调试模式

当传入 `--tile-index ROW,COL` 时，脚本不会启动生产者/消费者线程，而是调用 `run_single_tile_debug(args)`。

这个函数按线性方式处理一个 tile：

```text
WSIReader 读取指定 tile
        |
        v
DeepLIIF
        |
        v
细胞提取
        |
        v
SAM2
        |
        v
merge_connected_masks
        |
        v
save_mask_npy
```

调试模式跳过 YOLO 分类，适合验证某个具体 tile 上 DeepLIIF、细胞提取、SAM2 和后处理是否正常。

## 6. 主入口 `main()` 的编排职责

`main()` 是整个脚本的装配器。它负责：

1. 解析命令行参数。
2. 检查 CUDA 是否可用。
3. 创建输出目录。
4. 处理单 tile 调试模式。
5. 打开 WSI。
6. 枚举 tile。
7. 根据 ROI CSV 过滤 tile。
8. 根据 `--max-tiles` 限制处理数量。
9. 创建 `Bucket`、`Event`、`stats`、`StickyProgress`。
10. 创建并启动 `Producer` 和 `Consumer` 线程。
11. 等待线程结束。
12. 关闭进度条和 WSI。
13. 打印整体统计。

从面向对象角度看，`main()` 不直接关心每个模型如何推理，它只负责对象装配和生命周期管理。

## 7. 五阶段数据流总结

| 阶段 | 所属对象 | 输入 | 输出 | 下一阶段 |
| --- | --- | --- | --- | --- |
| 1. YOLO 处理 | `Producer` + `TileClassifier` | `tile_info` + PIL tile | `target_tiles` | DeepLIIF |
| 2. DeepLIIF 处理 | `Producer` + `DeepLIIFBatchInference` | PIL target tile | `Seg` / `Marker` | 细胞提取 |
| 3. 细胞提取 | `Producer` + cell extraction functions | `seg_np` + `marker_np` | `positive_cells_info` + `clusters` + `BucketItem` | Bucket / SAM2 |
| 4. SAM2 处理 | `Consumer` + SAM2 predictor | `tile_np` + `clusters` | `sam_mask` + `scores` | 后处理 |
| 5. 后处理 | `Consumer` + `AsyncSaver` | `sam_mask` + `scores` + `positive_cells_info` | merged mask `.npy` | 输出目录 |

## 8. 设计特点

这个脚本的设计重点有四个：

1. 使用 YOLO 做前置过滤，减少 DeepLIIF 和 SAM2 的无效计算。
2. 使用 `BucketItem` 作为明确的数据契约，只把 SAM2 需要的数据跨线程传递。
3. 使用 `Bucket` 的水位线机制平衡前后段速度，避免生产者无限制占用内存。
4. 使用 `AsyncSaver` 将磁盘写入从 SAM2 消费流程中剥离，提高吞吐稳定性。

因此，这份代码不是简单的线性脚本，而是一个由多个职责明确的对象协同工作的批处理流水线。
