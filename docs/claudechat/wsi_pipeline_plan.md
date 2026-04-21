# WSI 直接处理方案：从 .ndpi 原图直接读取 tile 进行 Pipeline 处理

## Context

当前 cd34_pipeline 处理流程是：先将图像切割成 512x512 的小图保存到磁盘 → 逐个读取 → DeepLIIF → cell extraction → SAM2。现在需要改为直接从 .ndpi 全扫描切片原图中按需读取 512x512 区域，结合 YOLO 分类模型过滤掉背景 tile，只对包含目标的 tile 执行完整 pipeline，最终将所有 tile 结果拼接成全切片级别的结果。

---

## 整体架构

```
.ndpi 文件
    │
    ▼
WSIReader: 打开 WSI，枚举所有 tile 坐标
    │
    ▼
TileClassifier: YOLO 批量分类（或读取已有 JSON map）
    │               │
    │               └──▶ tile_map.json（持久化，可复用）
    ▼
过滤：只保留 "target" tile
    │
    ▼
逐 tile 处理循环：
    ├── WSIReader.read_tile() → PIL Image (512×512)
    ├���─ DeepLIIF inference → Seg, Marker 等
    ├── cell extraction (extract_cells_from_seg 或 extract_connected_positive_regions)
    ├── filter_positive_cells
    ├── SAM2 segmentation
    ├── merge_connected_masks
    └── save_mask_npy → npy_masks/tile_{row}_{col}_{x}_{y}.npy
    │
    ▼
reconstruct_tiles(): 拼接所有 tile mask → 全切片实例分割结果
```

---

## 需要创建的文件（4个）

### 1. `cd34_pipeline/io/wsi_reader.py` — WSI 读取模块

基于 OpenSlide 的全切片图像读取器，参考 `/local1/yangwenjie/Pathological-image-slide-processing/src/image_slicer.py` 的 OpenSlide 使用方式。

```python
class WSIReader:
    def __init__(self, wsi_path: str, tile_size: int = 512, 
                 target_magnification: float = 20.0)
    
    def get_level_for_magnification(self) -> tuple[int, float]
    # 根据目标放大倍数找最佳金字塔层级
    
    def get_slide_info(self) -> dict
    # 返回 WSI 元数据（尺寸、层级数、放大倍数等）
    
    def enumerate_tiles(self) -> list[dict]
    # 生成所有 tile 坐标列表
    # 返回: [{row, col, x, y, x_level0, y_level0, actual_w, actual_h}, ...]
    # stride = tile_size（无重叠，因为 DeepLIIF 在 512x512 输入上只产生 1 个 tile）
    
    def read_tile(self, tile_info: dict) -> Image.Image
    # 从 WSI 读取单个 tile，边缘用白色填充到 tile_size
    # 使用 slide.read_region((x_level0, y_level0), level, (w, h))
    
    def read_tile_np(self, tile_info: dict) -> np.ndarray
    # 读取为 numpy RGB 数组
    
    def close(self)
    # 关闭 OpenSlide 句柄
```

**关键设计点**：
- tile 之间无重叠（stride = tile_size），因为 512x512 输入送入 DeepLIIF 时，InferenceTiler 只产生 1 个 tile（已通过代码确认：当 image_width == patch_size 时 overlap_width=0，center_width=patch_size）
- tile 命名使用 `tile_{row}_{col}_{x}_{y}` 格式（x 在前 y 在后），匹配 `tile_reconstruction.py:parse_tile_filename()` 的解析逻辑

### 2. `cd34_pipeline/io/tile_classifier.py` — YOLO 分类模块

使用 YOLO 模型（`data/models/yolo/yolo11n_cls_cd34_bg_target_20601.pt`）对 tile 进行 background/target 分类。

```python
class TileClassifier:
    def __init__(self, model_path: str, device: str = 'cuda',
                 batch_size: int = 64, imgsz: int = 512)
    
    def classify_tiles_from_wsi(self, wsi_reader, tile_list, 
                                 progress_callback=None) -> dict
    # 批量从 WSI 读取 tile → YOLO 推理 → 返回分类结果
    # YOLO model.predict() 支持直接传入 numpy 数组/PIL Image 列表
    
    @staticmethod
    def save_tile_map(tile_map, output_path)
    # 保存分类结果为 CSV（row,col,x,y,classification）
    
    @staticmethod
    def load_tile_map(csv_path) -> list[dict]
    # 加载已有的 CSV tile map
    
    @staticmethod
    def get_target_tiles(tile_map) -> list[dict]
    # 过滤返回 "target" 类别的 tile 列表
```

**CSV tile map 格式**（轻量、快速读写）：

文件名：`{wsi_stem}_tile_map.csv`

```csv
row,col,x,y,classification
1,1,0,0,background
1,2,512,0,background
5,12,5632,2048,target
5,13,6144,2048,target
```

只保存必要字段：行号、列号、坐标、分类结果。元数据（WSI 路径、tile_size、magnification 等）不存入 CSV，由 pipeline 运行时自行确定。
```

### 3. `cd34_pipeline/wsi_pipeline.py` — WSI Pipeline 主流程

参考 `pipeline.py` 的处理逻辑，封装为 WSI 模式的主函数。

```python
def main_wsi():
    # 1. 解析参数 + 验证
    # 2. 加载模型（DeepLIIF + SAM2 + YOLO）
    # 3. 打开 WSI，枚举 tile
    # 4. YOLO 分类（或加载已有 tile map）
    # 5. 逐 target tile 处理：
    #    a. read_tile → PIL (512×512)
    #    b. deepliif_engine.inference(tile_pil, tile_size=512, ...)
    #    c. extract_cells_from_seg / extract_connected_positive_regions
    #    d. filter_positive_cells
    #    e. SAM2: run_sam2_segmentation(predictor, tile_np, clusters, set_image=True)
    #    f. merge_connected_masks
    #    g. save_mask_npy → npy_masks/tile_{row}_{col}_{x}_{y}.npy
    # 6. reconstruct_tiles(npy_masks_dir) → 全切片结果
    # 7. 清理
```

**关键处理细节**：
- 每个 tile 必须调用 `predictor.set_image(tile_np)`，因为是不同的图像
- DeepLIIF 对 512×512 输入天然兼容（InferenceTiler 只产生 1 个 tile）
- 每处理完一个 tile 即释放 PIL/numpy 内存
- 支持 `--classify-only` 模式：只运行 YOLO 分类生成 CSV，不执行后续 pipeline
- 支持 `--tile-map` 参���：直接加载已有 CSV 跳过 YOLO 分类
- **支持 `--tile-index ROW,COL` 参数**：指定单个 tile 的行列号，只处理该 tile（跳过 YOLO 分类和拼接，直接读取指定位置的 tile 走 DeepLIIF+SAM2，方便调试和单张验证）

### 4. `scripts/run_wsi_pipeline.py` — CLI 入口

简单包装器，调用 `main_wsi()`。

---

## 需要修改的文件（2个）

### 5. `cd34_pipeline/config.py` — 添加 WSI 模式参数

在现有 `parse_arguments()` 中添加 WSI 参数组：

```python
# WSI 模式参数
group_wsi = parser.add_argument_group('WSI Mode')
--wsi-path          # .ndpi 文件路径
--target-magnification  # 目标放大倍数 (默认 20.0)
--yolo-model-path   # YOLO 模型路径 (默认 ./data/models/yolo/...)
--tile-map          # 已有 tile map CSV 路径
--yolo-batch-size   # YOLO 批量大小 (默认 64)
--classify-only     # 仅分类，不执行 pipeline
--skip-reconstruction  # 跳过最终拼接（调试用）
--tile-index ROW,COL   # 单独指定一个 tile 的行列号，只跑这一张
```

### 6. `cd34_pipeline/io/__init__.py` — 添加新模块导入

---

## 实施顺序

| 阶段 | 内容 | 依赖 |
|------|------|------|
| Phase 1 | 创建 `wsi_reader.py`，实现 WSIReader | 无 |
| Phase 2 | 创建 `tile_classifier.py`，实现 TileClassifier + JSON 格式 | wsi_reader |
| Phase 3 | 修改 `config.py`，添加 WSI 参数 | 无 |
| Phase 4 | 创建 `wsi_pipeline.py`，实现 main_wsi() | Phase 1-3 |
| Phase 5 | 创建 `run_wsi_pipeline.py` 入口脚本 | Phase 4 |
| Phase 6 | 集成测试（使用真实 .ndpi 文件） | Phase 5 |

---

## 验证方案

1. **WSIReader 单元测试**：打开 `/local1/yangwenjie/DataImg/CD34/DC2200155 A3 CD34.ndpi`，枚举 tile，读取几个 tile 验证尺寸 (512×512)
2. **YOLO 分类测试**：对所有 tile 分类，检查 CSV 输出，验证 target/background 比例合理
3. **单 tile 测试**（使用 `--tile-index` 参数）：
   ```bash
   python scripts/run_wsi_pipeline.py \
     --wsi-path /path/to/slide.ndpi \
     --output-dir ./output/wsi_test \
     --tile-index 5,12 \
     --save-deepliif-outputs
   ```
   只处理第 5 行第 12 列的 tile，可以快速验证单张 tile 的 DeepLIIF + SAM2 结果
4. **Pipeline 端到端测试**：
   ```bash
   python scripts/run_wsi_pipeline.py \
     --wsi-path /local1/yangwenjie/DataImg/CD34/DC2200155\ A3\ CD34.ndpi \
     --output-dir ./output/wsi_test \
     --deepliif-model-dir ./data/models/deepliif \
     --yolo-model-path ./data/models/yolo/yolo11n_cls_cd34_bg_target_20601.pt \
     --tile-size 512 \
     --target-magnification 20.0 \
     --use-connected-regions \
     --save-npy
   ```
5. 检查输出：
   - `{wsi_stem}_tile_map.csv` 存在且格式正确
   - `npy_masks/` 目录下有 tile NPY 文件
   - `reconstructed/` 目录下有全切片 mask
