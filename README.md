# CD34 Microvessel Detection Pipeline

基于 **DeepLIIF** + **SAM2** 的 CD34 免疫组化染色微血管实例分割工具。

## 项目结构

```
cd34-microvessel-detection/
├── pyproject.toml              # 项目依赖与构建配置
├── configs/
│   └── default.yaml            # 默认参数配置
│
├── cd34_pipeline/              # 核心包
│   ├── config.py               # 命令行参数解析
│   ├── pipeline.py             # 主流程编排
│   ├── deepliif/               # DeepLIIF 推理与后处理
│   ├── sam2_wrapper/           # SAM2 模型加载与推理
│   ├── cell/                   # 细胞提取、分类、掩码操作
│   ├── io/                     # 文件读写、LabelMe 导出、GeoJSON 导出
│   └── visualization/          # 可视化与对比图
│
├── scripts/                    # 入口脚本与工具
│   ├── pipeline/
│   │   ├── run_wsi_pipeline.py # WSI pipeline 入口
│   │   └── run_deepliif_only.py# DeepLIIF 独立测试
│   ├── analysis/
│   │   ├── analyze_wsi_masks.py
│   │   ├── visualize_reconstructed.py
│   │   └── generate_ppt.py
│   └── experimental/
│       ├── merge_tiles_demo.py
│       ├── test_interactive_sam2.py
│       ├── test_mask_only.py
│       └── visualize_results.py
│
├── docs/                       # 文档
│   └── parameters.md           # 参数详细说明
│
└── data/                       # 数据目录 (gitignored)
    ├── models/
    │   ├── deepliif/           # DeepLIIF 模型权重 (G1-G55.pt)
    │   └── sam2/               # SAM2 checkpoints
    ├── input/                  # 输入图像
    └── output/                 # 输出结果
```

## Pipeline 流程

```
输入图像 (H&E 染色)
    ↓
[DeepLIIF 推理] → DAPI, Hema, Marker, Seg
    ↓
[Prompt 构建] → Seg/Marker -5..5 加权 mask → 伪影/碎片/孤立小块过滤 → 强阳性点
    ↓
[SAM2 精细化] → weighted mask + 强阳性点 → 连通实例掩码
    ↓
[Center-valid 拼接] → GeoJSON
```

当前主流程只支持 `weighted-points`：Seg/Marker 加权 mask 加强阳性点，
不再保留逐连通域二值 mask 回退路径。

## 安装

```bash
# 克隆项目
git clone <repo-url> cd34-microvessel-detection
cd cd34-microvessel-detection

# 安装依赖 (SAM2 会自动作为 pip 依赖安装)
pip install -e .

# 如需 GeoJSON 导出功能
pip install -e ".[tile]"
```

## 使用

```bash
# 完整 pipeline
python scripts/pipeline/run_wsi_pipeline.py \
    --wsi-path data/input/slide.ndpi \
    --output-dir data/output

# 仅运行 DeepLIIF
python scripts/pipeline/run_deepliif_only.py --img data/input/sample.png
```

区域验证：

```bash
# 默认读取 config/cell_main.json
python -m cell.main

# 也可以临时覆盖 config 中的字段
python -m cell.main --weighted-dab-min-intensity 125
```

## 参数说明

详见 [docs/parameters.md](docs/parameters.md)。

## 依赖

- Python >= 3.10
- PyTorch >= 2.3.1
- [SAM2](https://github.com/facebookresearch/sam2) (自动安装)
- OpenCV, NumPy, Pillow, Hydra, Matplotlib
