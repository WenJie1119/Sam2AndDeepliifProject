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
│   ├── io/                     # 文件读写、LabelMe 导出、瓦片重建
│   └── visualization/          # 可视化与对比图
│
├── scripts/                    # 入口脚本与工具
│   ├── run_pipeline.py         # 主 pipeline 入口
│   ├── run_deepliif_only.py    # DeepLIIF 独立测试
│   ├── reconstruct_tiles.py    # 瓦片重建
│   └── ...
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
[细胞提取] → 前景检测 → R/B 分类 → Marker 增强 → 阳性细胞
    ↓
[SAM2 精细化] → 实例分割掩码
    ↓
[导出] → LabelMe JSON / CSV / 可视化对比图
```

## 安装

```bash
# 克隆项目
git clone <repo-url> cd34-microvessel-detection
cd cd34-microvessel-detection

# 安装依赖 (SAM2 会自动作为 pip 依赖安装)
pip install -e .

# 如需瓦片重建功能
pip install -e ".[tile]"
```

## 使用

```bash
# 完整 pipeline
python scripts/run_pipeline.py \
    --input-dir data/input \
    --output-dir data/output \
    --deepliif-model-dir data/models/deepliif \
    --sam-checkpoint data/models/sam2/sam2.1_hiera_large.pt \
    --sam-config configs/sam2.1/sam2.1_hiera_l.yaml \
    --export-labelme

# 仅运行 DeepLIIF
python scripts/run_deepliif_only.py --img data/input/sample.png

# 瓦片重建
python scripts/reconstruct_tiles.py \
    --tile-dir data/output/npy_masks \
    --output-dir data/output/reconstructed
```

## 参数说明

详见 [docs/parameters.md](docs/parameters.md)。

## 依赖

- Python >= 3.10
- PyTorch >= 2.3.1
- [SAM2](https://github.com/facebookresearch/sam2) (自动安装)
- OpenCV, NumPy, Pillow, Hydra, Matplotlib
