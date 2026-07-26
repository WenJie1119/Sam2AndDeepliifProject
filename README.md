# CD34 Microvessel Detection Pipeline

基于 **DeepLIIF** + **SAM2** 的 CD34 免疫组化染色微血管实例分割工具。

## 项目结构

```
cd34-microvessel-detection/
├── pyproject.toml              # 项目依赖与构建配置
├── config/
│   └── cell_main.json          # 当前主流程默认配置
│
├── cell/                       # 正式 WSI Pipeline 入口与运行编排
│   ├── main.py                 # Producer-Consumer 主流程
│   ├── deepliif.py             # DeepLIIF 批处理封装
│   ├── sam2.py                 # SAM2 批处理封装
│   └── postprocess.py          # 拼接、后处理与导出
│
├── cd34_pipeline/              # 底层算法、模型与 I/O
│   ├── deepliif/               # DeepLIIF 推理与后处理
│   ├── sam2_wrapper/           # SAM2 加载、推理与 weighted prompt
│   ├── cell/                   # 细胞提取、分类、掩码操作
│   ├── io/                     # WSI 读取、拼接与 GeoJSON 导出
│   └── visualization/          # 可视化与对比图
│
├── scripts/
│   ├── pipeline/               # 独立的稳定运行工具
│   ├── analysis/               # 结果分析与报告生成
│   ├── annotation/             # 标注格式转换
│   └── experimental/           # 单图、局部算法实验（非正式入口）
│
├── tests/                      # 正式单元测试
├── docs/
│   ├── architecture.md         # 当前架构说明
│   ├── parameters.md           # 参数详细说明
│   └── reports/                # 阶段报告与演示文稿
│
└── data/                       # 数据目录（gitignored）
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
# 正式完整 WSI Pipeline（默认读取 config/cell_main.json）
python -m cell.main

# 安装项目后也可以使用命令入口
cd34-pipeline

# 仅运行 DeepLIIF
python scripts/pipeline/run_deepliif_only.py --img data/input/sample.png
```

区域验证：

```bash
# 命令行参数会覆盖配置文件中的同名字段
python -m cell.main --weighted-dab-min-intensity 125
```

当前只有一套正式的完整 WSI Pipeline。`cell/` 负责入口和流程编排，
并调用 `cd34_pipeline/` 中的 DeepLIIF、SAM2、weighted prompt、WSI I/O
及 GeoJSON 导出功能。`scripts/experimental/` 中的脚本仅用于局部实验和调试。

## 参数说明

详见 [docs/parameters.md](docs/parameters.md)。

## 依赖

- Python >= 3.10
- PyTorch >= 2.3.1
- [SAM2](https://github.com/facebookresearch/sam2) (自动安装)
- OpenCV, NumPy, Pillow, Hydra, Matplotlib
