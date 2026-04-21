"""
cd34_pipeline - CD34 微血管检测 Pipeline

基于 DeepLIIF + SAM2 的 CD34 免疫组化染色微血管实例分割工具。

子包:
- deepliif: DeepLIIF 模型推理与后处理
- sam2_wrapper: SAM2 模型加载与推理封装
- cell: 细胞提取、分类与掩码操作
- io: 文件读写、CSV/GeoJSON 导出、瓦片重建
- visualization: 可视化与对比图生成
"""

__version__ = "1.0.0"
