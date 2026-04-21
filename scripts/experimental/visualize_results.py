#!/usr/bin/env python3
"""
可视化Tile合并结果

功能:
1. 拼接tile图像为完整大图
2. 在大图上绘制合并后的区域边界
3. 验证合并算法的正确性
"""

import os
import re
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_merged_results(
    labelme_dir: str,
    npy_dir: str,
    merged_csv: str,
    output_path: str,
    tile_size: int = 512
):
    """
    可视化合并结果 - 使用透明填充绘制实际mask区域
    
    Args:
        labelme_dir: labelme目录(包含tile图像)
        npy_dir: npy_masks目录
        merged_csv: 合并结果CSV文件
        output_path: 输出图像路径
        tile_size: tile大小
    """
    print("=" * 60)
    print("可视化Tile合并结果 - 透明填充模式")
    print("=" * 60)
    
    # 1. 扫描所有tile图像
    print("\n步骤1: 扫描tile图像...")
    tiles = []
    pattern = re.compile(r'tile_(\d+)_(\d+)_(\d+)_(\d+)\.png')
    
    for filename in os.listdir(labelme_dir):
        match = pattern.match(filename)
        if not match:
            continue
        
        row, col, offset_y, offset_x = map(int, match.groups())
        tiles.append({
            'filename': filename,
            'basename': filename.replace('.png', ''),
            'row': row,
            'col': col,
            'offset_y': offset_y,
            'offset_x': offset_x,
            'path': os.path.join(labelme_dir, filename)
        })
    
    if not tiles:
        print("错误: 没有找到tile图像!")
        return
    
    print(f"找到 {len(tiles)} 个tile图像")
    
    # 2. 计算大图尺寸
    print("\n步骤2: 计算大图尺寸...")
    min_y = min(t['offset_y'] for t in tiles)
    max_y = max(t['offset_y'] for t in tiles)
    min_x = min(t['offset_x'] for t in tiles)
    max_x = max(t['offset_x'] for t in tiles)
    
    height = max_y - min_y + tile_size
    width = max_x - min_x + tile_size
    
    print(f"大图尺寸: {width} x {height}")
    
    # 3. 拼接图像
    print("\n步骤3: 拼接tile图像...")
    big_image = Image.new('RGB', (width, height), color=(255, 255, 255))
    
    for tile in tiles:
        img = Image.open(tile['path'])
        x = tile['offset_x'] - min_x
        y = tile['offset_y'] - min_y
        big_image.paste(img, (x, y))
    
    print("图像拼接完成")
    
    # 4. 创建mask叠加层
    print("\n步骤4: 创建mask叠加层...")
    mask_overlay = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))
    
    # 5. 读取合并结果CSV以获取global_id映射
    print("\n步骤5: 读取合并结果...")
    df = pd.read_csv(merged_csv)
    print(f"读取到 {len(df)} 个合并后的区域")
    
    # 6. 从npy文件读取mask并绘制
    print("\n步骤6: 绘制mask区域...")
    
    # 生成每个region的颜色
    np.random.seed(42)
    region_colors = {}
    for _, row in df.iterrows():
        r, g, b = np.random.randint(0, 255, 3)
        region_colors[row['region_id']] = (r, g, b, 100)  # 透明度100
    
    # 创建tile basename到tile info的映射
    tile_map = {t['basename']: t for t in tiles}
    
    # 读取每个npy文件
    processed = 0
    for tile_basename in tile_map.keys():
        npy_path = os.path.join(npy_dir, f"{tile_basename}.npy")
        if not os.path.exists(npy_path):
            continue
        
        # 加载mask
        mask = np.load(npy_path)
        tile_info = tile_map[tile_basename]
        
        # 获取该tile在大图中的偏移
        offset_x_in_big = tile_info['offset_x'] - min_x
        offset_y_in_big = tile_info['offset_y'] - min_y
        
        # 找到这个tile中的所有instance对应的global_id
        # 从CSV中找 - 这里简化处理，直接根据区域的bbox判断
        for _, region_row in df.iterrows():
            region_id = region_row['region_id']
            
            # 检查这个region的bbox是否与当前tile重叠
            bbox_ymin = region_row['bbox_ymin']
            bbox_xmin = region_row['bbox_xmin']
            bbox_ymax = region_row['bbox_ymax']
            bbox_xmax = region_row['bbox_xmax']
            
            tile_y_start = tile_info['offset_y']
            tile_y_end = tile_info['offset_y'] + tile_size
            tile_x_start = tile_info['offset_x']
            tile_x_end = tile_info['offset_x'] + tile_size
            
            # 判断是否重叠
            if not (bbox_ymax < tile_y_start or bbox_ymin >= tile_y_end or
                    bbox_xmax < tile_x_start or bbox_xmin >= tile_x_end):
                # 有重叠，在mask中查找这个region
                # 遍历mask中的所有instance
                for inst_id in np.unique(mask):
                    if inst_id == 0:
                        continue
                    
                    inst_mask = (mask == inst_id)
                    inst_pixels = np.argwhere(inst_mask)
                    
                    if len(inst_pixels) == 0:
                        continue
                    
                    # 检查这个instance的全局坐标是否匹配region的bbox
                    inst_global_y = inst_pixels[:, 0] + tile_info['offset_y']
                    inst_global_x = inst_pixels[:, 1] + tile_info['offset_x']
                    
                    inst_bbox_ymin = inst_global_y.min()
                    inst_bbox_xmin = inst_global_x.min()
                    inst_bbox_ymax = inst_global_y.max()
                    inst_bbox_xmax = inst_global_x.max()
                    
                    # 如果bbox匹配(允许一点误差)，则绘制
                    if (abs(inst_bbox_ymin - bbox_ymin) <= 2 and
                        abs(inst_bbox_xmin - bbox_xmin) <= 2 and
                        abs(inst_bbox_ymax - bbox_ymax) <= 2 and
                        abs(inst_bbox_xmax - bbox_xmax) <= 2):
                        
                        # 绘制这个instance
                        color = region_colors[region_id]
                        
                        for y, x in inst_pixels:
                            img_y = y + offset_y_in_big
                            img_x = x + offset_x_in_big
                            if 0 <= img_y < height and 0 <= img_x < width:
                                mask_overlay.putpixel((img_x, img_y), color)
        
        processed += 1
        if processed % 5 == 0:
            print(f"  已处理 {processed}/{len(tile_map)} tiles...")
    
    print(f"  已处理 {processed}/{len(tile_map)} tiles")
    
    # 7. 合并图像和mask
    print("\n步骤7: 合并图像和mask叠加层...")
    big_image = big_image.convert('RGBA')
    result = Image.alpha_composite(big_image, mask_overlay)
    result = result.convert('RGB')
    
    # 8. 添加ID标注
    print("\n步骤8: 添加ID标注...")
    draw = ImageDraw.Draw(result)
    
    for _, row in df.iterrows():
        centroid_y = row['centroid_y'] - min_y
        centroid_x = row['centroid_x'] - min_x
        
        text = str(int(row['region_id']))
        
        # 白色文字+黑色描边
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw.text((centroid_x + dx, centroid_y + dy), text, fill=(0, 0, 0))
        draw.text((centroid_x, centroid_y), text, fill=(255, 255, 255))
    
    # 9. 保存结果
    print(f"\n步骤9: 保存可视化结果...")
    result.save(output_path)
    print(f"已保存到: {output_path}")
    
    # 10. 生成统计报告
    print("\n" + "=" * 60)
    print("可视化完成!")
    print("=" * 60)
    print(f"总区域数: {len(df)}")
    print(f"总面积: {df['area_pixels'].sum()} 像素")
    print(f"平均面积: {df['area_pixels'].mean():.1f} 像素")
    
    # 跨tile统计
    multi_tile = df[df['num_tiles'] > 1]
    if len(multi_tile) > 0:
        print(f"\n跨tile区域: {len(multi_tile)} 个")
    else:
        print("\n注意: 没有发现跨tile的区域")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="可视化Tile合并结果")
    parser.add_argument(
        "--labelme-dir",
        type=str,
        required=True,
        help="labelme目录(包含tile图像)"
    )
    parser.add_argument(
        "--npy-dir",
        type=str,
        required=True,
        help="npy_masks目录"
    )
    parser.add_argument(
        "--merged-csv",
        type=str,
        required=True,
        help="合并结果CSV文件"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出可视化图像路径"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile大小,默认512"
    )
    
    args = parser.parse_args()
    
    visualize_merged_results(
        args.labelme_dir,
        args.npy_dir,
        args.merged_csv,
        args.output,
        args.tile_size
    )


if __name__ == "__main__":
    main()
