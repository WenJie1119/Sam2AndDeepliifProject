#!/usr/bin/env python3
"""
Tile Mask合并Demo脚本 - 方案B: 边界合并法

处理pipeline_full_inference.py生成的tile mask,合并跨边界的连通区域,
并计算面积、个数等统计指标。

作者: Based on 方案B算法设计
日期: 2026-01-19
"""

import os
import json
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from scipy import ndimage


@dataclass
class TileInfo:
    """Tile信息"""
    filename: str
    row: int
    col: int
    offset_y: int
    offset_x: int
    mask: Optional[np.ndarray] = None
    num_instances: int = 0
    
    @property
    def tile_id(self) -> str:
        return self.filename


@dataclass
class Region:
    """连通区域信息"""
    local_id: int  # 在tile内部的ID
    tile_id: str  # 所属tile
    pixels: np.ndarray  # 像素坐标 (N, 2) [y, x]
    area: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # ymin, xmin, ymax, xmax
    is_boundary: bool
    boundary_edges: Set[str] = field(default_factory=set)


@dataclass
class GlobalRegion:
    """全局合并后的区域"""
    global_id: int
    area: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    perimeter: int
    circularity: float
    tiles: List[str]
    num_source_regions: int


class UnionFind:
    """并查集数据结构(带路径压缩和按秩合并)"""
    
    def __init__(self, elements: List[Tuple[str, int]]):
        """
        Args:
            elements: List of (tile_id, local_id) tuples
        """
        self.parent = {elem: elem for elem in elements}
        self.rank = {elem: 0 for elem in elements}
    
    def find(self, x: Tuple[str, int]) -> Tuple[str, int]:
        """查找根节点(带路径压缩)"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: Tuple[str, int], y: Tuple[str, int]) -> None:
        """合并两个集合(按秩合并)"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
    
    def get_groups(self) -> Dict[Tuple[str, int], List[Tuple[str, int]]]:
        """获取所有合并组"""
        groups = defaultdict(list)
        for elem in self.parent.keys():
            root = self.find(elem)
            groups[root].append(elem)
        return groups


class TileMaskMerger:
    """Tile Mask合并处理器"""
    
    def __init__(self, tile_size: int = 512, verbose: bool = True):
        self.tile_size = tile_size
        self.verbose = verbose
        self.tiles: List[TileInfo] = []
        self.all_regions: List[Region] = []
        self.global_regions: List[GlobalRegion] = []
    
    def log(self, message: str):
        """打印日志"""
        if self.verbose:
            print(message)
    
    def scan_tiles(self, input_dir: str) -> List[TileInfo]:
        """
        扫描目录中的所有tile文件
        
        文件名格式: tile_{row}_{col}_{y}_{x}.npy
        """
        self.log(f"\n=== 阶段1: 扫描Tile文件 ===")
        self.log(f"输入目录: {input_dir}")
        
        tiles = []
        pattern = re.compile(r'tile_(\d+)_(\d+)_(\d+)_(\d+)\.npy')
        
        for filename in os.listdir(input_dir):
            match = pattern.match(filename)
            if not match:
                continue
            
            row, col, offset_y, offset_x = map(int, match.groups())
            
            # 读取元数据
            meta_path = os.path.join(input_dir, filename.replace('.npy', '_meta.json'))
            num_instances = 0
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    num_instances = meta.get('num_instances', 0)
            
            tile = TileInfo(
                filename=filename,
                row=row,
                col=col,
                offset_y=offset_y,
                offset_x=offset_x,
                num_instances=num_instances
            )
            tiles.append(tile)
        
        # 按行列排序
        tiles.sort(key=lambda t: (t.row, t.col))
        
        self.tiles = tiles
        self.log(f"找到 {len(tiles)} 个tile文件")
        
        if tiles:
            rows = {t.row for t in tiles}
            cols = {t.col for t in tiles}
            self.log(f"行范围: {min(rows)} - {max(rows)} (共{len(rows)}行)")
            self.log(f"列范围: {min(cols)} - {max(cols)} (共{len(cols)}列)")
        
        return tiles
    
    def extract_regions_from_tiles(self, input_dir: str) -> List[Region]:
        """
        从所有tile中提取连通区域
        """
        self.log(f"\n=== 阶段2: 提取连通组件 ===")
        
        all_regions = []
        total_instances = 0
        boundary_instances = 0
        
        for idx, tile in enumerate(self.tiles):
            # 加载mask
            mask_path = os.path.join(input_dir, tile.filename)
            mask = np.load(mask_path)
            tile.mask = mask
            
            # 提取该tile的所有实例
            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids > 0]  # 跳过背景
            
            for inst_id in instance_ids:
                # 提取像素坐标
                pixels = np.argwhere(mask == inst_id)  # 返回 (N, 2) [y, x]
                area = len(pixels)
                
                # 计算属性
                centroid_y = pixels[:, 0].mean()
                centroid_x = pixels[:, 1].mean()
                
                bbox_ymin = pixels[:, 0].min()
                bbox_ymax = pixels[:, 0].max()
                bbox_xmin = pixels[:, 1].min()
                bbox_xmax = pixels[:, 1].max()
                
                # 检测边界
                is_boundary = False
                boundary_edges = set()
                
                if bbox_ymin == 0:
                    is_boundary = True
                    boundary_edges.add("top")
                if bbox_ymax == self.tile_size - 1:
                    is_boundary = True
                    boundary_edges.add("bottom")
                if bbox_xmin == 0:
                    is_boundary = True
                    boundary_edges.add("left")
                if bbox_xmax == self.tile_size - 1:
                    is_boundary = True
                    boundary_edges.add("right")
                
                region = Region(
                    local_id=int(inst_id),
                    tile_id=tile.filename,
                    pixels=pixels,
                    area=area,
                    centroid=(centroid_y, centroid_x),
                    bbox=(bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax),
                    is_boundary=is_boundary,
                    boundary_edges=boundary_edges
                )
                
                all_regions.append(region)
                total_instances += 1
                if is_boundary:
                    boundary_instances += 1
            
            if (idx + 1) % 100 == 0:
                self.log(f"  已处理 {idx + 1}/{len(self.tiles)} tiles...")
        
        self.all_regions = all_regions
        self.log(f"提取完成: 共 {total_instances} 个实例")
        self.log(f"  其中 {boundary_instances} 个接触边界 ({boundary_instances/total_instances*100:.1f}%)")
        
        return all_regions
    
    def build_tile_adjacency(self) -> Dict[str, List[Dict]]:
        """构建tile的邻接关系"""
        self.log(f"\n=== 阶段3: 构建Tile邻接关系 ===")
        
        # 创建(row, col) -> tile的映射
        grid = {}
        for tile in self.tiles:
            grid[(tile.row, tile.col)] = tile
        
        neighbors = defaultdict(list)
        
        for tile in self.tiles:
            r, c = tile.row, tile.col
            
            # 检查4个方向
            directions = [
                (r - 1, c, "top"),
                (r + 1, c, "bottom"),
                (r, c - 1, "left"),
                (r, c + 1, "right")
            ]
            
            for nr, nc, direction in directions:
                if (nr, nc) in grid:
                    neighbor = grid[(nr, nc)]
                    neighbors[tile.filename].append({
                        "tile": neighbor.filename,
                        "tile_obj": neighbor,
                        "direction": direction
                    })
        
        total_adjacencies = sum(len(v) for v in neighbors.values())
        self.log(f"构建完成: {total_adjacencies} 个邻接关系")
        
        return neighbors
    
    def find_merge_pairs_zipper(
        self, 
        neighbors: Dict[str, List[Dict]]
    ) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
        """
        使用经典拉链法查找需要合并的区域对
        
        核心思想:
        1. 只检查右边界和下边界(避免重复)
        2. 逐像素比较边界列/行
        3. 两边都是前景(>0)就记录合并关系
        """
        self.log(f"\n=== 阶段4: 边界检测与合并对查找(经典拉链法) ===")
        
        merge_pairs = []
        
        # 创建tile映射
        tile_map = {t.filename: t for t in self.tiles}
        
        # 只检查右侧和下侧邻居,避免重复
        checked_pairs = set()
        
        for tile_a in self.tiles:
            for neighbor_info in neighbors[tile_a.filename]:
                tile_b_filename = neighbor_info["tile"]
                tile_b = neighbor_info["tile_obj"]
                direction = neighbor_info["direction"]
                
                # 只处理右侧和下侧
                if direction not in ["right", "bottom"]:
                    continue
                
                # 避免重复检查
                pair_key = (tile_a.filename, tile_b_filename)
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)
                
                # 获取两个tile的mask
                mask_a = tile_a.mask
                mask_b = tile_b.mask
                
                if mask_a is None or mask_b is None:
                    continue
                
                # 根据方向提取边界
                if direction == "right":
                    # tile_b在tile_a右侧
                    # 提取tile_a的最右列和tile_b的最左列
                    boundary_a = mask_a[:, -1]  # 最右列
                    boundary_b = mask_b[:, 0]   # 最左列
                    
                    # 逐行检查
                    for y in range(len(boundary_a)):
                        id_a = boundary_a[y]
                        id_b = boundary_b[y]
                        
                        # 两边都是前景就合并
                        if id_a > 0 and id_b > 0:
                            uid_a = (tile_a.filename, int(id_a))
                            uid_b = (tile_b_filename, int(id_b))
                            merge_pairs.append((uid_a, uid_b))
                
                elif direction == "bottom":
                    # tile_b在tile_a下方
                    # 提取tile_a的最下行和tile_b的最上行
                    boundary_a = mask_a[-1, :]  # 最下行
                    boundary_b = mask_b[0, :]   # 最上行
                    
                    # 逐列检查
                    for x in range(len(boundary_a)):
                        id_a = boundary_a[x]
                        id_b = boundary_b[x]
                        
                        # 两边都是前景就合并
                        if id_a > 0 and id_b > 0:
                            uid_a = (tile_a.filename, int(id_a))
                            uid_b = (tile_b_filename, int(id_b))
                            merge_pairs.append((uid_a, uid_b))
        
        self.log(f"找到 {len(merge_pairs)} 对需要合并的边界像素")
        
        # 去重(可能有多个像素连接同一对region)
        unique_pairs = list(set(merge_pairs))
        self.log(f"去重后: {len(unique_pairs)} 对唯一的region合并关系")
        
        return unique_pairs
    
    def merge_regions_with_unionfind(
        self, 
        merge_pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]]
    ) -> Dict[Tuple[str, int], int]:
        """使用并查集合并区域并分配全局ID"""
        self.log(f"\n=== 阶段5: 执行区域合并(并查集) ===")
        
        # 创建所有region的UID
        all_uids = [(r.tile_id, r.local_id) for r in self.all_regions]
        
        # 初始化并查集
        uf = UnionFind(all_uids)
        
        # 执行合并
        for uid_a, uid_b in merge_pairs:
            uf.union(uid_a, uid_b)
        
        # 获取合并组
        groups = uf.get_groups()
        
        self.log(f"合并前: {len(all_uids)} 个region")
        self.log(f"合并后: {len(groups)} 个全局region")
        self.log(f"减少: {len(all_uids) - len(groups)} 个 ({(len(all_uids) - len(groups))/len(all_uids)*100:.1f}%)")
        
        # 分配全局ID
        global_id_map = {}
        for global_id, (root, members) in enumerate(groups.items(), start=1):
            for member in members:
                global_id_map[member] = global_id
        
        return global_id_map
    
    def generate_global_regions(
        self,
        global_id_map: Dict[Tuple[str, int], int]
    ) -> List[GlobalRegion]:
        """生成全局区域信息"""
        self.log(f"\n=== 阶段6: 生成全局区域信息 ===")
        
        # 按global_id分组
        groups = defaultdict(list)
        for region in self.all_regions:
            uid = (region.tile_id, region.local_id)
            gid = global_id_map[uid]
            groups[gid].append(region)
        
        # 创建tile映射
        tile_map = {t.filename: t for t in self.tiles}
        
        global_regions = []
        
        for global_id, region_list in groups.items():
            # 合并所有像素(转换到全局坐标)
            all_pixels = []
            for region in region_list:
                tile = tile_map[region.tile_id]
                global_pixels = region.pixels.copy()
                global_pixels[:, 0] += tile.offset_y
                global_pixels[:, 1] += tile.offset_x
                all_pixels.append(global_pixels)
            
            all_pixels = np.vstack(all_pixels)
            
            # 计算属性
            area = len(all_pixels)
            centroid_y = all_pixels[:, 0].mean()
            centroid_x = all_pixels[:, 1].mean()
            
            bbox = (
                all_pixels[:, 0].min(),
                all_pixels[:, 1].min(),
                all_pixels[:, 0].max(),
                all_pixels[:, 1].max()
            )
            
            # 计算周长(边界像素数)
            perimeter = self._count_boundary_pixels(all_pixels)
            
            # 计算圆形度
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            
            # 记录涉及的tiles
            tiles_list = list(set(r.tile_id for r in region_list))
            
            global_region = GlobalRegion(
                global_id=global_id,
                area=area,
                centroid=(centroid_y, centroid_x),
                bbox=bbox,
                perimeter=perimeter,
                circularity=circularity,
                tiles=tiles_list,
                num_source_regions=len(region_list)
            )
            
            global_regions.append(global_region)
        
        self.global_regions = global_regions
        self.log(f"生成完成: {len(global_regions)} 个全局区域")
        
        return global_regions
    
    @staticmethod
    def _count_boundary_pixels(pixels: np.ndarray) -> int:
        """计算边界像素数(周长估计)"""
        pixel_set = set(map(tuple, pixels))
        boundary_count = 0
        
        for y, x in pixels:
            # 检查8邻域
            is_boundary = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if (y + dy, x + dx) not in pixel_set:
                        is_boundary = True
                        break
                if is_boundary:
                    break
            
            if is_boundary:
                boundary_count += 1
        
        return boundary_count
    
    def compute_statistics(self, pixel_size_um: float = 0.5) -> Dict:
        """计算统计指标"""
        self.log(f"\n=== 阶段7: 计算统计指标 ===")
        
        areas = [r.area for r in self.global_regions]
        
        stats = {
            "total_regions": len(self.global_regions),
            "total_area_pixels": sum(areas),
            "total_area_um2": sum(areas) * (pixel_size_um ** 2),
            "average_area_pixels": np.mean(areas) if areas else 0,
            "median_area_pixels": np.median(areas) if areas else 0,
            "min_area": min(areas) if areas else 0,
            "max_area": max(areas) if areas else 0,
            "std_area": np.std(areas) if areas else 0,
            "pixel_size_um": pixel_size_um,
        }
        
        # 按大小分类
        small = sum(1 for a in areas if a < 100)
        medium = sum(1 for a in areas if 100 <= a < 500)
        large = sum(1 for a in areas if a >= 500)
        
        stats["size_distribution"] = {
            "small_(<100px)": small,
            "medium_(100-500px)": medium,
            "large_(>=500px)": large
        }
        
        # 跨tile统计
        multi_tile = sum(1 for r in self.global_regions if r.num_source_regions > 1)
        stats["multi_tile_regions"] = multi_tile
        stats["multi_tile_percentage"] = (multi_tile / len(self.global_regions) * 100) if self.global_regions else 0
        
        self.log(f"统计完成:")
        self.log(f"  总区域数: {stats['total_regions']}")
        self.log(f"  总面积: {stats['total_area_pixels']} 像素 ({stats['total_area_um2']:.2f} μm²)")
        self.log(f"  平均面积: {stats['average_area_pixels']:.1f} 像素")
        self.log(f"  跨tile区域: {multi_tile} ({stats['multi_tile_percentage']:.1f}%)")
        
        return stats
    
    def save_results(self, output_dir: str, stats: Dict):
        """保存结果"""
        self.log(f"\n=== 保存结果 ===")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存区域信息表(CSV)
        csv_path = os.path.join(output_dir, "regions.csv")
        df_data = []
        
        for region in self.global_regions:
            df_data.append({
                "region_id": region.global_id,
                "area_pixels": region.area,
                "area_um2": region.area * (stats["pixel_size_um"] ** 2),
                "centroid_y": region.centroid[0],
                "centroid_x": region.centroid[1],
                "bbox_ymin": region.bbox[0],
                "bbox_xmin": region.bbox[1],
                "bbox_ymax": region.bbox[2],
                "bbox_xmax": region.bbox[3],
                "perimeter": region.perimeter,
                "circularity": region.circularity,
                "num_tiles": len(region.tiles),
                "num_source_regions": region.num_source_regions
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)
        self.log(f"  ✓ 区域信息表: {csv_path}")
        
        # 2. 保存统计摘要(JSON)
        stats_path = os.path.join(output_dir, "statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        self.log(f"  ✓ 统计摘要: {stats_path}")
        
        # 3. 保存详细报告(TXT)
        report_path = os.path.join(output_dir, "report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Tile Mask 合并处理报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"输入Tiles数量: {len(self.tiles)}\n")
            f.write(f"合并前区域数: {len(self.all_regions)}\n")
            f.write(f"合并后区域数: {stats['total_regions']}\n")
            f.write(f"减少区域数: {len(self.all_regions) - stats['total_regions']}\n\n")
            
            f.write(f"总面积: {stats['total_area_pixels']} 像素\n")
            f.write(f"总面积: {stats['total_area_um2']:.2f} μm² (pixel_size={stats['pixel_size_um']}μm)\n\n")
            
            f.write(f"平均面积: {stats['average_area_pixels']:.1f} 像素\n")
            f.write(f"中位数面积: {stats['median_area_pixels']:.1f} 像素\n")
            f.write(f"最小面积: {stats['min_area']} 像素\n")
            f.write(f"最大面积: {stats['max_area']} 像素\n")
            f.write(f"标准差: {stats['std_area']:.1f} 像素\n\n")
            
            f.write("面积分布:\n")
            for key, val in stats['size_distribution'].items():
                f.write(f"  {key}: {val}\n")
            
            f.write(f"\n跨Tile区域: {stats['multi_tile_regions']} ({stats['multi_tile_percentage']:.1f}%)\n")
        
        self.log(f"  ✓ 详细报告: {report_path}")
    
    def run(self, input_dir: str, output_dir: str, pixel_size_um: float = 0.5):
        """运行完整流程"""
        print("\n" + "=" * 60)
        print(" Tile Mask 合并处理 - 方案B: 边界合并法")
        print("=" * 60)
        
        # 扫描tiles
        self.scan_tiles(input_dir)
        
        if not self.tiles:
            print("错误: 没有找到任何tile文件!")
            return
        
        # 提取区域
        self.extract_regions_from_tiles(input_dir)
        
        # 构建邻接关系
        neighbors = self.build_tile_adjacency()
        
        # 查找合并对
        merge_pairs = self.find_merge_pairs_zipper(neighbors)
        
        # 执行合并
        global_id_map = self.merge_regions_with_unionfind(merge_pairs)
        
        # 生成全局区域
        self.generate_global_regions(global_id_map)
        
        # 计算统计
        stats = self.compute_statistics(pixel_size_um)
        
        # 保存结果
        self.save_results(output_dir, stats)
        
        print("\n" + "=" * 60)
        print(" 处理完成!")
        print("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tile Mask合并Demo - 方案B")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="输入目录(包含tile npy文件)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.5,
        help="像素物理尺寸(μm), 默认0.5"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile大小(像素), 默认512"
    )
    
    args = parser.parse_args()
    
    # 创建处理器并运行
    merger = TileMaskMerger(tile_size=args.tile_size, verbose=True)
    merger.run(args.input_dir, args.output_dir, args.pixel_size)


if __name__ == "__main__":
    main()
