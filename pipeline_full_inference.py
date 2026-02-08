#!/usr/bin/env python3
"""
Full Pipeline: DeepLIIF -> SAM2

重构后的主入口脚本 - 只包含流程编排，具体实现在 src/ 模块中。
命令行接口与原版 100% 兼容。

Usage:
    python pipeline_full_inference.py --input-dir /path/to/images --output-dir /path/to/save
"""

import os
import sys
import shutil
import numpy as np
import torch

# Add paths for local modules
sys.path.insert(0, '/local1/yangwenjie/sam2')

# Import modular components
from src.config import (
    parse_arguments, 
    validate_config, 
    parse_size_thresh, 
    parse_large_noise_thresh,
    print_pipeline_header,
    print_pipeline_footer
)
from src.model_loader import load_all_models
from src.cell_extraction import (
    extract_cells_from_seg, 
    filter_positive_cells,
    get_clusters_from_cells,
    create_binary_mask_from_cells,
    renumber_cells,
    group_cells_by_distance
)
from src.mask_utils import (
    get_clusters_from_mask_image,
    generate_mask_from_cluster,
    generate_distinct_colors
)
from src.sam2_inference import (
    run_sam2_segmentation,
    run_sam2_grouped_segmentation,
    merge_connected_masks
)
from src.visualization import save_comparison, save_original_sam_comparison
from src.file_io import (
    get_image_files,
    read_image,
    save_deepliif_outputs,
    save_positive_cells_csv,
    export_labelme_annotation,
    save_sam2_mask_visualization,
    save_mask_npy,
    save_seg_probability_npy
)

# PIL for DeepLIIF compatibility
from PIL import Image
import cv2


def main():
    """主流程编排函数"""
    
    # ========== 1. 配置解析与验证 ==========
    args = parse_arguments()
    args = validate_config(args)
    
    torch.autograd.set_grad_enabled(False)
    print_pipeline_header(args)
    
    try:
        # ========== 2. 加载模型 ==========
        deepliif_engine, sam2_predictor = load_all_models(args)
        
        # ========== 3. 获取输入图像列表 ==========
        input_dir, image_files = get_image_files(args.input_dir)
        print(f"\nFound {len(image_files)} images to process.")
        
        # ========== 3.1 断点续传处理 ==========
        skip_set = set()  # 需要跳过的图像基础名集合
        if args.resume:
            labelme_dir = os.path.join(args.output_dir, "labelme")
            if os.path.exists(labelme_dir):
                # 获取所有已存在的 JSON 文件及其修改时间
                existing_jsons = []
                for f in os.listdir(labelme_dir):
                    if f.endswith('.json'):
                        json_path = os.path.join(labelme_dir, f)
                        mtime = os.path.getmtime(json_path)
                        base = os.path.splitext(f)[0]
                        existing_jsons.append((base, json_path, mtime))
                
                if existing_jsons:
                    # 按修改时间排序，找到最新的
                    existing_jsons.sort(key=lambda x: x[2], reverse=True)
                    latest_base, latest_path, _ = existing_jsons[0]
                    
                    # 删除最新的 JSON（可能不完整）
                    os.remove(latest_path)
                    print(f"\n[RESUME] Deleted last JSON (may be incomplete): {latest_path}")
                    
                    # 其余的都放入跳过集合
                    for base, path, _ in existing_jsons[1:]:
                        skip_set.add(base)
                    
                    print(f"[RESUME] Will skip {len(skip_set)} already processed images.")
                    print(f"[RESUME] Will re-process from: {latest_base}")
                else:
                    print(f"\n[RESUME] No existing JSON files found. Starting from beginning.")
            else:
                print(f"\n[RESUME] Output directory not found. Starting from beginning.")
        
        # ========== 4. 处理每张图像 ==========
        for idx, img_name in enumerate(image_files):
            base_name = os.path.splitext(img_name)[0]
            
            # 断点续传：跳过已处理的图像
            if args.resume and base_name in skip_set:
                print(f"\n--- [{idx+1}/{len(image_files)}] SKIP (already processed): {img_name} ---")
                continue
            
            print(f"\n--- Processing Image {idx+1}/{len(image_files)}: {img_name} ---")
            img_path = os.path.join(input_dir, img_name)
            
            # 4.1 读取原始图像
            original_pil = Image.open(img_path).convert('RGB')
            original_np = np.array(original_pil)
            print(f"  Image Size: {original_pil.size}")

            
            # ========== Step A: DeepLIIF 推理 ==========
            print("  > Running DeepLIIF inference...")
            size_thresh = parse_size_thresh(args.size_thresh)
            large_noise_thresh = parse_large_noise_thresh(args.large_noise_thresh)
            
            deepliif_results = deepliif_engine.inference(
                original_pil, 
                tile_size=args.tile_size, 
                seg_weights=args.seg_weights,
                resolution=args.resolution,
                do_postprocessing=args.enable_postprocessing,
                seg_thresh=args.seg_thresh,
                size_thresh=size_thresh,
                marker_thresh=args.marker_thresh,
                size_thresh_upper=args.size_thresh_upper,
                noise_thresh=args.noise_thresh,
                large_noise_thresh=large_noise_thresh,
                color_dapi=args.color_dapi,
                color_marker=args.color_marker
            )
            
            # 保存 DeepLIIF 中间结果
            if args.save_deepliif_outputs:
                save_deepliif_outputs(deepliif_results, args.output_dir, base_name)
            else:
                print("    (Skipping DeepLIIF intermediate file saving)")
            
            # 保存 Seg 概率图 (可选)
            if args.save_seg_npy and deepliif_results.get('Seg') is not None:
                print("  > Saving Seg probability map as npy...")
                seg_npy_dir = os.path.join(args.output_dir, "seg_probability")
                seg_npy_path = os.path.join(seg_npy_dir, f"{base_name}_seg.npy")
                save_seg_probability_npy(
                    deepliif_results.get('Seg'),
                    seg_npy_path,
                    metadata={
                        'image_name': img_name,
                        'seg_thresh': args.seg_thresh
                    }
                )
            
            # 检查 Seg 输出
            if deepliif_results.get('Seg') is None:
                print("    Error: No valid segmentation mask generated. Skipping this image.")
                continue
            
            # ========== Step B: 细胞提取与分类 ==========
            seg_img = deepliif_results.get('Seg')
            marker_img = deepliif_results.get('Marker')
            seg_np = np.array(seg_img)
            
            if marker_img is not None:
                marker_np = np.array(marker_img)
                print(f"  > Extracting cells from raw Seg with Marker classification...")
                
                all_cells_info = extract_cells_from_seg(
                    seg_np, marker_np, 
                    min_area=args.min_mask_area,
                    seg_thresh=args.seg_thresh,
                    marker_thresh=args.marker_thresh
                )
                
                # 可视化细胞提取过程 (如果启用)
                if hasattr(args, 'save_cell_extraction_vis') and args.save_cell_extraction_vis:
                    from src.cell_extraction import visualize_cell_extraction
                    vis_dir = os.path.join(args.output_dir, "cell_extraction_vis")
                    os.makedirs(vis_dir, exist_ok=True)
                    vis_path = os.path.join(vis_dir, f"{base_name}_cell_extraction.png")
                    print(f"  > Saving cell extraction visualization...")
                    visualize_cell_extraction(
                        seg_np, marker_np, all_cells_info,
                        output_path=vis_path,
                        seg_thresh=args.seg_thresh,
                        show_labels=True
                    )
                
                # 过滤阳性细胞 - 使用与 DeepLIIF 一致的动态阈值筛选
                # [已注释] 原第二层筛选：
                positive_cells_info = filter_positive_cells(all_cells_info, marker_sum_thresh=1000, marker_max_thresh=15)
                # 直接使用 is_positive 标志（由动态 marker_thresh 决定）
                # positive_cells_info = [c for c in all_cells_info if c.get('is_positive', False)]
                # print(f"    After dynamic marker filtering (is_positive=True): {len(positive_cells_info)} cells")
                
                num_positive = len(positive_cells_info)
                num_negative = len(all_cells_info) - num_positive
                print(f"    Total cells: {len(all_cells_info)} (Positive: {num_positive}, Negative: {num_negative})")
                
                clusters = get_clusters_from_cells(positive_cells_info)
                mask_np = create_binary_mask_from_cells(positive_cells_info, seg_np.shape)
            else:
                print(f"  > Marker not available, extracting all cells from Seg...")
                clusters = get_clusters_from_mask_image(seg_np, min_area=args.min_mask_area)
                positive_cells_info = None
                all_cells_info = None
                mask_np = np.zeros((seg_np.shape[0], seg_np.shape[1]), dtype=np.uint8)
            
            print(f"    Found {len(clusters)} positive cell regions for SAM2.")
            
            if len(clusters) == 0:
                print("    No regions found. Skipping SAM2.")
                
                # 移动没有阳性细胞的图像到专门的文件夹
                no_positive_dir = os.path.join(args.output_dir, "no_positive_cells")
                os.makedirs(no_positive_dir, exist_ok=True)
                
                # 移动原始图像
                dest_path = os.path.join(no_positive_dir, img_name)
                shutil.move(img_path, dest_path)
                print(f"    Moved image with no positive cells to: {dest_path}")
                
                # 如果已经生成了相关文件，也移动它们
                # 移动 background 图像 (如果存在)
                background_dir = os.path.join(args.output_dir, "background")
                if os.path.exists(background_dir):
                    bg_file = os.path.join(background_dir, f"{base_name}.png")
                    if os.path.exists(bg_file):
                        dest_bg = os.path.join(no_positive_dir, f"{base_name}_background.png")
                        shutil.move(bg_file, dest_bg)
                        print(f"    Moved background file to: {dest_bg}")
                
                # 移动 labelme 文件 (如果存在)
                labelme_dir = os.path.join(args.output_dir, "labelme")
                if os.path.exists(labelme_dir):
                    labelme_json = os.path.join(labelme_dir, f"{base_name}.json")
                    labelme_img = os.path.join(labelme_dir, f"{base_name}.png")
                    
                    if os.path.exists(labelme_json):
                        dest_json = os.path.join(no_positive_dir, f"{base_name}.json")
                        shutil.move(labelme_json, dest_json)
                        print(f"    Moved labelme JSON to: {dest_json}")
                    
                    if os.path.exists(labelme_img):
                        dest_img = os.path.join(no_positive_dir, f"{base_name}_labelme.png")
                        shutil.move(labelme_img, dest_img)
                        print(f"    Moved labelme image to: {dest_img}")
                
                continue
                
            # ========== Step C: SAM2 推理 (Mask-Only 模式) ==========
            image_set = False  # 跟踪是否已设置图像
            
            sam_mask_only = np.zeros((original_np.shape[0], original_np.shape[1]), dtype=np.uint8)
            scores_mask_only = []
            filtered_mask_only = []
            sam_mask_only_merged = np.zeros_like(sam_mask_only)
            scores_mask_only_merged = []
            mask_only_merge_mapping = {}
            merged_cells_info = positive_cells_info  # 默认使用原始 cells_info
            
            if not args.skip_mask_only:
                # 准备保存 SAM2 每一步中间结果的目录
                sam_steps_dir = None
                if hasattr(args, 'save_sam_steps') and args.save_sam_steps:
                    sam_steps_dir = os.path.join(args.output_dir, "sam2_steps", base_name)
                
                # 检查是否启用分组模式
                if hasattr(args, 'group_cells') and args.group_cells:
                    # ========== 分组模式 ==========
                    distance_threshold = getattr(args, 'group_distance', 50.0)
                    print(f"  > Grouping cells by distance (threshold={distance_threshold}px)...")
                    
                    cell_groups = group_cells_by_distance(positive_cells_info, distance_threshold)
                    print(f"    {len(positive_cells_info)} cells -> {len(cell_groups)} groups")
                    
                    # 显示分组详情
                    for group in cell_groups:
                        if len(group['member_ids']) > 1:
                            print(f"      Group {group['group_id']}: cells {group['member_ids']} (total: {group['total_pixels']} pixels)")
                    
                    # 保存分组信息 CSV
                    group_csv_dir = os.path.join(args.output_dir, "cell_groups")
                    os.makedirs(group_csv_dir, exist_ok=True)
                    group_csv_path = os.path.join(group_csv_dir, f"{base_name}_cell_groups.csv")
                    
                    with open(group_csv_path, 'w') as f:
                        f.write("group_id,num_cells,member_ids,total_pixels,center_y,center_x,distance_threshold,member_distances\n")
                        for group in cell_groups:
                            member_ids_str = ';'.join(map(str, group['member_ids']))
                            center_y, center_x = group['center']
                            
                            # 计算组内成员之间的距离
                            member_distances = []
                            member_cells = group['member_cells']
                            for i in range(len(member_cells)):
                                for j in range(i + 1, len(member_cells)):
                                    c1 = member_cells[i]['center']
                                    c2 = member_cells[j]['center']
                                    dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                                    member_distances.append(f"{member_cells[i]['id']}-{member_cells[j]['id']}:{dist:.1f}")
                            
                            distances_str = ';'.join(member_distances) if member_distances else 'single_cell'
                            
                            f.write(f"{group['group_id']},{len(group['member_ids'])},\"{member_ids_str}\",{group['total_pixels']},{center_y},{center_x},{distance_threshold},\"{distances_str}\"\n")
                    
                    print(f"    Saved grouping info to: {group_csv_path}")
                    
                    # 保存分组可视化图片
                    if hasattr(args, 'save_sam_steps') and args.save_sam_steps:
                        group_viz_path = os.path.join(group_csv_dir, f"{base_name}_cell_groups.png")
                        _save_grouping_visualization(
                            original_np, cell_groups, positive_cells_info, 
                            distance_threshold, group_viz_path
                        )
                        print(f"    Saved grouping visualization to: {group_viz_path}")
                    
                    print("  > Running SAM2 segmentation (Grouped mode)...")
                    sam_mask_only, scores_mask_only, filtered_mask_only = run_sam2_grouped_segmentation(
                        sam2_predictor, original_np, cell_groups,
                        min_area=args.min_mask_area,
                        set_image=True,
                        score_threshold=0.05,
                        save_steps_dir=sam_steps_dir
                    )
                else:
                    # ========== 原有逐个模式 ==========
                    print("  > Running SAM2 segmentation (Mask-Only mode)...")
                    sam_mask_only, scores_mask_only, filtered_mask_only = run_sam2_segmentation(
                        sam2_predictor, original_np, clusters, 
                        min_area=args.min_mask_area, prompt_mode='mask_only',
                        set_image=True,
                        score_threshold=0.05,
                        save_steps_dir=sam_steps_dir
                    )
                image_set = True
                
                # 合并连通掩码并过滤小面积区域
                print("  > Merging connected masks...")
                
                # 准备合并步骤保存目录
                merge_steps_dir = None
                if hasattr(args, 'save_sam_steps') and args.save_sam_steps:
                    merge_steps_dir = os.path.join(args.output_dir, "sam2_steps", base_name, "merge_filter")
                
                sam_mask_only_merged, scores_mask_only_merged, mask_only_merge_mapping, merged_cells_info = merge_connected_masks(
                    sam_mask_only, scores_mask_only, positive_cells_info,
                    min_area=200,  # 过滤小于 200 像素的零碎区域
                    save_steps_dir=merge_steps_dir,
                    original_image=original_np
                )
                
                # 保存合并区域的 CSV 统计信息
                if sam_mask_only_merged is not None and np.max(sam_mask_only_merged) > 0:
                    merged_regions_csv_path = os.path.join(args.output_dir, "merged_regions", f"{base_name}_merged_regions.csv")
                    os.makedirs(os.path.dirname(merged_regions_csv_path), exist_ok=True)
                    
                    # 统计每个区域的像素数
                    unique_ids = np.unique(sam_mask_only_merged)
                    unique_ids = unique_ids[unique_ids > 0]  # 排除背景
                    
                    with open(merged_regions_csv_path, 'w') as f:
                        f.write("region_id,pixel_count,avg_score,member_instances\n")
                        for region_id in sorted(unique_ids):
                            pixel_count = int(np.sum(sam_mask_only_merged == region_id))
                            # 从 scores_mask_only_merged 获取分数信息
                            score_info = next((s for s in scores_mask_only_merged if s[0] == region_id), None)
                            if score_info:
                                avg_score = score_info[1]
                                member_ids = score_info[2] if len(score_info) > 2 else []
                            else:
                                avg_score = 0.0
                                member_ids = []
                            f.write(f"{region_id},{pixel_count},{avg_score:.4f},\"{member_ids}\"\n")
                    
                    print(f"  > Saved merged regions CSV: {merged_regions_csv_path}")
                    print(f"    Total {len(unique_ids)} regions")
                    
            else:
                print("  > Skipping Mask-Only mode (--skip-mask-only)")
                if args.export_labelme:
                    print("    WARNING: LabelMe export requires Mask-Only mode! Export will be skipped.")
            
            # ========== Step D: 保存 SAM2 结果 ==========
            sam_out_dir = os.path.join(args.output_dir, "sam2_results", base_name)
            if args.save_sam_outputs:
                os.makedirs(sam_out_dir, exist_ok=True)
                _save_sam2_outputs(
                    sam_out_dir, original_np, positive_cells_info,
                    sam_mask_only, sam_mask_only_merged, filtered_mask_only
                )
            
            # ========== Step E: 保存对比图 ==========
            seg_overlaid_np = np.array(deepliif_results.get('SegOverlaid')) if deepliif_results.get('SegOverlaid') else None
            seg_refined_np = np.array(deepliif_results.get('SegRefined')) if deepliif_results.get('SegRefined') else None
            
            deepliif_params = {
                'seg_thresh': args.seg_thresh,
                'size_thresh': size_thresh if size_thresh != 'default' else 'auto',
                'size_thresh_upper': args.size_thresh_upper if args.size_thresh_upper else 'none',
                'marker_thresh': args.marker_thresh if args.marker_thresh else 'auto',
                'noise_thresh': args.noise_thresh,
                'large_noise_thresh': large_noise_thresh if large_noise_thresh not in ['default', None] else 'auto',
                'resolution': args.resolution,
            }
            
            sam_params = {
                'prompt_mode': 'mask_only',
                'mask_size': '256x256',
                'multimask_output': True,
                'num_prompts': len(clusters),
            }
            
            save_comparison(
                args.output_dir, base_name, original_np, 
                seg_np, seg_overlaid_np, seg_refined_np, mask_np,
                sam_mask_only_merged, sam_mask_only,  # 使用合并后的结果
                clusters=clusters,
                scores_box_mask=[],  # 已删除 box+mask 模式
                scores_mask_only=scores_mask_only,
                deepliif_params=deepliif_params,
                sam_params=sam_params,
                marker=marker_np if marker_img is not None else None,
                positive_cells_info=positive_cells_info,
                filtered_box_mask=[],
                filtered_mask_only=filtered_mask_only,
                merge_info=[],
                save_comparison_image=args.save_comparison,
                save_combined_image=args.save_combined
            )
            
            # ========== Step E.1: 保存原图与 SAM2 结果对比图 ==========
            if args.save_original_sam_comparison and not args.skip_mask_only:
                print("  > Saving original vs SAM2 comparison...")
                comparison_path = save_original_sam_comparison(
                    args.output_dir, base_name, original_np, sam_mask_only_merged
                )
                print(f"    Saved to: {comparison_path}")
            
            # ========== Step F: 保存 CSV ==========
            if positive_cells_info is not None and len(positive_cells_info) > 0 and args.save_csv:
                print("  > Saving positive cells info...")
                positive_cells_info = renumber_cells(positive_cells_info)
                
                comparison_dir = os.path.join(args.output_dir, "comparison")
                os.makedirs(comparison_dir, exist_ok=True)
                csv_path = f"{comparison_dir}/{base_name}_PositiveCells.csv"
                save_positive_cells_csv(csv_path, positive_cells_info)
                
                total_pixels = sum(c['pixel_count'] for c in positive_cells_info)
                total_marker = sum(c['marker_sum'] for c in positive_cells_info)
                avg_marker = total_marker / total_pixels if total_pixels > 0 else 0
                
                print(f"    Found {len(positive_cells_info)} positive cells.")
                print(f"    Total positive pixels: {total_pixels}")
                print(f"    Average marker value: {avg_marker:.2f}")
                print(f"    Saved CSV to comparison folder.")
            
            # ========== Step G: 导出 LabelMe (可选) ==========
            if args.export_labelme and not args.skip_mask_only:
                print("  > Exporting to LabelMe format...")
                json_path, num_shapes = export_labelme_annotation(
                    output_dir=args.output_dir,
                    base_name=base_name,
                    instance_mask=sam_mask_only_merged,
                    original_image_array=original_np,
                    cells_info=merged_cells_info,  # Use merged cells_info to match merged mask
                    include_image_data=args.labelme_include_imagedata,
                    original_image_path=img_path  # Pass original image path for absolute path in JSON
                )
                
                # 检查是否为空结果（SAM2 分割失败）
                if num_shapes == 0:
                    print("    WARNING: SAM2 produced no valid segmentations!")
                    
                    # 创建 sam2_failed 文件夹（与 labelme 同级）
                    sam2_failed_dir = os.path.join(args.output_dir, "sam2_failed")
                    os.makedirs(sam2_failed_dir, exist_ok=True)
                    
                    # 删除空的 JSON 文件（不再保存到 sam2_failed）
                    if os.path.exists(json_path):
                        os.remove(json_path)
                        print(f"    Deleted empty JSON: {json_path}")
                    
                    # 移动原始输入图像到 sam2_failed
                    dest_img = os.path.join(sam2_failed_dir, img_name)
                    if os.path.exists(img_path):
                        shutil.move(img_path, dest_img)
                        print(f"    Moved original image to: {dest_img}")
                    
                    # 如果 labelme 目录有复制的图像，也删除它
                    labelme_dir = os.path.join(args.output_dir, "labelme")
                    labelme_img = os.path.join(labelme_dir, f"{base_name}.png")
                    if os.path.exists(labelme_img):
                        os.remove(labelme_img)
                        print(f"    Removed duplicate from labelme/")

                else:
                    # 成功：移动原图到 labelme 目录
                    labelme_dir = os.path.join(args.output_dir, "labelme")
                    dest_img = os.path.join(labelme_dir, img_name)
                    # 如果原图还在输入目录，移动它
                    if os.path.exists(img_path):
                        shutil.move(img_path, dest_img)
                        print(f"    Moved original image to: {dest_img}")
                    print(f"    Use command: labelme {args.output_dir}/labelme/{base_name}.json")
            
            # ========== Step H: 保存 npy 格式 mask (可选) ==========
            if args.save_npy and not args.skip_mask_only:
                print("  > Saving instance mask as npy format...")
                npy_dir = os.path.join(args.output_dir, "npy_masks")
                npy_path = os.path.join(npy_dir, f"{base_name}.npy")
                
                # 准备元数据
                npy_metadata = {
                    'image_name': img_name,
                    'image_size': [original_np.shape[0], original_np.shape[1]],
                    'num_instances': int(np.max(sam_mask_only_merged)),
                    'tile_size': args.tile_size,
                }
                
                save_mask_npy(sam_mask_only_merged, npy_path, metadata=npy_metadata)
            
            print("  > Success.")
            
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print_pipeline_footer()


def _save_sam2_outputs(sam_out_dir: str, original_np: np.ndarray, 
                       positive_cells_info: list,
                       sam_mask_only: np.ndarray, sam_mask_only_merged: np.ndarray,
                       filtered_mask_only: list):
    """保存 SAM2 Mask-Only 输出结果 (辅助函数)"""
    
    # 保存 mask prompts
    mask_prompt_dir = os.path.join(sam_out_dir, "mask_prompts")
    os.makedirs(mask_prompt_dir, exist_ok=True)
    
    if positive_cells_info and len(positive_cells_info) > 0:
        h, w = original_np.shape[:2]
        colors = generate_distinct_colors(len(positive_cells_info))
        
        # Combined visualization
        combined_mask_viz = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, cell in enumerate(positive_cells_info):
            coords = cell['coords']
            color = colors[idx] if idx < len(colors) else (255, 0, 0)
            combined_mask_viz[coords[:, 0], coords[:, 1]] = color
            
            center_y, center_x = cell['center']
            cv2.putText(combined_mask_viz, str(idx + 1), (center_x - 8, center_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(combined_mask_viz, str(idx + 1), (center_x - 8, center_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.imwrite(f"{mask_prompt_dir}/mask_prompts_combined.png", 
                   cv2.cvtColor(combined_mask_viz, cv2.COLOR_RGB2BGR))
        
        # Low-res version
        low_res_combined = np.zeros((256, 256), dtype=np.float32)
        for idx, cell in enumerate(positive_cells_info):
            mask_input = generate_mask_from_cluster(cell['coords'], original_np.shape)
            low_res_combined = np.maximum(low_res_combined, mask_input[0])
        
        low_res_viz = ((low_res_combined + 10) / 20 * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(f"{mask_prompt_dir}/mask_prompts_256x256.png", low_res_viz)
        
        print(f"  > Saved mask prompts to {mask_prompt_dir}/")
    
    # 保存 mask-only 结果
    filtered_ids = set(inst_id for inst_id, _ in filtered_mask_only)
    
    # 原始 mask-only
    save_sam2_mask_visualization(
        sam_mask_only, 
        f"{sam_out_dir}/sam_mask_only.png",
        positive_cells_info,
        filtered_ids
    )
    
    # 合并后的 mask-only (过滤小面积后的结果)
    save_sam2_mask_visualization(
        sam_mask_only_merged, 
        f"{sam_out_dir}/sam_mask_only_merged.png",
        positive_cells_info,
        set()  # merged 版本已经过滤完毕
    )


def _save_grouping_visualization(original_image: np.ndarray, cell_groups: list, 
                                  cells_info: list, distance_threshold: float, 
                                  save_path: str):
    """
    保存分组过程的可视化图片。
    
    Args:
        original_image: 原始图像
        cell_groups: 分组列表
        cells_info: 所有细胞信息
        distance_threshold: 分组距离阈值
        save_path: 保存路径
    """
    h, w = original_image.shape[:2]
    
    # 创建可视化图像 (在原图上叠加)
    viz = original_image.copy()
    
    # 生成不同组的颜色
    np.random.seed(42)
    group_colors = np.random.randint(100, 255, size=(len(cell_groups) + 1, 3))
    
    # 绘制每个组
    for group in cell_groups:
        group_id = group['group_id']
        color = tuple(int(c) for c in group_colors[group_id])
        member_cells = group['member_cells']
        
        # 绘制组内所有细胞
        for cell in member_cells:
            coords = cell['coords']
            rows = coords[:, 0].astype(np.intp)
            cols = coords[:, 1].astype(np.intp)
            # 半透明叠加
            viz[rows, cols] = (np.array(viz[rows, cols]) * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        
        # 如果组内有多个细胞，绘制连线表示分组关系
        if len(member_cells) > 1:
            for i in range(len(member_cells)):
                for j in range(i + 1, len(member_cells)):
                    c1 = member_cells[i]['center']
                    c2 = member_cells[j]['center']
                    # 绘制连线
                    cv2.line(viz, (c1[1], c1[0]), (c2[1], c2[0]), color, 2)
        
        # 在组中心添加组 ID 标签
        center_y, center_x = group['center']
        label = f"G{group_id}({len(member_cells)})"
        cv2.putText(viz, label, (center_x - 20, center_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(viz, label, (center_x - 20, center_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # 添加图例
    cv2.putText(viz, f"Groups: {len(cell_groups)}, Threshold: {distance_threshold}px", 
               (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(viz, f"Groups: {len(cell_groups)}, Threshold: {distance_threshold}px", 
               (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.imwrite(save_path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
