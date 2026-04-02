#!/usr/bin/env python3
"""
Full Pipeline: DeepLIIF -> SAM2

主流程编排模块，具体实现分布在 cd34_pipeline 各子包中。

Usage:
    python scripts/run_pipeline.py --input-dir /path/to/images --output-dir /path/to/save
"""

import os
import sys
import numpy as np
import torch

# Import modular components
from cd34_pipeline.config import (
    parse_arguments,
    validate_config,
    parse_size_thresh,
    parse_large_noise_thresh,
    print_pipeline_header,
    print_pipeline_footer
)
from cd34_pipeline.sam2_wrapper.model_loader import load_all_models
from cd34_pipeline.cell.extraction import (
    extract_cells_from_seg,
    filter_positive_cells,
    get_clusters_from_cells,
    create_binary_mask_from_cells,
    renumber_cells,
    group_cells_by_distance
)
from cd34_pipeline.cell.mask_utils import (
    get_clusters_from_mask_image,
    generate_distinct_colors
)
from cd34_pipeline.sam2_wrapper.inference import (
    run_sam2_segmentation,
    run_sam2_grouped_segmentation,
    merge_connected_masks
)
from cd34_pipeline.visualization.visualize import (
    save_comparison,
    save_original_sam_comparison,
    save_pipeline_comparison,
    save_grouping_visualization
)
from cd34_pipeline.io.file_io import (
    get_image_files,
    read_image,
    save_deepliif_outputs,
    save_positive_cells_csv,
    export_labelme_annotation,
    save_sam2_mask_visualization,
    save_mask_npy,
    save_seg_probability_npy,
    prepare_resume_skip_set,
    handle_no_positive_cells,
    save_cell_groups_csv,
    save_merged_regions_csv,
    export_and_handle_labelme,
    save_sam2_outputs
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

        # ========== 3.1 断点续传：跳过已处理图像，从最后一张重新开始 ==========
        skip_set = prepare_resume_skip_set(args.output_dir, args.resume)

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

                # ===== 连通区域提取模式: Seg+Marker联合 =====
                if hasattr(args, 'use_connected_regions') and args.use_connected_regions:
                    print(f"  > [Connected Region Mode] Extracting connected positive regions (Seg+Marker)...")
                    from cd34_pipeline.cell.extraction import extract_connected_positive_regions

                    positive_cells_info = extract_connected_positive_regions(
                        seg_np, marker_np,
                        seg_thresh=args.seg_thresh,
                        marker_thresh=args.marker_thresh,
                        morphology_kernel=args.morphology_kernel,
                        min_area=args.min_mask_area
                    )

                    print(f"    Found {len(positive_cells_info)} connected positive regions")
                    print(f"    Parameters: seg_thresh={args.seg_thresh}, kernel={args.morphology_kernel}, min_area={args.min_mask_area}")

                    if positive_cells_info:
                        areas = [c['pixel_count'] for c in positive_cells_info]
                        print(f"    Region sizes: min={min(areas)}, max={max(areas)}, mean={np.mean(areas):.0f} pixels")

                    all_cells_info = positive_cells_info

                # ===== 原有模式: 从Seg提取单个细胞 =====
                else:
                    print(f"  > [Cell Mode] Extracting cells from raw Seg with Marker classification...")

                    all_cells_info = extract_cells_from_seg(
                        seg_np, marker_np,
                        min_area=args.min_mask_area,
                        seg_thresh=args.seg_thresh,
                        marker_thresh=args.marker_thresh
                    )

                # 可视化细胞提取过程 (如果启用) — 仅限原有细胞模式
                if hasattr(args, 'save_cell_extraction_vis') and args.save_cell_extraction_vis:
                    if not (hasattr(args, 'use_connected_regions') and args.use_connected_regions):
                        from cd34_pipeline.cell.extraction import visualize_cell_extraction
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

                # 过滤阳性细胞 + pipeline 可视化
                save_pipeline_vis_later = hasattr(args, 'save_pipeline_vis') and args.save_pipeline_vis

                if hasattr(args, 'use_connected_regions') and args.use_connected_regions:
                    # 连通区域模式：不需要二次过滤
                    positive_cells_info = all_cells_info
                    print(f"    [Connected Region Mode] All {len(positive_cells_info)} regions are positive (no filtering)")

                    # 连通区域模式的 pipeline 可视化
                    if save_pipeline_vis_later:
                        from cd34_pipeline.visualization.pipeline_viz import save_connected_region_visualization
                        save_connected_region_visualization(
                            seg_np, marker_np, positive_cells_info,
                            args.output_dir, base_name,
                            seg_thresh=args.seg_thresh,
                            marker_thresh=args.marker_thresh,
                            morphology_kernel=args.morphology_kernel,
                            original_image=original_np
                        )
                else:
                    # 原有的细胞过滤逻辑
                    filter_params = {
                        'marker_sum_thresh': 1000,
                        'marker_max_thresh': 30,
                        'min_pixel_count': 100
                    }
                    positive_cells_info = filter_positive_cells(
                        all_cells_info,
                        marker_sum_thresh=filter_params['marker_sum_thresh'],
                        marker_max_thresh=filter_params['marker_max_thresh'],
                        min_pixel_count=filter_params['min_pixel_count']
                    )

                    # 原有模式的 pipeline 可视化
                    if save_pipeline_vis_later:
                        from cd34_pipeline.visualization.pipeline_viz import save_pipeline_visualization
                        save_pipeline_visualization(
                            seg_np, marker_np, all_cells_info,
                            args.output_dir, base_name,
                            seg_thresh=args.seg_thresh,
                            marker_thresh=args.marker_thresh,
                            original_image=original_np,
                            filtered_cells_info=positive_cells_info,
                            filter_params=filter_params
                        )

                num_positive = len(positive_cells_info)
                num_negative = len(all_cells_info) - num_positive
                print(f"    Total cells: {len(all_cells_info)} (Positive: {num_positive}, Negative: {num_negative})")

                # 无阳性细胞则跳过后续处理
                if num_positive == 0:
                    print("    No positive cells after filtering. Skipping SAM2.")
                    handle_no_positive_cells(img_path, img_name, args.output_dir, base_name)
                    continue

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
                handle_no_positive_cells(img_path, img_name, args.output_dir, base_name)
                continue

            # ========== Step C: SAM2 推理 (Mask-Only 模式) ==========
            image_set = False

            sam_mask_only = np.zeros((original_np.shape[0], original_np.shape[1]), dtype=np.uint8)
            scores_mask_only = []
            filtered_mask_only = []
            sam_mask_only_merged = np.zeros_like(sam_mask_only)
            scores_mask_only_merged = []
            mask_only_merge_mapping = {}
            merged_cells_info = positive_cells_info

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

                    for group in cell_groups:
                        if len(group['member_ids']) > 1:
                            print(f"      Group {group['group_id']}: cells {group['member_ids']} (total: {group['total_pixels']} pixels)")

                    # 保存分组信息
                    group_csv_path = save_cell_groups_csv(cell_groups, args.output_dir, base_name, distance_threshold)

                    # 保存分组可视化图片
                    if hasattr(args, 'save_sam_steps') and args.save_sam_steps:
                        group_csv_dir = os.path.join(args.output_dir, "cell_groups")
                        group_viz_path = os.path.join(group_csv_dir, f"{base_name}_cell_groups.png")
                        save_grouping_visualization(
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
                merge_steps_dir = None
                if hasattr(args, 'save_sam_steps') and args.save_sam_steps:
                    merge_steps_dir = os.path.join(args.output_dir, "sam2_steps", base_name, "merge_filter")

                sam_mask_only_merged, scores_mask_only_merged, mask_only_merge_mapping, merged_cells_info = merge_connected_masks(
                    sam_mask_only, scores_mask_only, positive_cells_info,
                    min_area=200,
                    save_steps_dir=merge_steps_dir,
                    original_image=original_np
                )

                # 保存合并区域统计
                save_merged_regions_csv(sam_mask_only_merged, scores_mask_only_merged, args.output_dir, base_name)

            else:
                print("  > Skipping Mask-Only mode (--skip-mask-only)")
                if args.export_labelme:
                    print("    WARNING: LabelMe export requires Mask-Only mode! Export will be skipped.")

            # ========== Step D: 保存 SAM2 结果 ==========
            if args.save_sam_outputs:
                sam_out_dir = os.path.join(args.output_dir, "sam2_results", base_name)
                os.makedirs(sam_out_dir, exist_ok=True)
                save_sam2_outputs(
                    sam_out_dir, original_np, positive_cells_info,
                    sam_mask_only, sam_mask_only_merged, filtered_mask_only
                )

            # ========== Step E: 保存对比图 ==========
            save_pipeline_comparison(
                args.output_dir, base_name, original_np,
                deepliif_results, mask_np,
                sam_mask_only_merged, sam_mask_only,
                clusters, scores_mask_only,
                positive_cells_info, filtered_mask_only,
                marker_np if marker_img is not None else None, args
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
                export_and_handle_labelme(
                    args.output_dir, base_name, sam_mask_only_merged,
                    original_np, merged_cells_info,
                    img_path, img_name, args
                )

            # ========== Step H: 保存 npy 格式 mask (可选) ==========
            if args.save_npy and not args.skip_mask_only:
                print("  > Saving instance mask as npy format...")
                npy_dir = os.path.join(args.output_dir, "npy_masks")
                npy_path = os.path.join(npy_dir, f"{base_name}.npy")

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


if __name__ == "__main__":
    main()
