#!/usr/bin/env python3
"""
sam2_inference.py — SAM2 推理模块

包含：
- 多种 SAM2 推理模式
- 掩码后处理与合并
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from cd34_pipeline.cell.mask_utils import (
    generate_mask_from_cluster,
    merge_overlapping_cells
)


def run_sam2_segmentation(predictor, image: np.ndarray, clusters: list,
                           min_area: int = 10,
                           set_image: bool = True, score_threshold: float = 0.0,
                           debug_dir: str = None) -> tuple:
    """
    对候选区域（clusters）逐个执行 SAM2 实例分割（mask_only 模式）。

    Args:
        predictor: SAM2ImagePredictor 实例
        image: RGB 图像数组
        clusters: 候选区域坐标列表，每个元素是一个 (N, 2) 的坐标数组
        min_area: 送入 SAM 前的最小 cluster 面积
        set_image: 是否调用 predictor.set_image() 对图像进行编码
        score_threshold: 最低置信度阈值，低于此分数的结果会被过滤（0.0 表��不过滤）
        debug_dir: 若提供路径，则保存每个实例的中间结果用于区域调试

    Returns:
        combined_mask: 实例分割掩码，每个像素值为对应的实例 ID
        scores_list: 保留的实例列表，每项为 (instance_id, score) 元组
        filtered_list: 被过滤掉的实例列表，每项为 (instance_id, score) 元组
    """
    import os

    # 对图像进行 SAM2 编码（只需执行一次）
    if set_image:
        predictor.set_image(image)

    h, w = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint16)  # 实例分割结果掩码
    score_map = np.zeros((h, w), dtype=np.float32)      # 每个像素对应的最高置信度
    scores_list = []    # 保留的实例及其分数
    filtered_list = []  # 被过滤的实例及其分数

    # 准备保存目录
    if debug_dir:
        sam2_debug_dir = os.path.join(debug_dir, "sam2_steps")
        os.makedirs(sam2_debug_dir, exist_ok=True)
        steps_summary = []  # 存储所有步骤的汇总信息

    # 逐个处理每个候选区域
    for idx, cluster in enumerate(clusters):
        if len(cluster) < min_area:
            continue

        try:
            # 根据 cluster 坐标生成 256x256 的掩码 prompt
            mask_input = generate_mask_from_cluster(cluster, image.shape)

            # 仅使用掩码作为提示
            masks, scores, low_res_masks = predictor.predict(
                mask_input=mask_input,
                multimask_output=True
            )

            # 从 3 个候选中选择置信度最高的掩码
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            best_mask = masks[best_idx].astype(bool)
            mask_area = np.sum(best_mask)

            inst_id = idx + 1  # 实例 ID 从 1 开始

            # 保存每一步的中间结果（用于调试可视化）
            if debug_dir:
                _save_sam2_step_results(
                    sam2_debug_dir, inst_id, image, cluster, mask_input,
                    masks, scores, best_idx
                )
                steps_summary.append({
                    'instance_id': int(inst_id),
                    'cluster_size': int(len(cluster)),
                    'scores': [float(s) for s in scores],
                    'best_idx': int(best_idx),
                    'best_score': float(best_score),
                    'mask_areas': [int(np.sum(m)) for m in masks],
                    'best_mask_area': int(mask_area)
                })

            # 置信度过滤：低于阈值的实例不纳入最终结果
            if score_threshold > 0 and best_score < score_threshold:
                print(f"      Instance {inst_id}: score={best_score:.4f}, area={mask_area} pixels [FILTERED: score<{score_threshold}]")
                filtered_list.append((inst_id, best_score))
                continue

            # 置信度优先合并：仅在新实例分数高于已有像素分数时才覆盖
            overwrite_mask = best_mask & (best_score > score_map)
            combined_mask[overwrite_mask] = inst_id
            score_map[overwrite_mask] = best_score

            scores_list.append((inst_id, best_score))

            # 统计合并后该实例的实际像素数（可能因重叠被部分覆盖）
            final_area = np.sum(combined_mask == inst_id)
            print(f"      Instance {inst_id}: score={best_score:.4f}, area={mask_area} pixels [final: {final_area} pixels]")

        except Exception as e:
            import traceback
            print(f"    SAM2 Error on cluster {idx}: {e}")
            print(f"      Cluster dtype: {cluster.dtype}, shape: {cluster.shape}")
            print(f"      Traceback: {traceback.format_exc()}")

    # 保存所有步骤的汇总信息到 JSON 文件
    if debug_dir:
        import json
        summary_path = os.path.join(sam2_debug_dir, "steps_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(steps_summary, f, indent=2)
        print(f"      Saved {len(steps_summary)} step details to {sam2_debug_dir}/")

    return combined_mask, scores_list, filtered_list


def run_sam2_grouped_segmentation(predictor, image: np.ndarray, cell_groups: list,
                                   min_area: int = 10, set_image: bool = True,
                                   score_threshold: float = 0.0,
                                   debug_dir: str = None) -> tuple:
    """
    Run SAM2 segmentation on grouped cells.

    Each group contains multiple nearby cells, and we send merged mask prompt
    for each group to SAM2.

    Args:
        predictor: SAM2ImagePredictor instance
        image: RGB image array
        cell_groups: List of cell group dicts from group_cells_by_distance()
        min_area: Minimum area threshold
        set_image: Whether to call predictor.set_image()
        score_threshold: Minimum score threshold for filtering
        debug_dir: If provided, save intermediate results for region debugging.

    Returns:
        combined_mask: Instance segmentation mask (group_id for each pixel)
        scores_list: List of (group_id, score) tuples
        filtered_list: List of (group_id, score) tuples for filtered instances
    """
    import os
    
    if set_image:
        predictor.set_image(image)
        
    h, w = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint16)
    score_map = np.zeros((h, w), dtype=np.float32)
    scores_list = []
    filtered_list = []
    steps_summary = []

    if debug_dir:
        grouped_debug_dir = os.path.join(debug_dir, "sam2_grouped_steps")
        os.makedirs(grouped_debug_dir, exist_ok=True)

    for group in cell_groups:
        group_id = group['group_id']
        merged_coords = group['merged_coords']
        member_ids = group['member_ids']
        total_pixels = group['total_pixels']
        
        if total_pixels < min_area:
            continue
        
        try:
            # Generate merged mask prompt from all cells in the group
            mask_input = generate_mask_from_cluster(merged_coords, image.shape)
            
            # Get center point from EACH member cell as multiple point prompts
            member_cells = group.get('member_cells', [])
            if member_cells and len(member_cells) > 0:
                # 方案 B: 每个成员细胞一个中心点
                point_list = []
                for cell in member_cells:
                    cy, cx = cell['center']
                    point_list.append([cx, cy])
                point_coords = np.array(point_list, dtype=np.float32)
                point_labels = np.ones(len(point_list), dtype=np.int32)  # 全部是正点
            else:
                # 回退到组中心点（如果没有 member_cells 信息）
                center_y, center_x = group['center']
                point_coords = np.array([[center_x, center_y]], dtype=np.float32)
                point_labels = np.array([1], dtype=np.int32)
            
            # SAM2 inference with mask + multiple center points
            masks, scores, low_res_masks = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                multimask_output=True
            )
            
            # ========== 面积优先选择策略 ==========
            # 计算每个候选 mask 的面积比例（相对于输入）
            mask_areas = [int(np.sum(m)) for m in masks]
            area_ratios = [a / total_pixels if total_pixels > 0 else 0 for a in mask_areas]
            
            # 理想面积比例范围：0.5 ~ 5.0（允许 SAM2 适度调整边界）
            # 优先选择面积比例接近 1.5 的 mask（SAM2 通常会稍微扩张）
            ideal_ratio = 1.5
            min_ratio = 0.3
            max_ratio = 10.0
            
            # 计算每个 mask 的综合得分：面积合理性 + 置信度
            selection_scores = []
            for i, (score, ratio) in enumerate(zip(scores, area_ratios)):
                if ratio < min_ratio or ratio > max_ratio:
                    # 面积比例异常，大幅降低分数
                    combined_score = -1.0
                else:
                    # 面积偏离度（越接近 ideal_ratio 越好）
                    ratio_penalty = abs(np.log(ratio / ideal_ratio))  # log 尺度惩罚
                    # 综合得分 = 置信度 - 面积偏离惩罚
                    combined_score = float(score) - 0.3 * ratio_penalty
                selection_scores.append(combined_score)
            
            best_idx = int(np.argmax(selection_scores))
            best_score = float(scores[best_idx])
            best_mask = masks[best_idx].astype(bool)
            mask_area = mask_areas[best_idx]
            best_ratio = area_ratios[best_idx]
            
            # 如果所有 mask 的面积比例都异常，则跳过这个组
            if max(selection_scores) < -0.5:
                print(f"      Group {group_id} (cells: {member_ids}): ALL masks have abnormal area ratio! Ratios={[f'{r:.1f}' for r in area_ratios]} [SKIPPED]")
                filtered_list.append((group_id, best_score))
                continue
            
            # Save step results
            if debug_dir:
                _save_grouped_step_results(
                    grouped_debug_dir, group_id, image, merged_coords, mask_input,
                    masks, scores, best_idx, member_ids, member_cells
                )
                steps_summary.append({
                    'group_id': int(group_id),
                    'member_ids': [int(m) for m in member_ids],
                    'num_cells': len(member_ids),
                    'total_input_pixels': int(total_pixels),
                    'scores': [float(s) for s in scores],
                    'selection_scores': [float(s) for s in selection_scores],
                    'area_ratios': [float(r) for r in area_ratios],
                    'best_idx': int(best_idx),
                    'best_score': float(best_score),
                    'best_ratio': float(best_ratio),
                    'mask_areas': mask_areas,
                    'best_mask_area': int(mask_area)
                })
            
            # Filter low confidence results
            if score_threshold > 0 and best_score < score_threshold:
                print(f"      Group {group_id} (cells: {member_ids}): score={best_score:.4f}, area={mask_area} [FILTERED: score<{score_threshold}]")
                filtered_list.append((group_id, best_score))
                continue
            
            # Confidence-priority merge
            overwrite_mask = best_mask & (best_score > score_map)
            combined_mask[overwrite_mask] = group_id
            score_map[overwrite_mask] = best_score
            
            scores_list.append((group_id, best_score))
            
            final_area = np.sum(combined_mask == group_id)
            cells_str = str(member_ids) if len(member_ids) <= 3 else f"{member_ids[:3]}...({len(member_ids)} cells)"
            num_points = len(point_coords)
            print(f"      Group {group_id} {cells_str}: score={best_score:.4f}, ratio={best_ratio:.2f}x, area={mask_area} [final: {final_area}] ({num_points}pts)")
            
        except Exception as e:
            import traceback
            print(f"    SAM2 Error on group {group_id}: {e}")
            print(f"      Member cells: {member_ids}")
            print(f"      Traceback: {traceback.format_exc()}")
    
    # Save summary
    if debug_dir:
        import json
        summary_path = os.path.join(grouped_debug_dir, "grouped_steps_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(steps_summary, f, indent=2)
        print(f"      Saved {len(steps_summary)} group details to {grouped_debug_dir}/")
    
    return combined_mask, scores_list, filtered_list


def run_sam2_multi_image_batch(predictor, images: list, clusters_list: list,
                                min_area: int = 10,
                                prompt_batch_size: int = 64,
                                score_threshold: float = 0.0) -> list:
    """
    多图批量 SAM2 推理。

    不调用 predictor 的批量 set-image 封装接口。这里直接调用 SAM2 的
    forward_image() 批量执行 encoder，然后把所有图片上的有效
    mask prompts 展平成全局 prompt batch，按 prompt 所属图片索引
    对齐 image/high-res features 后批量执行 prompt_encoder + mask_decoder。

    Args:
        predictor: SAM2ImagePredictor instance
        images: M 张 RGB 图像列表，每张 (H, W, 3) np.ndarray
        clusters_list: M 个 cluster 列表，clusters_list[i] 对应 images[i]
        min_area: 送入 SAM 前的最小 cluster 面积
        prompt_batch_size: 全局 prompt 批大小
        score_threshold: 置信度阈值，低于此值的实例被过滤

    Returns:
        长度 M 的列表，每项为 (combined_mask, scores_list, filtered_list)
    """
    num_images = len(images)
    if num_images == 0:
        return []
    if len(clusters_list) != num_images:
        raise ValueError(
            f"clusters_list length ({len(clusters_list)}) must match images length ({num_images})"
        )

    orig_hws = []
    for image in images:
        if not isinstance(image, np.ndarray):
            raise TypeError("images must be np.ndarray arrays in RGB HWC format")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "images must be RGB arrays with shape (H, W, 3); "
                f"got {image.shape}"
            )
        orig_hws.append(tuple(image.shape[:2]))

    image_groups = {}
    for img_idx, orig_hw in enumerate(orig_hws):
        image_groups.setdefault(orig_hw, []).append(img_idx)

    image_group_positions = np.empty(num_images, dtype=np.int64)
    combined_masks_by_hw = {}
    score_maps_by_hw = {}
    prompt_records_by_hw = {orig_hw: [] for orig_hw in image_groups}
    for orig_hw, img_indices in image_groups.items():
        img_indices_arr = np.asarray(img_indices, dtype=np.int64)
        image_group_positions[img_indices_arr] = np.arange(len(img_indices_arr))
        h_img, w_img = orig_hw
        combined_masks_by_hw[orig_hw] = np.zeros(
            (len(img_indices), h_img, w_img), dtype=np.uint16
        )
        score_maps_by_hw[orig_hw] = np.zeros(
            (len(img_indices), h_img, w_img), dtype=np.float32
        )

    score_records = []
    filtered_records = []

    for img_idx, clusters in enumerate(clusters_list):
        if clusters is None:
            continue
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) < min_area:
                continue
            prompt_records_by_hw[orig_hws[img_idx]].append(
                (img_idx, cluster_idx, cluster)
            )

    num_prompts = sum(len(records) for records in prompt_records_by_hw.values())

    with torch.no_grad():
        # 1. 手动 batch 编码所有图像，避免走 predictor 的批量 set-image 封装。
        predictor.reset_predictor()
        predictor._orig_hw = orig_hws

        can_stack_images = (
            len({image.shape for image in images}) == 1
            and all(image.dtype == np.uint8 for image in images)
            and hasattr(predictor._transforms, "transforms")
        )
        if can_stack_images:
            img_batch = torch.as_tensor(np.stack(images, axis=0))
            img_batch = img_batch.permute(0, 3, 1, 2).contiguous().float()
            img_batch = predictor._transforms.transforms(img_batch.div_(255.0))
            img_batch = img_batch.to(predictor.device)
        else:
            img_batch = predictor._transforms.forward_batch(images).to(
                predictor.device
            )
        assert (
            len(img_batch.shape) == 4 and img_batch.shape[1] == 3
        ), f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"

        backbone_out = predictor.model.forward_image(img_batch)
        _, vision_feats, _, feat_sizes = (
            predictor.model._prepare_backbone_features(backbone_out)
        )
        if predictor.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + predictor.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(num_images, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]
        predictor._features = {
            "image_embed": feats[-1],
            "high_res_feats": feats[:-1],
        }
        predictor._is_image_set = True
        predictor._is_batch = True

        if num_prompts == 0:
            return [
                (
                    combined_masks_by_hw[orig_hws[i]][image_group_positions[i]],
                    [],
                    [],
                )
                for i in range(num_images)
            ]

        if prompt_batch_size is None or prompt_batch_size <= 0:
            prompt_batch_size = num_prompts

        image_embeddings_all = predictor._features["image_embed"]
        high_res_features_all = predictor._features["high_res_feats"]
        dense_pe = predictor.model.sam_prompt_encoder.get_dense_pe()
        mask_input_size = predictor.model.sam_prompt_encoder.mask_input_size

        # 2. 全局 prompt batch：两阶段处理。
        #    Phase 1: 在 CPU 上预构建所有 mask tensor + 元数据（GPU 空闲一次性完成）
        #    Phase 2: 紧凑 GPU 循环（最小化 GPU 间隙）
        for orig_hw, prompt_records in prompt_records_by_hw.items():
            if not prompt_records:
                continue

            h_img, w_img = orig_hw
            group_size = len(image_groups[orig_hw])
            combined_masks = combined_masks_by_hw[orig_hw]
            score_maps = score_maps_by_hw[orig_hw]

            # ── Phase 1: Pre-build all mask tensors at low resolution ──
            # Build at mask_input_size (256×256) directly instead of full
            # resolution (512×512) + F.interpolate → 4x less allocation.
            mask_h, mask_w = mask_input_size
            scale_h = mask_h / h_img
            scale_w = mask_w / w_img

            prebuild: list[tuple] = []
            for batch_start in range(0, len(prompt_records), prompt_batch_size):
                batch = prompt_records[batch_start:batch_start + prompt_batch_size]
                batch_len = len(batch)

                batch_img_indices = np.fromiter(
                    (item[0] for item in batch), dtype=np.int64, count=batch_len
                )
                batch_group_indices = image_group_positions[batch_img_indices]
                batch_inst_ids = np.fromiter(
                    (item[1] + 1 for item in batch), dtype=np.int64, count=batch_len
                )

                batch_clusters = [
                    np.asarray(item[2], dtype=np.int64).reshape(-1, 2)
                    for item in batch
                ]
                cluster_lengths = np.fromiter(
                    (len(cluster) for cluster in batch_clusters),
                    dtype=np.int64,
                    count=batch_len,
                )
                full_masks = np.full(
                    (batch_len, mask_h, mask_w), -5.0, dtype=np.float32
                )
                if cluster_lengths.sum() > 0:
                    all_coords = np.concatenate(batch_clusters, axis=0)
                    all_prompt_indices = np.repeat(
                        np.arange(batch_len), cluster_lengths
                    )
                    rows = (all_coords[:, 0] * scale_h).astype(np.intp)
                    cols = (all_coords[:, 1] * scale_w).astype(np.intp)
                    valid = (
                        (rows >= 0) & (rows < mask_h) &
                        (cols >= 0) & (cols < mask_w)
                    )
                    full_masks[
                        all_prompt_indices[valid], rows[valid], cols[valid]
                    ] = 5.0

                prebuild.append((
                    batch_len, batch_img_indices, batch_group_indices,
                    batch_inst_ids, full_masks,
                ))

            # ── Phase 2: Tight GPU loop (only GPU + minimal .cpu()) ──
            gpu_results: list[tuple] = []   # collect for Phase 3
            for (batch_len, batch_img_indices, batch_group_indices,
                 batch_inst_ids, mask_np) in prebuild:

                # Upload pre-built low-res mask (no F.interpolate needed)
                mask_tensor = torch.as_tensor(
                    mask_np, dtype=torch.float, device=predictor.device
                ).unsqueeze(1)

                sparse_embeddings, dense_embeddings = (
                    predictor.model.sam_prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=mask_tensor,
                    )
                )

                batch_img_indices_t = torch.as_tensor(
                    batch_img_indices, dtype=torch.long, device=predictor.device
                )
                high_res_features = [
                    feat_level.index_select(0, batch_img_indices_t)
                    for feat_level in high_res_features_all
                ]
                low_res_masks, iou_predictions, _, _ = (
                    predictor.model.sam_mask_decoder(
                        image_embeddings=image_embeddings_all.index_select(
                            0, batch_img_indices_t
                        ),
                        image_pe=dense_pe,
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                        repeat_image=False,
                        high_res_features=high_res_features,
                    )
                )

                # Select best mask on GPU, transfer only the best one.
                # Old: (B,3) float + (B,3,H,W) float → 12x more data
                # New: (B,) float + (B,H,W) bool
                batch_range_t = torch.arange(
                    batch_len, device=iou_predictions.device)
                best_indices_t = iou_predictions.argmax(dim=1)
                best_scores = (
                    iou_predictions[batch_range_t, best_indices_t]
                    .float().cpu().numpy()
                )
                masks_t = predictor._transforms.postprocess_masks(
                    low_res_masks, orig_hw
                )
                best_masks = (
                    (masks_t[batch_range_t, best_indices_t]
                     > predictor.mask_threshold)
                    .cpu().numpy()
                )

                del mask_tensor, sparse_embeddings, dense_embeddings
                del low_res_masks, iou_predictions, masks_t

                gpu_results.append((
                    best_scores, best_masks,
                    batch_img_indices, batch_group_indices, batch_inst_ids,
                ))

            # ── Phase 3: Single-pass numpy winner selection (CPU) ──
            if gpu_results:
                all_scores = np.concatenate(
                    [r[0] for r in gpu_results])
                all_masks = np.concatenate(
                    [r[1] for r in gpu_results])
                all_img_idx = np.concatenate(
                    [r[2] for r in gpu_results])
                all_grp_idx = np.concatenate(
                    [r[3] for r in gpu_results])
                all_inst_ids = np.concatenate(
                    [r[4] for r in gpu_results])
                del gpu_results

                total = len(all_scores)
                passed = (
                    all_scores >= score_threshold
                    if score_threshold > 0
                    else np.ones(total, dtype=bool)
                )

                passed_pos = np.where(passed)[0]
                filtered_pos = np.where(~passed)[0]
                score_records.extend(zip(
                    all_img_idx[passed_pos].tolist(),
                    all_inst_ids[passed_pos].tolist(),
                    all_scores[passed_pos].tolist(),
                ))
                filtered_records.extend(zip(
                    all_img_idx[filtered_pos].tolist(),
                    all_inst_ids[filtered_pos].tolist(),
                    all_scores[filtered_pos].tolist(),
                ))

                candidate_scores = np.where(
                    all_masks & passed[:, None, None],
                    all_scores[:, None, None],
                    -np.inf,
                )
                all_score_maps = np.full(
                    (group_size, h_img, w_img), -np.inf, dtype=np.float32
                )
                np.maximum.at(
                    all_score_maps, all_grp_idx, candidate_scores
                )

                image_best_scores = all_score_maps[all_grp_idx]
                winner_ids_sparse = np.where(
                    candidate_scores == image_best_scores,
                    all_inst_ids[:, None, None],
                    0,
                )
                all_winner_ids = np.zeros(
                    (group_size, h_img, w_img), dtype=np.uint16
                )
                np.maximum.at(
                    all_winner_ids, all_grp_idx, winner_ids_sparse
                )

                update = (
                    np.isfinite(all_score_maps) &
                    (all_score_maps > score_maps)
                )
                combined_masks[update] = all_winner_ids[update]
                score_maps[update] = all_score_maps[update]

    scores_lists = [[] for _ in range(num_images)]
    filtered_lists = [[] for _ in range(num_images)]
    for img_idx, inst_id, score in score_records:
        scores_lists[img_idx].append((inst_id, score))
    for img_idx, inst_id, score in filtered_records:
        filtered_lists[img_idx].append((inst_id, score))

    return [
        (
            combined_masks_by_hw[orig_hws[i]][image_group_positions[i]],
            scores_lists[i],
            filtered_lists[i],
        )
        for i in range(num_images)
    ]


def _save_grouped_step_results(save_dir: str, group_id: int, image: np.ndarray,
                                merged_coords: np.ndarray, mask_input: np.ndarray,
                                masks: np.ndarray, scores: np.ndarray,
                                best_idx: int, member_ids: list, member_cells: list):
    """保存分组 SAM2 推理的中间结果。"""
    import os
    
    group_dir = os.path.join(save_dir, f"group_{group_id:03d}")
    os.makedirs(group_dir, exist_ok=True)
    
    h, w = image.shape[:2]
    
    # 1. 保存输入 mask prompt
    mask_input_viz = ((mask_input[0] + 10) / 20 * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(group_dir, "input_mask_prompt_256x256.png"), mask_input_viz)
    
    # 2. 保存合并后的输入区域可视化
    cluster_viz = np.zeros((h, w, 3), dtype=np.uint8)
    rows = merged_coords[:, 0].astype(np.intp)
    cols = merged_coords[:, 1].astype(np.intp)
    cluster_viz[rows, cols] = (0, 255, 0)
    cv2.imwrite(os.path.join(group_dir, "input_merged_cells.png"), cv2.cvtColor(cluster_viz, cv2.COLOR_RGB2BGR))
    
    # 3. 保存 3 个候选 mask
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask_bool = mask.astype(bool)
        mask_viz = np.zeros((h, w, 3), dtype=np.uint8)
        mask_viz[mask_bool] = colors[i]
        
        selected_mark = "_SELECTED" if i == best_idx else ""
        cv2.imwrite(os.path.join(group_dir, f"mask_{i}_score{score:.4f}{selected_mark}.png"),
                   cv2.cvtColor(mask_viz, cv2.COLOR_RGB2BGR))
    
    # 4. 收集所有成员细胞的中心点
    member_centers = []
    for cell in member_cells:
        if 'center' in cell:
            cy, cx = cell['center']
            member_centers.append([int(cx), int(cy)])
    
    # 5. 保存元数据
    import json
    metadata = {
        'group_id': group_id,
        'member_ids': [int(m) for m in member_ids],
        'num_cells': len(member_ids),
        'merged_coords_count': len(merged_coords),
        'point_prompts': member_centers,  # 每个成员细胞的中心点
        'num_points': len(member_centers),
        'prompt_mode': 'multi_point+mask',
        'scores': [float(s) for s in scores],
        'best_idx': best_idx,
        'best_score': float(scores[best_idx]),
        'mask_areas': [int(np.sum(m)) for m in masks]
    }
    with open(os.path.join(group_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)


def _save_sam2_step_results(save_dir: str, inst_id: int, image: np.ndarray,
                             cluster: np.ndarray, mask_input: np.ndarray,
                             masks: np.ndarray, scores: np.ndarray,
                             best_idx: int):
    """
    保存 SAM2 单步推理的中间结果。

    Args:
        save_dir: 保存目录
        inst_id: 实例 ID
        image: 原始图像
        cluster: 输入的 cluster 坐标
        mask_input: 输入给 SAM2 的 mask prompt (256x256)
        masks: SAM2 输出的 3 个候选 mask
        scores: SAM2 输出的 3 个 scores
        best_idx: 选择的最佳 mask 索引
    """
    import os
    
    inst_dir = os.path.join(save_dir, f"instance_{inst_id:03d}")
    os.makedirs(inst_dir, exist_ok=True)
    
    h, w = image.shape[:2]
    
    # 1. 保存输入 mask prompt (256x256 低分辨率)
    mask_input_viz = ((mask_input[0] + 10) / 20 * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(inst_dir, "input_mask_prompt_256x256.png"), mask_input_viz)
    
    # 2. 保存输入 cluster 的可视化
    cluster_viz = np.zeros((h, w, 3), dtype=np.uint8)
    # 确保坐标为整数类型
    rows = cluster[:, 0].astype(np.intp)
    cols = cluster[:, 1].astype(np.intp)
    cluster_viz[rows, cols] = (0, 255, 0)  # 绿色标记输入区域
    cv2.imwrite(os.path.join(inst_dir, "input_cluster.png"), cv2.cvtColor(cluster_viz, cv2.COLOR_RGB2BGR))
    
    # 3. 保存 3 个候选 mask
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # RGB: 红、绿、蓝
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # 确保 mask 是布尔类型 (SAM2 返回的 mask 可能不是布尔类型)
        mask_bool = mask.astype(bool)
        
        # 单独保存每个 mask
        mask_viz = np.zeros((h, w, 3), dtype=np.uint8)
        mask_viz[mask_bool] = colors[i]
        
        # 叠加到原图
        overlay = image.copy()
        overlay[mask_bool] = (np.array(overlay[mask_bool]) * 0.5 + np.array(colors[i]) * 0.5).astype(np.uint8)
        
        selected_mark = " [SELECTED]" if i == best_idx else ""
        
        # 保存 mask 二值图
        cv2.imwrite(os.path.join(inst_dir, f"mask_{i}_score{score:.4f}{selected_mark.replace(' ', '_').replace('[', '').replace(']', '')}.png"), 
                   cv2.cvtColor(mask_viz, cv2.COLOR_RGB2BGR))
        
        # 保存叠加图
        cv2.imwrite(os.path.join(inst_dir, f"overlay_{i}_score{score:.4f}.png"), 
                   cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # 4. 保存对比图 (3 个 mask 并排)
    comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # 确保 mask 是布尔类型
        mask_bool = mask.astype(bool)
        panel = image.copy()
        panel[mask_bool] = (np.array(panel[mask_bool]) * 0.5 + np.array(colors[i]) * 0.5).astype(np.uint8)
        # 添加 score 文字
        cv2.putText(panel, f"Score: {score:.4f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if i == best_idx:
            cv2.putText(panel, "BEST", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        comparison[:, i*w:(i+1)*w] = panel
    
    cv2.imwrite(os.path.join(inst_dir, "comparison_3_masks.png"), 
               cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    # 5. 保存元数据
    import json
    metadata = {
        'instance_id': inst_id,
        'prompt_mode': 'mask_only',
        'cluster_size': len(cluster),
        'scores': [float(s) for s in scores],
        'best_idx': best_idx,
        'best_score': float(scores[best_idx]),
        'mask_areas': [int(np.sum(m)) for m in masks]
    }
    with open(os.path.join(inst_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)


def run_sam2_merged_box_mask(predictor, image: np.ndarray, cells_info: list,
                              min_area: int = 10, padding: int = 10,
                              set_image: bool = True) -> tuple:
    """
    Run SAM2 segmentation with merged overlapping boxes and masks.
    
    This function:
    1. Detects overlapping cells based on their bounding boxes
    2. Merges overlapping cells into single prompts
    3. Sends merged box+mask prompts to SAM2
    
    Args:
        predictor: SAM2ImagePredictor instance
        image: RGB image array
        cells_info: List of cell info dicts
        min_area: Minimum cell area threshold
        padding: Bounding box padding
        set_image: Whether to call predictor.set_image()
    
    Returns:
        combined_mask: Instance segmentation mask
        scores_list: List of (merged_id, score, member_ids) tuples
        merge_info: List of dicts with merge details for visualization
    """
    if set_image:
        predictor.set_image(image)
        
    h, w = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint16)
    scores_list = []
    merge_info = []
    
    # Merge overlapping cells
    merged_cells = merge_overlapping_cells(cells_info, padding=padding)
    
    print(f"      Merged {len(cells_info)} cells into {len(merged_cells)} groups")
    
    for idx, merged_cell in enumerate(merged_cells):
        if merged_cell['pixel_count'] < min_area:
            continue
            
        try:
            mask_input = generate_mask_from_cluster(merged_cell['coords'], image.shape)
            box = np.clip(merged_cell['box'], [0, 0, 0, 0], [w, h, w, h]).astype(np.int64)
            
            masks, scores, _ = predictor.predict(
                box=box,
                mask_input=mask_input,
                multimask_output=True
            )
            
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            best_mask = masks[best_idx].astype(bool)
            mask_area = np.sum(best_mask)
            
            inst_id = idx + 1
            combined_mask[best_mask] = inst_id
            
            member_str = ','.join(map(str, merged_cell['member_ids']))
            is_merged_str = "[MERGED]" if merged_cell['is_merged'] else ""
            print(f"      Group {inst_id} ({member_str}): score={best_score:.4f}, area={mask_area} pixels {is_merged_str}")
            
            scores_list.append((inst_id, best_score, merged_cell['member_ids']))
            merge_info.append({
                'group_id': inst_id,
                'member_ids': merged_cell['member_ids'],
                'is_merged': merged_cell['is_merged'],
                'box': merged_cell['box'],
                'center': merged_cell['center'],
                'score': best_score,
                'area': mask_area
            })
            
        except Exception as e:
            print(f"    SAM2 Error on merged group {idx}: {e}")
            
    return combined_mask, scores_list, merge_info


def merge_connected_masks(instance_mask: np.ndarray, scores_list: list,
                          positive_cells_info: list = None,
                          min_area: int = 0,
                          debug_dir: str = None,
                          debug_prefix: str = None,
                          original_image: np.ndarray = None) -> tuple:
    """
    将重叠或相连的 mask 实例合并为单个实例。

    参数:
        instance_mask: 实例分割掩膜 (H, W)，像素值为实例 ID
        scores_list: (instance_id, score) 元组列表
        positive_cells_info: 可选的细胞信息字典列表
        min_area: 连通区域最小面积阈值（0 = 不过滤）
        debug_dir: 若提供，保存合并/过滤过程的区域调试可视化图
        debug_prefix: 若提供，将调试文件写成外层 pipeline 步骤名
            （例如 step5_01_merge_filter_*.png）
        original_image: 用于叠加可视化的原始图像

    返回:
        merged_mask: 合并后的新实例掩膜
        merged_scores: (new_id, avg_score, member_ids) 元组列表
        merge_mapping: 旧实例 ID -> 新合并 ID 的映射字典
        merged_cells_info: 合并后的细胞信息字典列表（长度等于新实例数量）
    """
    import os

    def _merge_debug_image_name(order: int, legacy_step: int, stem: str) -> str:
        if debug_prefix:
            return f"{debug_prefix}_{order:02d}_merge_filter_{stem}.png"
        return f"step{legacy_step}_{stem}.png"

    def _merge_debug_summary_name(order: int) -> str:
        if debug_prefix:
            return f"{debug_prefix}_{order:02d}_merge_filter_summary.json"
        return "merge_summary.json"
    
    if instance_mask is None or np.max(instance_mask) == 0:
        return instance_mask, scores_list, {}, positive_cells_info
    
    h, w = instance_mask.shape
    
    # 创建二值掩膜
    binary_mask = (instance_mask > 0).astype(np.uint8) * 255

    # 查找连通组件
    num_labels, labels = cv2.connectedComponents(binary_mask)

    # 构建连通组件到实例的映射，并记录面积信息（向量化）
    # 用 bincount 一次算出所有组件面积
    comp_areas = np.bincount(labels.ravel(), minlength=num_labels)  # (num_labels,)
    # 构建 (comp_id, inst_id) 配对，批量提取每个组件包含的实例
    flat_labels = labels.ravel()
    flat_inst = instance_mask.ravel()
    valid_px = flat_inst > 0  # 仅前景像素
    if valid_px.any():
        pairs = np.stack([flat_labels[valid_px], flat_inst[valid_px]], axis=1)  # (N, 2)
        unique_pairs = np.unique(pairs, axis=0)  # 去重 (comp_id, inst_id) 配对
        component_info = {}
        for comp_id in range(1, num_labels):
            mask = unique_pairs[:, 0] == comp_id
            if mask.any():
                member_ids = unique_pairs[mask, 1].tolist()
                component_info[comp_id] = {
                    'member_ids': member_ids,
                    'area': int(comp_areas[comp_id])
                }
    else:
        component_info = {}
    
    # 若提供了 debug_dir，准备可视化
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        
        # 1. 保存合并前的可视化（SAM2 原始输出）
        pre_merge_viz = _create_instance_mask_visualization(instance_mask, h, w, "SAM2 Raw Output")
        cv2.imwrite(os.path.join(
            debug_dir, _merge_debug_image_name(1, 1, "sam2_raw_output")),
                   cv2.cvtColor(pre_merge_viz, cv2.COLOR_RGB2BGR))
        
        # 若有原始图像，创建叠加可视化
        if original_image is not None:
            overlay = _create_mask_overlay(original_image, instance_mask)
            cv2.imwrite(os.path.join(
                debug_dir, _merge_debug_image_name(2, 1, "sam2_raw_overlay")),
                       cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # 创建合并后的掩膜（使用 uint16 以支持 >255 个实例）
    merged_mask = np.zeros(instance_mask.shape, dtype=np.uint16)
    merged_scores = []
    merge_mapping = {}
    merged_cells_info = []
    filtered_count = 0
    filtered_regions = []  # 记录被过滤的区域，用于可视化
    
    score_dict = {inst_id: score for inst_id, score in scores_list}
    
    new_id = 0
    for comp_id, info in component_info.items():
        member_ids = info['member_ids']
        area = info['area']
        comp_mask = labels == comp_id
        
        # 面积过滤
        if min_area > 0 and area < min_area:
            print(f"      [FILTERED] Connected component: area={area} pixels (< min_area={min_area})")
            filtered_count += 1
            filtered_regions.append({'comp_id': comp_id, 'area': area, 'member_ids': member_ids})
            continue
        
        new_id += 1
        merged_mask[comp_mask] = new_id
        
        member_scores = [score_dict.get(m_id, 0.0) for m_id in member_ids]
        avg_score = sum(member_scores) / len(member_scores) if member_scores else 0.0
        
        merged_scores.append((new_id, avg_score, member_ids))
        
        for m_id in member_ids:
            merge_mapping[m_id] = new_id
        
        # 从成员细胞创建合并后的细胞信息
        if positive_cells_info is not None:
            # 收集所有成员细胞（1-based 索引）
            member_cells = []
            for m_id in member_ids:
                if m_id <= len(positive_cells_info):
                    member_cells.append(positive_cells_info[m_id - 1])
            
            if member_cells:
                # 判断阳性：任一成员为阳性则标记为阳性
                is_positive = any(c.get('is_positive', True) for c in member_cells)
                
                # 合并细胞信息
                merged_cell = {
                    'id': new_id,
                    'is_positive': is_positive,
                    'pixel_count': sum(c.get('pixel_count', 0) for c in member_cells),
                    'marker_sum': sum(c.get('marker_sum', 0) for c in member_cells),
                    'marker_mean': np.mean([c.get('marker_mean', 0) for c in member_cells]),
                    'marker_max': max(c.get('marker_max', 0) for c in member_cells),
                    'marker_min': min(c.get('marker_min', 255) for c in member_cells),
                    'center': member_cells[0]['center'],  # 使用第一个细胞的中心点
                    'member_ids': member_ids,
                    'area': area  # 添加实际合并后的面积
                }
                merged_cells_info.append(merged_cell)
            else:
                # 未找到有效的成员细胞，创建占位信息
                merged_cells_info.append({
                    'id': new_id,
                    'is_positive': True,  # 默认为阳性
                    'pixel_count': area,
                    'marker_sum': 0,
                    'marker_mean': 0,
                    'marker_max': 0,
                    'marker_min': 0,
                    'center': (0, 0),
                    'member_ids': member_ids,
                    'area': area
                })
        
        if len(member_ids) > 1:
            print(f"      [MERGE] Connected component {new_id}: merged instances {member_ids} -> avg_score={avg_score:.4f}, area={area}")
    
    if filtered_count > 0:
        print(f"      [AREA FILTER] Removed {filtered_count} small regions (min_area={min_area})")
    print(f"      Merge result: {len(scores_list)} instances -> {new_id} connected regions")
    
    # 保存合并/过滤的可视化结果
    if debug_dir:
        # 2. 保存被过滤区域的可视化
        if filtered_regions:
            filtered_viz = _create_filtered_regions_visualization(labels, filtered_regions, h, w)
            cv2.imwrite(os.path.join(
                debug_dir, _merge_debug_image_name(3, 2, "filtered_regions")),
                       cv2.cvtColor(filtered_viz, cv2.COLOR_RGB2BGR))
        
        # 3. 保存合并后的可视化
        post_merge_viz = _create_instance_mask_visualization(merged_mask, h, w, "Merged & Filtered")
        cv2.imwrite(os.path.join(
            debug_dir, _merge_debug_image_name(4, 3, "merged_result")),
                   cv2.cvtColor(post_merge_viz, cv2.COLOR_RGB2BGR))
        
        # 若有原始图像，创建最终叠加可视化
        if original_image is not None:
            final_overlay = _create_mask_overlay(original_image, merged_mask)
            cv2.imwrite(os.path.join(
                debug_dir, _merge_debug_image_name(5, 3, "merged_overlay")),
                       cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR))
        
        # 4. 保存合并汇总 JSON
        import json
        summary = {
            'total_sam2_instances': len(scores_list),
            'total_connected_components': num_labels - 1,
            'filtered_count': filtered_count,
            'final_regions': new_id,
            'min_area_threshold': min_area,
            'filtered_regions': [{'area': r['area'], 'member_ids': [int(m) for m in r['member_ids']]} 
                                 for r in filtered_regions],
            'merged_regions': [{'new_id': int(s[0]), 'avg_score': float(s[1]), 
                               'member_ids': [int(m) for m in s[2]]} for s in merged_scores]
        }
        with open(os.path.join(
                debug_dir, _merge_debug_summary_name(6)), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"      Saved merge/filter visualization to {debug_dir}/")
    
    # 若未提供 positive_cells_info，返回 None
    if positive_cells_info is None:
        merged_cells_info = None
    
    return merged_mask, merged_scores, merge_mapping, merged_cells_info


def _create_instance_mask_visualization(mask: np.ndarray, h: int, w: int, title: str = "") -> np.ndarray:
    """Create colorful visualization of instance mask."""
    viz = np.zeros((h, w, 3), dtype=np.uint8)
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids > 0]
    
    # Generate distinct colors
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(len(unique_ids) + 1, 3))
    
    for idx, inst_id in enumerate(unique_ids):
        inst_mask = mask == inst_id
        viz[inst_mask] = colors[idx]
        
        # Add instance ID label
        coords = np.argwhere(inst_mask)
        if len(coords) > 0:
            center_y, center_x = int(coords[:, 0].mean()), int(coords[:, 1].mean())
            cv2.putText(viz, str(int(inst_id)), (center_x - 5, center_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return viz


def _create_mask_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create overlay of mask on original image."""
    overlay = image.copy()
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids > 0]
    
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(len(unique_ids) + 1, 3))
    
    for idx, inst_id in enumerate(unique_ids):
        inst_mask = mask == inst_id
        color = colors[idx].tolist()
        overlay[inst_mask] = (np.array(overlay[inst_mask]) * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    
    return overlay


def _create_filtered_regions_visualization(labels: np.ndarray, filtered_regions: list, 
                                            h: int, w: int) -> np.ndarray:
    """Create visualization showing filtered (removed) regions in red."""
    viz = np.zeros((h, w, 3), dtype=np.uint8)
    
    for region in filtered_regions:
        comp_id = region['comp_id']
        comp_mask = labels == comp_id
        viz[comp_mask] = (255, 0, 0)  # Red for filtered regions
        
        # Add area label
        coords = np.argwhere(comp_mask)
        if len(coords) > 0:
            center_y, center_x = int(coords[:, 0].mean()), int(coords[:, 1].mean())
            cv2.putText(viz, str(region['area']), (center_x - 10, center_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return viz


def run_sam2_mask_with_point(predictor, image: np.ndarray, cells_info: list, 
                              min_area: int = 10, set_image: bool = True) -> tuple:
    """
    Run SAM2 segmentation using our custom mask + cell center point as prompts.
    
    Args:
        predictor: SAM2ImagePredictor instance
        image: RGB image array
        cells_info: List of cell info dicts (must contain 'center' and 'coords' keys)
        min_area: Minimum cell area threshold
        set_image: Whether to call predictor.set_image()
    
    Returns:
        combined_mask: Instance segmentation mask
        scores_list: List of (instance_id, score) tuples
    """
    if set_image:
        predictor.set_image(image)
        
    h, w = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint16)
    scores_list = []

    for idx, cell in enumerate(cells_info):
        if cell.get('pixel_count', 0) < min_area:
            continue
            
        try:
            center_y, center_x = cell['center']
            point_coords = np.array([[center_x, center_y]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)
            
            mask_input = generate_mask_from_cluster(cell['coords'], image.shape)
            
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                multimask_output=True
            )
            
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            best_mask = masks[best_idx].astype(bool)
            mask_area = np.sum(best_mask)
            
            combined_mask[best_mask] = idx + 1
            scores_list.append((idx + 1, best_score))
            
            print(f"      Instance {idx+1}: score={best_score:.4f}, area={mask_area} pixels (mode=point+our_mask)")
            
        except Exception as e:
            print(f"    SAM2 Error on cell {idx}: {e}")
            
    return combined_mask, scores_list


def run_sam2_point_iterative(predictor, image: np.ndarray, cells_info: list, 
                              min_area: int = 10, set_image: bool = True) -> tuple:
    """
    Run SAM2 segmentation using point prompts with iterative refinement.
    
    Args:
        predictor: SAM2ImagePredictor instance
        image: RGB image array
        cells_info: List of cell info dicts (must contain 'center' key)
        min_area: Minimum cell area threshold
        set_image: Whether to call predictor.set_image()
    
    Returns:
        combined_mask_pass1: Instance mask from point-only pass
        combined_mask_pass2: Instance mask from point+mask pass
        scores_pass1: List of (instance_id, score) from first pass
        scores_pass2: List of (instance_id, score) from second pass
    """
    if set_image:
        predictor.set_image(image)
        
    h, w = image.shape[:2]
    
    combined_mask_pass1 = np.zeros((h, w), dtype=np.uint16)
    combined_mask_pass2 = np.zeros((h, w), dtype=np.uint16)
    scores_pass1 = []
    scores_pass2 = []

    for idx, cell in enumerate(cells_info):
        if cell.get('pixel_count', 0) < min_area:
            continue
            
        try:
            center_y, center_x = cell['center']
            point_coords = np.array([[center_x, center_y]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)
            
            # Pass 1: Point only
            masks_p1, scores_p1, low_res_masks_p1 = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )
            
            best_idx_p1 = int(np.argmax(scores_p1))
            best_score_p1 = float(scores_p1[best_idx_p1])
            best_mask_p1 = masks_p1[best_idx_p1].astype(bool)
            best_low_res_p1 = low_res_masks_p1[best_idx_p1:best_idx_p1+1]
            area_p1 = np.sum(best_mask_p1)
            
            combined_mask_pass1[best_mask_p1] = idx + 1
            scores_pass1.append((idx + 1, best_score_p1))
            
            # Pass 2: Point + low_res_mask from Pass 1
            masks_p2, scores_p2, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=best_low_res_p1,
                multimask_output=True
            )
            
            best_idx_p2 = int(np.argmax(scores_p2))
            best_score_p2 = float(scores_p2[best_idx_p2])
            best_mask_p2 = masks_p2[best_idx_p2].astype(bool)
            area_p2 = np.sum(best_mask_p2)
            
            combined_mask_pass2[best_mask_p2] = idx + 1
            scores_pass2.append((idx + 1, best_score_p2))
            
            print(f"      Instance {idx+1}: Pass1(point) score={best_score_p1:.4f} area={area_p1} | "
                  f"Pass2(point+mask) score={best_score_p2:.4f} area={area_p2}")
            
        except Exception as e:
            print(f"    SAM2 Error on cell {idx}: {e}")
            
    return combined_mask_pass1, combined_mask_pass2, scores_pass1, scores_pass2


def run_sam2_segmentation_batch(predictor, image: np.ndarray, clusters: list,
                                 min_area: int = 10,
                                 set_image: bool = True, batch_size: int = 32,
                                 score_threshold: float = 0.0) -> tuple:
    """
    批量版本的 SAM2 推理（mask_only 模式），一次处理多个 prompt 以提升速度。

    与 run_sam2_segmentation 产出完全一致（IoU≈1.0），仅推理方式不同：
    直接调用 prompt_encoder + mask_decoder，手动设 repeat_image=True
    让 mask decoder 广播单张 image embedding 到 batch 维度。

    Args:
        predictor: SAM2ImagePredictor instance
        image: RGB image array
        clusters: List of cluster coordinate arrays
        min_area: Minimum cluster size before SAM inference
        set_image: Whether to call predictor.set_image()
        batch_size: Number of prompts to process at once
        score_threshold: Minimum score threshold

    Returns:
        combined_mask: Instance segmentation mask
        scores_list: List of (instance_id, score) tuples
        filtered_list: List of (instance_id, score) tuples for filtered instances
    """
    if set_image:
        predictor.set_image(image)

    h, w = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint16)
    score_map = np.zeros((h, w), dtype=np.float32)
    scores_list = []
    filtered_list = []

    # 1. Filter clusters before constructing SAM prompts.
    valid_clusters = []
    for idx, cluster in enumerate(clusters):
        if len(cluster) < min_area:
            continue
        valid_clusters.append((idx, cluster))

    if len(valid_clusters) == 0:
        return combined_mask, scores_list, filtered_list

    # 2. 分批处理
    for batch_start in range(0, len(valid_clusters), batch_size):
        batch = valid_clusters[batch_start:batch_start + batch_size]

        # 准备 mask prompt tensor: (B, 1, 256, 256) — 批量构建，无逐个 for 循环
        B = len(batch)
        h_img, w_img = image.shape[:2]
        full_masks = np.full((B, h_img, w_img), -5.0, dtype=np.float32)
        # 将所有 cluster 坐标拼接，用 batch_idx 区分所属 batch
        all_coords = []
        all_batch_idx = []
        for i, (_, cluster) in enumerate(batch):
            coords = cluster if np.issubdtype(cluster.dtype, np.integer) else cluster.astype(np.int64)
            all_coords.append(coords)
            all_batch_idx.append(np.full(len(coords), i, dtype=np.intp))
        all_coords = np.concatenate(all_coords, axis=0)
        all_batch_idx = np.concatenate(all_batch_idx)
        rows = all_coords[:, 0].astype(np.intp)
        cols = all_coords[:, 1].astype(np.intp)
        valid = (rows >= 0) & (rows < h_img) & (cols >= 0) & (cols < w_img)
        full_masks[all_batch_idx[valid], rows[valid], cols[valid]] = 5.0
        # batch resize: (B, H, W) -> (B, 1, 256, 256)，使用 F.interpolate 代替逐张 cv2.resize
        mask_tensor = torch.as_tensor(full_masks, dtype=torch.float, device=predictor.device)
        mask_tensor = F.interpolate(mask_tensor.unsqueeze(1), size=(256, 256), mode='area')

        # mask-only batch forward：直接调用 prompt_encoder + mask_decoder
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=mask_tensor,
        )

        high_res_features = [
            feat_level[-1].unsqueeze(0)
            for feat_level in predictor._features["high_res_feats"]
        ]
        low_res_masks, iou_predictions, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=True,
            high_res_features=high_res_features,
        )

        # 后处理：upscale + threshold
        all_masks = predictor._transforms.postprocess_masks(
            low_res_masks, predictor._orig_hw[-1]
        )
        all_masks = all_masks > predictor.mask_threshold

        all_masks_np = all_masks.float().detach().cpu().numpy()
        all_scores_np = iou_predictions.float().detach().cpu().numpy()

        # 3. 后处理 — 向量化，无逐个 for 循环
        batch_idx = np.arange(B)
        orig_indices = np.array([orig_idx for orig_idx, _ in batch])
        inst_ids = orig_indices + 1  # (B,)

        # 每个样本选最佳 mask：(B,) best index, (B,) best score, (B, H, W) best masks
        best_indices = np.argmax(all_scores_np, axis=1)  # (B,)
        best_scores = all_scores_np[batch_idx, best_indices]  # (B,)
        best_masks = all_masks_np[batch_idx, best_indices].astype(bool)  # (B, H, W)
        mask_areas = best_masks.sum(axis=(1, 2))  # (B,)

        # 置信度过滤：向量化分离 passed / filtered
        passed = (best_scores >= score_threshold) if score_threshold > 0 else np.ones(B, dtype=bool)

        # 记录被过滤的（批量构建列表）
        filtered_mask = ~passed
        if filtered_mask.any():
            f_ids = inst_ids[filtered_mask]
            f_scores = best_scores[filtered_mask]
            f_areas = mask_areas[filtered_mask]
            filtered_list.extend(list(zip(f_ids.tolist(), f_scores.tolist())))
            for fid, fs, fa in zip(f_ids, f_scores, f_areas):
                print(f"      Instance {fid}: score={fs:.4f}, area={fa} pixels [FILTERED: score<{score_threshold}]")

        # 置信度优先合并 — 使用 score volume 向量化
        passed_idx = np.where(passed)[0]
        if len(passed_idx) > 0:
            p_masks = best_masks[passed_idx]      # (P, H, W)
            p_scores = best_scores[passed_idx]    # (P,)
            p_iids = inst_ids[passed_idx]         # (P,)

            # 构建 score volume: 每个像素取覆盖它的 mask 中 score 最高的
            score_vol = np.where(p_masks, p_scores[:, None, None], -np.inf)  # (P, H, W)
            winner = np.argmax(score_vol, axis=0)  # (H, W) — 每个像素的最佳实例索引
            winner_scores = np.take_along_axis(
                score_vol, winner[np.newaxis, :, :], axis=0
            )[0]  # (H, W)

            # 仅更新：有 mask 覆盖 且 score > 已有 score_map 的像素
            any_covered = p_masks.any(axis=0)  # (H, W)
            update = any_covered & (winner_scores > score_map)
            combined_mask[update] = p_iids[winner[update]]
            score_map[update] = winner_scores[update]

            # 记录 scores_list + 日志
            scores_list.extend(list(zip(p_iids.tolist(), p_scores.tolist())))
            for pi in passed_idx:
                final_area = np.sum(combined_mask == inst_ids[pi])
                print(f"      Instance {inst_ids[pi]}: score={best_scores[pi]:.4f}, area={mask_areas[pi]} pixels [final: {final_area} pixels]")

    return combined_mask, scores_list, filtered_list
