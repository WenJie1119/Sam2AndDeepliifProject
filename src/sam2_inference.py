#!/usr/bin/env python3
"""
sam2_inference.py — SAM2 推理模块

包含：
- 多种 SAM2 推理模式
- 掩码后处理与合并
"""

import numpy as np
import cv2

from .mask_utils import (
    generate_mask_from_cluster,
    get_bounding_box_from_cluster,
    merge_overlapping_cells
)


def run_sam2_segmentation(predictor, image: np.ndarray, clusters: list, 
                           min_area: int = 10, prompt_mode: str = 'box+mask', 
                           set_image: bool = True, score_threshold: float = 0.0,
                           save_steps_dir: str = None) -> tuple:
    """
    Run SAM2 segmentation on clusters.
    
    Args:
        predictor: SAM2ImagePredictor instance
        image: RGB image array
        clusters: List of cluster coordinate arrays
        min_area: Minimum cluster size
        prompt_mode: 'box+mask' or 'mask_only'
        set_image: Whether to call predictor.set_image()
        score_threshold: Minimum score threshold for filtering (0.0 = no filtering)
        save_steps_dir: If provided, save intermediate results for each instance to this directory
    
    Returns:
        combined_mask: Instance segmentation mask
        scores_list: List of (instance_id, score) tuples
        filtered_list: List of (instance_id, score) tuples for filtered instances
    """
    import os
    
    if set_image:
        predictor.set_image(image)
        
    h, w = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.float32)
    scores_list = []
    filtered_list = []
    
    # 准备保存目录
    if save_steps_dir:
        os.makedirs(save_steps_dir, exist_ok=True)
        # 存储所有步骤的汇总信息
        steps_summary = []

    for idx, cluster in enumerate(clusters):
        if len(cluster) < min_area:
            continue
            
        try:
            mask_input = generate_mask_from_cluster(cluster, image.shape)
            
            if prompt_mode == 'box+mask':
                box = get_bounding_box_from_cluster(cluster, padding=10)
                box = np.clip(box, [0, 0, 0, 0], [w, h, w, h]).astype(np.int64)
                masks, scores, low_res_masks = predictor.predict(
                    box=box,
                    mask_input=mask_input,
                    multimask_output=True
                )
            else:  # mask_only
                masks, scores, low_res_masks = predictor.predict(
                    mask_input=mask_input,
                    multimask_output=True
                )
            
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            best_mask = masks[best_idx].astype(bool)
            mask_area = np.sum(best_mask)
            
            inst_id = idx + 1
            
            # 保存每一步的中间结果
            if save_steps_dir:
                _save_sam2_step_results(
                    save_steps_dir, inst_id, image, cluster, mask_input,
                    masks, scores, best_idx, prompt_mode
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
            
            # Filter low confidence results (if threshold > 0)
            if score_threshold > 0 and best_score < score_threshold:
                print(f"      Instance {inst_id}: score={best_score:.4f}, area={mask_area} pixels (mode={prompt_mode}) [FILTERED: score<{score_threshold}]")
                filtered_list.append((inst_id, best_score))
                continue
            
            # Confidence-priority merge
            overwrite_mask = best_mask & (best_score > score_map)
            combined_mask[overwrite_mask] = inst_id
            score_map[overwrite_mask] = best_score
            
            scores_list.append((inst_id, best_score))
            
            final_area = np.sum(combined_mask == inst_id)
            print(f"      Instance {inst_id}: score={best_score:.4f}, area={mask_area} pixels (mode={prompt_mode}) [final: {final_area} pixels]")
            
        except Exception as e:
            import traceback
            print(f"    SAM2 Error on cluster {idx}: {e}")
            print(f"      Cluster dtype: {cluster.dtype}, shape: {cluster.shape}")
            print(f"      Traceback: {traceback.format_exc()}")
    
    # 保存汇总信息
    if save_steps_dir:
        import json
        summary_path = os.path.join(save_steps_dir, "steps_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(steps_summary, f, indent=2)
        print(f"      Saved {len(steps_summary)} step details to {save_steps_dir}/")
            
    return combined_mask, scores_list, filtered_list


def run_sam2_grouped_segmentation(predictor, image: np.ndarray, cell_groups: list,
                                   min_area: int = 10, set_image: bool = True,
                                   score_threshold: float = 0.0,
                                   save_steps_dir: str = None) -> tuple:
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
        save_steps_dir: If provided, save intermediate results
    
    Returns:
        combined_mask: Instance segmentation mask (group_id for each pixel)
        scores_list: List of (group_id, score) tuples
        filtered_list: List of (group_id, score) tuples for filtered instances
    """
    import os
    
    if set_image:
        predictor.set_image(image)
        
    h, w = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.float32)
    scores_list = []
    filtered_list = []
    steps_summary = []
    
    if save_steps_dir:
        os.makedirs(save_steps_dir, exist_ok=True)
    
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
            
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            best_mask = masks[best_idx].astype(bool)
            mask_area = np.sum(best_mask)
            
            # Save step results
            if save_steps_dir:
                _save_grouped_step_results(
                    save_steps_dir, group_id, image, merged_coords, mask_input,
                    masks, scores, best_idx, member_ids
                )
                steps_summary.append({
                    'group_id': int(group_id),
                    'member_ids': [int(m) for m in member_ids],
                    'num_cells': len(member_ids),
                    'total_input_pixels': int(total_pixels),
                    'scores': [float(s) for s in scores],
                    'best_idx': int(best_idx),
                    'best_score': float(best_score),
                    'mask_areas': [int(np.sum(m)) for m in masks],
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
            print(f"      Group {group_id} {cells_str}: score={best_score:.4f}, area={mask_area} [final: {final_area}] (mode=multi_point+mask, {num_points}pts)")
            
        except Exception as e:
            import traceback
            print(f"    SAM2 Error on group {group_id}: {e}")
            print(f"      Member cells: {member_ids}")
            print(f"      Traceback: {traceback.format_exc()}")
    
    # Save summary
    if save_steps_dir:
        import json
        summary_path = os.path.join(save_steps_dir, "grouped_steps_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(steps_summary, f, indent=2)
        print(f"      Saved {len(steps_summary)} group details to {save_steps_dir}/")
    
    return combined_mask, scores_list, filtered_list


def _save_grouped_step_results(save_dir: str, group_id: int, image: np.ndarray,
                                merged_coords: np.ndarray, mask_input: np.ndarray,
                                masks: np.ndarray, scores: np.ndarray,
                                best_idx: int, member_ids: list):
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
                             best_idx: int, prompt_mode: str):
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
        prompt_mode: 使用的 prompt 模式
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
        'prompt_mode': prompt_mode,
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
    combined_mask = np.zeros((h, w), dtype=np.uint8)
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
                          save_steps_dir: str = None,
                          original_image: np.ndarray = None) -> tuple:
    """
    Merge overlapping or connected mask instances into single instances.
    
    Args:
        instance_mask: Instance segmentation mask (H, W) with instance IDs
        scores_list: List of (instance_id, score) tuples
        positive_cells_info: Optional list of cell info dicts
        min_area: Minimum area threshold for connected regions (0 = no filtering)
        save_steps_dir: If provided, save visualization of merge/filter process
        original_image: Original image for overlay visualization
    
    Returns:
        merged_mask: New instance mask with merged instances
        merged_scores: List of (new_id, avg_score, member_ids) tuples
        merge_mapping: Dict mapping old instance IDs to new merged IDs
        merged_cells_info: List of merged cell info dicts (same length as new instance count)
    """
    import os
    
    if instance_mask is None or np.max(instance_mask) == 0:
        return instance_mask, scores_list, {}, positive_cells_info
    
    h, w = instance_mask.shape
    
    # Create binary mask
    binary_mask = (instance_mask > 0).astype(np.uint8) * 255
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary_mask)
    
    # Build component to instances mapping with area info
    component_info = {}
    for comp_id in range(1, num_labels):
        comp_mask = labels == comp_id
        area = int(np.sum(comp_mask))
        instance_ids = np.unique(instance_mask[comp_mask])
        instance_ids = instance_ids[instance_ids > 0]
        if len(instance_ids) > 0:
            component_info[comp_id] = {
                'member_ids': list(instance_ids),
                'area': area
            }
    
    # Prepare visualization if save_steps_dir is provided
    if save_steps_dir:
        os.makedirs(save_steps_dir, exist_ok=True)
        
        # 1. Save pre-merge visualization (SAM2 raw output)
        pre_merge_viz = _create_instance_mask_visualization(instance_mask, h, w, "SAM2 Raw Output")
        cv2.imwrite(os.path.join(save_steps_dir, "step1_sam2_raw_output.png"), 
                   cv2.cvtColor(pre_merge_viz, cv2.COLOR_RGB2BGR))
        
        # If original image available, create overlay
        if original_image is not None:
            overlay = _create_mask_overlay(original_image, instance_mask)
            cv2.imwrite(os.path.join(save_steps_dir, "step1_sam2_raw_overlay.png"), 
                       cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # Create merged mask (use uint16 to support >255 instances)
    merged_mask = np.zeros(instance_mask.shape, dtype=np.uint16)
    merged_scores = []
    merge_mapping = {}
    merged_cells_info = []
    filtered_count = 0
    filtered_regions = []  # Track filtered regions for visualization
    
    score_dict = {inst_id: score for inst_id, score in scores_list}
    
    new_id = 0
    for comp_id, info in component_info.items():
        member_ids = info['member_ids']
        area = info['area']
        comp_mask = labels == comp_id
        
        # Area filtering
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
        
        # Create merged cell info from member cells
        if positive_cells_info is not None:
            # Collect all member cells (1-based index)
            member_cells = []
            for m_id in member_ids:
                if m_id <= len(positive_cells_info):
                    member_cells.append(positive_cells_info[m_id - 1])
            
            if member_cells:
                # Determine is_positive: positive if ANY member is positive
                is_positive = any(c.get('is_positive', True) for c in member_cells)
                
                # Merge cell info
                merged_cell = {
                    'id': new_id,
                    'is_positive': is_positive,
                    'pixel_count': sum(c.get('pixel_count', 0) for c in member_cells),
                    'marker_sum': sum(c.get('marker_sum', 0) for c in member_cells),
                    'marker_mean': np.mean([c.get('marker_mean', 0) for c in member_cells]),
                    'marker_max': max(c.get('marker_max', 0) for c in member_cells),
                    'marker_min': min(c.get('marker_min', 255) for c in member_cells),
                    'center': member_cells[0]['center'],  # Use first cell's center
                    'member_ids': member_ids,
                    'area': area  # Add actual merged area
                }
                merged_cells_info.append(merged_cell)
            else:
                # No valid member cells found, create placeholder
                merged_cells_info.append({
                    'id': new_id,
                    'is_positive': True,  # Default to positive
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
    
    # Save merge/filter visualization
    if save_steps_dir:
        # 2. Save filtered regions visualization
        if filtered_regions:
            filtered_viz = _create_filtered_regions_visualization(labels, filtered_regions, h, w)
            cv2.imwrite(os.path.join(save_steps_dir, "step2_filtered_regions.png"), 
                       cv2.cvtColor(filtered_viz, cv2.COLOR_RGB2BGR))
        
        # 3. Save post-merge visualization
        post_merge_viz = _create_instance_mask_visualization(merged_mask, h, w, "Merged & Filtered")
        cv2.imwrite(os.path.join(save_steps_dir, "step3_merged_result.png"), 
                   cv2.cvtColor(post_merge_viz, cv2.COLOR_RGB2BGR))
        
        # If original image available, create final overlay
        if original_image is not None:
            final_overlay = _create_mask_overlay(original_image, merged_mask)
            cv2.imwrite(os.path.join(save_steps_dir, "step3_merged_overlay.png"), 
                       cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR))
        
        # 4. Save merge summary JSON
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
        with open(os.path.join(save_steps_dir, "merge_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"      Saved merge/filter visualization to {save_steps_dir}/")
    
    # If no positive_cells_info provided, return None for merged_cells_info
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
    combined_mask = np.zeros((h, w), dtype=np.uint8)
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
    
    combined_mask_pass1 = np.zeros((h, w), dtype=np.uint8)
    combined_mask_pass2 = np.zeros((h, w), dtype=np.uint8)
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
                                 min_area: int = 10, prompt_mode: str = 'box+mask', 
                                 set_image: bool = True, batch_size: int = 32,
                                 score_threshold: float = 0.3) -> tuple:
    """
    批量版本的 SAM2 推理，一次处理多个 prompt 以提升速度。
    
    Args:
        predictor: SAM2ImagePredictor instance
        image: RGB image array
        clusters: List of cluster coordinate arrays
        min_area: Minimum cluster size
        prompt_mode: 'box+mask' or 'mask_only'
        set_image: Whether to call predictor.set_image()
        batch_size: Number of prompts to process at once
        score_threshold: Minimum score threshold (for mask_only mode)
    
    Returns:
        combined_mask: Instance segmentation mask
        scores_list: List of (instance_id, score) tuples
        filtered_list: List of (instance_id, score) tuples for filtered instances
    """
    if set_image:
        predictor.set_image(image)
        
    h, w = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.float32)
    scores_list = []
    filtered_list = []
    
    # 1. 过滤有效的 clusters
    valid_clusters = []
    for idx, cluster in enumerate(clusters):
        if len(cluster) >= min_area:
            valid_clusters.append((idx, cluster))
    
    if len(valid_clusters) == 0:
        return combined_mask, scores_list, filtered_list
    
    # 2. 分批处理
    total_batches = (len(valid_clusters) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(valid_clusters))
        batch = valid_clusters[batch_start:batch_end]
        
        # 准备批量输入
        masks_list = []
        boxes_list = []
        original_indices = []
        
        for idx, cluster in batch:
            mask_input = generate_mask_from_cluster(cluster, image.shape)
            masks_list.append(mask_input)
            original_indices.append(idx)
            
            if prompt_mode == 'box+mask':
                box = get_bounding_box_from_cluster(cluster, padding=10)
                box = np.clip(box, [0, 0, 0, 0], [w, h, w, h]).astype(np.int64)
                boxes_list.append(box)
        
        try:
            # 堆叠成批量 tensor
            masks_batch = np.concatenate(masks_list, axis=0)  # (B, 256, 256)
            masks_batch = masks_batch[:, np.newaxis, :, :]    # (B, 1, 256, 256)
            
            if prompt_mode == 'box+mask':
                boxes_batch = np.stack(boxes_list, axis=0)    # (B, 4)
                
                # 批量推理
                all_masks, all_scores, _ = predictor.predict(
                    box=boxes_batch,
                    mask_input=masks_batch,
                    multimask_output=True
                )
            else:  # mask_only
                all_masks, all_scores, _ = predictor.predict(
                    mask_input=masks_batch,
                    multimask_output=True
                )
            
            # all_masks: (B, 3, H, W) 或 (3, H, W) 如果 B=1
            # all_scores: (B, 3) 或 (3,) 如果 B=1
            
            # 确保维度正确
            if len(all_masks.shape) == 3:
                all_masks = all_masks[np.newaxis, :]
                all_scores = all_scores[np.newaxis, :]
            
            # 3. 后处理每个结果
            for i, orig_idx in enumerate(original_indices):
                best_idx = int(np.argmax(all_scores[i]))
                best_score = float(all_scores[i, best_idx])
                best_mask = all_masks[i, best_idx].astype(bool)
                mask_area = np.sum(best_mask)
                
                inst_id = orig_idx + 1
                
                # 置信度过滤 (mask_only 模式)
                if prompt_mode == 'mask_only' and best_score < score_threshold:
                    print(f"      Instance {inst_id}: score={best_score:.4f}, area={mask_area} pixels (mode={prompt_mode}) [FILTERED: score<{score_threshold}]")
                    filtered_list.append((inst_id, best_score))
                    continue
                
                # 置信度优先合并
                overwrite_mask = best_mask & (best_score > score_map)
                combined_mask[overwrite_mask] = inst_id
                score_map[overwrite_mask] = best_score
                
                scores_list.append((inst_id, best_score))
                
                final_area = np.sum(combined_mask == inst_id)
                print(f"      Instance {inst_id}: score={best_score:.4f}, area={mask_area} pixels (mode={prompt_mode}) [final: {final_area} pixels]")
                
        except Exception as e:
            # 如果批量推理失败，回退到逐个处理
            print(f"    Batch inference failed: {e}, falling back to sequential...")
            for idx, cluster in batch:
                try:
                    mask_input = generate_mask_from_cluster(cluster, image.shape)
                    
                    if prompt_mode == 'box+mask':
                        box = get_bounding_box_from_cluster(cluster, padding=10)
                        box = np.clip(box, [0, 0, 0, 0], [w, h, w, h]).astype(np.int64)
                        masks, scores, _ = predictor.predict(
                            box=box,
                            mask_input=mask_input,
                            multimask_output=True
                        )
                    else:
                        masks, scores, _ = predictor.predict(
                            mask_input=mask_input,
                            multimask_output=True
                        )
                    
                    best_idx = int(np.argmax(scores))
                    best_score = float(scores[best_idx])
                    best_mask = masks[best_idx].astype(bool)
                    mask_area = np.sum(best_mask)
                    inst_id = idx + 1
                    
                    if prompt_mode == 'mask_only' and best_score < score_threshold:
                        filtered_list.append((inst_id, best_score))
                        continue
                    
                    overwrite_mask = best_mask & (best_score > score_map)
                    combined_mask[overwrite_mask] = inst_id
                    score_map[overwrite_mask] = best_score
                    scores_list.append((inst_id, best_score))
                    
                except Exception as e2:
                    print(f"    SAM2 Error on cluster {idx}: {e2}")
    
    return combined_mask, scores_list, filtered_list
