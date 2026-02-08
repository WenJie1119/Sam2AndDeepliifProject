#!/usr/bin/env python3
"""
SAM2 segmentation script for red pixel coordinate regions.
Reads original images and red pixel coordinate CSV files,
clusters coordinates into regions, generates prompts, and saves segmentation results.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import DBSCAN

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Add sam2 to path
sys.path.insert(0, '/local1/yangwenjie/sam2')

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_coordinates(csv_path: str) -> NDArray[np.int64]:
    """Load red pixel coordinates from CSV file."""
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=(0, 1), dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def cluster_coordinates(
    coords: NDArray, eps: float = 10, min_samples: int = 5
) -> list[NDArray]:
    """
    Cluster coordinates into separate regions using DBSCAN.
    Returns list of cluster coordinate arrays.
    """
    if len(coords) < min_samples:
        return [coords] if len(coords) > 0 else []

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_

    unique_labels = set(labels) - {-1}  # Exclude noise
    clusters = []
    for label in unique_labels:
        mask = labels == label
        cluster = coords[mask].astype(coords.dtype)
        clusters.append(cluster)
    return clusters


def get_clusters_from_mask(mask_path: str, min_area: int = 10) -> list[NDArray]:
    """
    Generate clusters from a binary mask using connected components.
    """
    if not os.path.exists(mask_path):
        return []

    # Read mask, handle 4-channel or RGB
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        if mask.shape[2] == 4:  # RGBA
            mask = mask[..., 3]  # Use alpha
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    
    # Threshold
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Connected components
    num_labels, labels = cv2.connectedComponents(binary_mask)
    
    clusters = []
    # Label 0 is background
    for i in range(1, num_labels):
        # cv2.connectedComponents returns (row, col) coordinates for where mask is true
        points = np.argwhere(labels == i)
        if len(points) >= min_area:
            clusters.append(points)
            
    return clusters


def generate_prompts_from_cluster(
    cluster_coords: NDArray, num_points: int = 5, rng: np.random.Generator | None = None
) -> NDArray | None:
    """
    Generate prompt points from a cluster.
    Returns points in (x, y) format for SAM2.
    """
    if len(cluster_coords) == 0:
        return None

    if rng is None:
        rng = np.random.default_rng()

    centroid = cluster_coords.mean(axis=0).astype(cluster_coords.dtype)

    if len(cluster_coords) <= num_points:
        points = cluster_coords.copy()
    else:
        num_to_sample = min(num_points - 1, len(cluster_coords))
        indices = rng.choice(len(cluster_coords), num_to_sample, replace=False)
        indices = indices.astype(np.intp)
        points = np.vstack([centroid.reshape(1, -1), cluster_coords[indices]])

    # Convert (row, col) to (x, y)
    return points[:, ::-1].astype(np.float32)


def generate_mask_from_cluster(
    cluster_coords: NDArray, image_shape: tuple[int, ...], target_size: int = 256
) -> NDArray[np.float32]:
    """
    Generate a low-resolution binary mask from cluster coordinates for SAM2.
    Returns mask of shape (1, target_size, target_size).
    """
    h, w = image_shape[:2]
    full_mask = np.zeros((h, w), dtype=np.float32)

    rows = cluster_coords[:, 0].astype(np.intp)
    cols = cluster_coords[:, 1].astype(np.intp)
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    full_mask[rows[valid], cols[valid]] = 1.0

    low_res_mask = cv2.resize(full_mask, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return low_res_mask[np.newaxis, :, :]


def get_bounding_box_from_cluster(
    cluster_coords: NDArray, padding: int = 10
) -> NDArray[np.int64] | None:
    """Get bounding box from cluster in (x1, y1, x2, y2) format."""
    if len(cluster_coords) == 0:
        return None

    min_row = int(cluster_coords[:, 0].min())
    max_row = int(cluster_coords[:, 0].max())
    min_col = int(cluster_coords[:, 1].min())
    max_col = int(cluster_coords[:, 1].max())

    return np.array([
        max(0, min_col - padding),
        max(0, min_row - padding),
        max_col + padding,
        max_row + padding
    ], dtype=np.int64)


def segment_image(
    predictor: SAM2ImagePredictor,
    image: NDArray,
    clusters: list[NDArray],
    use_box: bool = True,
    points_per_cluster: int = 3,
    prompt_mode: str = "point_only",
    rng: np.random.Generator | None = None,
) -> NDArray[np.uint8]:
    """
    Segment image using SAM2 with prompts from coordinate clusters.
    Returns combined mask with unique labels for each cluster.
    """
    predictor.set_image(image)
    h, w = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    if rng is None:
        rng = np.random.default_rng()

    for idx, cluster in enumerate(clusters):
        if len(cluster) < 3:
            continue

        try:
            point_coords, point_labels, mask_input = None, None, None

            if prompt_mode in ("point_only", "mixed"):
                point_coords = generate_prompts_from_cluster(cluster, num_points=points_per_cluster, rng=rng)
                point_labels = np.ones(len(point_coords), dtype=np.int32)

            if prompt_mode in ("mask_only", "mixed"):
                mask_input = generate_mask_from_cluster(cluster, image.shape)

            box = get_bounding_box_from_cluster(cluster, padding=15) if use_box else None
            if box is not None:
                box = np.clip(box, [0, 0, 0, 0], [w, h, w, h]).astype(np.int64)

            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                mask_input=mask_input,
                multimask_output=True,
            )

            best_mask_idx = int(np.argmax(scores))
            best_mask = masks[best_mask_idx]
            # Ensure mask is boolean type for indexing
            if best_mask.dtype != bool:
                best_mask = best_mask.astype(bool)
            combined_mask[best_mask] = idx + 1

        except Exception as e:
            import traceback
            print(f"  Warning: Failed to segment cluster {idx}: {e}")
            print(f"  Traceback: {traceback.format_exc()}")

    return combined_mask


def generate_distinct_colors(n):
    """Generate n visually distinct colors using HSV color space."""
    if n == 0:
        return []
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        hsv_color = np.uint8([[[hue, 255, 255]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
        colors.append(tuple(map(int, rgb_color)))
    return colors


def visualize_clusters(image, clusters):
    """Visualize all cluster points with distinct colors on the image."""
    vis_image = image.copy()
    n_clusters = len(clusters)
    if n_clusters == 0:
        return vis_image

    colors = generate_distinct_colors(n_clusters)
    for idx, cluster in enumerate(clusters):
        rows = cluster[:, 0].astype(np.intp)
        cols = cluster[:, 1].astype(np.intp)
        h, w = image.shape[:2]
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        vis_image[rows[valid], cols[valid]] = colors[idx]

    return vis_image


def visualize_prompts(image, clusters, points_per_cluster=3):
    """Visualize SAM input prompt points on the image."""
    vis_image = image.copy()
    n_clusters = len(clusters)
    if n_clusters == 0:
        return vis_image

    colors = generate_distinct_colors(n_clusters)
    for idx, cluster in enumerate(clusters):
        if len(cluster) < 3:
            continue
        prompt_points = generate_prompts_from_cluster(cluster, num_points=points_per_cluster)
        if prompt_points is not None:
            for pt in prompt_points:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(vis_image, (x, y), 6, colors[idx], -1)
                cv2.circle(vis_image, (x, y), 6, (0, 0, 0), 2)

    return vis_image


def create_overlay(image, mask, color=(255, 0, 0), alpha=1.0):
    """Create overlay visualization of mask on image."""
    overlay = image.copy()
    mask_region = mask > 0
    if alpha < 1.0:
        overlay[mask_region] = (overlay[mask_region] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    else:
        overlay[mask_region] = color
    return overlay


def create_instance_visualization(mask):
    """Create colorful instance segmentation visualization."""
    num_instances = int(mask.max())
    h, w = mask.shape
    instance_vis = np.zeros((h, w, 3), dtype=np.uint8)

    if num_instances > 0:
        colors = generate_distinct_colors(num_instances)
        for i in range(1, num_instances + 1):
            instance_vis[mask == i] = colors[i - 1]

    return instance_vis


def create_comparison_image(images, labels):
    """Create side-by-side comparison image with labels."""
    h, w = images[0].shape[:2]

    # Convert to BGR for OpenCV
    bgr_images = []
    for img in images:
        if len(img.shape) == 2:
            # Grayscale to BGR
            bgr_images.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        else:
            bgr_images.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(min(h, w) / 500, 2.0))
    thickness = max(1, int(font_scale * 2))

    for img, label in zip(bgr_images, labels):
        cv2.putText(img, label, (10, 30), font, font_scale, (255, 255, 255), thickness + 2)
        cv2.putText(img, label, (10, 30), font, font_scale, (0, 0, 0), thickness)

    return np.concatenate(bgr_images, axis=1)


def save_image(path, image, is_rgb=True):
    """Save image, converting from RGB to BGR if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if is_rgb and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)


def create_input_mask_image(image, clusters):
    """Create input mask visualization (white pixels on black background)."""
    h, w = image.shape[:2]
    input_mask = np.zeros((h, w), dtype=np.uint8)
    for cluster in clusters:
        rows = cluster[:, 0].astype(np.intp)
        cols = cluster[:, 1].astype(np.intp)
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        input_mask[rows[valid], cols[valid]] = 255
    # Convert to RGB for consistency
    return cv2.cvtColor(input_mask, cv2.COLOR_GRAY2RGB)


def create_mixed_input_image(image, clusters, points_per_cluster=3):
    """Create mixed input visualization: mask + points overlay."""
    h, w = image.shape[:2]
    # Start with input mask
    vis_image = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw mask pixels in white
    for cluster in clusters:
        rows = cluster[:, 0].astype(np.intp)
        cols = cluster[:, 1].astype(np.intp)
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        vis_image[rows[valid], cols[valid]] = [255, 255, 255]

    # Draw prompt points on top
    n_clusters = len(clusters)
    if n_clusters > 0:
        colors = generate_distinct_colors(n_clusters)
        for idx, cluster in enumerate(clusters):
            if len(cluster) < 3:
                continue
            prompt_points = generate_prompts_from_cluster(cluster, num_points=points_per_cluster)
            if prompt_points is not None:
                for pt in prompt_points:
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(vis_image, (x, y), 8, colors[idx], -1)
                    cv2.circle(vis_image, (x, y), 8, (0, 0, 0), 2)

    return vis_image


def save_results(output_dir, image_name, image, mask, clusters, points_per_cluster=3, prompt_mode="point_only"):
    """Save all segmentation results."""
    base_name = os.path.splitext(image_name)[0]

    # Precompute visualizations
    cluster_vis = visualize_clusters(image, clusters)
    overlay_red = create_overlay(image, mask, alpha=1.0)
    overlay_blend = create_overlay(image, mask, alpha=0.5)
    instance_vis = create_instance_visualization(mask)

    # Save SAM outputs
    save_image(f"{output_dir}/sam_output_masks/{base_name}_mask.png",
               (mask > 0).astype(np.uint8) * 255, is_rgb=False)
    save_image(f"{output_dir}/sam_output_overlays_red/{base_name}_overlay.png", overlay_red)
    save_image(f"{output_dir}/sam_output_overlays_blend/{base_name}_overlay.png", overlay_blend)
    save_image(f"{output_dir}/sam_output_instances/{base_name}_instance.png", instance_vis)

    # Save input visualizations
    save_image(f"{output_dir}/sam_input_clusters/{base_name}_clusters.png", cluster_vis)

    # Create comparison based on prompt mode
    if prompt_mode == "point_only":
        # Input: point prompts visualization
        points_vis = visualize_prompts(image, clusters, points_per_cluster)
        save_image(f"{output_dir}/sam_input_points/{base_name}_points.png", points_vis)

        comparison = create_comparison_image(
            [image, points_vis, overlay_blend],
            ["Original", "Input Points", "SAM Output"]
        )

    elif prompt_mode == "mask_only":
        # Input: mask visualization
        input_mask_vis = create_input_mask_image(image, clusters)
        save_image(f"{output_dir}/sam_input_masks/{base_name}_input_mask.png",
                   input_mask_vis[:, :, 0], is_rgb=False)

        comparison = create_comparison_image(
            [image, input_mask_vis, overlay_blend],
            ["Original", "Input Mask", "SAM Output"]
        )

    else:  # mixed
        # Input: both points and mask
        points_vis = visualize_prompts(image, clusters, points_per_cluster)
        input_mask_vis = create_input_mask_image(image, clusters)
        mixed_input_vis = create_mixed_input_image(image, clusters, points_per_cluster)

        save_image(f"{output_dir}/sam_input_points/{base_name}_points.png", points_vis)
        save_image(f"{output_dir}/sam_input_masks/{base_name}_input_mask.png",
                   input_mask_vis[:, :, 0], is_rgb=False)

        comparison = create_comparison_image(
            [image, mixed_input_vis, overlay_blend],
            ["Original", "Input (Mask+Points)", "SAM Output"]
        )

    # Save comparison
    os.makedirs(f"{output_dir}/comparison", exist_ok=True)
    cv2.imwrite(f"{output_dir}/comparison/{base_name}_comparison.png", comparison)

    print(f"  Saved results for {base_name}")


def process_image(predictor, image_path, data_source_path, output_dirs,
                  # Cluster params
                  cluster_eps, cluster_min_samples, 
                  # Common params
                  points_per_cluster, prompt_modes,
                  # Method
                  method="dbscan"):
    """Process a single image with all prompt modes."""
    image_name = os.path.basename(image_path)

    # Load image
    image = np.array(Image.open(image_path).convert("RGB"))
    
    clusters = []
    if method == "dbscan":
        coords = load_coordinates(data_source_path)
        if len(coords) == 0:
            print(f"  No coordinates found, skipping...")
            return
        # Cluster coordinates
        clusters = cluster_coordinates(coords, eps=cluster_eps, min_samples=cluster_min_samples)
        print(f"  Image: {image.shape}, Coords: {len(coords)}, Clusters: {len(clusters)} (DBSCAN)")
        
    elif method == "connected_components":
        clusters = get_clusters_from_mask(data_source_path)
        print(f"  Image: {image.shape}, Clusters: {len(clusters)} (CC)")

    if len(clusters) == 0:
        print(f"  No valid clusters found, skipping...")
        return

    # Process each prompt mode
    for prompt_mode in prompt_modes:
        mask = segment_image(predictor, image, clusters, use_box=True,
                           points_per_cluster=points_per_cluster,
                           prompt_mode=prompt_mode)

        num_regions = len(np.unique(mask)) - 1
        print(f"    [{prompt_mode}] Segmented {num_regions} regions")

        save_results(output_dirs[prompt_mode], image_name, image, mask,
                    clusters, points_per_cluster, prompt_mode)


def main():
    parser = argparse.ArgumentParser(description='SAM2 segmentation for red pixel regions')
    parser.add_argument('--data-dir', type=str,
                        default="/local1/yangwenjie/DataImg/CD34Deepliif/ywj_data",
                        help='Data directory containing original_images')
    parser.add_argument('--method', type=str, default='dbscan',
                        choices=['dbscan', 'connected_components'],
                        help='Method to generate region proposals')
    parser.add_argument('--output-dir', type=str,
                        default="/local1/yangwenjie/DataImg/CD34SAM2",
                        help='Output directory for results')
    parser.add_argument('--checkpoint', type=str,
                        default="/local1/yangwenjie/sam2/checkpoints/sam2.1_hiera_large.pt",
                        help='SAM2 checkpoint path')
    parser.add_argument('--config', type=str,
                        default="configs/sam2.1/sam2.1_hiera_l.yaml",
                        help='SAM2 config path')
    parser.add_argument('--cluster-eps', type=float, default=15,
                        help='DBSCAN epsilon for clustering')
    parser.add_argument('--cluster-min-samples', type=int, default=5,
                        help='DBSCAN minimum samples per cluster')
    parser.add_argument('--points-per-cluster', type=int, default=3,
                        help='Number of prompt points per cluster')
    parser.add_argument('--modes', type=str, nargs='+',
                        default=['point_only', 'mask_only', 'mixed'],
                        choices=['point_only', 'mask_only', 'mixed'],
                        help='Prompt modes to run')

    args = parser.parse_args()

    # Paths
    original_images_dir = os.path.join(args.data_dir, "original_images")
    coords_dir = os.path.join(args.data_dir, "original_images_results_red_coordes_results")
    masks_dir = os.path.join(args.data_dir, "original_images_results")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load SAM2 model
    print("Loading SAM2 model...")
    sam2_model = build_sam2(args.config, args.checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("Model loaded successfully!")

    # Get image files
    image_files = sorted([f for f in os.listdir(original_images_dir) if f.endswith('.png')])
    print(f"Found {len(image_files)} images to process")

    # Create output directories
    output_dirs = {mode: os.path.join(args.output_dir, mode) for mode in args.modes}

    # Process each image
    for img_idx, image_name in enumerate(image_files):
        print(f"\n[{img_idx + 1}/{len(image_files)}] Processing: {image_name}")

        image_path = os.path.join(original_images_dir, image_name)
        base_name = os.path.splitext(image_name)[0]
        
        data_source_path = ""
        if args.method == "dbscan":
            data_source_path = os.path.join(coords_dir, f"{base_name}_SegRefined_coords.csv")
            if not os.path.exists(data_source_path):
                print(f"  Warning: Coordinates file not found: {data_source_path}")
                continue
        elif args.method == "connected_components":
             # Pattern seems to be [name]_SegRefined.png based on find_by_name search
             # e.g. tile_21_202_10240_102912_SegRefined.png
             # Wait, user file pattern check.
             # The files found were in `original_images_results/` and had `_SegRefined.png` suffix.
             # Let's construct it carefully.
             data_source_path = os.path.join(masks_dir, f"{base_name}_SegRefined.png")
             if not os.path.exists(data_source_path):
                print(f"  Warning: Mask file not found: {data_source_path}")
                continue

        process_image(predictor, image_path, data_source_path, output_dirs,
                     args.cluster_eps, args.cluster_min_samples,
                     args.points_per_cluster, args.modes,
                     method=args.method)

    print(f"\n{'='*60}")
    print(f"All results saved to: {args.output_dir}")
    for mode in args.modes:
        print(f"  - {mode}/")
    print("Done!")


if __name__ == "__main__":
    main()
