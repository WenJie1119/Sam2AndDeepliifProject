#!/usr/bin/env python3
"""Run one tile through the weighted-prompt SAM2 debug flow.

The top-level PNG files form one globally numbered visualization sequence.
Supporting arrays are kept in the data subdirectory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cd34_pipeline.sam2_wrapper.model_loader import load_sam2


DEFAULT_TILE_DIR = Path(
    "debug_output/debug_marker_keep31/debug_vis/tile_39_12_4608_14976"
)

COLORS = {
    -5: np.array([30, 30, 160], dtype=np.uint8),
    0: np.array([180, 180, 180], dtype=np.uint8),
    1: np.array([255, 230, 130], dtype=np.uint8),
    2: np.array([255, 190, 80], dtype=np.uint8),
    3: np.array([255, 140, 40], dtype=np.uint8),
    4: np.array([230, 70, 25], dtype=np.uint8),
    5: np.array([180, 0, 0], dtype=np.uint8),
}


VISUAL_STEPS = (
    ("input_original", "01_input_original.png", "原始输入图"),
    ("deepliif_seg", "02_deepliif_seg.png", "DeepLIIF Seg 输出"),
    ("deepliif_marker", "03_deepliif_marker.png", "DeepLIIF Marker 输出"),
    ("raw_prompt", "04_prompt_raw_heatmap.png", "Seg + Marker 原始 weighted prompt"),
    ("artifact_weak_mid", "05_artifact_weak_mid_mask.png", "提取 logit 1..3 的弱/中阳性区域"),
    ("artifact_closed", "06_artifact_after_close.png", "形态学 close 后的候选区域"),
    ("artifact_grouped", "07_artifact_after_dilate.png", "dilate 后的分组种子"),
    ("artifact_components", "08_artifact_connected_components.png", "artifact 候选连通域"),
    ("artifact_decisions", "09_artifact_component_decisions.png", "artifact 连通域保留/删除决策"),
    ("artifact_mask", "10_artifact_mask.png", "最终 artifact 删除 mask"),
    ("artifact_filtered_prompt", "11_prompt_after_artifact_filter.png", "删除 artifact 后的 prompt"),
    ("small_fragment_mask", "12_small_weak_fragment_mask.png", "细小弱阳性碎片 mask"),
    ("small_fragment_overlay", "13_small_weak_fragments_overlay.png", "细小弱阳性碎片位置"),
    ("small_fragment_filtered_prompt", "14_prompt_after_small_fragment_filter.png", "删除细小弱阳性碎片后的 prompt"),
    ("weighted_prompt_heatmap", "15_prompt_final_heatmap.png", "修补、填洞和边界处理后的最终 prompt"),
    ("weighted_prompt_overlay", "16_prompt_final_overlay.png", "最终 prompt 叠加到原图"),
    ("weighted_prompt_mask_input", "17_sam2_mask_input_256.png", "送入 SAM2 的 256x256 mask input"),
    ("weighted_best_mask", "18_sam2_weighted_best_mask.png", "仅 weighted prompt 的 SAM2 最优 mask"),
    ("weighted_best_overlay", "19_sam2_weighted_best_overlay.png", "仅 weighted prompt 的 SAM2 最优结果"),
    ("strong_positive_points", "20_strong_positive_points.png", "最终 prompt 提取的强阳性点"),
    ("points_only_best_mask", "21_sam2_points_only_best_mask.png", "仅强阳性点的 SAM2 最优 mask"),
    ("points_only_best_overlay", "22_sam2_points_only_best_overlay.png", "仅强阳性点的 SAM2 最优结果"),
    ("weighted_plus_points_best_mask", "23_sam2_weighted_plus_points_best_mask.png", "weighted prompt + 强阳性点的 SAM2 最优 mask"),
    ("weighted_plus_points_best_overlay", "24_sam2_weighted_plus_points_best_overlay.png", "weighted prompt + 强阳性点的 SAM2 最优结果"),
)
VISUAL_FILES = {key: filename for key, filename, _ in VISUAL_STEPS}


def visual_path(out_dir: Path, key: str) -> Path:
    return out_dir / VISUAL_FILES[key]


def data_path(out_dir: Path, filename: str) -> Path:
    path = out_dir / "data" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.uint8)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def colorize_logits(logits: np.ndarray) -> np.ndarray:
    vis = np.zeros((*logits.shape, 3), dtype=np.uint8)
    for value, color in COLORS.items():
        vis[logits == value] = color
    return vis


def colorize_component_labels(labels: np.ndarray) -> np.ndarray:
    num_labels = int(labels.max()) + 1
    rng = np.random.default_rng(42)
    colors = rng.integers(60, 256, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = np.array([30, 30, 160], dtype=np.uint8)
    return colors[labels]


def draw_component_decisions(
        image: np.ndarray,
        components: list[dict],
) -> np.ndarray:
    overlay = image.copy()
    for component in components:
        x, y, box_w, box_h = component["bbox_xywh"]
        selected = component["selected"]
        color = (255, 0, 0) if selected else (0, 210, 0)
        label = "DROP" if selected else "KEEP"
        text = f"{label} s={component['score']}"
        cv2.rectangle(
            overlay,
            (x, y),
            (x + box_w - 1, y + box_h - 1),
            color,
            thickness=2,
        )
        text_y = max(12, y - 4)
        cv2.putText(
            overlay,
            text,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            text,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            lineType=cv2.LINE_AA,
        )
    return overlay


def overlay_mask(
        image: np.ndarray,
        mask: np.ndarray,
        color: tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.55,
) -> np.ndarray:
    overlay = image.copy()
    pixels = mask.astype(bool)
    if pixels.any():
        color_arr = np.array(color, dtype=np.float32)
        overlay[pixels] = (
            overlay[pixels].astype(np.float32) * (1.0 - alpha)
            + color_arr * alpha
        ).clip(0, 255).astype(np.uint8)
    return overlay


def seg_logits(
        seg: np.ndarray,
        seg_thresh: int,
        foreground_green_max: int,
) -> tuple[np.ndarray, dict]:
    r = seg[:, :, 0].astype(np.int16)
    g = seg[:, :, 1].astype(np.int16)
    b = seg[:, :, 2].astype(np.int16)
    foreground = (r + b > seg_thresh) & (g <= foreground_green_max)
    positive = foreground & (r >= b)
    negative_blue = foreground & (b > r)

    out = np.full(seg.shape[:2], -5, dtype=np.int16)
    out[positive & (r < 150)] = 0
    out[positive & (r >= 150) & (r < 170)] = 1
    out[positive & (r >= 170) & (r < 190)] = 2
    out[positive & (r >= 190) & (r < 210)] = 3
    out[positive & (r >= 210) & (r < 235)] = 4
    out[positive & (r >= 235)] = 5

    ambiguous = positive & ((r - b) < 20) & (out > 1)
    out[ambiguous] -= 1
    out[negative_blue] = -5

    stats = {
        "seg_foreground_px": int(foreground.sum()),
        "seg_blue_negative_px": int(negative_blue.sum()),
        "seg_positive_any_px": int((positive & (out > 0)).sum()),
    }
    for value in (1, 2, 3, 4, 5):
        stats[f"seg_logit_{value}_px"] = int((out == value).sum())
    return out, stats


def marker_logits(
        marker: np.ndarray,
        marker_thresh: int,
        marker_max: int,
) -> tuple[np.ndarray, dict]:
    out = np.full(marker.shape, -5, dtype=np.int16)
    positive = marker > marker_thresh
    if positive.any():
        denom = max(1, marker_max - marker_thresh)
        norm = (marker.astype(np.float32) - marker_thresh) / denom
        bins = np.ceil(np.clip(norm, 0, 1) * 5).astype(np.int16)
        bins = np.clip(bins, 1, 5)
        out[positive] = bins[positive]

    stats = {
        "marker_thresh": int(marker_thresh),
        "roi_marker_max": int(marker_max),
        "tile_marker_max": int(marker.max()),
        "marker_positive_px": int(positive.sum()),
    }
    for value in (1, 2, 3, 4, 5):
        stats[f"marker_logit_{value}_px"] = int((out == value).sum())
    return out, stats


def fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    flood = mask.astype(np.uint8) * 255
    filled_from_border = flood.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(filled_from_border, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(filled_from_border) > 0
    return mask | holes


def max_pool_lowres(logits: np.ndarray, target_size: int) -> np.ndarray:
    h, w = logits.shape
    low = np.full((target_size, target_size), -5, dtype=np.int16)
    rr = (np.arange(h) * target_size // h).astype(np.intp)
    cc = (np.arange(w) * target_size // w).astype(np.intp)
    rows = np.repeat(rr, w)
    cols = np.tile(cc, h)
    np.maximum.at(low, (rows, cols), logits.reshape(-1))
    return low.astype(np.float32)


def make_odd_kernel_size(value: int) -> int:
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    return value


def logit_counts(logits: np.ndarray, mask: np.ndarray) -> dict:
    values = logits[mask].astype(np.int16)
    counts = {}
    for value in (-5, 0, 1, 2, 3, 4, 5):
        counts[str(value)] = int((values == value).sum())
    return counts


def suppress_artifact_prompt(
        raw: np.ndarray,
        args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Suppress diffuse weak/mid-positive prompt clusters before SAM2."""
    filtered = raw.copy()
    artifact_mask = np.zeros(raw.shape, dtype=bool)
    h, w = raw.shape

    if not args.enable_artifact_filter:
        empty_labels = np.zeros(raw.shape, dtype=np.int32)
        return filtered, artifact_mask, {
            "weak_mid_mask": artifact_mask.copy(),
            "closed_weak_mid_mask": artifact_mask.copy(),
            "grouped_seed": artifact_mask.copy(),
            "component_labels": empty_labels,
        }, {
            "artifact_filter_enabled": False,
        }

    weak_mid = (
        (raw >= args.artifact_min_logit)
        & (raw <= args.artifact_max_logit)
    )
    kernel_size = make_odd_kernel_size(args.artifact_group_kernel)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed_weak_mid = cv2.morphologyEx(
        weak_mid.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
        iterations=args.artifact_close_iterations,
    )
    seed = cv2.dilate(
        closed_weak_mid,
        kernel,
        iterations=args.artifact_dilate_iterations,
    ).astype(bool)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        seed.astype(np.uint8), connectivity=8)
    components = []
    selected_components = []

    positive = raw > 0
    tile_area = h * w
    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        box_w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        box_h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        region = labels == label_id
        positive_region = region & positive
        positive_area = int(positive_region.sum())
        if positive_area < args.artifact_min_area:
            continue

        values = raw[positive_region].astype(np.int16)
        weak_mid_count = int(
            ((values >= args.artifact_min_logit)
             & (values <= args.artifact_max_logit)).sum())
        strong_count = int((values >= args.artifact_strong_logit).sum())
        weak_mid_ratio = weak_mid_count / max(positive_area, 1)
        strong_ratio = strong_count / max(positive_area, 1)
        mean_logit = float(values.mean()) if positive_area else 0.0
        touches_border = (
            x == 0 or y == 0 or x + box_w >= w or y + box_h >= h
        )

        score = 0
        if weak_mid_ratio >= args.artifact_weak_mid_ratio:
            score += 2
        if strong_ratio <= args.artifact_max_strong_ratio:
            score += 2
        if mean_logit <= args.artifact_max_mean_logit:
            score += 1
        if touches_border:
            score += 1

        selected = score >= args.artifact_score_threshold
        component = {
            "label_id": int(label_id),
            "selected": bool(selected),
            "score": int(score),
            "positive_area": positive_area,
            "area_ratio": float(positive_area / tile_area),
            "bbox_xywh": [x, y, box_w, box_h],
            "touches_border": bool(touches_border),
            "weak_mid_ratio": float(weak_mid_ratio),
            "strong_ratio": float(strong_ratio),
            "mean_logit": mean_logit,
            "centroid_xy": [
                float(centroids[label_id][0]),
                float(centroids[label_id][1]),
            ],
        }
        components.append(component)
        if selected:
            selected_components.append(component)
            artifact_mask |= region

    if artifact_mask.any() and args.artifact_suppress_dilate > 0:
        suppress_kernel_size = make_odd_kernel_size(
            args.artifact_suppress_kernel)
        suppress_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (suppress_kernel_size, suppress_kernel_size),
        )
        artifact_mask = cv2.dilate(
            artifact_mask.astype(np.uint8),
            suppress_kernel,
            iterations=args.artifact_suppress_dilate,
        ).astype(bool)

    removed_positive = artifact_mask & positive
    filtered[artifact_mask] = -5

    stats_out = {
        "artifact_filter_enabled": True,
        "artifact_group_kernel": int(kernel_size),
        "artifact_min_area": int(args.artifact_min_area),
        "artifact_weak_mid_ratio_threshold": float(
            args.artifact_weak_mid_ratio),
        "artifact_max_strong_ratio_threshold": float(
            args.artifact_max_strong_ratio),
        "artifact_max_mean_logit_threshold": float(
            args.artifact_max_mean_logit),
        "artifact_score_threshold": int(args.artifact_score_threshold),
        "artifact_seed_px": int(seed.sum()),
        "artifact_mask_px": int(artifact_mask.sum()),
        "artifact_removed_positive_px": int(removed_positive.sum()),
        "artifact_removed_logit_counts": logit_counts(raw, removed_positive),
        "artifact_candidate_count": int(len(components)),
        "artifact_selected_count": int(len(selected_components)),
        "artifact_components": components,
    }
    debug_out = {
        "weak_mid_mask": weak_mid,
        "closed_weak_mid_mask": closed_weak_mid.astype(bool),
        "grouped_seed": seed,
        "component_labels": labels,
    }
    return filtered, artifact_mask, debug_out, stats_out


def suppress_small_weak_fragments(
        raw: np.ndarray,
        args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Remove small nonnegative components without a strong logit core."""
    filtered = raw.copy()
    fragment_mask = np.zeros(raw.shape, dtype=bool)
    if not args.enable_small_fragment_filter:
        return filtered, fragment_mask, {
            "small_fragment_filter_enabled": False,
        }

    # Logit 0 is the uncertain/gray part of the prompt. Include it in both
    # connectivity and area so gray pixels belonging to a small fragment are
    # removed together with its positive pixels.
    nonnegative = raw >= 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        nonnegative.astype(np.uint8), connectivity=8)
    selected_components = []

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area > args.small_fragment_max_area:
            continue

        component_mask = labels == label_id
        values = raw[component_mask].astype(np.int16)
        max_logit = int(values.max())
        if max_logit > args.small_fragment_max_logit:
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        box_w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        box_h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        fragment_mask |= component_mask
        selected_components.append({
            "label_id": int(label_id),
            "area": area,
            "bbox_xywh": [x, y, box_w, box_h],
            "mean_logit": float(values.mean()),
            "max_logit": max_logit,
        })

    filtered[fragment_mask] = -5
    stats_out = {
        "small_fragment_filter_enabled": True,
        "small_fragment_max_area": int(args.small_fragment_max_area),
        "small_fragment_max_logit": int(args.small_fragment_max_logit),
        "small_fragment_removed_count": int(len(selected_components)),
        "small_fragment_removed_px": int(fragment_mask.sum()),
        "small_fragment_removed_logit_counts": logit_counts(
            raw, fragment_mask),
        "small_fragment_components": selected_components,
    }
    return filtered, fragment_mask, stats_out


def build_weighted_prompt(
        seg: np.ndarray,
        marker: np.ndarray,
        args: argparse.Namespace,
        marker_max: int,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    seg_logit, seg_stats = seg_logits(
        seg, args.seg_thresh, args.foreground_green_max)
    marker_logit, marker_stats = marker_logits(
        marker, args.marker_thresh, marker_max)

    raw = np.maximum(seg_logit, marker_logit).astype(np.int16)
    raw_positive_support = raw > 0
    filtered_raw, artifact_mask, artifact_debug, artifact_stats = (
        suppress_artifact_prompt(raw, args))
    cleaned_raw, small_fragment_mask, small_fragment_stats = (
        suppress_small_weak_fragments(filtered_raw, args))
    support = cleaned_raw > 0

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (args.repair_kernel, args.repair_kernel))
    closed = cv2.morphologyEx(
        support.astype(np.uint8),
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=args.repair_iterations,
    ).astype(bool)
    repair = closed & ~support

    filled = fill_holes(closed)
    lumen = filled & ~closed

    band_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(
        filled.astype(np.uint8),
        band_kernel,
        iterations=args.uncertain_iterations,
    ).astype(bool)
    uncertain = dilated & ~filled

    final = cleaned_raw.copy()
    final[repair & (final < args.repair_logit)] = args.repair_logit
    final[lumen & (final < args.lumen_logit)] = args.lumen_logit
    final[uncertain & (final < 0)] = 0
    final[artifact_mask] = -5
    final[small_fragment_mask] = -5
    final = np.clip(final, -5, 5).astype(np.int16)

    mask_input = max_pool_lowres(final, args.target_size)[None, :, :]

    stats = {
        **seg_stats,
        **marker_stats,
        **artifact_stats,
        **small_fragment_stats,
        "raw_positive_support_px": int(raw_positive_support.sum()),
        "filtered_positive_support_px": int(support.sum()),
        "repair_px": int(repair.sum()),
        "lumen_px": int(lumen.sum()),
        "uncertain_band_px": int(uncertain.sum()),
        "final_nonnegative_px": int((final >= 0).sum()),
    }
    for value in (-5, 0, 1, 2, 3, 4, 5):
        stats[f"raw_logit_{value}_px"] = int((raw == value).sum())
        stats[f"final_logit_{value}_px"] = int((final == value).sum())
    debug = {
        "raw_logits": raw,
        "artifact_mask": artifact_mask,
        "filtered_raw_logits": filtered_raw,
        "small_fragment_mask": small_fragment_mask,
        "cleaned_raw_logits": cleaned_raw,
        **artifact_debug,
    }
    return final, mask_input.astype(np.float32), stats, debug


def save_weighted_prompt_outputs(
        out_dir: Path,
        image: np.ndarray,
        logits_512: np.ndarray,
        mask_input: np.ndarray,
        stats: dict,
        debug: dict | None = None,
) -> None:
    if debug is not None:
        raw_logits = debug.get("raw_logits")
        if raw_logits is not None:
            raw_heat = colorize_logits(raw_logits)
            save_rgb(visual_path(out_dir, "raw_prompt"), raw_heat)

        weak_mid_mask = debug.get("weak_mid_mask")
        if weak_mid_mask is not None:
            save_gray(
                visual_path(out_dir, "artifact_weak_mid"),
                weak_mid_mask.astype(np.uint8) * 255,
            )

        closed_weak_mid_mask = debug.get("closed_weak_mid_mask")
        if closed_weak_mid_mask is not None:
            save_gray(
                visual_path(out_dir, "artifact_closed"),
                closed_weak_mid_mask.astype(np.uint8) * 255,
            )

        grouped_seed = debug.get("grouped_seed")
        if grouped_seed is not None:
            save_gray(
                visual_path(out_dir, "artifact_grouped"),
                grouped_seed.astype(np.uint8) * 255,
            )

        component_labels = debug.get("component_labels")
        if component_labels is not None:
            save_rgb(
                visual_path(out_dir, "artifact_components"),
                colorize_component_labels(component_labels),
            )
            save_rgb(
                visual_path(out_dir, "artifact_decisions"),
                draw_component_decisions(
                    image,
                    stats.get("artifact_components", []),
                ),
            )

        artifact_mask = debug.get("artifact_mask")
        if artifact_mask is not None:
            save_gray(
                visual_path(out_dir, "artifact_mask"),
                artifact_mask.astype(np.uint8) * 255,
            )

        filtered_raw_logits = debug.get("filtered_raw_logits")
        if filtered_raw_logits is not None:
            save_rgb(
                visual_path(out_dir, "artifact_filtered_prompt"),
                colorize_logits(filtered_raw_logits),
            )

        small_fragment_mask = debug.get("small_fragment_mask")
        if small_fragment_mask is not None:
            save_gray(
                visual_path(out_dir, "small_fragment_mask"),
                small_fragment_mask.astype(np.uint8) * 255,
            )
            save_rgb(
                visual_path(out_dir, "small_fragment_overlay"),
                overlay_mask(
                    image,
                    small_fragment_mask,
                    color=(255, 0, 255),
                    alpha=0.75,
                ),
            )

        cleaned_raw_logits = debug.get("cleaned_raw_logits")
        if cleaned_raw_logits is not None:
            save_rgb(
                visual_path(out_dir, "small_fragment_filtered_prompt"),
                colorize_logits(cleaned_raw_logits),
            )

    heat = colorize_logits(logits_512)
    active = logits_512 >= 0
    overlay = image.copy()
    overlay[active] = (
        overlay[active].astype(np.float32) * 0.45
        + heat[active].astype(np.float32) * 0.55
    ).clip(0, 255).astype(np.uint8)

    save_rgb(visual_path(out_dir, "weighted_prompt_heatmap"), heat)
    save_rgb(visual_path(out_dir, "weighted_prompt_overlay"), overlay)
    save_rgb(
        visual_path(out_dir, "weighted_prompt_mask_input"),
        colorize_logits(mask_input[0].astype(np.int16)),
    )
    np.save(data_path(out_dir, "weighted_prompt_logits_512.npy"),
            logits_512.astype(np.float32))
    np.save(data_path(out_dir, "weighted_prompt_mask_input_256.npy"),
            mask_input.astype(np.float32))


def choose_strong_positive_points(
        logits_512: np.ndarray,
        min_area: int,
        max_points: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    strong = logits_512 >= 5
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        strong.astype(np.uint8), connectivity=8)

    components = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = centroids[label_id]
        ys, xs = np.where(labels == label_id)
        nearest_idx = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
        x = int(xs[nearest_idx])
        y = int(ys[nearest_idx])
        components.append({
            "label_id": int(label_id),
            "area": area,
            "bbox_xywh": [
                int(stats[label_id, cv2.CC_STAT_LEFT]),
                int(stats[label_id, cv2.CC_STAT_TOP]),
                int(stats[label_id, cv2.CC_STAT_WIDTH]),
                int(stats[label_id, cv2.CC_STAT_HEIGHT]),
            ],
            "point_xy": [x, y],
        })

    components.sort(key=lambda item: item["area"], reverse=True)
    if max_points > 0:
        components = components[:max_points]

    point_coords = np.array(
        [item["point_xy"] for item in components], dtype=np.float32)
    point_labels = np.ones((len(components),), dtype=np.int32)
    return point_coords, point_labels, components


def save_points_overlay(
        out_path: Path,
        image: np.ndarray,
        components: list[dict],
) -> None:
    overlay = image.copy()
    for idx, item in enumerate(components, start=1):
        x, y = item["point_xy"]
        cv2.circle(overlay, (x, y), 6, (0, 0, 0), thickness=3)
        cv2.circle(overlay, (x, y), 5, (255, 255, 0), thickness=-1)
        cv2.putText(
            overlay,
            str(idx),
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            str(idx),
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            lineType=cv2.LINE_AA,
        )
    save_rgb(out_path, overlay)


def run_sam2_prompt(
        predictor,
        image: np.ndarray,
        mask_input: np.ndarray,
        out_dir: Path,
        label: str,
) -> dict:
    masks, scores, low_res_masks = predictor.predict(
        mask_input=mask_input,
        multimask_output=True,
    )
    scores = np.asarray(scores, dtype=np.float32)
    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx].astype(bool)

    save_gray(visual_path(out_dir, f"{label}_best_mask"),
              best_mask.astype(np.uint8) * 255)
    save_rgb(visual_path(out_dir, f"{label}_best_overlay"),
             overlay_mask(image, best_mask, color=(255, 0, 0), alpha=0.55))
    np.save(data_path(out_dir, f"{label}_low_res_masks.npy"), low_res_masks)

    summary = {
        "label": label,
        "scores": [float(score) for score in scores],
        "best_idx": best_idx,
        "best_score": float(scores[best_idx]),
        "candidate_areas": [int(mask.astype(bool).sum()) for mask in masks],
        "best_area": int(best_mask.sum()),
    }
    return summary


def run_sam2_point_prompt(
        predictor,
        image: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        out_dir: Path,
        label: str,
        mask_input: np.ndarray | None = None,
) -> dict:
    if len(point_coords) == 0:
        summary = {
            "label": label,
            "point_count": 0,
            "scores": [],
            "best_idx": None,
            "best_score": None,
            "candidate_areas": [],
            "best_area": 0,
        }
        return summary

    masks, scores, low_res_masks = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=mask_input,
        multimask_output=True,
    )
    scores = np.asarray(scores, dtype=np.float32)
    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx].astype(bool)

    save_gray(visual_path(out_dir, f"{label}_best_mask"),
              best_mask.astype(np.uint8) * 255)
    save_rgb(visual_path(out_dir, f"{label}_best_overlay"),
             overlay_mask(image, best_mask, color=(255, 0, 0), alpha=0.55))
    np.save(data_path(out_dir, f"{label}_low_res_masks.npy"), low_res_masks)

    summary = {
        "label": label,
        "point_count": int(len(point_coords)),
        "points_xy": point_coords.astype(int).tolist(),
        "scores": [float(score) for score in scores],
        "best_idx": best_idx,
        "best_score": float(scores[best_idx]),
        "candidate_areas": [int(mask.astype(bool).sum()) for mask in masks],
        "best_area": int(best_mask.sum()),
        "used_mask_input": mask_input is not None,
    }
    return summary


def write_step_index(out_dir: Path) -> None:
    lines = [
        "# Single Image Pipeline Debug",
        "",
        "顶层 PNG 按编号就是当前 pipeline 的执行顺序，每个编号只对应一张图：",
        "",
    ]
    lines.extend(
        f"- `{filename}`: {description}。"
        for _, filename, description in VISUAL_STEPS
    )
    lines.extend([
        "",
        "颜色说明：深蓝 `-5` 是强背景/负提示，灰色 `0` 是不确定边界带，",
        "浅黄到深红 `1..5` 是从弱到强的正提示。",
        "",
        "候选分数、连通域决策和参数统一写入 `run_summary.json`；",
        "复查用的 NPY 数组写入 `data/`。不再生成候选结果重复图或多图拼图。",
        "",
    ])
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def clean_generated_files(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    for path in out_dir.iterdir():
        if path.name.endswith("_person.png"):
            continue
        if path.name == "data" and path.is_dir():
            shutil.rmtree(path)
            continue
        if path.is_file() and (
                path.name.startswith("step")
                or (len(path.name) > 3
                    and path.name[:2].isdigit()
                    and path.name[2] == "_")
                or path.name in {"README.md", "run_summary.json"}):
            path.unlink()


def copy_new_pipeline_inputs(tile_dir: Path, out_dir: Path) -> None:
    copies = {
        "step1_original.png": VISUAL_FILES["input_original"],
        "step2_01_deepliif_Seg.png": VISUAL_FILES["deepliif_seg"],
        "step2_02_deepliif_Marker.png": VISUAL_FILES["deepliif_marker"],
    }
    for src_name, dst_name in copies.items():
        copy_if_exists(tile_dir / src_name, out_dir / dst_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-dir", type=Path, default=DEFAULT_TILE_DIR)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("debug_output/single_image_pipeline_test"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sam-config",
                        default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam-checkpoint", type=Path,
                        default=Path("data/models/sam2/sam2.1_hiera_large.pt"))
    parser.add_argument("--seg-thresh", type=int, default=120)
    parser.add_argument("--foreground-green-max", type=int, default=80)
    parser.add_argument("--marker-thresh", type=int, default=100)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--repair-kernel", type=int, default=5)
    parser.add_argument("--repair-iterations", type=int, default=1)
    parser.add_argument("--repair-logit", type=int, default=1)
    parser.add_argument("--lumen-logit", type=int, default=1)
    parser.add_argument("--uncertain-iterations", type=int, default=1)
    parser.add_argument("--enable-artifact-filter", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--artifact-min-logit", type=int, default=1)
    parser.add_argument("--artifact-max-logit", type=int, default=3)
    parser.add_argument("--artifact-strong-logit", type=int, default=4)
    parser.add_argument("--artifact-group-kernel", type=int, default=11)
    parser.add_argument("--artifact-close-iterations", type=int, default=1)
    parser.add_argument("--artifact-dilate-iterations", type=int, default=1)
    parser.add_argument("--artifact-min-area", type=int, default=700)
    parser.add_argument("--artifact-weak-mid-ratio", type=float, default=0.75)
    parser.add_argument("--artifact-max-strong-ratio", type=float, default=0.25)
    parser.add_argument("--artifact-max-mean-logit", type=float, default=2.8)
    parser.add_argument("--artifact-score-threshold", type=int, default=5)
    parser.add_argument("--artifact-suppress-kernel", type=int, default=7)
    parser.add_argument("--artifact-suppress-dilate", type=int, default=1)
    parser.add_argument("--enable-small-fragment-filter",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--small-fragment-max-area", type=int, default=100)
    parser.add_argument("--small-fragment-max-logit", type=int, default=3)
    parser.add_argument("--point-min-area", type=int, default=20)
    parser.add_argument("--max-positive-points", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tile_dir = args.tile_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_generated_files(out_dir)
    copy_new_pipeline_inputs(tile_dir, out_dir)

    image = read_rgb(tile_dir / "step1_original.png")
    seg = read_rgb(tile_dir / "step2_01_deepliif_Seg.png")
    marker = read_gray(tile_dir / "step2_02_deepliif_Marker.png")
    marker_max = max(int(marker.max()), args.marker_thresh + 1)

    prompt_logits, weighted_input, prompt_stats, prompt_debug = build_weighted_prompt(
        seg, marker, args, marker_max)
    prompt_stats.update({
        "tile_dir": str(tile_dir),
        "scheme": "single_image_seg_marker_weighted_prompt",
        "sam_config": args.sam_config,
        "sam_checkpoint": str(args.sam_checkpoint),
        "device": args.device,
    })
    save_weighted_prompt_outputs(
        out_dir, image, prompt_logits, weighted_input, prompt_stats,
        prompt_debug)

    predictor = load_sam2(
        args.sam_config, str(args.sam_checkpoint), args.device)
    predictor.set_image(image)

    summary = {
        "tile_dir": str(tile_dir),
        "output_dir": str(out_dir),
        "marker_max_used": int(marker_max),
        "prompt": prompt_stats,
        "weighted": run_sam2_prompt(
            predictor, image, weighted_input, out_dir, "weighted"),
    }

    point_coords, point_labels, point_components = choose_strong_positive_points(
        prompt_logits, args.point_min_area, args.max_positive_points)
    save_points_overlay(
        visual_path(out_dir, "strong_positive_points"),
        image,
        point_components,
    )
    summary["strong_positive_points"] = {
        "point_min_area": int(args.point_min_area),
        "max_positive_points": int(args.max_positive_points),
        "point_count": int(len(point_components)),
        "components": point_components,
    }

    summary["points_only"] = run_sam2_point_prompt(
        predictor,
        image,
        point_coords,
        point_labels,
        out_dir,
        "points_only",
        mask_input=None,
    )
    summary["weighted_plus_points"] = run_sam2_point_prompt(
        predictor,
        image,
        point_coords,
        point_labels,
        out_dir,
        "weighted_plus_points",
        mask_input=weighted_input,
    )

    write_step_index(out_dir)

    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(out_dir)


if __name__ == "__main__":
    main()
