"""Build the Seg/Marker weighted mask and positive-point SAM2 prompt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from cd34_pipeline.cell.extraction import (
    MARKER_MIN_KEEP_INTENSITY,
    compute_marker_two_stage_multi_otsu_details,
    enforce_marker_min_keep_threshold,
)


@dataclass(frozen=True)
class WeightedPromptConfig:
    """Parameters for the weighted-mask + strong-positive-point prompt."""

    seg_thresh: int = 120
    foreground_green_max: int = 80
    marker_thresh: Optional[int] = None
    marker_max: Optional[int] = None
    enable_dab_filter: bool = False
    enable_dab_strong_support: bool = True
    dab_strong_support_neighborhood_kernel: int = 21
    dab_min_intensity: int = 160
    dab_normalization_percentile: float = 99.5
    enable_dab_hsv_brown_filter: bool = True
    dab_hsv_brown_hue_min: int = 0
    dab_hsv_brown_hue_max: int = 35
    dab_hsv_brown_saturation_min: int = 30
    dab_hsv_brown_value_min: int = 20
    dab_hsv_brown_white_value_min: int = 245
    dab_hsv_brown_white_saturation_max: int = 25
    dab_hsv_brown_exclude_seg_blue: bool = True
    dab_hsv_brown_seg_blue_dilate_kernel: int = 3
    dapi_lumen_dark_max: int = 15
    dapi_lumen_support_logit_min: int = 1
    dapi_lumen_wall_closing_kernel: int = 5
    target_size: int = 256
    repair_kernel: int = 5
    repair_iterations: int = 1
    repair_logit: int = 1
    lumen_logit: int = 1
    uncertain_iterations: int = 1
    enable_artifact_filter: bool = True
    artifact_min_logit: int = 1
    artifact_max_logit: int = 3
    artifact_strong_logit: int = 4
    artifact_group_kernel: int = 11
    artifact_close_iterations: int = 1
    artifact_dilate_iterations: int = 1
    artifact_min_area: int = 700
    artifact_weak_mid_ratio: float = 0.75
    artifact_max_strong_ratio: float = 0.25
    artifact_max_mean_logit: float = 2.8
    artifact_score_threshold: int = 5
    artifact_suppress_kernel: int = 7
    artifact_suppress_dilate: int = 1
    enable_small_fragment_filter: bool = True
    small_fragment_max_area: int = 100
    small_fragment_max_logit: int = 3
    enable_isolated_fragment_filter: bool = True
    isolated_fragment_max_area: int = 200
    isolated_fragment_min_gap: int = 8
    isolated_fragment_neighbor_min_area: int = 700
    point_min_area: int = 20
    max_positive_points: int = 30
    enable_lumen_points: bool = True
    lumen_point_support_logit_min: int = 1
    lumen_point_closing_kernel: int = 7
    lumen_point_min_area: int = 8
    lumen_point_max_area: int = 1200
    lumen_point_ring_kernel: int = 5
    lumen_point_min_wall_ratio: float = 0.40
    lumen_point_fill_logit: int = 2
    max_lumen_points: int = 3
    enable_dab_lumen_fill: bool = True
    dab_lumen_wall_min_intensity: int = 160
    dab_lumen_interior_max_intensity: int = 90
    dab_lumen_near_wall_kernel: int = 21
    dab_lumen_ring_kernel: int = 9
    dab_lumen_min_area: int = 80
    dab_lumen_max_area: int = 8000
    dab_lumen_min_wall_ratio: float = 0.18
    dab_lumen_min_boundary_ratio: float = 0.45
    dab_lumen_min_border_boundary_ratio: float = 0.22
    dab_lumen_macro_closing_kernel: int = 31
    dab_lumen_macro_min_overlap: float = 0.50
    dab_lumen_macro_min_wall_ratio: float = 0.30
    dab_lumen_use_white_interior: bool = True
    dab_lumen_white_value_min: int = 210
    dab_lumen_white_saturation_max: float = 0.18
    dab_lumen_white_channel_delta_max: int = 35
    dab_lumen_max_aspect_ratio: float = 8.0
    max_dab_lumen_points: int = 3


@dataclass
class WeightedPromptResult:
    """Arrays and metadata needed by the SAM2 consumer and debug renderer."""

    logits: np.ndarray
    mask_input: np.ndarray
    point_coords: np.ndarray
    point_labels: np.ndarray
    point_components: list[dict]
    stats: dict
    debug: dict


def _marker_gray(marker: np.ndarray) -> np.ndarray:
    if marker.ndim == 2:
        return marker.astype(np.uint8, copy=False)
    if marker.ndim == 3 and marker.shape[2] >= 3:
        return cv2.cvtColor(marker[:, :, :3], cv2.COLOR_RGB2GRAY)
    raise ValueError("DeepLIIF Marker must be a grayscale or RGB image")


def _dapi_gray(dapi: np.ndarray) -> np.ndarray:
    if dapi.ndim == 2:
        gray = dapi
    elif dapi.ndim == 3 and dapi.shape[2] >= 3:
        gray = cv2.cvtColor(dapi[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("DeepLIIF DAPI must be a grayscale or RGB image")
    return np.clip(gray, 0, 255).astype(np.uint8)


def _rgb_float(tile_rgb: np.ndarray) -> np.ndarray:
    if tile_rgb.ndim != 3 or tile_rgb.shape[2] < 3:
        raise ValueError("Original tile must be an RGB image")
    rgb = tile_rgb[:, :, :3]
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255.0
    rgb = rgb.astype(np.float32)
    if rgb.max(initial=0.0) > 1.0:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _dab_channel_u8(tile_rgb: np.ndarray,
                    percentile: float) -> tuple[np.ndarray, float]:
    """Return a scikit-image-compatible HED DAB channel normalized to 0-255."""
    rgb = _rgb_float(tile_rgb)
    rgb = np.maximum(rgb, 1e-6)
    hed_from_rgb = np.array([
        [1.87798274, -1.00767869, -0.55611582],
        [-0.06590806, 1.13473037, -0.13552180],
        [-0.60190736, -0.48041419, 1.57358807],
    ], dtype=np.float32)
    stains = (np.log(rgb) / np.log(1e-6)) @ hed_from_rgb
    dab = np.maximum(stains[:, :, 2], 0.0)
    positive = dab[dab > 0]
    if positive.size:
        norm_value = float(np.percentile(
            positive, np.clip(percentile, 0.0, 100.0)))
    else:
        norm_value = 0.0
    if norm_value <= 1e-8:
        norm_value = float(dab.max(initial=0.0))
    if norm_value <= 1e-8:
        return np.zeros(dab.shape, dtype=np.uint8), 0.0
    dab_u8 = np.round(np.clip(dab / norm_value, 0.0, 1.0) * 255)
    return dab_u8.astype(np.uint8), norm_value


def _hue_range_mask(
        hue: np.ndarray,
        hue_min: int,
        hue_max: int,
) -> np.ndarray:
    """Return an OpenCV-HSV hue mask, supporting ranges that wrap at 179."""
    hue_min = int(np.clip(hue_min, 0, 179))
    hue_max = int(np.clip(hue_max, 0, 179))
    if hue_min <= hue_max:
        return (hue >= hue_min) & (hue <= hue_max)
    return (hue >= hue_min) | (hue <= hue_max)


def _hsv_brown_keep_mask(
        tile_rgb: np.ndarray,
        config: WeightedPromptConfig,
) -> np.ndarray:
    """Return original-RGB HSV brown candidates used to confirm DAB pixels."""
    rgb = np.round(_rgb_float(tile_rgb) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    saturation = hsv[:, :, 1].astype(np.int16)
    value = hsv[:, :, 2].astype(np.int16)

    hue_ok = _hue_range_mask(
        hue,
        config.dab_hsv_brown_hue_min,
        config.dab_hsv_brown_hue_max,
    )
    saturation_ok = (
        saturation >= int(config.dab_hsv_brown_saturation_min))
    value_ok = value >= int(config.dab_hsv_brown_value_min)
    near_white = (
        (value >= int(config.dab_hsv_brown_white_value_min))
        & (saturation <= int(config.dab_hsv_brown_white_saturation_max))
    )
    return hue_ok & saturation_ok & value_ok & ~near_white


def _seg_blue_exclusion_mask(
        seg: Optional[np.ndarray],
        shape: tuple[int, int],
        config: WeightedPromptConfig,
) -> np.ndarray:
    """Return DeepLIIF-Seg blue negative support to exclude from DAB keep."""
    excluded = np.zeros(shape, dtype=bool)
    if seg is None or not config.dab_hsv_brown_exclude_seg_blue:
        return excluded
    if seg.ndim != 3 or seg.shape[2] < 3:
        raise ValueError("DeepLIIF Seg must be an RGB image")
    if seg.shape[:2] != shape:
        raise ValueError("DeepLIIF Seg shape must match DAB filter shape")

    r = seg[:, :, 0].astype(np.int16)
    g = seg[:, :, 1].astype(np.int16)
    b = seg[:, :, 2].astype(np.int16)
    foreground = (
        (r + b > config.seg_thresh)
        & (g <= config.foreground_green_max)
    )
    excluded = foreground & (b > r)
    kernel_size = _odd_kernel(config.dab_hsv_brown_seg_blue_dilate_kernel)
    if kernel_size > 1 and excluded.any():
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        excluded = cv2.dilate(
            excluded.astype(np.uint8), kernel, iterations=1).astype(bool)
    return excluded


def _dab_filter_masks(
        tile_rgb: np.ndarray,
        seg: Optional[np.ndarray],
        config: WeightedPromptConfig,
) -> tuple[dict, dict]:
    """Build the HED-DAB, HSV-brown, Seg-blue, and final keep masks."""
    dab_u8, norm_value = _dab_channel_u8(
        tile_rgb, config.dab_normalization_percentile)
    hed_keep = dab_u8 >= config.dab_min_intensity
    if config.enable_dab_hsv_brown_filter:
        hsv_brown_keep = _hsv_brown_keep_mask(tile_rgb, config)
    else:
        hsv_brown_keep = np.ones(dab_u8.shape, dtype=bool)
    seg_blue_excluded = _seg_blue_exclusion_mask(
        seg, dab_u8.shape, config)
    final_keep = hed_keep & hsv_brown_keep & ~seg_blue_excluded

    debug = {
        "dab_intensity": dab_u8,
        "dab_hed_intensity_keep_mask": hed_keep,
        "dab_hsv_brown_keep_mask": hsv_brown_keep,
        "dab_seg_blue_excluded_mask": seg_blue_excluded,
        "dab_intensity_keep_mask": final_keep,
        "dab_keep_mask": final_keep,
    }
    stats = {
        "dab_normalization_value": float(norm_value),
        "dab_hed_intensity_keep_px": int(hed_keep.sum()),
        "dab_hsv_brown_filter_enabled": bool(
            config.enable_dab_hsv_brown_filter),
        "dab_hsv_brown_keep_px": int(hsv_brown_keep.sum()),
        "dab_hsv_brown_hue_min": int(config.dab_hsv_brown_hue_min),
        "dab_hsv_brown_hue_max": int(config.dab_hsv_brown_hue_max),
        "dab_hsv_brown_saturation_min": int(
            config.dab_hsv_brown_saturation_min),
        "dab_hsv_brown_value_min": int(config.dab_hsv_brown_value_min),
        "dab_hsv_brown_white_value_min": int(
            config.dab_hsv_brown_white_value_min),
        "dab_hsv_brown_white_saturation_max": int(
            config.dab_hsv_brown_white_saturation_max),
        "dab_hsv_brown_exclude_seg_blue": bool(
            config.dab_hsv_brown_exclude_seg_blue),
        "dab_hsv_brown_seg_blue_dilate_kernel": int(
            _odd_kernel(config.dab_hsv_brown_seg_blue_dilate_kernel)),
        "dab_seg_blue_excluded_px": int(seg_blue_excluded.sum()),
        "dab_final_keep_px": int(final_keep.sum()),
    }
    return debug, stats


def _odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def _multi_otsu_thresholds_from_counts(
        counts: np.ndarray,
        min_intensity: int,
        max_intensity: int) -> Optional[tuple[int, int]]:
    """Return 3-class Multi-Otsu thresholds from a 256-bin histogram."""
    min_intensity = int(np.clip(min_intensity, 0, 255))
    max_intensity = int(np.clip(max_intensity, 0, 255))
    if min_intensity >= max_intensity:
        return None

    counts = np.asarray(counts, dtype=np.float64)
    support = np.flatnonzero(counts[min_intensity:max_intensity + 1])
    if support.size < 3:
        return None
    support = support + min_intensity

    cumulative_counts = np.cumsum(counts)
    cumulative_sums = np.cumsum(counts * np.arange(counts.size))

    def _range_count_sum(start: int, stop: int) -> tuple[float, float]:
        count = cumulative_counts[stop]
        total = cumulative_sums[stop]
        if start > 0:
            count -= cumulative_counts[start - 1]
            total -= cumulative_sums[start - 1]
        return count, total

    best_score = -np.inf
    best_thresholds = (int(support[0]), int(support[-2]))
    for low_threshold in range(min_intensity, max_intensity - 1):
        low_count, low_sum = _range_count_sum(
            min_intensity, low_threshold)
        if low_count <= 0:
            continue

        for high_threshold in range(low_threshold + 1, max_intensity):
            mid_count, mid_sum = _range_count_sum(
                low_threshold + 1, high_threshold)
            high_count, high_sum = _range_count_sum(
                high_threshold + 1, max_intensity)
            if mid_count <= 0 or high_count <= 0:
                continue

            score = (
                (low_sum * low_sum / low_count)
                + (mid_sum * mid_sum / mid_count)
                + (high_sum * high_sum / high_count)
            )
            if score > best_score:
                best_score = score
                best_thresholds = (low_threshold, high_threshold)

    if not np.isfinite(best_score):
        return None
    return int(best_thresholds[0]), int(best_thresholds[1])


def _near_white_value_threshold_details(
        value: np.ndarray,
        balanced_mask: np.ndarray,
        fallback_value_min: int) -> dict:
    """Choose near-white HSV V threshold from the post-peak tail."""
    fallback_value_min = int(np.clip(fallback_value_min, 0, 255))
    values = np.clip(value[balanced_mask], 0, 255).astype(np.uint8)
    if values.size == 0:
        return {
            "threshold": fallback_value_min,
            "source": "fallback_no_balanced_pixels",
            "peak_intensity": None,
            "tail_range": [None, None],
            "multiotsu_thresholds": [None, None],
        }

    counts = np.bincount(values, minlength=256)[:256].astype(np.int64)
    support = np.flatnonzero(counts)
    peak_intensity = int(support[np.argmax(counts[support])])
    tail_min = min(255, peak_intensity + 1)
    tail_support = support[support >= tail_min]
    if tail_support.size < 3:
        return {
            "threshold": fallback_value_min,
            "source": "fallback_insufficient_peak_tail",
            "peak_intensity": peak_intensity,
            "tail_range": [tail_min, 255],
            "multiotsu_thresholds": [None, None],
        }

    thresholds = _multi_otsu_thresholds_from_counts(counts, tail_min, 255)
    if thresholds is None:
        return {
            "threshold": fallback_value_min,
            "source": "fallback_multiotsu_failed",
            "peak_intensity": peak_intensity,
            "tail_range": [tail_min, 255],
            "multiotsu_thresholds": [None, None],
        }

    low_threshold, high_threshold = thresholds
    return {
        "threshold": int(high_threshold),
        "source": "auto_peak_tail_multiotsu",
        "peak_intensity": peak_intensity,
        "tail_range": [tail_min, 255],
        "multiotsu_thresholds": [int(low_threshold), int(high_threshold)],
    }


def _seg_logits(seg: np.ndarray, config: WeightedPromptConfig) -> tuple[np.ndarray, dict]:
    if seg.ndim != 3 or seg.shape[2] < 3:
        raise ValueError("DeepLIIF Seg must be an RGB image")

    r = seg[:, :, 0].astype(np.int16)
    g = seg[:, :, 1].astype(np.int16)
    b = seg[:, :, 2].astype(np.int16)
    foreground = (
        (r + b > config.seg_thresh)
        & (g <= config.foreground_green_max)
    )
    positive = foreground & (r >= b)
    negative_blue = foreground & (b > r)

    logits = np.full(seg.shape[:2], -5, dtype=np.int16)
    logits[positive & (r < 150)] = 0
    logits[positive & (r >= 150) & (r < 170)] = 1
    logits[positive & (r >= 170) & (r < 190)] = 2
    logits[positive & (r >= 190) & (r < 210)] = 3
    logits[positive & (r >= 210) & (r < 235)] = 4
    logits[positive & (r >= 235)] = 5

    ambiguous = positive & ((r - b) < 20) & (logits > 1)
    logits[ambiguous] -= 1
    logits[negative_blue] = -5

    stats = {
        "seg_foreground_px": int(foreground.sum()),
        "seg_blue_negative_px": int(negative_blue.sum()),
        "seg_positive_any_px": int((positive & (logits > 0)).sum()),
    }
    for value in range(1, 6):
        stats[f"seg_logit_{value}_px"] = int((logits == value).sum())
    return logits, stats


def _marker_logits(marker: np.ndarray, config: WeightedPromptConfig) -> tuple[np.ndarray, dict]:
    marker = _marker_gray(marker)
    marker_thresh = config.marker_thresh
    marker_threshold_source = "fixed"
    marker_threshold_details = None
    if marker_thresh is None:
        marker_threshold_details = compute_marker_two_stage_multi_otsu_details(
            marker)
        marker_thresh = marker_threshold_details["keep_threshold"]
        marker_threshold_source = "auto_two_stage_multiotsu"
    marker_thresh = enforce_marker_min_keep_threshold(marker_thresh)

    marker_max = config.marker_max
    if marker_max is None:
        marker_max = int(marker.max())
    marker_max = max(int(marker_max), marker_thresh + 1)

    logits = np.full(marker.shape, -5, dtype=np.int16)
    positive = marker > marker_thresh
    if positive.any():
        denominator = max(1, marker_max - marker_thresh)
        normalized = (
            marker.astype(np.float32) - marker_thresh
        ) / denominator
        bins = np.ceil(np.clip(normalized, 0, 1) * 5).astype(np.int16)
        logits[positive] = np.clip(bins[positive], 1, 5)

    stats = {
        "marker_thresh": int(marker_thresh),
        "marker_threshold_source": marker_threshold_source,
        "marker_min_keep_intensity": int(MARKER_MIN_KEEP_INTENSITY),
        "marker_effective_keep_min_intensity": int(marker_thresh + 1),
        "marker_max_used": int(marker_max),
        "tile_marker_max": int(marker.max()),
        "marker_positive_px": int(positive.sum()),
    }
    if marker_threshold_details is not None:
        stats.update({
            "marker_two_stage_outer_thresholds": (
                marker_threshold_details["outer_thresholds"]),
            "marker_two_stage_middle_thresholds": (
                marker_threshold_details["middle_thresholds"]),
        })
    for value in range(1, 6):
        stats[f"marker_logit_{value}_px"] = int((logits == value).sum())
    return logits, stats


def _logit_counts(logits: np.ndarray, mask: np.ndarray) -> dict:
    values = logits[mask].astype(np.int16)
    return {
        str(value): int((values == value).sum())
        for value in (-5, 0, 1, 2, 3, 4, 5)
    }


def _optional_bool_mask(mask: Optional[np.ndarray],
                        shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=bool)
    if mask.shape != shape:
        raise ValueError("Mask shape must match logits shape")
    return mask.astype(bool, copy=False)


def _dapi_dark_mask(dapi: np.ndarray,
                    config: WeightedPromptConfig) -> tuple[np.ndarray, np.ndarray]:
    gray = _dapi_gray(dapi)
    dark = gray <= int(config.dapi_lumen_dark_max)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark = cv2.morphologyEx(
        dark.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1
    ).astype(bool)
    return dark, gray


def _empty_dapi_lumen_rescue(shape: tuple[int, int],
                             *, enabled: bool) -> tuple[dict, dict]:
    empty = np.zeros(shape, dtype=bool)
    stats = {
        "dapi_lumen_rescue_enabled": bool(enabled),
        "dapi_lumen_dark_max": None,
        "dapi_lumen_support_logit_min": None,
        "dapi_lumen_candidate_count": 0,
        "dapi_lumen_accepted_count": 0,
        "dapi_lumen_protected_prompt_px": 0,
        "dapi_lumen_filled_px": 0,
        "dapi_lumen_point_count": 0,
        "dapi_lumen_components": [],
    }
    debug = {
        "dapi_intensity": np.zeros(shape, dtype=np.uint8),
        "dapi_dark_mask": empty.copy(),
        "dapi_lumen_wall_mask": empty.copy(),
        "dapi_lumen_near_wall_mask": empty.copy(),
        "dapi_lumen_candidate_mask": empty.copy(),
        "dapi_lumen_accepted_mask": empty.copy(),
        "dapi_lumen_protected_prompt_mask": empty.copy(),
    }
    return stats, debug


def _dapi_lumen_rescue_from_prompt(
        raw: np.ndarray,
        dapi: Optional[np.ndarray],
        config: WeightedPromptConfig,
) -> tuple[dict, dict]:
    enabled = bool(config.enable_dab_lumen_fill and dapi is not None)
    stats, debug = _empty_dapi_lumen_rescue(raw.shape, enabled=enabled)
    if not enabled:
        return stats, debug
    if dapi.shape[:2] != raw.shape:
        raise ValueError("DeepLIIF DAPI shape must match Seg/Marker prompt shape")

    dark_mask, dapi_gray = _dapi_dark_mask(dapi, config)
    debug["dapi_intensity"] = dapi_gray
    debug["dapi_dark_mask"] = dark_mask
    stats["dapi_lumen_dark_max"] = int(config.dapi_lumen_dark_max)
    stats["dapi_lumen_support_logit_min"] = int(
        config.dapi_lumen_support_logit_min)

    support = raw >= int(config.dapi_lumen_support_logit_min)
    if not support.any() or not dark_mask.any():
        return stats, debug

    wall_size = _odd_kernel(config.dapi_lumen_wall_closing_kernel)
    wall_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (wall_size, wall_size))
    wall_closed = cv2.morphologyEx(
        support.astype(np.uint8), cv2.MORPH_CLOSE, wall_kernel,
        iterations=1).astype(bool)

    near_size = _odd_kernel(config.dab_lumen_near_wall_kernel)
    near_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (near_size, near_size))
    near_wall = cv2.dilate(
        wall_closed.astype(np.uint8), near_kernel, iterations=1).astype(bool)
    candidate_base = dark_mask & ~wall_closed
    candidate_seed = candidate_base & near_wall
    _, dark_component_labels = cv2.connectedComponents(
        candidate_base.astype(np.uint8), connectivity=8)
    seed_ids = np.unique(dark_component_labels[candidate_seed])
    seed_ids = seed_ids[seed_ids > 0]
    candidate_mask = np.isin(dark_component_labels, seed_ids)

    num_labels, labels, component_stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask.astype(np.uint8), connectivity=8)
    ring_size = _odd_kernel(config.dab_lumen_ring_kernel)
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ring_size, ring_size))
    macro_size = _odd_kernel(config.dab_lumen_macro_closing_kernel)
    macro_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (macro_size, macro_size))
    macro_closed = cv2.morphologyEx(
        wall_closed.astype(np.uint8), cv2.MORPH_CLOSE, macro_kernel,
        iterations=1).astype(bool)
    macro_holes = _fill_holes(macro_closed) & ~macro_closed
    border = _border_mask(raw.shape, width=max(1, ring_size // 2))

    prompt_support = raw >= 0
    _, prompt_labels = cv2.connectedComponents(
        prompt_support.astype(np.uint8), connectivity=8)

    accepted_mask = np.zeros(raw.shape, dtype=bool)
    protected_prompt = np.zeros(raw.shape, dtype=bool)
    components = []
    height, width = raw.shape

    for label_id in range(1, num_labels):
        area = int(component_stats[label_id, cv2.CC_STAT_AREA])
        x = int(component_stats[label_id, cv2.CC_STAT_LEFT])
        y = int(component_stats[label_id, cv2.CC_STAT_TOP])
        box_w = int(component_stats[label_id, cv2.CC_STAT_WIDTH])
        box_h = int(component_stats[label_id, cv2.CC_STAT_HEIGHT])
        component = labels == label_id
        touches_border = (
            x == 0 or y == 0
            or x + box_w >= width or y + box_h >= height
        )

        dilated = cv2.dilate(
            component.astype(np.uint8), ring_kernel, iterations=1).astype(bool)
        ring = dilated & ~component
        ring_area = int(ring.sum())
        wall_pixels = ring & wall_closed
        border_pixels = ring & border if touches_border else np.zeros_like(ring)
        wall_ratio = (
            float(wall_pixels.sum()) / float(ring_area)
            if ring_area else 0.0
        )
        boundary_ratio = (
            float((wall_pixels | border_pixels).sum()) / float(ring_area)
            if ring_area else 0.0
        )

        macro_overlap_ratio = (
            float((component & macro_holes).sum()) / float(area)
            if area else 0.0
        )
        macro_ring = cv2.dilate(
            component.astype(np.uint8), macro_kernel, iterations=1).astype(bool)
        macro_ring &= ~component
        macro_ring_area = int(macro_ring.sum())
        macro_wall_ratio = (
            float((macro_ring & wall_closed).sum()) / float(macro_ring_area)
            if macro_ring_area else 0.0
        )
        macro_supported = (
            macro_overlap_ratio >= config.dab_lumen_macro_min_overlap
            and macro_wall_ratio >= config.dab_lumen_macro_min_wall_ratio
        )
        aspect_ratio = (
            float(max(box_w, box_h)) / float(max(1, min(box_w, box_h)))
        )

        reject_reasons = []
        if area < config.dab_lumen_min_area:
            reject_reasons.append("area_too_small")
        if area > config.dab_lumen_max_area:
            reject_reasons.append("area_too_large")
        if aspect_ratio > config.dab_lumen_max_aspect_ratio:
            reject_reasons.append("too_elongated")
        min_boundary_ratio = (
            config.dab_lumen_min_border_boundary_ratio
            if touches_border else config.dab_lumen_min_boundary_ratio
        )
        if wall_ratio < config.dab_lumen_min_wall_ratio and not macro_supported:
            reject_reasons.append("weak_seg_marker_wall")
        if boundary_ratio < min_boundary_ratio and not macro_supported:
            reject_reasons.append("weak_boundary_support")

        protect_context = cv2.dilate(
            component.astype(np.uint8), near_kernel, iterations=1).astype(bool)
        protect_context &= ~component
        protected_ids = np.unique(prompt_labels[protect_context & prompt_support])
        protected_ids = protected_ids[protected_ids > 0]

        dist = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 3)
        point_y, point_x = np.unravel_index(int(np.argmax(dist)), dist.shape)
        component_info = {
            "label_id": int(label_id),
            "accepted": not reject_reasons,
            "selected": not reject_reasons,
            "reject_reasons": reject_reasons,
            "area": area,
            "bbox_xywh": [x, y, box_w, box_h],
            "ring_area": ring_area,
            "wall_ratio": wall_ratio,
            "boundary_ratio": boundary_ratio,
            "macro_overlap_ratio": macro_overlap_ratio,
            "macro_wall_ratio": macro_wall_ratio,
            "macro_supported": bool(macro_supported),
            "touches_border": bool(touches_border),
            "aspect_ratio": aspect_ratio,
            "protected_prompt_component_ids": [
                int(value) for value in protected_ids.tolist()
            ],
            "point_xy": [int(point_x), int(point_y)],
            "kind": "dapi_lumen",
        }
        components.append(component_info)
        if reject_reasons:
            continue

        accepted_mask |= component
        for prompt_id in protected_ids:
            protected_prompt |= prompt_labels == int(prompt_id)

    selected_components = [
        component for component in components if component["selected"]
    ]
    selected_components.sort(
        key=lambda item: (
            item["touches_border"],
            item["boundary_ratio"],
            item["area"],
        ),
        reverse=True,
    )
    point_components = selected_components
    if config.max_dab_lumen_points > 0:
        point_components = point_components[:config.max_dab_lumen_points]
    for component in selected_components:
        component["selected_for_point"] = component in point_components

    stats.update({
        "dapi_lumen_wall_closing_kernel": int(
            config.dapi_lumen_wall_closing_kernel),
        "dapi_lumen_near_wall_kernel": int(config.dab_lumen_near_wall_kernel),
        "dapi_lumen_ring_kernel": int(config.dab_lumen_ring_kernel),
        "dapi_lumen_min_area": int(config.dab_lumen_min_area),
        "dapi_lumen_max_area": int(config.dab_lumen_max_area),
        "dapi_lumen_min_wall_ratio": float(config.dab_lumen_min_wall_ratio),
        "dapi_lumen_min_boundary_ratio": float(
            config.dab_lumen_min_boundary_ratio),
        "dapi_lumen_min_border_boundary_ratio": float(
            config.dab_lumen_min_border_boundary_ratio),
        "dapi_lumen_max_aspect_ratio": float(
            config.dab_lumen_max_aspect_ratio),
        "dapi_lumen_candidate_count": int(num_labels - 1),
        "dapi_lumen_accepted_count": int(len(selected_components)),
        "dapi_lumen_protected_prompt_px": int(protected_prompt.sum()),
        "dapi_lumen_filled_px": int(accepted_mask.sum()),
        "dapi_lumen_point_count": int(len(point_components)),
        "dapi_lumen_components": components,
    })
    debug.update({
        "dapi_lumen_wall_mask": wall_closed,
        "dapi_lumen_near_wall_mask": near_wall,
        "dapi_lumen_candidate_mask": candidate_mask,
        "dapi_lumen_accepted_mask": accepted_mask,
        "dapi_lumen_protected_prompt_mask": protected_prompt,
    })
    return stats, debug


def _suppress_weak_dab_prompt(
        raw: np.ndarray,
        tile_rgb: Optional[np.ndarray],
        dapi: Optional[np.ndarray],
        config: WeightedPromptConfig,
        seg: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    filtered = raw.copy()
    empty = np.zeros(raw.shape, dtype=bool)
    if not config.enable_dab_filter:
        dapi_stats, dapi_debug = _empty_dapi_lumen_rescue(
            raw.shape, enabled=False)
        return filtered, empty, {
            "dab_intensity": np.zeros(raw.shape, dtype=np.uint8),
            "dab_hed_intensity_keep_mask": empty.copy(),
            "dab_hsv_brown_keep_mask": empty.copy(),
            "dab_seg_blue_excluded_mask": empty.copy(),
            "dab_intensity_keep_mask": empty.copy(),
            "dab_keep_mask": empty.copy(),
            "dab_removed_mask": empty.copy(),
            "dab_filtered_logits": filtered.copy(),
            **dapi_debug,
        }, {
            "dab_filter_enabled": False,
            **dapi_stats,
        }

    if tile_rgb is None:
        raise ValueError(
            "DAB prompt filtering requires the original RGB tile image")
    if tile_rgb.shape[:2] != raw.shape:
        raise ValueError(
            "Original RGB tile shape must match Seg/Marker prompt shape")

    dab_debug, dab_filter_stats = _dab_filter_masks(tile_rgb, seg, config)
    keep_mask = dab_debug["dab_keep_mask"]
    dapi_stats, dapi_debug = _dapi_lumen_rescue_from_prompt(
        raw, dapi, config)
    protected_prompt = dapi_debug["dapi_lumen_protected_prompt_mask"]
    filled_lumen = dapi_debug["dapi_lumen_accepted_mask"]

    prompt_candidate = raw >= 0
    removed_mask = prompt_candidate & ~keep_mask & ~protected_prompt
    filtered[removed_mask] = -5
    fill_value = int(config.lumen_point_fill_logit)
    if fill_value > -5 and filled_lumen.any():
        filtered[filled_lumen & (filtered < fill_value)] = fill_value

    stats = {
        "dab_filter_enabled": True,
        "dab_min_intensity": int(config.dab_min_intensity),
        "dab_normalization_percentile": float(
            config.dab_normalization_percentile),
        "dab_intensity_keep_px": int(keep_mask.sum()),
        "dab_prompt_candidate_px": int(prompt_candidate.sum()),
        "dab_prompt_kept_px": int(
            (prompt_candidate & (keep_mask | protected_prompt)).sum()),
        "dab_prompt_kept_by_dapi_lumen_px": int(
            (prompt_candidate & protected_prompt & ~keep_mask).sum()),
        "dab_prompt_removed_px": int(removed_mask.sum()),
        "dab_prompt_removed_logit_counts": _logit_counts(raw, removed_mask),
        **dab_filter_stats,
        **dapi_stats,
    }
    debug = {
        "dab_removed_mask": removed_mask,
        "dab_filtered_logits": filtered.copy(),
        **dab_debug,
        **dapi_debug,
    }
    return filtered, removed_mask, debug, stats


def _strong_dab_support_logits(
        dab_u8: np.ndarray,
        keep_mask: np.ndarray,
        config: WeightedPromptConfig,
) -> np.ndarray:
    logits = np.full(dab_u8.shape, -5, dtype=np.int16)
    if not keep_mask.any():
        return logits

    denominator = max(1, 255 - int(config.dab_min_intensity))
    normalized = (
        dab_u8.astype(np.float32) - float(config.dab_min_intensity)
    ) / float(denominator)
    bins = np.ceil(np.clip(normalized, 0.0, 1.0) * 5.0).astype(np.int16)
    logits[keep_mask] = np.clip(bins[keep_mask], 1, 5)
    return logits


def _add_strong_dab_prompt_support(
        filtered: np.ndarray,
        dab_debug: dict,
        config: WeightedPromptConfig,
) -> tuple[np.ndarray, dict, dict]:
    empty = np.zeros(filtered.shape, dtype=bool)
    empty_logits = np.full(filtered.shape, -5, dtype=np.int16)
    neighborhood_kernel = _odd_kernel(config.dab_strong_support_neighborhood_kernel)
    if not (config.enable_dab_filter and config.enable_dab_strong_support):
        return filtered.copy(), {
            "dab_support_logits": empty_logits,
            "dab_added_mask": empty.copy(),
            "dab_upgraded_mask": empty.copy(),
            "dab_strong_support_context_mask": empty.copy(),
            "dab_augmented_logits": filtered.copy(),
        }, {
            "dab_strong_support_enabled": False,
            "dab_strong_support_neighborhood_kernel": neighborhood_kernel,
            "dab_strong_support_anchor_px": 0,
            "dab_strong_support_context_px": 0,
            "dab_prompt_blocked_by_context_px": 0,
            "dab_prompt_added_px": 0,
            "dab_prompt_upgraded_px": 0,
        }

    dab_u8 = dab_debug.get("dab_intensity")
    keep_mask = dab_debug.get("dab_keep_mask")
    if dab_u8 is None or keep_mask is None:
        return filtered.copy(), {
            "dab_support_logits": empty_logits,
            "dab_added_mask": empty.copy(),
            "dab_upgraded_mask": empty.copy(),
            "dab_strong_support_context_mask": empty.copy(),
            "dab_augmented_logits": filtered.copy(),
        }, {
            "dab_strong_support_enabled": True,
            "dab_strong_support_neighborhood_kernel": neighborhood_kernel,
            "dab_strong_support_anchor_px": 0,
            "dab_strong_support_context_px": 0,
            "dab_prompt_blocked_by_context_px": 0,
            "dab_prompt_added_px": 0,
            "dab_prompt_upgraded_px": 0,
        }

    keep_mask = keep_mask.astype(bool, copy=False)
    support_logits = _strong_dab_support_logits(dab_u8, keep_mask, config)
    anchor = filtered > 0
    if anchor.any():
        context_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (neighborhood_kernel, neighborhood_kernel))
        context = cv2.dilate(
            anchor.astype(np.uint8),
            context_kernel,
            iterations=1,
        ).astype(bool)
    else:
        context = empty.copy()
    blocked_by_context = keep_mask & ~context
    support_logits[blocked_by_context] = -5
    augmented = np.maximum(filtered, support_logits).astype(np.int16)
    added_mask = keep_mask & (filtered < 0) & (support_logits >= 0)
    upgraded_mask = (
        keep_mask
        & (filtered >= 0)
        & (support_logits > filtered)
    )

    stats = {
        "dab_strong_support_enabled": True,
        "dab_strong_support_neighborhood_kernel": neighborhood_kernel,
        "dab_strong_support_anchor_px": int(anchor.sum()),
        "dab_strong_support_context_px": int(context.sum()),
        "dab_prompt_blocked_by_context_px": int(blocked_by_context.sum()),
        "dab_prompt_added_px": int(added_mask.sum()),
        "dab_prompt_upgraded_px": int(upgraded_mask.sum()),
    }
    for value in range(1, 6):
        stats[f"dab_support_logit_{value}_px"] = int(
            (support_logits == value).sum())

    return augmented, {
        "dab_support_logits": support_logits,
        "dab_added_mask": added_mask,
        "dab_upgraded_mask": upgraded_mask,
        "dab_strong_support_context_mask": context,
        "dab_augmented_logits": augmented.copy(),
    }, stats


def _suppress_artifacts(
        raw: np.ndarray,
        config: WeightedPromptConfig,
        protected_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    filtered = raw.copy()
    empty = np.zeros(raw.shape, dtype=bool)
    protected = _optional_bool_mask(protected_mask, raw.shape)
    if not config.enable_artifact_filter:
        return filtered, empty, {
            "weak_mid_mask": empty.copy(),
            "closed_weak_mid_mask": empty.copy(),
            "grouped_seed": empty.copy(),
            "component_labels": np.zeros(raw.shape, dtype=np.int32),
        }, {"artifact_filter_enabled": False}

    weak_mid = (
        (raw >= config.artifact_min_logit)
        & (raw <= config.artifact_max_logit)
    )
    kernel_size = _odd_kernel(config.artifact_group_kernel)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(
        weak_mid.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
        iterations=config.artifact_close_iterations,
    )
    seed = cv2.dilate(
        closed,
        kernel,
        iterations=config.artifact_dilate_iterations,
    ).astype(bool)

    num_labels, labels, component_stats, centroids = (
        cv2.connectedComponentsWithStats(seed.astype(np.uint8), connectivity=8)
    )
    artifact_mask = np.zeros(raw.shape, dtype=bool)
    components = []
    selected_count = 0
    positive = raw > 0
    protect_kernel_size = _odd_kernel(config.artifact_suppress_kernel)
    protect_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (protect_kernel_size, protect_kernel_size))
    protected_context = cv2.dilate(
        protected.astype(np.uint8), protect_kernel, iterations=1
    ).astype(bool) if protected.any() else protected
    height, width = raw.shape

    for label_id in range(1, num_labels):
        x = int(component_stats[label_id, cv2.CC_STAT_LEFT])
        y = int(component_stats[label_id, cv2.CC_STAT_TOP])
        box_w = int(component_stats[label_id, cv2.CC_STAT_WIDTH])
        box_h = int(component_stats[label_id, cv2.CC_STAT_HEIGHT])
        region = labels == label_id
        positive_region = region & positive
        positive_area = int(positive_region.sum())
        if positive_area < config.artifact_min_area:
            continue
        protected_overlap_ratio = (
            float((region & protected_context).sum()) / float(region.sum())
            if region.any() else 0.0
        )
        protected_border_lumen = bool(
            protected_overlap_ratio > 0.0 and (
                x == 0 or y == 0
                or x + box_w >= width or y + box_h >= height
            )
        )

        values = raw[positive_region].astype(np.int16)
        weak_mid_ratio = float((
            (values >= config.artifact_min_logit)
            & (values <= config.artifact_max_logit)
        ).mean())
        strong_ratio = float((values >= config.artifact_strong_logit).mean())
        mean_logit = float(values.mean())
        touches_border = (
            x == 0 or y == 0
            or x + box_w >= width or y + box_h >= height
        )

        score = 0
        if weak_mid_ratio >= config.artifact_weak_mid_ratio:
            score += 2
        if strong_ratio <= config.artifact_max_strong_ratio:
            score += 2
        if mean_logit <= config.artifact_max_mean_logit:
            score += 1
        if touches_border:
            score += 1
        selected = score >= config.artifact_score_threshold
        if protected_border_lumen:
            selected = False
        if selected:
            artifact_mask |= region
            selected_count += 1

        components.append({
            "label_id": int(label_id),
            "selected": bool(selected),
            "score": int(score),
            "positive_area": positive_area,
            "bbox_xywh": [x, y, box_w, box_h],
            "touches_border": bool(touches_border),
            "weak_mid_ratio": weak_mid_ratio,
            "strong_ratio": strong_ratio,
            "mean_logit": mean_logit,
            "protected_overlap_ratio": protected_overlap_ratio,
            "protected_border_lumen": protected_border_lumen,
            "centroid_xy": [
                float(centroids[label_id][0]),
                float(centroids[label_id][1]),
            ],
        })

    artifact_mask &= ~protected
    if artifact_mask.any() and config.artifact_suppress_dilate > 0:
        suppress_size = _odd_kernel(config.artifact_suppress_kernel)
        suppress_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (suppress_size, suppress_size))
        artifact_mask = cv2.dilate(
            artifact_mask.astype(np.uint8),
            suppress_kernel,
            iterations=config.artifact_suppress_dilate,
        ).astype(bool)
        artifact_mask &= ~protected

    removed_positive = artifact_mask & positive
    filtered[artifact_mask] = -5
    stats = {
        "artifact_filter_enabled": True,
        "artifact_candidate_count": int(len(components)),
        "artifact_selected_count": int(selected_count),
        "artifact_mask_px": int(artifact_mask.sum()),
        "artifact_protected_px": int(protected.sum()),
        "artifact_removed_positive_px": int(removed_positive.sum()),
        "artifact_removed_logit_counts": _logit_counts(raw, removed_positive),
        "artifact_components": components,
    }
    debug = {
        "weak_mid_mask": weak_mid,
        "closed_weak_mid_mask": closed.astype(bool),
        "grouped_seed": seed,
        "component_labels": labels,
    }
    return filtered, artifact_mask, debug, stats


def _suppress_small_fragments(
        raw: np.ndarray,
        config: WeightedPromptConfig,
        protected_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    filtered = raw.copy()
    fragment_mask = np.zeros(raw.shape, dtype=bool)
    protected = _optional_bool_mask(protected_mask, raw.shape)
    if not config.enable_small_fragment_filter:
        return filtered, fragment_mask, {
            "small_fragment_filter_enabled": False,
        }

    # Include logit 0 so gray pixels attached to a small weak fragment are
    # removed together with its positive pixels.
    nonnegative = raw >= 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        nonnegative.astype(np.uint8), connectivity=8)
    components = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area > config.small_fragment_max_area:
            continue
        component = labels == label_id
        if np.any(component & protected):
            continue
        values = raw[component].astype(np.int16)
        max_logit = int(values.max())
        if max_logit > config.small_fragment_max_logit:
            continue
        fragment_mask |= component
        components.append({
            "label_id": int(label_id),
            "area": area,
            "bbox_xywh": [
                int(stats[label_id, cv2.CC_STAT_LEFT]),
                int(stats[label_id, cv2.CC_STAT_TOP]),
                int(stats[label_id, cv2.CC_STAT_WIDTH]),
                int(stats[label_id, cv2.CC_STAT_HEIGHT]),
            ],
            "mean_logit": float(values.mean()),
            "max_logit": max_logit,
        })

    filtered[fragment_mask] = -5
    return filtered, fragment_mask, {
        "small_fragment_filter_enabled": True,
        "small_fragment_removed_count": int(len(components)),
        "small_fragment_removed_px": int(fragment_mask.sum()),
        "small_fragment_protected_px": int(protected.sum()),
        "small_fragment_removed_logit_counts": _logit_counts(
            raw, fragment_mask),
        "small_fragment_components": components,
    }


def _suppress_isolated_fragments(
        raw: np.ndarray,
        config: WeightedPromptConfig,
        protected_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    filtered = raw.copy()
    fragment_mask = np.zeros(raw.shape, dtype=bool)
    protected = _optional_bool_mask(protected_mask, raw.shape)
    if not config.enable_isolated_fragment_filter:
        return filtered, fragment_mask, {
            "isolated_fragment_filter_enabled": False,
        }

    nonnegative = raw >= 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        nonnegative.astype(np.uint8), connectivity=8)

    large_support = np.zeros(raw.shape, dtype=bool)
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= config.isolated_fragment_neighbor_min_area:
            large_support |= labels == label_id

    if large_support.any():
        distance_to_large = cv2.distanceTransform(
            (~large_support).astype(np.uint8), cv2.DIST_L2, 3)
    else:
        distance_to_large = np.full(raw.shape, np.inf, dtype=np.float32)

    components = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area > config.isolated_fragment_max_area:
            continue

        component = labels == label_id
        if np.any(component & protected):
            continue
        min_distance = float(distance_to_large[component].min())
        if min_distance <= config.isolated_fragment_min_gap:
            continue

        values = raw[component].astype(np.int16)
        fragment_mask |= component
        components.append({
            "label_id": int(label_id),
            "area": area,
            "bbox_xywh": [
                int(stats[label_id, cv2.CC_STAT_LEFT]),
                int(stats[label_id, cv2.CC_STAT_TOP]),
                int(stats[label_id, cv2.CC_STAT_WIDTH]),
                int(stats[label_id, cv2.CC_STAT_HEIGHT]),
            ],
            "mean_logit": float(values.mean()),
            "max_logit": int(values.max()),
            "distance_to_large_support": (
                min_distance if np.isfinite(min_distance) else None),
        })

    filtered[fragment_mask] = -5
    return filtered, fragment_mask, {
        "isolated_fragment_filter_enabled": True,
        "isolated_fragment_max_area": int(config.isolated_fragment_max_area),
        "isolated_fragment_min_gap": int(config.isolated_fragment_min_gap),
        "isolated_fragment_neighbor_min_area": int(
            config.isolated_fragment_neighbor_min_area),
        "isolated_fragment_removed_count": int(len(components)),
        "isolated_fragment_removed_px": int(fragment_mask.sum()),
        "isolated_fragment_protected_px": int(protected.sum()),
        "isolated_fragment_removed_logit_counts": _logit_counts(
            raw, fragment_mask),
        "isolated_fragment_components": components,
    }


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    # Flood from a padded exterior border. The single-image prototype flooded
    # from tile pixel (0, 0), which turns almost the whole tile into a "hole"
    # whenever that pixel is foreground or foreground blocks it from another
    # exterior pocket.
    padded = np.pad(mask.astype(np.uint8), 1, constant_values=0) * 255
    exterior = padded.copy()
    flood_mask = np.zeros(
        (exterior.shape[0] + 2, exterior.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(exterior, flood_mask, (0, 0), 255)
    holes = exterior[1:-1, 1:-1] == 0
    return mask | holes


def _max_pool_lowres(logits: np.ndarray, target_size: int) -> np.ndarray:
    height, width = logits.shape
    lowres = np.full((target_size, target_size), -5, dtype=np.int16)
    rows = (np.arange(height) * target_size // height).astype(np.intp)
    cols = (np.arange(width) * target_size // width).astype(np.intp)
    np.maximum.at(
        lowres,
        (np.repeat(rows, width), np.tile(cols, height)),
        logits.reshape(-1),
    )
    return lowres.astype(np.float32)


def _strong_positive_points(
        logits: np.ndarray,
        min_area: int,
        max_points: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    strong = logits >= 5
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        strong.astype(np.uint8), connectivity=8)
    components = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = centroids[label_id]
        ys, xs = np.where(labels == label_id)
        nearest = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
        components.append({
            "label_id": int(label_id),
            "area": area,
            "bbox_xywh": [
                int(stats[label_id, cv2.CC_STAT_LEFT]),
                int(stats[label_id, cv2.CC_STAT_TOP]),
                int(stats[label_id, cv2.CC_STAT_WIDTH]),
                int(stats[label_id, cv2.CC_STAT_HEIGHT]),
            ],
            "point_xy": [int(xs[nearest]), int(ys[nearest])],
            "kind": "strong",
        })

    components.sort(key=lambda item: item["area"], reverse=True)
    if max_points > 0:
        components = components[:max_points]
    coords = np.asarray(
        [component["point_xy"] for component in components],
        dtype=np.float32,
    ).reshape(-1, 2)
    labels_out = np.ones((len(coords),), dtype=np.int32)
    return coords, labels_out, components


def _lowres_to_image_point(
        point_xy: tuple[int, int],
        image_shape: tuple[int, int],
        lowres_shape: tuple[int, int],
) -> list[float]:
    x, y = point_xy
    height, width = image_shape
    low_h, low_w = lowres_shape
    image_x = (float(x) + 0.5) * float(width) / float(low_w) - 0.5
    image_y = (float(y) + 0.5) * float(height) / float(low_h) - 0.5
    image_x = float(np.clip(image_x, 0.0, max(0.0, width - 1.0)))
    image_y = float(np.clip(image_y, 0.0, max(0.0, height - 1.0)))
    return [image_x, image_y]


def _lumen_positive_points_from_lowres(
        lowres_logits: np.ndarray,
        image_shape: tuple[int, int],
        config: WeightedPromptConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict, dict]:
    empty_mask = np.zeros(lowres_logits.shape, dtype=bool)
    empty_debug = {
        "lumen_point_support_mask_256": empty_mask.copy(),
        "lumen_point_closed_mask_256": empty_mask.copy(),
        "lumen_point_candidate_mask_256": empty_mask.copy(),
        "lumen_point_accepted_mask_256": empty_mask.copy(),
    }
    if not config.enable_lumen_points:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            [],
            {
                "lumen_points_enabled": False,
                "lumen_point_candidate_count": 0,
                "lumen_point_accepted_count": 0,
            },
            empty_debug,
        )

    support = lowres_logits >= config.lumen_point_support_logit_min
    if not support.any():
        debug = dict(empty_debug)
        debug["lumen_point_support_mask_256"] = support
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            [],
            {
                "lumen_points_enabled": True,
                "lumen_point_candidate_count": 0,
                "lumen_point_accepted_count": 0,
            },
            debug,
        )

    close_size = _odd_kernel(config.lumen_point_closing_kernel)
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size))
    closed = cv2.morphologyEx(
        support.astype(np.uint8),
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    ).astype(bool)
    filled = _fill_holes(closed)
    candidate_mask = filled & ~closed

    ring_size = _odd_kernel(config.lumen_point_ring_kernel)
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ring_size, ring_size))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask.astype(np.uint8), connectivity=8)
    components = []
    accepted = []
    accepted_mask = np.zeros(candidate_mask.shape, dtype=bool)
    height, width = lowres_logits.shape

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        box_w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        box_h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        component = labels == label_id
        touches_border = (
            x == 0 or y == 0
            or x + box_w >= width or y + box_h >= height
        )
        dilated = cv2.dilate(
            component.astype(np.uint8), ring_kernel, iterations=1).astype(bool)
        ring = dilated & ~component
        ring_area = int(ring.sum())
        wall_ratio = (
            float((ring & support).sum()) / float(ring_area)
            if ring_area else 0.0
        )

        reject_reasons = []
        if area < config.lumen_point_min_area:
            reject_reasons.append("area_too_small")
        if area > config.lumen_point_max_area:
            reject_reasons.append("area_too_large")
        if touches_border:
            reject_reasons.append("touches_border")
        if wall_ratio < config.lumen_point_min_wall_ratio:
            reject_reasons.append("weak_wall_support")

        dist = cv2.distanceTransform(
            component.astype(np.uint8), cv2.DIST_L2, 3)
        point_y, point_x = np.unravel_index(int(np.argmax(dist)), dist.shape)
        image_point = _lowres_to_image_point(
            (int(point_x), int(point_y)), image_shape, lowres_logits.shape)
        component_info = {
            "label_id": int(label_id),
            "accepted": not reject_reasons,
            "selected": False,
            "reject_reasons": reject_reasons,
            "area": area,
            "bbox_xywh_256": [x, y, box_w, box_h],
            "ring_area": ring_area,
            "wall_ratio": wall_ratio,
            "touches_border": bool(touches_border),
            "point_xy_256": [int(point_x), int(point_y)],
            "point_xy": [
                int(round(image_point[0])),
                int(round(image_point[1])),
            ],
            "kind": "lumen",
        }
        components.append(component_info)
        if not reject_reasons:
            accepted.append((component_info, image_point, component))

    accepted.sort(
        key=lambda item: (item[0]["wall_ratio"], item[0]["area"]),
        reverse=True,
    )
    if config.max_lumen_points > 0:
        accepted = accepted[:config.max_lumen_points]

    selected_components = []
    coords = []
    for component_info, image_point, component in accepted:
        component_info["selected"] = True
        accepted_mask |= component
        selected_components.append(component_info)
        coords.append(image_point)

    coords_out = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    labels_out = np.ones((len(coords_out),), dtype=np.int32)
    stats_out = {
        "lumen_points_enabled": True,
        "lumen_point_support_logit_min": int(
            config.lumen_point_support_logit_min),
        "lumen_point_closing_kernel": int(config.lumen_point_closing_kernel),
        "lumen_point_min_area": int(config.lumen_point_min_area),
        "lumen_point_max_area": int(config.lumen_point_max_area),
        "lumen_point_ring_kernel": int(config.lumen_point_ring_kernel),
        "lumen_point_min_wall_ratio": float(
            config.lumen_point_min_wall_ratio),
        "max_lumen_points": int(config.max_lumen_points),
        "lumen_point_candidate_count": int(num_labels - 1),
        "lumen_point_accepted_count": int(len(selected_components)),
        "lumen_point_components": components,
    }
    debug_out = {
        "lumen_point_support_mask_256": support,
        "lumen_point_closed_mask_256": closed,
        "lumen_point_candidate_mask_256": candidate_mask,
        "lumen_point_accepted_mask_256": accepted_mask,
    }
    return coords_out, labels_out, selected_components, stats_out, debug_out


def _border_mask(shape: tuple[int, int], width: int = 1) -> np.ndarray:
    height, width_img = shape
    border = np.zeros(shape, dtype=bool)
    width = max(1, int(width))
    border[:width, :] = True
    border[-width:, :] = True
    border[:, :width] = True
    border[:, -width:] = True
    return border


def _near_white_mask_and_threshold_details(
        tile_rgb: np.ndarray,
        config: WeightedPromptConfig) -> tuple[np.ndarray, dict]:
    rgb = np.round(_rgb_float(tile_rgb) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    value = hsv[:, :, 2].astype(np.int16)
    channel_max = rgb.max(axis=2).astype(np.int16)
    channel_min = rgb.min(axis=2).astype(np.int16)
    balanced = (
        (saturation <= float(config.dab_lumen_white_saturation_max))
        & ((channel_max - channel_min)
           <= int(config.dab_lumen_white_channel_delta_max))
    )
    threshold_details = _near_white_value_threshold_details(
        value, balanced, int(config.dab_lumen_white_value_min))
    value_min = int(threshold_details["threshold"])
    white = (
        balanced
        & (value >= value_min)
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(
        white.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1
    ).astype(bool)
    return closed, threshold_details


def _near_white_mask(tile_rgb: np.ndarray,
                     config: WeightedPromptConfig) -> np.ndarray:
    mask, _ = _near_white_mask_and_threshold_details(tile_rgb, config)
    return mask


def _dab_lumen_points(
        dab_u8: np.ndarray,
        existing_lumen_mask: np.ndarray,
        forbidden_mask: np.ndarray,
        config: WeightedPromptConfig,
        tile_rgb: Optional[np.ndarray] = None,
        wall_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict, dict]:
    shape = existing_lumen_mask.shape if dab_u8 is None else dab_u8.shape
    empty_mask = np.zeros(shape, dtype=bool)
    empty_debug = {
        "dab_lumen_wall_mask": empty_mask.copy(),
        "dab_lumen_white_mask": empty_mask.copy(),
        "dab_lumen_near_wall_mask": empty_mask.copy(),
        "dab_lumen_candidate_mask": empty_mask.copy(),
        "dab_lumen_accepted_mask": empty_mask.copy(),
    }
    if not config.enable_dab_lumen_fill:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            [],
            {
                "dab_lumen_fill_enabled": False,
                "dab_lumen_candidate_count": 0,
                "dab_lumen_accepted_count": 0,
            },
            empty_debug,
        )

    if (dab_u8 is None or not dab_u8.any()) and wall_mask is None:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            [],
            {
                "dab_lumen_fill_enabled": True,
                "dab_lumen_candidate_count": 0,
                "dab_lumen_accepted_count": 0,
            },
            empty_debug,
        )

    if wall_mask is not None:
        if wall_mask.shape != shape:
            raise ValueError("DAB lumen wall mask shape must match tile shape")
        wall = wall_mask.astype(bool, copy=False)
    else:
        wall = dab_u8 >= config.dab_lumen_wall_min_intensity
    wall_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5))
    wall_closed = cv2.morphologyEx(
        wall.astype(np.uint8), cv2.MORPH_CLOSE, wall_kernel,
        iterations=1).astype(bool)
    near_size = _odd_kernel(config.dab_lumen_near_wall_kernel)
    near_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (near_size, near_size))
    near_wall = cv2.dilate(
        wall_closed.astype(np.uint8), near_kernel, iterations=1).astype(bool)
    white_mask = empty_mask.copy()
    candidate_source = "dab_dark"
    near_white_threshold_details = None
    if (config.dab_lumen_use_white_interior and tile_rgb is not None):
        if tile_rgb.shape[:2] != shape:
            raise ValueError(
                "Original RGB tile shape must match DAB lumen shape")
        white_mask, near_white_threshold_details = (
            _near_white_mask_and_threshold_details(tile_rgb, config))
        candidate_source = "rgb_white"
        candidate_base = white_mask
    elif dab_u8 is not None:
        dark = dab_u8 <= config.dab_lumen_interior_max_intensity
        candidate_base = dark
    else:
        candidate_base = empty_mask.copy()
    candidate_base = candidate_base & ~wall_closed & ~forbidden_mask
    candidate_seed = candidate_base & near_wall
    _, candidate_base_labels = cv2.connectedComponents(
        candidate_base.astype(np.uint8), connectivity=8)
    seed_ids = np.unique(candidate_base_labels[candidate_seed])
    seed_ids = seed_ids[seed_ids > 0]
    candidate_mask = np.isin(candidate_base_labels, seed_ids)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask.astype(np.uint8), connectivity=8)
    ring_size = _odd_kernel(config.dab_lumen_ring_kernel)
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ring_size, ring_size))
    macro_size = _odd_kernel(config.dab_lumen_macro_closing_kernel)
    macro_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (macro_size, macro_size))
    macro_closed = cv2.morphologyEx(
        wall_closed.astype(np.uint8), cv2.MORPH_CLOSE, macro_kernel,
        iterations=1).astype(bool)
    macro_holes = _fill_holes(macro_closed) & ~macro_closed
    _, macro_hole_labels = cv2.connectedComponents(
        macro_holes.astype(np.uint8), connectivity=8)
    border = _border_mask(shape, width=max(1, ring_size // 2))

    components = []
    accepted = []
    accepted_mask = np.zeros(shape, dtype=bool)
    height, width = shape

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        box_w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        box_h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        component = labels == label_id
        touches_top = y == 0
        touches_bottom = y + box_h >= height
        touches_left = x == 0
        touches_right = x + box_w >= width
        border_touch_count = int(
            touches_top + touches_bottom + touches_left + touches_right)
        touches_border = border_touch_count > 0
        area_ratio = float(area) / float(max(1, height * width))
        dilated = cv2.dilate(
            component.astype(np.uint8), ring_kernel, iterations=1).astype(bool)
        ring = dilated & ~component
        ring_area = int(ring.sum())
        wall_pixels = ring & wall_closed
        border_pixels = ring & border if touches_border else empty_mask
        wall_ratio = (
            float(wall_pixels.sum()) / float(ring_area)
            if ring_area else 0.0
        )
        boundary_ratio = (
            float((wall_pixels | border_pixels).sum()) / float(ring_area)
            if ring_area else 0.0
        )
        existing_overlap_ratio = (
            float((component & existing_lumen_mask).sum()) / float(area)
            if area else 0.0
        )
        macro_overlap_ratio = (
            float((component & macro_holes).sum()) / float(area)
            if area else 0.0
        )
        macro_ring = cv2.dilate(
            component.astype(np.uint8), macro_kernel, iterations=1).astype(bool)
        macro_ring &= ~component
        macro_ring_area = int(macro_ring.sum())
        macro_wall_ratio = (
            float((macro_ring & wall_closed).sum()) / float(macro_ring_area)
            if macro_ring_area else 0.0
        )
        macro_supported = (
            macro_overlap_ratio >= config.dab_lumen_macro_min_overlap
            and macro_wall_ratio >= config.dab_lumen_macro_min_wall_ratio
        )
        fill_component = component
        fill_kind = candidate_source
        if macro_supported and candidate_source != "rgb_white":
            overlap_labels = macro_hole_labels[component & macro_holes]
            overlap_labels = overlap_labels[overlap_labels > 0]
            if overlap_labels.size:
                label_counts = np.bincount(overlap_labels)
                macro_label_id = int(np.argmax(label_counts))
                fill_component = macro_hole_labels == macro_label_id
                fill_kind = "macro_hole"
        aspect_ratio = (
            float(max(box_w, box_h)) / float(max(1, min(box_w, box_h)))
        )

        reject_reasons = []
        if area < config.dab_lumen_min_area:
            reject_reasons.append("area_too_small")
        if area > config.dab_lumen_max_area:
            reject_reasons.append("area_too_large")
        if aspect_ratio > config.dab_lumen_max_aspect_ratio:
            reject_reasons.append("too_elongated")
        min_boundary_ratio = (
            config.dab_lumen_min_border_boundary_ratio
            if touches_border else config.dab_lumen_min_boundary_ratio
        )
        weak_wall = wall_ratio < config.dab_lumen_min_wall_ratio
        weak_boundary = boundary_ratio < min_boundary_ratio
        if (touches_border and border_touch_count >= 3
                and not macro_supported):
            reject_reasons.append("touches_many_borders")
        if (touches_border and border_touch_count > 1
                and area_ratio > 0.05 and not macro_supported):
            reject_reasons.append("border_candidate_too_large")
        if weak_wall and not macro_supported:
            reject_reasons.append("weak_dab_wall")
        if weak_boundary and not macro_supported:
            reject_reasons.append("weak_boundary_support")
        if existing_overlap_ratio >= 0.50:
            reject_reasons.append("already_covered_by_prompt_lumen")

        point_mask = fill_component if macro_supported else component
        dist = cv2.distanceTransform(
            point_mask.astype(np.uint8), cv2.DIST_L2, 3)
        point_y, point_x = np.unravel_index(int(np.argmax(dist)), dist.shape)
        fill_yx = np.argwhere(fill_component)
        if fill_yx.size:
            fill_y0, fill_x0 = fill_yx.min(axis=0)
            fill_y1, fill_x1 = fill_yx.max(axis=0) + 1
            fill_bbox = [
                int(fill_x0),
                int(fill_y0),
                int(fill_x1 - fill_x0),
                int(fill_y1 - fill_y0),
            ]
        else:
            fill_bbox = [x, y, box_w, box_h]
        component_info = {
            "label_id": int(label_id),
            "accepted": not reject_reasons,
            "selected": False,
            "reject_reasons": reject_reasons,
            "area": area,
            "bbox_xywh": [x, y, box_w, box_h],
            "ring_area": ring_area,
            "wall_ratio": wall_ratio,
            "boundary_ratio": boundary_ratio,
            "macro_overlap_ratio": macro_overlap_ratio,
            "macro_wall_ratio": macro_wall_ratio,
            "macro_supported": bool(macro_supported),
            "fill_kind": fill_kind,
            "fill_area": int(fill_component.sum()),
            "fill_bbox_xywh": fill_bbox,
            "touches_border": bool(touches_border),
            "border_touch_count": int(border_touch_count),
            "area_ratio": area_ratio,
            "aspect_ratio": aspect_ratio,
            "existing_overlap_ratio": existing_overlap_ratio,
            "point_xy": [int(point_x), int(point_y)],
            "kind": "dab_lumen",
        }
        components.append(component_info)
        if not reject_reasons:
            accepted.append((component_info, fill_component))

    accepted.sort(
        key=lambda item: (
            item[0]["touches_border"],
            item[0]["boundary_ratio"],
            item[0]["area"],
        ),
        reverse=True,
    )
    if config.max_dab_lumen_points > 0:
        accepted = accepted[:config.max_dab_lumen_points]

    selected_components = []
    coords = []
    for component_info, component in accepted:
        component_info["selected"] = True
        accepted_mask |= component
        selected_components.append(component_info)
        coords.append(component_info["point_xy"])

    coords_out = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    labels_out = np.ones((len(coords_out),), dtype=np.int32)
    stats_out = {
        "dab_lumen_fill_enabled": True,
        "dab_lumen_wall_min_intensity": int(
            config.dab_lumen_wall_min_intensity),
        "dab_lumen_interior_max_intensity": int(
            config.dab_lumen_interior_max_intensity),
        "dab_lumen_near_wall_kernel": int(config.dab_lumen_near_wall_kernel),
        "dab_lumen_ring_kernel": int(config.dab_lumen_ring_kernel),
        "dab_lumen_min_area": int(config.dab_lumen_min_area),
        "dab_lumen_max_area": int(config.dab_lumen_max_area),
        "dab_lumen_min_wall_ratio": float(config.dab_lumen_min_wall_ratio),
        "dab_lumen_min_boundary_ratio": float(
            config.dab_lumen_min_boundary_ratio),
        "dab_lumen_min_border_boundary_ratio": float(
            config.dab_lumen_min_border_boundary_ratio),
        "dab_lumen_macro_closing_kernel": int(
            config.dab_lumen_macro_closing_kernel),
        "dab_lumen_macro_min_overlap": float(
            config.dab_lumen_macro_min_overlap),
        "dab_lumen_macro_min_wall_ratio": float(
            config.dab_lumen_macro_min_wall_ratio),
        "dab_lumen_candidate_source": candidate_source,
        "dab_lumen_use_white_interior": bool(
            config.dab_lumen_use_white_interior),
        "dab_lumen_white_value_min": int(
            config.dab_lumen_white_value_min),
        "dab_lumen_white_effective_value_min": (
            int(near_white_threshold_details["threshold"])
            if near_white_threshold_details is not None else None),
        "dab_lumen_white_threshold_source": (
            near_white_threshold_details["source"]
            if near_white_threshold_details is not None else None),
        "dab_lumen_white_peak_value_intensity": (
            near_white_threshold_details["peak_intensity"]
            if near_white_threshold_details is not None else None),
        "dab_lumen_white_peak_tail_multiotsu_thresholds": (
            near_white_threshold_details["multiotsu_thresholds"]
            if near_white_threshold_details is not None else None),
        "dab_lumen_white_saturation_max": float(
            config.dab_lumen_white_saturation_max),
        "dab_lumen_white_channel_delta_max": int(
            config.dab_lumen_white_channel_delta_max),
        "dab_lumen_max_aspect_ratio": float(
            config.dab_lumen_max_aspect_ratio),
        "max_dab_lumen_points": int(config.max_dab_lumen_points),
        "dab_lumen_candidate_count": int(num_labels - 1),
        "dab_lumen_accepted_count": int(len(selected_components)),
        "dab_lumen_components": components,
    }
    debug_out = {
        "dab_lumen_wall_mask": wall_closed,
        "dab_lumen_white_mask": white_mask,
        "dab_lumen_near_wall_mask": near_wall,
        "dab_lumen_macro_hole_mask": macro_holes,
        "dab_lumen_candidate_mask": candidate_mask,
        "dab_lumen_accepted_mask": accepted_mask,
    }
    return coords_out, labels_out, selected_components, stats_out, debug_out


def build_weighted_prompt(
        seg: np.ndarray,
        marker: np.ndarray,
        config: WeightedPromptConfig,
        tile_rgb: Optional[np.ndarray] = None,
        dapi: Optional[np.ndarray] = None,
) -> WeightedPromptResult:
    """Create the exact prompt used by the weighted-mask + points flow."""
    seg_logits, seg_stats = _seg_logits(seg, config)
    marker_logits, marker_stats = _marker_logits(marker, config)
    pre_dab_raw = np.maximum(seg_logits, marker_logits).astype(np.int16)
    filtered, dab_removed_mask, dab_debug, dab_stats = _suppress_weak_dab_prompt(
        pre_dab_raw, tile_rgb, dapi, config, seg=seg)
    raw, dab_support_debug, dab_support_stats = _add_strong_dab_prompt_support(
        filtered, dab_debug, config)

    dapi_lumen_filled_mask = _optional_bool_mask(
        dab_debug.get("dapi_lumen_accepted_mask"), raw.shape)
    dapi_lumen_protected_prompt_mask = _optional_bool_mask(
        dab_debug.get("dapi_lumen_protected_prompt_mask"), raw.shape)
    dapi_lumen_rescue_mask = (
        dapi_lumen_filled_mask | dapi_lumen_protected_prompt_mask)

    dab_lumen_wall_mask = dab_debug.get("dab_keep_mask")
    if dab_lumen_wall_mask is not None and np.any(dab_lumen_wall_mask):
        dab_lumen_wall_mask = dab_lumen_wall_mask.astype(bool, copy=False)
        dab_lumen_wall_source = "dab_keep_mask"
    else:
        dab_lumen_wall_mask = None
        dab_lumen_wall_source = "dab_u8_threshold"
    (
        dab_lumen_coords,
        dab_lumen_labels,
        dab_lumen_components,
        dab_lumen_stats,
        dab_lumen_debug,
    ) = _dab_lumen_points(
        dab_debug.get("dab_intensity"),
        existing_lumen_mask=dapi_lumen_filled_mask,
        forbidden_mask=dapi_lumen_rescue_mask,
        config=config,
        tile_rgb=tile_rgb,
        wall_mask=dab_lumen_wall_mask,
    )
    dab_lumen_accepted_mask = _optional_bool_mask(
        dab_lumen_debug.get("dab_lumen_accepted_mask"), raw.shape)
    dab_lumen_protected_prompt_mask = np.zeros(raw.shape, dtype=bool)
    if dab_lumen_accepted_mask.any():
        protect_size = _odd_kernel(config.dab_lumen_near_wall_kernel)
        protect_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (protect_size, protect_size))
        dab_lumen_protected_prompt_mask = cv2.dilate(
            dab_lumen_accepted_mask.astype(np.uint8),
            protect_kernel,
            iterations=1,
        ).astype(bool)
        dab_lumen_protected_prompt_mask &= raw >= 0
    dab_lumen_rescue_mask = (
        dab_lumen_accepted_mask | dab_lumen_protected_prompt_mask)
    dab_lumen_stats.update({
        "dab_lumen_wall_source": dab_lumen_wall_source,
        "dab_lumen_filled_px": int(dab_lumen_accepted_mask.sum()),
        "dab_lumen_protected_prompt_px": int(
            dab_lumen_protected_prompt_mask.sum()),
        "dab_lumen_filter_rescue_px": int(dab_lumen_rescue_mask.sum()),
    })

    protected_filter_mask = dapi_lumen_rescue_mask | dab_lumen_rescue_mask
    protected_prompt_mask = (
        dapi_lumen_protected_prompt_mask | dab_lumen_protected_prompt_mask)
    protected_lumen_mask = dapi_lumen_filled_mask | dab_lumen_accepted_mask

    artifact_filtered, artifact_mask, artifact_debug, artifact_stats = (
        _suppress_artifacts(raw, config, protected_filter_mask)
    )
    cleaned, fragment_mask, fragment_stats = _suppress_small_fragments(
        artifact_filtered, config, protected_filter_mask)
    cleaned, isolated_fragment_mask, isolated_fragment_stats = (
        _suppress_isolated_fragments(
            cleaned, config, protected_filter_mask))
    if protected_filter_mask.any():
        artifact_mask &= ~protected_filter_mask
        fragment_mask &= ~protected_filter_mask
        isolated_fragment_mask &= ~protected_filter_mask
        restore_wall = protected_prompt_mask & (raw > cleaned)
        cleaned[restore_wall] = raw[restore_wall]
        fill_value = int(config.lumen_point_fill_logit)
        if fill_value > -5:
            cleaned[
                protected_lumen_mask & (cleaned < fill_value)
            ] = fill_value
    support = cleaned > 0

    repair_size = _odd_kernel(config.repair_kernel)
    repair_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (repair_size, repair_size))
    closed = cv2.morphologyEx(
        support.astype(np.uint8),
        cv2.MORPH_CLOSE,
        repair_kernel,
        iterations=config.repair_iterations,
    ).astype(bool)
    repair = closed & ~support
    filled = _fill_holes(closed)
    lumen = filled & ~closed

    band_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(
        filled.astype(np.uint8),
        band_kernel,
        iterations=config.uncertain_iterations,
    ).astype(bool)
    uncertain = dilated & ~filled

    final = cleaned.copy()
    final[repair & (final < config.repair_logit)] = config.repair_logit
    final[lumen & (final < config.lumen_logit)] = config.lumen_logit
    final[uncertain & (final < 0)] = 0
    final[artifact_mask] = -5
    final[fragment_mask] = -5
    final[isolated_fragment_mask] = -5
    final = np.clip(final, -5, 5).astype(np.int16)
    lowres_logits = _max_pool_lowres(final, config.target_size)
    point_coords, point_labels, point_components = _strong_positive_points(
        final, config.point_min_area, config.max_positive_points)
    strong_point_count = len(point_coords)
    lumen_coords, lumen_labels, lumen_components, lumen_stats, lumen_debug = (
        _lumen_positive_points_from_lowres(
            lowres_logits, final.shape, config)
    )
    lumen_filled_mask = np.zeros(final.shape, dtype=bool)
    if config.lumen_point_fill_logit > -5:
        selected_lumen = lumen_debug["lumen_point_accepted_mask_256"]
        if selected_lumen.any():
            lumen_filled_mask = cv2.resize(
                selected_lumen.astype(np.uint8),
                (final.shape[1], final.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            fill_value = int(config.lumen_point_fill_logit)
            final[lumen_filled_mask & (final < fill_value)] = fill_value
            final = np.clip(final, -5, 5).astype(np.int16)
            lowres_logits = _max_pool_lowres(final, config.target_size)
    dapi_point_components = [
        component for component in dab_stats.get("dapi_lumen_components", [])
        if component.get("selected_for_point", False)
    ]
    dapi_coords = np.asarray(
        [component["point_xy"] for component in dapi_point_components],
        dtype=np.float32,
    ).reshape(-1, 2)
    dapi_labels = np.ones((len(dapi_coords),), dtype=np.int32)
    mask_input = lowres_logits[None, :, :]
    if len(lumen_coords):
        point_coords = np.concatenate([point_coords, lumen_coords], axis=0)
        point_labels = np.concatenate([point_labels, lumen_labels], axis=0)
        point_components = point_components + lumen_components
    if len(dab_lumen_coords):
        point_coords = np.concatenate([point_coords, dab_lumen_coords], axis=0)
        point_labels = np.concatenate([point_labels, dab_lumen_labels], axis=0)
        point_components = point_components + dab_lumen_components
    if len(dapi_coords):
        point_coords = np.concatenate([point_coords, dapi_coords], axis=0)
        point_labels = np.concatenate([point_labels, dapi_labels], axis=0)
        point_components = point_components + dapi_point_components

    stats = {
        **seg_stats,
        **marker_stats,
        **dab_stats,
        **dab_support_stats,
        **dab_lumen_stats,
        **artifact_stats,
        **fragment_stats,
        **isolated_fragment_stats,
        "prompt_mode": "weighted-points",
        "pre_dab_raw_positive_support_px": int((pre_dab_raw > 0).sum()),
        "raw_positive_support_px": int((raw > 0).sum()),
        "filtered_positive_support_px": int(support.sum()),
        "repair_px": int(repair.sum()),
        "lumen_px": int(lumen.sum()),
        "lumen_point_fill_logit": int(config.lumen_point_fill_logit),
        "lumen_point_filled_px": int(lumen_filled_mask.sum()),
        "dapi_lumen_filled_px": int(dapi_lumen_filled_mask.sum()),
        "dapi_lumen_filter_rescue_px": int(dapi_lumen_rescue_mask.sum()),
        "protected_filter_px": int(protected_filter_mask.sum()),
        "uncertain_band_px": int(uncertain.sum()),
        "final_nonnegative_px": int((final >= 0).sum()),
        "strong_positive_point_count": int(strong_point_count),
        **lumen_stats,
    }
    stats["total_positive_point_count"] = int(len(point_coords))
    for value in (-5, 0, 1, 2, 3, 4, 5):
        stats[f"pre_dab_raw_logit_{value}_px"] = int(
            (pre_dab_raw == value).sum())
        stats[f"raw_logit_{value}_px"] = int((raw == value).sum())
        stats[f"final_logit_{value}_px"] = int((final == value).sum())

    return WeightedPromptResult(
        logits=final,
        mask_input=mask_input.astype(np.float32),
        point_coords=point_coords,
        point_labels=point_labels,
        point_components=point_components,
        stats=stats,
        debug={
            "pre_dab_raw_logits": pre_dab_raw,
            "raw_logits": raw,
            "dab_removed_mask": dab_removed_mask,
            **dab_support_debug,
            "artifact_filtered_logits": artifact_filtered,
            "artifact_mask": artifact_mask,
            "small_fragment_mask": fragment_mask,
            "isolated_fragment_mask": isolated_fragment_mask,
            "cleaned_logits": cleaned,
            "lumen_point_filled_mask": lumen_filled_mask,
            "dapi_lumen_filled_mask": dapi_lumen_filled_mask,
            "dab_lumen_protected_prompt_mask": (
                dab_lumen_protected_prompt_mask),
            "protected_filter_mask": protected_filter_mask,
            **lumen_debug,
            **dab_lumen_debug,
            **dab_debug,
            **artifact_debug,
        },
    )


def colorize_logits(logits: np.ndarray) -> np.ndarray:
    colors = {
        -5: (30, 30, 160),
        0: (180, 180, 180),
        1: (255, 230, 130),
        2: (255, 190, 80),
        3: (255, 140, 40),
        4: (230, 70, 25),
        5: (180, 0, 0),
    }
    output = np.zeros((*logits.shape, 3), dtype=np.uint8)
    for value, color in colors.items():
        output[logits == value] = color
    return output
