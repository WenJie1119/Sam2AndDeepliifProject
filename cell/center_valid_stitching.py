"""
Debug-only center-valid stitching for DeepLIIF Seg/Marker outputs.

This module keeps DeepLIIF inference unchanged: every tile is still inferred as
a full tile, while only the center-valid region of each output is copied into a
stitched canvas for ROI-level seed extraction diagnostics.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ValidCrop:
    local_x0: int
    local_y0: int
    local_x1: int
    local_y1: int
    global_x0: int
    global_y0: int
    global_x1: int
    global_y1: int

    @property
    def width(self) -> int:
        return self.local_x1 - self.local_x0

    @property
    def height(self) -> int:
        return self.local_y1 - self.local_y0


def _as_rgb_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = arr.clip(0, 255).astype(np.uint8)
    return arr


def _save_rgb(path: str, img_rgb: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(_as_rgb_uint8(img_rgb)).save(path)


def _save_gray(path: str, img_gray: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.asarray(img_gray)
    if arr.dtype != np.uint8:
        arr = arr.clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _valid_crop(tile_info: dict, seg_shape: tuple[int, ...],
                tile_size: int, overlap: int) -> Optional[ValidCrop]:
    """Return the center-valid crop for one tile output."""
    h, w = seg_shape[:2]
    left = overlap // 2
    top = overlap // 2
    right = overlap - left
    bottom = overlap - top

    # Normal case: 64:448 for tile_size=512, overlap=128.
    local_x0 = min(left, w)
    local_y0 = min(top, h)
    local_x1 = min(tile_size - right, w)
    local_y1 = min(tile_size - bottom, h)

    # If an edge tile is smaller than the nominal tile size, do not address
    # padded pixels beyond the real WSI extent.
    actual_w = int(tile_info.get("actual_w", w))
    actual_h = int(tile_info.get("actual_h", h))
    local_x1 = min(local_x1, actual_w, w)
    local_y1 = min(local_y1, actual_h, h)

    if local_x1 <= local_x0 or local_y1 <= local_y0:
        return None

    global_x0 = int(tile_info["x"]) + local_x0
    global_y0 = int(tile_info["y"]) + local_y0
    global_x1 = int(tile_info["x"]) + local_x1
    global_y1 = int(tile_info["y"]) + local_y1
    return ValidCrop(
        local_x0=local_x0,
        local_y0=local_y0,
        local_x1=local_x1,
        local_y1=local_y1,
        global_x0=global_x0,
        global_y0=global_y0,
        global_x1=global_x1,
        global_y1=global_y1,
    )


def _build_stitched_canvases(records: list[dict], tile_size: int,
                             overlap: int) -> tuple[np.ndarray, np.ndarray,
                                                     np.ndarray, dict]:
    prepared: list[tuple[dict, np.ndarray, np.ndarray, ValidCrop]] = []
    for record in records:
        seg_np = _as_rgb_uint8(record["seg_np"])
        marker_np = _as_rgb_uint8(record["marker_np"])
        crop = _valid_crop(record["tile_info"], seg_np.shape, tile_size, overlap)
        if crop is None:
            continue
        prepared.append((record, seg_np, marker_np, crop))

    if not prepared:
        raise ValueError("no valid DeepLIIF tile outputs to stitch")

    min_x = min(crop.global_x0 for _, _, _, crop in prepared)
    min_y = min(crop.global_y0 for _, _, _, crop in prepared)
    max_x = max(crop.global_x1 for _, _, _, crop in prepared)
    max_y = max(crop.global_y1 for _, _, _, crop in prepared)
    canvas_w = max_x - min_x
    canvas_h = max_y - min_y
    if canvas_w <= 0 or canvas_h <= 0:
        raise ValueError("invalid stitched canvas extent")

    seg_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    marker_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    owner_canvas = np.full((canvas_h, canvas_w), -1, dtype=np.int32)

    overlap_pixels = 0
    sources = []
    for tile_id, (record, seg_np, marker_np, crop) in enumerate(prepared, start=1):
        src_y = slice(crop.local_y0, crop.local_y1)
        src_x = slice(crop.local_x0, crop.local_x1)
        dst_x0 = crop.global_x0 - min_x
        dst_y0 = crop.global_y0 - min_y
        dst_y = slice(dst_y0, dst_y0 + crop.height)
        dst_x = slice(dst_x0, dst_x0 + crop.width)

        existing = owner_canvas[dst_y, dst_x] >= 0
        overlap_pixels += int(existing.sum())

        seg_canvas[dst_y, dst_x] = seg_np[src_y, src_x]
        marker_canvas[dst_y, dst_x] = marker_np[src_y, src_x]
        owner_canvas[dst_y, dst_x] = tile_id

        tile_info = record["tile_info"]
        sources.append({
            "tile_id": tile_id,
            "tile_name": record.get("tile_name", ""),
            "row": int(tile_info.get("row", -1)),
            "col": int(tile_info.get("col", -1)),
            "role": tile_info.get("debug_role", ""),
            "valid_local": [
                crop.local_x0, crop.local_y0, crop.local_x1, crop.local_y1,
            ],
            "valid_global": [
                crop.global_x0, crop.global_y0, crop.global_x1, crop.global_y1,
            ],
        })

    metadata = {
        "canvas_origin_xy": [min_x, min_y],
        "canvas_size_wh": [canvas_w, canvas_h],
        "tile_size": int(tile_size),
        "overlap": int(overlap),
        "halo": int(overlap // 2),
        "valid_size": int(tile_size - overlap),
        "source_tile_count": len(prepared),
        "hole_pixels": int((owner_canvas < 0).sum()),
        "overlap_pixels": int(overlap_pixels),
        "sources": sources,
    }
    return seg_canvas, marker_canvas, owner_canvas, metadata


def _positive_region_overlay(base_rgb: np.ndarray,
                             regions_info: list[dict]) -> np.ndarray:
    overlay = _as_rgb_uint8(base_rgb).copy()
    for region in regions_info:
        coords = region.get("coords")
        if coords is None or len(coords) == 0:
            continue
        rid = int(region.get("id", 0))
        color = np.array([
            (rid * 67) % 256,
            (rid * 137) % 256,
            (rid * 221) % 256,
        ], dtype=np.float32)
        rows = coords[:, 0].astype(np.intp)
        cols = coords[:, 1].astype(np.intp)
        valid = (
            (rows >= 0) & (rows < overlay.shape[0]) &
            (cols >= 0) & (cols < overlay.shape[1])
        )
        rows = rows[valid]
        cols = cols[valid]
        if rows.size == 0:
            continue
        overlay[rows, cols] = (
            overlay[rows, cols].astype(np.float32) * 0.35 +
            color * 0.65
        ).clip(0, 255).astype(np.uint8)

        cy, cx = region.get("center", (None, None))
        if cy is not None and cx is not None:
            cv2.putText(
                overlay,
                str(rid),
                (int(cx) - 4, int(cy) + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return overlay


def write_center_valid_debug_outputs(records: list[dict], output_root: str,
                                     tile_size: int, overlap: int,
                                     seg_thresh: int,
                                     marker_thresh: Optional[int],
                                     marker_percentile_factor: float,
                                     morphology_kernel: int,
                                     min_area: int) -> dict:
    """
    Stitch center-valid DeepLIIF outputs and write debug-region artifacts.

    Args:
        records: list of dicts containing tile_info, tile_name, seg_np,
            and marker_np.
        output_root: pipeline output directory.

    Returns:
        JSON-serializable metadata for the stitched debug outputs.
    """
    from cd34_pipeline.cell.extraction import (
        compute_marker_threshold,
        compute_posneg_mask,
        extract_connected_positive_regions,
    )

    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be in [0, tile_size)")

    out_dir = os.path.join(output_root, "debug_region")
    os.makedirs(out_dir, exist_ok=True)

    seg_canvas, marker_canvas, owner_canvas, metadata = _build_stitched_canvases(
        records, tile_size=tile_size, overlap=overlap)

    _save_rgb(os.path.join(out_dir, "08_stitched_deepliif_seg.png"), seg_canvas)
    _save_rgb(os.path.join(out_dir, "09_stitched_deepliif_marker.png"),
              marker_canvas)

    if marker_canvas.ndim == 3:
        marker_gray = cv2.cvtColor(marker_canvas, cv2.COLOR_RGB2GRAY)
    else:
        marker_gray = marker_canvas.copy()

    effective_marker_thresh = marker_thresh
    if effective_marker_thresh is None:
        effective_marker_thresh = compute_marker_threshold(
            marker_gray, percentile_factor=marker_percentile_factor)

    posneg_mask, is_foreground, _ = compute_posneg_mask(seg_canvas, seg_thresh)
    seg_positive = posneg_mask == 2
    marker_positive = is_foreground & (marker_gray > effective_marker_thresh)
    combined_positive = seg_positive & marker_positive

    seg_pos_vis = np.zeros_like(seg_canvas, dtype=np.uint8)
    seg_pos_vis[seg_positive] = [0, 255, 0]
    _save_rgb(os.path.join(out_dir, "10_stitched_seg_positive.png"),
              seg_pos_vis)

    _save_gray(
        os.path.join(out_dir, "11_stitched_combined_positive.png"),
        combined_positive.astype(np.uint8) * 255,
    )

    regions_info = extract_connected_positive_regions(
        seg_canvas,
        marker_canvas,
        seg_thresh=seg_thresh,
        marker_thresh=effective_marker_thresh,
        marker_percentile_factor=marker_percentile_factor,
        morphology_kernel=morphology_kernel,
        min_area=min_area,
    )
    _save_rgb(
        os.path.join(out_dir, "12_stitched_positive_regions.png"),
        _positive_region_overlay(seg_canvas, regions_info),
    )

    owner_vis = np.zeros((*owner_canvas.shape, 3), dtype=np.uint8)
    assigned = owner_canvas >= 0
    owner_vis[assigned, 0] = (owner_canvas[assigned] * 67 % 256).astype(np.uint8)
    owner_vis[assigned, 1] = (owner_canvas[assigned] * 137 % 256).astype(np.uint8)
    owner_vis[assigned, 2] = (owner_canvas[assigned] * 221 % 256).astype(np.uint8)
    _save_rgb(os.path.join(out_dir, "13_stitched_owner_map.png"), owner_vis)

    metadata.update({
        "marker_thresh": int(effective_marker_thresh),
        "seg_positive_pixels": int(seg_positive.sum()),
        "marker_positive_pixels": int(marker_positive.sum()),
        "combined_positive_pixels": int(combined_positive.sum()),
        "positive_region_count": len(regions_info),
    })
    with open(os.path.join(out_dir, "stitched_deepliif_metadata.json"),
              "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata
