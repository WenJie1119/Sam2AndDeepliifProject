"""
cell/debug_vis.py -- Step-by-step debug visualization for single-tile debug mode.

Triggered by --debug-vis when --tile-index or --tile-um is given.
Each tile gets its own sub-directory under {output_dir}/debug_vis/{tile_name}/.

File naming convention:
    step1_original.png                 -- original RGB tile
    step2_{N}_deepliif_{KEY}.png       -- DeepLIIF outputs (Seg/Marker/DAPI/Hema/Lap2)
    step3_{N}_{sub}.png                -- Connected-region extraction intermediates
    step3_7_sam2_prompt_*.png          -- Final prompt regions sent to SAM2
    step4_sam2_raw_{M}inst.png         -- SAM2 raw instance mask overlay
    step4_sam2_steps/instance_XXX/...  -- per-prompt SAM2 details
    step5_merged_{K}inst.png           -- post-merge instance overlay
    step7_sam2_merge_diff.png          -- SAM2 raw vs merged diff
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image


__all__ = ["DebugVisualizer"]


def _colorize_instances(tile_np: np.ndarray, inst_mask: np.ndarray,
                        contour_thickness: int = 1) -> np.ndarray:
    """Overlay instance mask with per-ID distinct colors + contours."""
    vis = tile_np.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
    max_id = int(inst_mask.max()) if inst_mask.size else 0
    for inst_id in range(1, max_id + 1):
        pixels = inst_mask == inst_id
        if not np.any(pixels):
            continue
        color = np.array([(inst_id * 67) % 256,
                          (inst_id * 137) % 256,
                          (inst_id * 221) % 256], dtype=np.int32)
        vis[pixels] = (vis[pixels].astype(np.int32) * 0.4
                       + color * 0.6).clip(0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(pixels.astype(np.uint8),
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color.tolist(), contour_thickness)
    return vis


class DebugVisualizer:
    """Saves step-by-step artefacts for one tile under one directory."""

    def __init__(self, output_root: str, tile_name: str):
        stem = os.path.splitext(tile_name)[0]
        self.dir = os.path.join(output_root, "debug_vis", stem)
        os.makedirs(self.dir, exist_ok=True)
        self.tile_stem = stem
        print(f"  [debug-vis] Directory: {self.dir}")

    # ----- shortcuts ------------------------------------------------------
    def _path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def _save_rgb(self, name: str, img_rgb: np.ndarray) -> None:
        cv2.imwrite(self._path(name), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    def _save_gray(self, name: str, img_gray: np.ndarray) -> None:
        cv2.imwrite(self._path(name), img_gray)

    def _remove_prefixed_pngs(self, prefix: str) -> None:
        for name in os.listdir(self.dir):
            if name.startswith(prefix) and name.endswith(".png"):
                os.remove(self._path(name))

    # ----- step 1 ---------------------------------------------------------
    def step1_original(self, tile_np: np.ndarray) -> None:
        Image.fromarray(tile_np[:, :, :3]).save(self._path("step1_original.png"))

    # ----- step 2 ---------------------------------------------------------
    def step2_deepliif(self, dl_result: dict) -> None:
        """Save DeepLIIF outputs. Order: Seg, Marker, DAPI, Hema, Lap2."""
        order = ["Seg", "Marker", "DAPI", "Hema", "Lap2"]
        idx = 1
        for key in order:
            img = dl_result.get(key)
            if img is None:
                continue
            fname = f"step2_{idx}_deepliif_{key}.png"
            if isinstance(img, Image.Image):
                img.save(self._path(fname))
            elif isinstance(img, np.ndarray):
                if img.ndim == 2:
                    cv2.imwrite(self._path(fname), img)
                else:
                    cv2.imwrite(self._path(fname),
                                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            idx += 1

    # ----- step 3 ---------------------------------------------------------
    def step3_connected_region(self, tile_np: np.ndarray,
                               seg_np: np.ndarray, marker_np: np.ndarray,
                               cells_info: list,
                               seg_thresh: int, marker_thresh: Optional[int],
                               marker_percentile_factor: float,
                               morphology_kernel: int) -> None:
        """
        Re-compute every intermediate of extract_connected_positive_regions()
        purely for visualization. The actual pipeline runs unchanged.
        """
        from cd34_pipeline.cell.extraction import (
            compute_marker_threshold,
            compute_posneg_mask,
        )

        if marker_np.ndim == 3:
            marker_gray = cv2.cvtColor(marker_np, cv2.COLOR_RGB2GRAY)
        else:
            marker_gray = marker_np.copy()

        posneg_mask, is_foreground, _ = compute_posneg_mask(seg_np, seg_thresh)
        seg_positive = (posneg_mask == 2)

        if marker_thresh is None:
            marker_thresh = compute_marker_threshold(
                marker_gray, percentile_factor=marker_percentile_factor)
        marker_positive = is_foreground & (marker_gray > marker_thresh)

        h, w = seg_np.shape[:2]

        # 3_1 foreground (Seg)
        self._save_gray("step3_1_seg_foreground.png",
                        (is_foreground.astype(np.uint8) * 255))

        # 3_2 posneg pixels
        posneg_vis = np.zeros((h, w, 3), dtype=np.uint8)
        posneg_vis[posneg_mask == 2] = [255, 0, 0]   # positive = red
        posneg_vis[posneg_mask == 1] = [0, 0, 255]   # negative = blue
        self._save_rgb("step3_2_posneg_pixels.png", posneg_vis)

        # 3_3 seg positive only (green)
        seg_pos_vis = np.zeros((h, w, 3), dtype=np.uint8)
        seg_pos_vis[seg_positive] = [0, 255, 0]
        self._save_rgb("step3_3_seg_positive.png", seg_pos_vis)

        # 3_4 marker threshold comparison
        # green = Seg positive and Marker positive
        # red = Seg positive but Marker negative
        # yellow = Marker positive outside Seg positive
        marker_vis = np.zeros((h, w, 3), dtype=np.uint8)
        marker_vis[seg_positive & ~marker_positive] = [255, 0, 0]
        marker_vis[marker_positive & ~seg_positive] = [255, 255, 0]
        marker_vis[seg_positive & marker_positive] = [0, 255, 0]
        self._save_rgb("step3_4_marker_enhanced.png", marker_vis)

        # 3_5 combined positive (Seg and Marker intersection)
        combined = (seg_positive & marker_positive).astype(np.uint8) * 255
        self._save_gray("step3_5_combined_positive.png", combined)

        # 3_6 morphology close (kernel ellipse, 2 iterations) -> open 3x3
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morphology_kernel, morphology_kernel))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        self._save_gray("step3_6_morph_close.png", opened)
        self._remove_prefixed_pngs("step3_7_connected_regions_")

        n_regions = len(cells_info)

        # 3_8 positive cells overlay on original tile (with centroid labels)
        cells_overlay = tile_np[:, :, :3].copy()
        for r in cells_info:
            coords = r.get('coords')
            if coords is None or len(coords) == 0:
                continue
            rows_i, cols_i = coords[:, 0], coords[:, 1]
            cells_overlay[rows_i, cols_i] = (
                cells_overlay[rows_i, cols_i].astype(np.float32) * 0.5
                + np.array([0, 255, 0], dtype=np.float32) * 0.5
            ).astype(np.uint8)
            cy, cx = r.get('center', (0, 0))
            cv2.putText(cells_overlay, str(r.get('id', '')),
                        (int(cx) - 4, int(cy) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
        self._save_rgb(f"step3_8_positive_cells_{n_regions}.png", cells_overlay)

    def step3_sam2_prompt(self, tile_np: np.ndarray,
                          cells_info: list,
                          target_size: int = 256) -> None:
        """Save the final mask prompts that are sent to SAM2."""
        if tile_np.ndim == 2:
            overlay = cv2.cvtColor(tile_np, cv2.COLOR_GRAY2RGB)
        else:
            overlay = tile_np[:, :, :3].copy()

        h, w = overlay.shape[:2]
        label_mask = np.zeros((h, w), dtype=np.uint16)
        prompt_mask = np.zeros((target_size, target_size), dtype=np.uint8)
        n_prompts = len(cells_info)
        self._remove_prefixed_pngs("step3_7_sam2_prompt_regions_")

        for prompt_id, cell in enumerate(cells_info, start=1):
            coords = cell.get('coords')
            if coords is None or len(coords) == 0:
                continue
            rows = coords[:, 0].astype(np.intp)
            cols = coords[:, 1].astype(np.intp)
            valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
            rows = rows[valid]
            cols = cols[valid]
            if rows.size == 0:
                continue

            label_mask[rows, cols] = prompt_id
            low_rows = (rows * target_size / h).astype(np.intp)
            low_cols = (cols * target_size / w).astype(np.intp)
            low_valid = (
                (low_rows >= 0) & (low_rows < target_size) &
                (low_cols >= 0) & (low_cols < target_size)
            )
            prompt_mask[low_rows[low_valid], low_cols[low_valid]] = 255

        prompt_pixels = label_mask > 0
        overlay[prompt_pixels] = (
            overlay[prompt_pixels].astype(np.float32) * 0.45
            + np.array([0, 255, 0], dtype=np.float32) * 0.55
        ).clip(0, 255).astype(np.uint8)

        for prompt_id, cell in enumerate(cells_info, start=1):
            pixels = (label_mask == prompt_id).astype(np.uint8)
            if not np.any(pixels):
                continue
            contours, _ = cv2.findContours(
                pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 0), 1)
            center_y, center_x = cell.get('center', (0, 0))
            label = str(cell.get('original_id', cell.get('id', prompt_id)))
            cv2.putText(overlay, label,
                        (int(center_x) - 4, int(center_y) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)

        self._save_rgb(f"step3_7_sam2_prompt_regions_{n_prompts}.png", overlay)
        self._save_gray("step3_7_sam2_prompt_mask_256.png", prompt_mask)

    # ----- step 4 ---------------------------------------------------------
    def sam2_steps_parent(self) -> str:
        """Return the directory that run_sam2_segmentation() should treat as
        its `debug_dir`. It will create `{this}/sam2_steps/` internally."""
        return self.dir

    def step4_sam2_raw(self, tile_np: np.ndarray, sam_mask: np.ndarray) -> None:
        n_inst = int(sam_mask.max()) if sam_mask.size else 0
        vis = _colorize_instances(tile_np, sam_mask, contour_thickness=1)
        self._save_rgb(f"step4_sam2_raw_{n_inst}inst.png", vis)

    # ----- step 5 ---------------------------------------------------------
    def step5_merged(self, tile_np: np.ndarray, merged_mask: np.ndarray) -> None:
        n_inst = int(merged_mask.max()) if merged_mask.size else 0
        vis = _colorize_instances(tile_np, merged_mask, contour_thickness=2)
        self._save_rgb(f"step5_merged_{n_inst}inst.png", vis)

    # ----- step 7 ---------------------------------------------------------
    def step7_sam2_merge_diff(self, tile_np: np.ndarray,
                              sam_mask: np.ndarray, merged_mask: np.ndarray,
                              cells_info: list) -> None:
        # Inline implementation matching the old save_sam2_merge_diff.
        diff_vis = tile_np[:, :, :3].copy()
        removed = (sam_mask > 0) & (merged_mask == 0)
        kept = merged_mask > 0

        diff_vis[removed] = [255, 0, 0]
        diff_vis[kept] = (
            diff_vis[kept].astype(np.float32) * 0.6
            + np.array([0, 255, 0], dtype=np.float32) * 0.4
        ).astype(np.uint8)

        for cell in cells_info:
            center_y, center_x = cell['center']
            label = str(cell.get('original_id', cell.get('id', '')))
            origin = (int(center_x) - 5, int(center_y) + 5)
            cv2.putText(diff_vis, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        self._save_rgb("step7_sam2_merge_diff.png", diff_vis)
