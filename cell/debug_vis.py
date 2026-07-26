"""
cell/debug_vis.py -- Step-by-step tile visualization for region debug mode.

Triggered by --debug-region-um. Each tile gets its own sub-directory under
{output_dir}/debug_vis/{tile_name}/.

File naming convention:
    step1_original.png                 -- original RGB tile
    step2_{NN}_deepliif_{KEY}.png      -- DeepLIIF outputs (Seg/Marker/DAPI)
    step2_03_* .. step2_11_*           -- threshold and source-tile curves
    step3_{NN}_weighted_*.png/json     -- weighted-prompt construction flow
    step4_sam2_raw_{M}inst.png         -- SAM2 raw instance mask overlay
    step4_sam2_steps/instance_XXX/...  -- per-prompt SAM2 details
    step5_{NN}_merge_filter_*.png/json -- SAM2 merge/filter intermediates
    step5_merged_{K}inst.png           -- post-merge instance overlay
    step7_sam2_merge_diff.png          -- SAM2 raw vs merged diff
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from threading import Lock
from typing import Optional

import cv2
import numpy as np
from PIL import Image


__all__ = ["DebugVisualizer", "compute_seg_positive_r_histogram"]


_PLOT_LOCK = Lock()
_MIN_PEAK_PROMINENCE_FRACTION = 0.01


def _curve_extrema(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the peak and trough indices for every distinct wave.

    Flat extrema are represented by the midpoint of the plateau.  The outer
    zero-only tails are excluded. A small prominence threshold prevents
    one-bin histogram noise from being treated as a separate wave; each trough
    is the lowest point between two retained peaks.
    """
    support = np.flatnonzero(counts)
    if support.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    start = int(support[0])
    stop = int(support[-1]) + 1
    supported_counts = counts[start:stop]
    if supported_counts.size < 3:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    # Split the curve into constant-value runs. Comparing each run with the
    # values immediately on either side detects both point and plateau extrema.
    run_starts = np.r_[
        0, np.flatnonzero(np.diff(supported_counts) != 0) + 1]
    run_stops = np.r_[run_starts[1:], supported_counts.size]
    peaks: list[int] = []
    for run_index in range(1, len(run_starts) - 1):
        run_start = int(run_starts[run_index])
        run_stop = int(run_stops[run_index])
        value = supported_counts[run_start]
        left = supported_counts[run_start - 1]
        right = supported_counts[run_stop]
        midpoint = start + (run_start + run_stop - 1) // 2
        if value > left and value > right:
            peaks.append(midpoint)

    minimum_prominence = max(
        1.0,
        float(np.ptp(supported_counts))
        * _MIN_PEAK_PROMINENCE_FRACTION,
    )
    retained_peaks: list[int] = []
    for peak in peaks:
        local_peak = peak - start
        peak_value = supported_counts[local_peak]

        left_minimum = peak_value
        cursor = local_peak - 1
        while cursor >= 0 and supported_counts[cursor] <= peak_value:
            left_minimum = min(left_minimum, supported_counts[cursor])
            cursor -= 1

        right_minimum = peak_value
        cursor = local_peak + 1
        while (cursor < supported_counts.size
               and supported_counts[cursor] <= peak_value):
            right_minimum = min(right_minimum, supported_counts[cursor])
            cursor += 1

        prominence = peak_value - max(left_minimum, right_minimum)
        if prominence >= minimum_prominence:
            retained_peaks.append(peak)

    troughs: list[int] = []
    for left_peak, right_peak in zip(retained_peaks, retained_peaks[1:]):
        between = counts[left_peak + 1:right_peak]
        if between.size == 0:
            continue
        minimum_offsets = np.flatnonzero(between == between.min())
        # Use the midpoint when the first minimum is a flat plateau.
        plateau_stop = 0
        while (plateau_stop + 1 < minimum_offsets.size
               and minimum_offsets[plateau_stop + 1]
               == minimum_offsets[plateau_stop] + 1):
            plateau_stop += 1
        midpoint = int(
            (minimum_offsets[0] + minimum_offsets[plateau_stop]) // 2)
        troughs.append(left_peak + 1 + midpoint)

    return (np.asarray(retained_peaks, dtype=np.int64),
            np.asarray(troughs, dtype=np.int64))


def _global_curve_extrema(
        counts: np.ndarray) -> tuple[Optional[int], Optional[int]]:
    """Return global extrema for the legacy scalar result fields."""
    support = np.flatnonzero(counts)
    if support.size == 0:
        return None, None
    start = int(support[0])
    stop = int(support[-1]) + 1
    supported_counts = counts[start:stop]
    return (start + int(np.argmax(supported_counts)),
            start + int(np.argmin(supported_counts)))


def _annotate_curve_extrema(ax, intensities: np.ndarray, counts: np.ndarray,
                            peaks: np.ndarray,
                            troughs: np.ndarray) -> None:
    """Mark every local peak and trough directly on a curve plot."""
    extrema = [
        (peaks, "Peak", "#b2182b", "^"),
        (troughs, "Trough", "#2166ac", "v"),
    ]
    for indices, label, color, marker in extrema:
        if indices.size == 0:
            continue
        ax.scatter(
            intensities[indices], counts[indices],
            s=40, marker=marker, color=color, edgecolor="white",
            linewidth=0.7, label=f"{label}s ({indices.size})", zorder=4,
        )

    # Keep the plot legible by using compact labels and alternating their
    # horizontal direction. The markers themselves identify every extremum.
    combined = [
        (int(index), label, color, ordinal)
        for indices, label, color, _ in extrema
        for ordinal, index in enumerate(indices)
    ]
    for index, label, color, ordinal in combined:
        x = int(intensities[index])
        y = int(counts[index])
        x_span = int(intensities[-1]) - int(intensities[0])
        near_right_edge = x >= int(intensities[-1]) - max(1, x_span // 12)
        align_right = near_right_edge or ordinal % 2 == 1
        is_peak = label.startswith("Peak")
        ax.annotate(
            f"{label[0]}({x}, {y:,})",
            xy=(x, y),
            xytext=(-5 if align_right else 5, -9 if is_peak else 9),
            textcoords="offset points",
            ha="right" if align_right else "left",
            va="top" if is_peak else "bottom",
            fontsize=7,
            color=color,
            bbox={"facecolor": "white", "alpha": 0.72,
                  "edgecolor": "none", "pad": 0.8},
            zorder=5,
        )


def _load_matplotlib_pyplot():
    """Load pyplot with a writable cache dir for threaded debug workers."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cd34_matplotlib")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def compute_seg_positive_r_histogram(
        seg_np: np.ndarray, seg_thresh: int) -> tuple[np.ndarray, np.ndarray]:
    """Count R intensities for pixels classified as positive in a Seg image."""
    if seg_np.ndim != 3 or seg_np.shape[2] < 3:
        raise ValueError("DeepLIIF Seg image must be an RGB array")

    r_channel = seg_np[:, :, 0].astype(np.int16)
    g_channel = seg_np[:, :, 1].astype(np.int16)
    b_channel = seg_np[:, :, 2].astype(np.int16)
    positive_mask = (
        (r_channel + b_channel > seg_thresh)
        & (g_channel <= 80)
        & (r_channel >= b_channel)
    )
    counts = np.bincount(
        r_channel[positive_mask], minlength=256
    )[:256].astype(np.int64)
    return counts, positive_mask


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
        print(f"  [debug-tile] Directory: {self.dir}")

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

    def _remove_prefixed_files(self, prefix: str) -> None:
        for name in os.listdir(self.dir):
            path = self._path(name)
            if os.path.isfile(path) and name.startswith(prefix):
                os.remove(path)

    def clear_step5_merge_outputs(self) -> None:
        """Remove stale SAM2 merge/filter debug artifacts before rewriting."""
        for name in os.listdir(self.dir):
            path = self._path(name)
            if os.path.isdir(path) and name in {"merge_steps", "step5_merge_steps"}:
                shutil.rmtree(path)
            elif os.path.isfile(path) and name.startswith("step5_"):
                os.remove(path)

    def clear_downstream_outputs(self) -> None:
        """Remove stale prompt/SAM2/merge artifacts when a tile is skipped."""
        for name in os.listdir(self.dir):
            path = self._path(name)
            if os.path.isdir(path) and name in {
                "step4_sam2_steps",
                "merge_steps",
                "step5_merge_steps",
            }:
                shutil.rmtree(path)
            elif os.path.isfile(path) and (
                name.startswith("step3_")
                or name.startswith("step4_")
                or name.startswith("step5_")
                or name.startswith("step7_")
            ):
                os.remove(path)

    # ----- step 1 ---------------------------------------------------------
    def step1_original(self, tile_np: np.ndarray) -> None:
        Image.fromarray(tile_np[:, :, :3]).save(self._path("step1_original.png"))

    # ----- step 2 ---------------------------------------------------------
    def step2_deepliif(self, dl_result: dict) -> None:
        """Save only DeepLIIF outputs used downstream. Order: Seg, Marker, DAPI."""
        disabled_keys = {"Hema", "Lap2"}
        for stale_prefix in (
            "step2_1_deepliif_",
            "step2_01_deepliif_",
            "step2_2_deepliif_",
            "step2_02_deepliif_",
        ):
            self._remove_prefixed_pngs(stale_prefix)
        for name in os.listdir(self.dir):
            if not (name.endswith(".png") and "_deepliif_" in name):
                continue
            key = name.rsplit("_deepliif_", 1)[1][:-4]
            if key in disabled_keys:
                os.remove(self._path(name))

        order = ["Seg", "Marker", "DAPI"]
        idx = 1
        for key in order:
            img = dl_result.get(key)
            if img is None:
                continue
            fname = f"step2_{idx:02d}_deepliif_{key}.png"
            if isinstance(img, Image.Image):
                img.save(self._path(fname))
            elif isinstance(img, np.ndarray):
                if img.ndim == 2:
                    cv2.imwrite(self._path(fname), img)
                else:
                    cv2.imwrite(self._path(fname),
                                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            idx += 1

    def step2_seg_positive_r_intensity(self, seg_np: np.ndarray,
                                       seg_thresh: int) -> dict:
        """Save the R-intensity distribution of Seg-positive pixels."""
        counts, positive_mask = compute_seg_positive_r_histogram(
            seg_np, seg_thresh)
        intensities = np.arange(256, dtype=np.int16)
        total = int(counts.sum())
        positive_r_support = np.flatnonzero(counts)
        min_r_intensity = (
            int(positive_r_support[0]) if positive_r_support.size else None
        )
        plot_intensities = intensities[150:]
        plot_counts = counts[150:]
        peak_indices, trough_indices = _curve_extrema(plot_counts)
        peak_index, trough_index = _global_curve_extrema(plot_counts)
        peak_intensities = plot_intensities[peak_indices].astype(int).tolist()
        trough_intensities = (
            plot_intensities[trough_indices].astype(int).tolist())
        peak_intensity = (
            int(plot_intensities[peak_index])
            if peak_index is not None else None
        )
        trough_intensity = (
            int(plot_intensities[trough_index])
            if trough_index is not None else None
        )

        self._remove_prefixed_files("step2_6_seg_positive_r_intensity_")
        self._remove_prefixed_files("step2_03_seg_positive_r_intensity_")
        self._remove_prefixed_files("step2_06_seg_positive_r_intensity_")
        self._remove_prefixed_pngs("step2_7_deepliif_Seg_filtered_")
        self._remove_prefixed_pngs("step2_07_deepliif_Seg_filtered_")
        self._remove_prefixed_pngs("step2_8_seg_positive_only_")
        self._remove_prefixed_pngs("step2_08_seg_positive_only_")

        csv_name = "step2_03_seg_positive_r_intensity_counts.csv"
        with open(self._path(csv_name), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["r_intensity", "pixel_count"])
            writer.writerows(zip(intensities.tolist(), counts.tolist()))

        # Debug tiles are processed by worker threads, while matplotlib is not
        # thread-safe. Serialize plot creation without slowing the main flow.
        with _PLOT_LOCK:
            plt = _load_matplotlib_pyplot()

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(
                plot_intensities, plot_counts,
                color="#d62728", linewidth=1.6,
            )
            ax.fill_between(
                plot_intensities, plot_counts,
                color="#d62728", alpha=0.12,
            )
            _annotate_curve_extrema(
                ax, plot_intensities, plot_counts,
                peak_indices, trough_indices)
            if min_r_intensity is not None and min_r_intensity >= 150:
                ax.axvline(
                    min_r_intensity,
                    color="#4d4d4d",
                    linestyle=":",
                    linewidth=1.4,
                    label=f"Min R: {min_r_intensity}",
                )
            ax.set_xlim(150, 255)
            ax.set_xticks(np.arange(150, 256, 10))
            ax.margins(y=0.15)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("R intensity")
            ax.set_ylabel("Pixel count")
            ax.set_title("DeepLIIF Seg-positive R intensity distribution")
            ax.text(
                0.99, 0.97,
                f"Positive rule: R+B>{seg_thresh}, G<=80, R>=B\n"
                f"Positive pixels: {total:,}\n"
                f"Min R under rule: "
                f"{min_r_intensity if min_r_intensity is not None else 'n/a'}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85,
                      "edgecolor": "#cccccc"},
            )
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper left")
            fig.tight_layout()
            fig.savefig(
                self._path("step2_03_seg_positive_r_intensity_curve.png"),
                dpi=150,
            )
            plt.close(fig)

        return {
            "positive_pixel_count": total,
            "min_r_intensity": min_r_intensity,
            "peak_r_intensity": peak_intensity,
            "trough_r_intensity": trough_intensity,
            "peak_r_intensities": peak_intensities,
            "trough_r_intensities": trough_intensities,
            "positive_mask": positive_mask,
        }

    def step2_marker_intensity(self, marker_np: np.ndarray,
                               marker_thresh: Optional[int],
                               marker_percentile_factor: float = 1.0) -> dict:
        """Save the non-zero Marker grayscale-intensity distribution."""
        from cd34_pipeline.cell.extraction import (
            compute_marker_range_otsu_threshold,
            compute_marker_two_stage_multi_otsu_details,
            enforce_marker_min_keep_threshold,
        )

        if marker_np.ndim == 3 and marker_np.shape[2] >= 3:
            marker_gray = cv2.cvtColor(
                marker_np[:, :, :3], cv2.COLOR_RGB2GRAY)
        elif marker_np.ndim == 2:
            marker_gray = marker_np
        else:
            raise ValueError("DeepLIIF Marker image must be grayscale or RGB")

        marker_gray = np.clip(marker_gray, 0, 255).astype(np.uint8)
        nonzero_values = marker_gray[marker_gray > 0]
        counts = np.bincount(nonzero_values, minlength=256)[:256].astype(
            np.int64)
        intensities = np.arange(256, dtype=np.int16)
        total = int(nonzero_values.size)
        plot_intensities = intensities
        plot_counts = counts
        peak_indices, trough_indices = _curve_extrema(plot_counts)
        peak_index, trough_index = _global_curve_extrema(plot_counts)
        peak_intensities = plot_intensities[peak_indices].astype(int).tolist()
        trough_intensities = (
            plot_intensities[trough_indices].astype(int).tolist())
        peak_intensity = (
            int(plot_intensities[peak_index])
            if peak_index is not None else None
        )
        trough_intensity = (
            int(plot_intensities[trough_index])
            if trough_index is not None else None
        )
        effective_thresh = marker_thresh
        threshold_source = "fixed"
        two_stage_details = compute_marker_two_stage_multi_otsu_details(
            marker_gray)
        if effective_thresh is None:
            effective_thresh = two_stage_details["keep_threshold"]
            threshold_source = "auto_two_stage_multiotsu"
        effective_thresh = enforce_marker_min_keep_threshold(effective_thresh)
        effective_keep_min_intensity = int(effective_thresh + 1)

        multiotsu_low_thresh, multiotsu_high_thresh = (
            two_stage_details["outer_thresholds"])
        multiotsu_mid_split_thresh = compute_marker_range_otsu_threshold(
            marker_gray,
            multiotsu_low_thresh + 1,
            multiotsu_high_thresh,
        )
        middle_low_thresh, middle_high_thresh = (
            two_stage_details["middle_thresholds"])
        retained_mask = marker_gray > effective_thresh
        multiotsu_mid_high_mask = marker_gray > multiotsu_low_thresh
        multiotsu_high_mask = marker_gray > multiotsu_high_thresh
        for stale_prefix in (
            "step2_9_marker_nonzero_intensity_",
            "step2_04_marker_nonzero_intensity_",
            "step2_09_marker_nonzero_intensity_",
            "step2_05_deepliif_Marker_filtered_",
            "step2_10_deepliif_Marker_filtered_",
            "step2_06_marker_positive_mask_",
            "step2_06a_marker_multiotsu_",
            "step2_06b_marker_positive_mask_multiotsu_",
            "step2_06c_deepliif_Marker_filtered_multiotsu_",
            "step2_06d_marker_multiotsu_mid_high_color_",
            "step2_06e_marker_positive_mask_multiotsu_mid_split_",
            "step2_06f_deepliif_Marker_filtered_multiotsu_mid_split_",
            "step2_06g_marker_multiotsu_mid_3class_color_",
            "step2_06h_marker_positive_mask_multiotsu_mid_3class_upper2_",
            "step2_06i_marker_positive_mask_multiotsu_mid_3class_top_",
            "step2_11_marker_positive_mask_",
            "step2_06_deepliif_Marker_filtered_out_",
        ):
            self._remove_prefixed_files(stale_prefix)

        csv_name = "step2_04_marker_nonzero_intensity_counts.csv"
        with open(self._path(csv_name), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["marker_intensity", "nonzero_pixel_count"])
            writer.writerows(zip(intensities.tolist(), counts.tolist()))

        with _PLOT_LOCK:
            plt = _load_matplotlib_pyplot()

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(
                plot_intensities, plot_counts,
                color="#ff7f0e", linewidth=1.6,
            )
            ax.fill_between(
                plot_intensities, plot_counts,
                color="#ff7f0e", alpha=0.12,
            )
            _annotate_curve_extrema(
                ax, plot_intensities, plot_counts,
                peak_indices, trough_indices)
            ax.axvline(
                effective_thresh,
                color="#1f77b4",
                linestyle="--",
                linewidth=1.5,
                label=f"Marker keep: >= {effective_keep_min_intensity}",
            )
            ax.axvline(
                multiotsu_low_thresh,
                color="#2ca02c",
                linestyle=":",
                linewidth=1.8,
                label=f"Multi-Otsu low/mid: {multiotsu_low_thresh}",
            )
            ax.axvline(
                multiotsu_high_thresh,
                color="#d62728",
                linestyle=":",
                linewidth=1.8,
                label=f"Multi-Otsu mid/high: {multiotsu_high_thresh}",
            )
            ax.axvline(
                multiotsu_mid_split_thresh,
                color="#9467bd",
                linestyle="-.",
                linewidth=1.4,
                label=f"Mid split: {multiotsu_mid_split_thresh}",
            )
            ax.axvline(
                middle_low_thresh,
                color="#bcbd22",
                linestyle="--",
                linewidth=1.2,
                label=f"Mid 3-class low/mid: {middle_low_thresh}",
            )
            ax.axvline(
                middle_high_thresh,
                color="#17becf",
                linestyle="--",
                linewidth=1.2,
                label=f"Mid 3-class mid/high: {middle_high_thresh}",
            )
            ax.set_xlim(0, 255)
            ax.margins(y=0.15)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Marker grayscale intensity")
            ax.set_ylabel("Non-zero pixel count")
            ax.set_title("DeepLIIF Marker non-zero intensity distribution")
            ax.text(
                0.99, 0.03,
                f"Threshold source: {threshold_source}\n"
                f"Retained rule: marker >= {effective_keep_min_intensity}\n"
                f"Multi-Otsu mid+high: marker > {multiotsu_low_thresh}\n"
                f"Mid split: {multiotsu_mid_split_thresh}\n"
                f"Mid 3-class: {middle_low_thresh}, {middle_high_thresh}\n"
                f"Non-zero pixels: {total:,}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85,
                      "edgecolor": "#cccccc"},
            )
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(
                self._path("step2_04_marker_nonzero_intensity_curve.png"),
                dpi=150,
            )
            plt.close(fig)

        removed_mask = (marker_gray > 0) & ~retained_mask
        if marker_np.ndim == 3:
            filtered_marker = marker_np[:, :, :3].copy()
            filtered_marker[~retained_mask] = 0
            self._save_rgb(
                "step2_05_deepliif_Marker_filtered_"
                f"ge{effective_keep_min_intensity}.png",
                filtered_marker,
            )
            removed_marker = marker_np[:, :, :3].copy()
            removed_marker[~removed_mask] = 0
            self._save_rgb(
                "step2_06_deepliif_Marker_filtered_out_"
                f"lt{effective_keep_min_intensity}.png",
                removed_marker,
            )
        else:
            filtered_marker = marker_gray.copy()
            filtered_marker[~retained_mask] = 0
            self._save_gray(
                "step2_05_deepliif_Marker_filtered_"
                f"ge{effective_keep_min_intensity}.png",
                filtered_marker,
            )
            removed_marker = marker_gray.copy()
            removed_marker[~removed_mask] = 0
            self._save_gray(
                "step2_06_deepliif_Marker_filtered_out_"
                f"lt{effective_keep_min_intensity}.png",
                removed_marker,
            )
        mid_low_mask = (
            (marker_gray > multiotsu_low_thresh)
            & (marker_gray <= multiotsu_mid_split_thresh)
        )
        mid_high_mask = (
            (marker_gray > multiotsu_mid_split_thresh)
            & (marker_gray <= multiotsu_high_thresh)
        )
        multiotsu_mid_split_keep_mask = marker_gray > multiotsu_mid_split_thresh
        middle_low_mask = (
            (marker_gray > multiotsu_low_thresh)
            & (marker_gray <= middle_low_thresh)
        )
        middle_mid_mask = (
            (marker_gray > middle_low_thresh)
            & (marker_gray <= middle_high_thresh)
        )
        middle_high_mask = (
            (marker_gray > middle_high_thresh)
            & (marker_gray <= multiotsu_high_thresh)
        )
        middle_upper2_keep_mask = marker_gray > middle_low_thresh
        middle_top_keep_mask = marker_gray > middle_high_thresh

        return {
            "nonzero_pixel_count": total,
            "retained_marker_pixel_count": int(retained_mask.sum()),
            "removed_marker_pixel_count": int(
                np.count_nonzero(marker_gray) - retained_mask.sum()),
            "multiotsu_thresholds": [
                int(multiotsu_low_thresh),
                int(multiotsu_high_thresh),
            ],
            "multiotsu_mid_split_threshold": int(
                multiotsu_mid_split_thresh),
            "multiotsu_middle_3class_thresholds": [
                int(middle_low_thresh),
                int(middle_high_thresh),
            ],
            "multiotsu_mid_low_marker_pixel_count": int(
                mid_low_mask.sum()),
            "multiotsu_mid_high_only_marker_pixel_count": int(
                mid_high_mask.sum()),
            "multiotsu_mid_split_keep_marker_pixel_count": int(
                multiotsu_mid_split_keep_mask.sum()),
            "multiotsu_middle_low_marker_pixel_count": int(
                middle_low_mask.sum()),
            "multiotsu_middle_mid_marker_pixel_count": int(
                middle_mid_mask.sum()),
            "multiotsu_middle_high_marker_pixel_count": int(
                middle_high_mask.sum()),
            "multiotsu_middle_upper2_keep_marker_pixel_count": int(
                middle_upper2_keep_mask.sum()),
            "multiotsu_middle_top_keep_marker_pixel_count": int(
                middle_top_keep_mask.sum()),
            "multiotsu_mid_high_marker_pixel_count": int(
                multiotsu_mid_high_mask.sum()),
            "multiotsu_high_marker_pixel_count": int(
                multiotsu_high_mask.sum()),
            "peak_marker_intensity": peak_intensity,
            "trough_marker_intensity": trough_intensity,
            "peak_marker_intensities": peak_intensities,
            "trough_marker_intensities": trough_intensities,
            "marker_thresh": int(effective_thresh),
            "marker_effective_keep_min_intensity": (
                effective_keep_min_intensity),
            "marker_threshold_source": threshold_source,
        }

    def step2_dapi_dark_intensity(self, dapi_np: np.ndarray,
                                  dark_max: int) -> dict:
        """Save DAPI grayscale distribution and no-nucleus dark candidate mask."""
        if not 0 <= dark_max <= 255:
            raise ValueError("dark_max must be between 0 and 255")

        from cd34_pipeline.sam2_wrapper.weighted_prompt import (
            WeightedPromptConfig,
            _dapi_dark_mask,
            _dapi_gray,
        )

        dapi_gray = _dapi_gray(dapi_np)
        dark_mask, _ = _dapi_dark_mask(
            dapi_np, WeightedPromptConfig(dapi_lumen_dark_max=dark_max))

        intensities = np.arange(256, dtype=np.int16)
        counts = np.bincount(
            dapi_gray.reshape(-1), minlength=256)[:256].astype(np.int64)
        dark_counts = np.where(intensities <= dark_max, counts, 0)
        peak_indices, trough_indices = _curve_extrema(counts)
        peak_index, trough_index = _global_curve_extrema(counts)
        peak_intensity = (
            int(intensities[peak_index]) if peak_index is not None else None
        )
        trough_intensity = (
            int(intensities[trough_index]) if trough_index is not None else None
        )

        for stale_prefix in (
            "step2_07_dapi_",
            "step2_08_dapi_",
            "step2_09_near_white_",
            "step2_08_near_white_",
        ):
            self._remove_prefixed_files(stale_prefix)

        self._save_gray("step2_07_dapi_gray.png", dapi_gray)
        self._save_gray(
            "step2_07_dapi_dark_mask.png",
            dark_mask.astype(np.uint8) * 255,
        )

        csv_name = "step2_08_dapi_intensity_counts.csv"
        with open(self._path(csv_name), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dapi_intensity",
                "pixel_count",
                "dark_candidate_pixel_count",
            ])
            writer.writerows(zip(
                intensities.tolist(),
                counts.tolist(),
                dark_counts.tolist(),
            ))

        with _PLOT_LOCK:
            plt = _load_matplotlib_pyplot()

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(
                intensities, counts,
                color="#4c78a8", linewidth=1.6,
                label="DAPI grayscale pixels",
            )
            ax.fill_between(
                intensities,
                counts,
                where=intensities <= dark_max,
                color="#4c78a8",
                alpha=0.18,
                label="DAPI-dark candidate range",
            )
            _annotate_curve_extrema(
                ax, intensities, counts, peak_indices, trough_indices)
            ax.axvline(
                dark_max,
                color="#1f77b4",
                linestyle="--",
                linewidth=1.5,
                label=f"DAPI dark max: {dark_max}",
            )
            ax.set_xlim(0, 255)
            ax.margins(y=0.15)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("DAPI grayscale intensity")
            ax.set_ylabel("Pixel count")
            ax.set_title("DeepLIIF DAPI no-nucleus dark candidate distribution")
            ax.text(
                0.99, 0.97,
                f"Rule: DAPI <= {dark_max}\n"
                f"Dark candidate pixels: {int(dark_mask.sum()):,}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85,
                      "edgecolor": "#cccccc"},
            )
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(
                self._path("step2_08_dapi_intensity_curve.png"),
                dpi=150,
            )
            plt.close(fig)

        return {
            "dapi_dark_max": int(dark_max),
            "dapi_dark_pixel_count": int(dark_mask.sum()),
            "peak_dapi_intensity": peak_intensity,
            "trough_dapi_intensity": trough_intensity,
        }

    def step2_dab_intensity(
            self,
            tile_np: np.ndarray,
            seg_np: Optional[np.ndarray],
            config) -> dict:
        """Save DAB intensity and the final DAB/HSV/Seg-blue keep rule."""
        dab_min_intensity = int(config.dab_min_intensity)
        if not 0 <= dab_min_intensity <= 255:
            raise ValueError("dab_min_intensity must be between 0 and 255")

        from cd34_pipeline.sam2_wrapper.weighted_prompt import (
            _dab_filter_masks,
        )

        dab_debug, dab_stats = _dab_filter_masks(tile_np, seg_np, config)
        dab_u8 = dab_debug["dab_intensity"]
        hed_keep_mask = dab_debug["dab_hed_intensity_keep_mask"]
        hsv_brown_keep_mask = dab_debug["dab_hsv_brown_keep_mask"]
        seg_blue_excluded_mask = dab_debug["dab_seg_blue_excluded_mask"]
        keep_mask = dab_debug["dab_keep_mask"]
        norm_value = float(dab_stats["dab_normalization_value"])
        dab_normalization_percentile = float(
            config.dab_normalization_percentile)

        intensities = np.arange(256, dtype=np.int16)
        all_counts = np.bincount(
            dab_u8.reshape(-1), minlength=256)[:256].astype(np.int64)
        hed_keep_counts = np.bincount(
            dab_u8[hed_keep_mask].reshape(-1), minlength=256
        )[:256].astype(np.int64)
        hsv_brown_keep_counts = np.bincount(
            dab_u8[hsv_brown_keep_mask].reshape(-1), minlength=256
        )[:256].astype(np.int64)
        seg_blue_excluded_counts = np.bincount(
            dab_u8[seg_blue_excluded_mask].reshape(-1), minlength=256
        )[:256].astype(np.int64)
        keep_counts = np.bincount(
            dab_u8[keep_mask].reshape(-1), minlength=256
        )[:256].astype(np.int64)

        peak_indices, trough_indices = _curve_extrema(all_counts)
        peak_index, trough_index = _global_curve_extrema(all_counts)
        peak_intensities = intensities[peak_indices].astype(int).tolist()
        trough_intensities = (
            intensities[trough_indices].astype(int).tolist())
        peak_intensity = (
            int(intensities[peak_index])
            if peak_index is not None else None
        )
        trough_intensity = (
            int(intensities[trough_index])
            if trough_index is not None else None
        )

        for stale_prefix in (
            "step2_07_dab_brown_intensity_",
            "step2_09_dab_",
            "step2_10_dab_",
            "step2_11_dab_",
            "step2_12_dab_brown_intensity_",
        ):
            self._remove_prefixed_files(stale_prefix)

        self._save_gray("step2_09_dab_intensity.png", dab_u8)
        self._save_gray(
            "step2_09_dab_hed_keep_mask.png",
            hed_keep_mask.astype(np.uint8) * 255,
        )
        self._save_gray(
            "step2_09_hsv_brown_keep_mask.png",
            hsv_brown_keep_mask.astype(np.uint8) * 255,
        )
        self._save_gray(
            "step2_09_seg_blue_excluded_mask.png",
            seg_blue_excluded_mask.astype(np.uint8) * 255,
        )
        self._save_gray(
            "step2_09_dab_hsv_brown_intersection_mask.png",
            keep_mask.astype(np.uint8) * 255,
        )
        self._save_gray(
            "step2_09_dab_keep_mask.png",
            keep_mask.astype(np.uint8) * 255,
        )
        self._save_gray(
            "step2_09_dab_filtered_out_low_intensity.png",
            (~keep_mask).astype(np.uint8) * 255,
        )

        csv_name = "step2_10_dab_intensity_counts.csv"
        with open(self._path(csv_name), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dab_intensity",
                "all_pixel_count",
                "hed_dab_keep_pixel_count",
                "hsv_brown_keep_pixel_count",
                "seg_blue_excluded_pixel_count",
                "dab_keep_pixel_count",
            ])
            writer.writerows(zip(
                intensities.tolist(),
                all_counts.tolist(),
                hed_keep_counts.tolist(),
                hsv_brown_keep_counts.tolist(),
                seg_blue_excluded_counts.tolist(),
                keep_counts.tolist(),
            ))

        keep_total = int(keep_mask.sum())
        with _PLOT_LOCK:
            plt = _load_matplotlib_pyplot()

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(
                intensities, all_counts,
                color="#8c510a", linewidth=1.7,
                label="All DAB intensity pixels",
            )
            ax.fill_between(
                intensities,
                keep_counts,
                color="#bf812d",
                alpha=0.30,
                label="Retained by DAB + HSV-brown + non-blue",
            )
            if peak_index is not None:
                peak_x = int(intensities[peak_index])
                peak_y = int(all_counts[peak_index])
                ax.scatter(
                    [peak_x], [peak_y],
                    s=44, marker="^", color="#b2182b",
                    edgecolor="white", linewidth=0.7,
                    label="Global peak", zorder=4,
                )
                peak_align_right = peak_x >= 245
                ax.annotate(
                    f"Peak({peak_x}, {peak_y:,})",
                    xy=(peak_x, peak_y),
                    xytext=(-5 if peak_align_right else 5, -9),
                    textcoords="offset points",
                    ha="right" if peak_align_right else "left",
                    va="top",
                    fontsize=8,
                    color="#b2182b",
                    bbox={"facecolor": "white", "alpha": 0.72,
                          "edgecolor": "none", "pad": 0.8},
                    zorder=5,
                )
            if trough_index is not None:
                trough_x = int(intensities[trough_index])
                trough_y = int(all_counts[trough_index])
                ax.scatter(
                    [trough_x], [trough_y],
                    s=44, marker="v", color="#2166ac",
                    edgecolor="white", linewidth=0.7,
                    label="Global trough", zorder=4,
                )
            ax.axvline(
                dab_min_intensity,
                color="#1f77b4",
                linestyle="--",
                linewidth=1.5,
                label=f"DAB threshold: {dab_min_intensity}",
            )
            ax.set_xlim(0, 255)
            ax.margins(y=0.15)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Normalized DAB intensity")
            ax.set_ylabel("Pixel count")
            ax.set_title("Original tile normalized HED-DAB intensity")
            ax.text(
                0.99, 0.97,
                f"Final keep pixels: {keep_total:,}\n"
                f"HED-DAB keep pixels: {int(hed_keep_mask.sum()):,}\n"
                f"HSV brown pixels: {int(hsv_brown_keep_mask.sum()):,}\n"
                f"Seg-blue excluded: {int(seg_blue_excluded_mask.sum()):,}\n"
                f"Normalization: p{dab_normalization_percentile:g} "
                f"= {norm_value:.6g}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85,
                      "edgecolor": "#cccccc"},
            )
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper left")
            fig.tight_layout()
            fig.savefig(
                self._path("step2_10_dab_intensity_curve.png"),
                dpi=150,
            )
            plt.close(fig)

        return {
            "dab_intensity_keep_pixel_count": keep_total,
            "dab_hed_intensity_keep_pixel_count": int(
                hed_keep_mask.sum()),
            "dab_hsv_brown_keep_pixel_count": int(
                hsv_brown_keep_mask.sum()),
            "dab_seg_blue_excluded_pixel_count": int(
                seg_blue_excluded_mask.sum()),
            "dab_min_intensity": int(dab_min_intensity),
            "dab_normalization_percentile": float(
                dab_normalization_percentile),
            "dab_normalization_value": float(norm_value),
            "peak_dab_intensity": peak_intensity,
            "trough_dab_intensity": trough_intensity,
            "peak_dab_intensities": peak_intensities,
            "trough_dab_intensities": trough_intensities,
        }

    def step2_near_white_intensity(
            self,
            tile_np: np.ndarray,
            value_min: int,
            saturation_max: float,
            channel_delta_max: int) -> dict:
        """Save near-white RGB region mask and HSV value distribution."""
        if not 0 <= value_min <= 255:
            raise ValueError("value_min must be between 0 and 255")
        if not 0 <= saturation_max <= 1:
            raise ValueError("saturation_max must be in [0, 1]")
        if not 0 <= channel_delta_max <= 255:
            raise ValueError("channel_delta_max must be between 0 and 255")

        from cd34_pipeline.sam2_wrapper.weighted_prompt import (
            WeightedPromptConfig,
            _near_white_mask,
            _near_white_value_threshold_details,
            _rgb_float,
        )

        config = WeightedPromptConfig(
            dab_lumen_white_value_min=value_min,
            dab_lumen_white_saturation_max=saturation_max,
            dab_lumen_white_channel_delta_max=channel_delta_max,
        )
        rgb = np.round(_rgb_float(tile_np) * 255).astype(np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        value = hsv[:, :, 2].astype(np.int16)
        channel_max = rgb.max(axis=2).astype(np.int16)
        channel_min = rgb.min(axis=2).astype(np.int16)
        balanced_mask = (
            (saturation <= saturation_max)
            & ((channel_max - channel_min) <= channel_delta_max)
        )
        threshold_details = _near_white_value_threshold_details(
            value, balanced_mask, value_min)
        effective_value_min = int(threshold_details["threshold"])
        raw_near_white_mask = balanced_mask & (value >= effective_value_min)
        near_white_mask = _near_white_mask(tile_np, config)

        intensities = np.arange(256, dtype=np.int16)
        balanced_counts = np.bincount(
            value[balanced_mask], minlength=256)[:256].astype(np.int64)
        near_white_counts = np.where(
            intensities >= effective_value_min, balanced_counts, 0)
        peak_indices, trough_indices = _curve_extrema(balanced_counts)
        peak_index, trough_index = _global_curve_extrema(balanced_counts)
        peak_intensity = (
            int(intensities[peak_index])
            if peak_index is not None else None
        )
        trough_intensity = (
            int(intensities[trough_index])
            if trough_index is not None else None
        )

        for stale_prefix in (
            "step2_08_near_white_",
            "step2_09_near_white_",
        ):
            self._remove_prefixed_files(stale_prefix)

        self._save_gray(
            "step2_08_near_white_mask.png",
            near_white_mask.astype(np.uint8) * 255,
        )

        csv_name = "step2_09_near_white_value_intensity_counts.csv"
        with open(self._path(csv_name), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "value_intensity",
                "low_saturation_balanced_pixel_count",
                "near_white_pixel_count",
            ])
            writer.writerows(zip(
                intensities.tolist(),
                balanced_counts.tolist(),
                near_white_counts.tolist(),
            ))

        raw_total = int(raw_near_white_mask.sum())
        closed_total = int(near_white_mask.sum())
        balanced_total = int(balanced_mask.sum())
        with _PLOT_LOCK:
            plt = _load_matplotlib_pyplot()

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(
                intensities, balanced_counts,
                color="#5ab4ac", linewidth=1.7,
                label="Low-saturation balanced pixels",
            )
            ax.fill_between(
                intensities,
                balanced_counts,
                where=intensities >= effective_value_min,
                color="#5ab4ac",
                alpha=0.18,
                label="Retained by adaptive near-white threshold",
            )
            _annotate_curve_extrema(
                ax, intensities, balanced_counts,
                peak_indices, trough_indices)
            tail_thresholds = threshold_details["multiotsu_thresholds"]
            if threshold_details["peak_intensity"] is not None:
                ax.axvline(
                    int(threshold_details["peak_intensity"]),
                    color="#6a3d9a",
                    linestyle=":",
                    linewidth=1.3,
                    label=f"Dominant peak: "
                          f"{threshold_details['peak_intensity']}",
                )
            if tail_thresholds[0] is not None:
                ax.axvline(
                    int(tail_thresholds[0]),
                    color="#ff7f00",
                    linestyle=":",
                    linewidth=1.3,
                    label=f"Tail Multi-Otsu low/mid: {tail_thresholds[0]}",
                )
            ax.axvline(
                effective_value_min,
                color="#1f77b4",
                linestyle="--",
                linewidth=1.5,
                label=f"Tail Multi-Otsu mid/high: {effective_value_min}",
            )
            if effective_value_min != value_min:
                ax.axvline(
                    value_min,
                    color="#4d4d4d",
                    linestyle=":",
                    linewidth=1.1,
                    label=f"Configured fallback: {value_min}",
                )
            ax.set_xlim(0, 255)
            ax.margins(y=0.15)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("HSV value intensity")
            ax.set_ylabel("Pixel count")
            ax.set_title("Original tile near-white value distribution")
            ax.text(
                0.99, 0.97,
                f"Rule: V>={effective_value_min}, S<={saturation_max:g}, "
                f"RGB delta<={channel_delta_max}\n"
                f"Threshold source: {threshold_details['source']}\n"
                f"Balanced pixels: {balanced_total:,}\n"
                f"Raw near-white pixels: {raw_total:,}\n"
                f"Closed near-white pixels: {closed_total:,}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85,
                      "edgecolor": "#cccccc"},
            )
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper left")
            fig.tight_layout()
            fig.savefig(
                self._path("step2_09_near_white_value_intensity_curve.png"),
                dpi=150,
            )
            plt.close(fig)

        return {
            "balanced_near_white_candidate_pixel_count": balanced_total,
            "raw_near_white_pixel_count": raw_total,
            "near_white_pixel_count": closed_total,
            "near_white_value_min": int(effective_value_min),
            "near_white_configured_value_min": int(value_min),
            "near_white_threshold_source": threshold_details["source"],
            "near_white_peak_tail_multiotsu_thresholds": (
                threshold_details["multiotsu_thresholds"]),
            "near_white_saturation_max": float(saturation_max),
            "near_white_channel_delta_max": int(channel_delta_max),
            "peak_near_white_value_intensity": peak_intensity,
            "trough_near_white_value_intensity": trough_intensity,
        }

    # ----- step 3 ---------------------------------------------------------
    def step3_connected_region(self, tile_np: np.ndarray,
                               seg_np: np.ndarray, marker_np: np.ndarray,
                               cells_info: list,
                               seg_thresh: int,
                               marker_thresh: Optional[int],
                               marker_percentile_factor: float,
                               morphology_kernel: int,
                               masks: Optional[dict] = None) -> None:
        """
        Visualize the same masks used by extraction.
        """
        if masks is None:
            from cd34_pipeline.cell.extraction import (
                compute_positive_masks,
            )
            masks = compute_positive_masks(
                seg_np,
                marker_np,
                seg_thresh=seg_thresh,
                marker_thresh=marker_thresh,
                marker_percentile_factor=marker_percentile_factor,
            )
        seg_positive = masks["seg_positive"]
        marker_positive = masks["marker_positive"]

        h, w = seg_np.shape[:2]
        for stale_prefix in (
            "step3_1_",
            "step3_2_",
            "step3_3_",
            "step3_4_",
            "step3_04_",
            "step3_5_",
            "step3_05_",
            "step3_6_",
            "step3_06_",
            "step3_8_positive_cells_",
            "step3_07_positive_cells_",
        ):
            self._remove_prefixed_pngs(stale_prefix)

        # 3_04 marker threshold comparison
        # green = both; red = Seg only; yellow = supplementary Marker only.
        # All three categories enter the OR-combined downstream mask.
        marker_vis = np.zeros((h, w, 3), dtype=np.uint8)
        marker_vis[seg_positive & ~marker_positive] = [255, 0, 0]
        marker_vis[marker_positive & ~seg_positive] = [255, 255, 0]
        marker_vis[seg_positive & marker_positive] = [0, 255, 0]
        self._save_rgb("step3_04_marker_enhanced.png", marker_vis)

        # 3_05 combined positive (Seg OR Marker)
        combined = masks["combined_positive"].astype(np.uint8) * 255
        self._save_gray("step3_05_combined_positive.png", combined)

        # 3_06 morphology close (kernel ellipse, 2 iterations) -> open 3x3
        opened = masks.get("morph_opened")
        if opened is None:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (morphology_kernel, morphology_kernel))
            closed = cv2.morphologyEx(
                combined, cv2.MORPH_CLOSE, kernel, iterations=2)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        self._save_gray("step3_06_morph_close.png", opened)
        self._remove_prefixed_pngs("step3_7_connected_regions_")

        n_regions = len(cells_info)

        # 3_07 positive cells overlay on original tile (with centroid labels)
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
        self._save_rgb(f"step3_07_positive_cells_{n_regions}.png", cells_overlay)

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
        self._remove_prefixed_files("step3_7_sam2_prompt_")
        self._remove_prefixed_files("step3_08_sam2_prompt_")

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

        self._save_rgb(
            f"step3_08_sam2_prompt_01_regions_{n_prompts}.png", overlay)
        self._save_gray("step3_08_sam2_prompt_02_mask_256.png", prompt_mask)

    def step3_weighted_prompt(self, tile_np: np.ndarray, result) -> None:
        """Save the weighted mask and strong points actually sent to SAM2."""
        from cd34_pipeline.sam2_wrapper.weighted_prompt import colorize_logits

        self._remove_prefixed_files("step3_")
        pre_dab_raw = result.debug.get(
            "pre_dab_raw_logits", result.debug["raw_logits"])
        self._save_rgb("step3_01_weighted_raw_heatmap.png",
                       colorize_logits(pre_dab_raw))

        if result.stats.get("dab_filter_enabled"):
            self._save_gray(
                "step3_02_weighted_dab_intensity.png",
                result.debug["dab_intensity"],
            )
            self._save_gray(
                "step3_03_weighted_dab_intensity_keep_mask.png",
                result.debug["dab_intensity_keep_mask"].astype(np.uint8) * 255,
            )
            for key, filename in (
                ("dab_hed_intensity_keep_mask",
                 "step3_03a_weighted_dab_hed_keep_mask.png"),
                ("dab_hsv_brown_keep_mask",
                 "step3_03b_weighted_hsv_brown_keep_mask.png"),
                ("dab_seg_blue_excluded_mask",
                 "step3_03c_weighted_seg_blue_excluded_mask.png"),
            ):
                mask = result.debug.get(key)
                if mask is not None:
                    self._save_gray(filename, mask.astype(np.uint8) * 255)
            self._save_gray(
                "step3_04_weighted_dab_removed_prompt_mask.png",
                result.debug["dab_removed_mask"].astype(np.uint8) * 255,
            )
            self._save_rgb(
                "step3_05_weighted_dab_filtered_heatmap.png",
                colorize_logits(result.debug["dab_filtered_logits"]),
            )
            keep_overlay = tile_np[:, :, :3].copy()
            keep = (
                (result.debug["dab_keep_mask"]
                 | result.debug.get("dapi_lumen_protected_prompt_mask",
                                    np.zeros(pre_dab_raw.shape, dtype=bool)))
                & (pre_dab_raw >= 0)
            )
            keep_overlay[keep] = (
                keep_overlay[keep].astype(np.float32) * 0.45
                + np.array([0, 220, 0], dtype=np.float32) * 0.55
            ).clip(0, 255).astype(np.uint8)
            self._save_rgb(
                "step3_06_weighted_dab_filter_overlay.png",
                keep_overlay,
            )
            for key, filename in (
                ("dapi_dark_mask", "step3_07_weighted_dapi_dark_mask.png"),
                ("dapi_lumen_candidate_mask",
                 "step3_08_weighted_dapi_lumen_candidates.png"),
                ("dapi_lumen_accepted_mask",
                 "step3_09_weighted_dapi_lumen_accepted.png"),
                ("dapi_lumen_protected_prompt_mask",
                 "step3_10_weighted_dapi_lumen_protected_prompt.png"),
            ):
                mask = result.debug.get(key)
                if mask is not None:
                    self._save_gray(filename, mask.astype(np.uint8) * 255)
            if result.stats.get("dab_strong_support_enabled"):
                support_logits = result.debug.get("dab_support_logits")
                if support_logits is not None:
                    self._save_rgb(
                        "step3_11_weighted_dab_support_heatmap.png",
                        colorize_logits(support_logits),
                    )
                added_mask = result.debug.get("dab_added_mask")
                if added_mask is not None:
                    self._save_gray(
                        "step3_12_weighted_dab_added_prompt_mask.png",
                        added_mask.astype(np.uint8) * 255,
                    )
                context_mask = result.debug.get(
                    "dab_strong_support_context_mask")
                if context_mask is not None:
                    self._save_gray(
                        "step3_12a_weighted_dab_support_context_mask.png",
                        context_mask.astype(np.uint8) * 255,
                    )
                upgraded_mask = result.debug.get("dab_upgraded_mask")
                if upgraded_mask is not None:
                    self._save_gray(
                        "step3_13_weighted_dab_upgraded_prompt_mask.png",
                        upgraded_mask.astype(np.uint8) * 255,
                    )
                augmented_logits = result.debug.get("dab_augmented_logits")
                if augmented_logits is not None:
                    self._save_rgb(
                        "step3_14_weighted_dab_augmented_heatmap.png",
                        colorize_logits(augmented_logits),
                    )
            for key, filename in (
                ("dab_lumen_wall_mask",
                 "step3_14a_weighted_dab_lumen_wall.png"),
                ("dab_lumen_white_mask",
                 "step3_14b_weighted_dab_lumen_white.png"),
                ("dab_lumen_near_wall_mask",
                 "step3_14c_weighted_dab_lumen_near_wall.png"),
                ("dab_lumen_candidate_mask",
                 "step3_14d_weighted_dab_lumen_candidates.png"),
                ("dab_lumen_accepted_mask",
                 "step3_14e_weighted_dab_lumen_accepted.png"),
                ("dab_lumen_protected_prompt_mask",
                 "step3_14f_weighted_dab_lumen_protected_prompt.png"),
            ):
                mask = result.debug.get(key)
                if mask is not None:
                    self._save_gray(filename, mask.astype(np.uint8) * 255)

        raw = result.debug["raw_logits"]

        artifact_mask = result.debug["artifact_mask"]
        self._save_gray(
            "step3_15_weighted_artifact_mask.png",
            artifact_mask.astype(np.uint8) * 255,
        )
        decisions = tile_np[:, :, :3].copy()
        for component in result.stats.get("artifact_components", []):
            x, y, width, height = component["bbox_xywh"]
            selected = component["selected"]
            color = (255, 0, 0) if selected else (0, 210, 0)
            label = "DROP" if selected else "KEEP"
            cv2.rectangle(
                decisions, (x, y), (x + width - 1, y + height - 1),
                color, thickness=2)
            cv2.putText(
                decisions, f"{label} s={component['score']}",
                (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, color, 1, cv2.LINE_AA)
        self._save_rgb("step3_16_weighted_artifact_decisions.png", decisions)

        fragment_mask = result.debug["small_fragment_mask"]
        self._save_gray(
            "step3_17_weighted_small_fragments.png",
            fragment_mask.astype(np.uint8) * 255,
        )
        isolated_fragment_mask = result.debug.get("isolated_fragment_mask")
        if isolated_fragment_mask is not None:
            self._save_gray(
                "step3_18_weighted_isolated_fragments.png",
                isolated_fragment_mask.astype(np.uint8) * 255,
            )
        self._save_rgb(
            "step3_19_weighted_cleaned_heatmap.png",
            colorize_logits(result.debug["cleaned_logits"]),
        )
        self._save_rgb(
            "step3_20_weighted_final_heatmap.png",
            colorize_logits(result.logits),
        )

        overlay = tile_np[:, :, :3].copy()
        heatmap = colorize_logits(result.logits)
        active = result.logits >= 0
        overlay[active] = (
            overlay[active].astype(np.float32) * 0.45
            + heatmap[active].astype(np.float32) * 0.55
        ).clip(0, 255).astype(np.uint8)
        self._save_rgb("step3_21_weighted_final_overlay.png", overlay)
        mask_input_heatmap = colorize_logits(
            result.mask_input[0].astype(np.int16))
        self._save_rgb(
            "step3_22_weighted_mask_input_256.png",
            mask_input_heatmap,
        )
        dab_keep_mask = result.debug.get("dab_keep_mask")
        if dab_keep_mask is not None and np.any(dab_keep_mask):
            dab_keep_256 = cv2.resize(
                dab_keep_mask.astype(np.uint8),
                (mask_input_heatmap.shape[1], mask_input_heatmap.shape[0]),
                interpolation=cv2.INTER_AREA,
            ) > 0
            dab_overlay = mask_input_heatmap.copy()
            dab_overlay[dab_keep_256] = (
                dab_overlay[dab_keep_256].astype(np.float32) * 0.45
                + np.array([0, 255, 255], dtype=np.float32) * 0.55
            ).clip(0, 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                dab_keep_256.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(dab_overlay, contours, -1, (255, 255, 255), 1)
            self._save_rgb(
                "step3_23_weighted_mask_input_dab_keep_overlay_256.png",
                dab_overlay,
            )

        lumen_candidate = result.debug.get("lumen_point_candidate_mask_256")
        lumen_accepted = result.debug.get("lumen_point_accepted_mask_256")
        if lumen_candidate is not None and lumen_accepted is not None:
            self._save_gray(
                "step3_24_weighted_lumen_point_candidates_256.png",
                lumen_candidate.astype(np.uint8) * 255,
            )
            self._save_gray(
                "step3_25_weighted_lumen_point_accepted_256.png",
                lumen_accepted.astype(np.uint8) * 255,
            )
            lumen_overlay = colorize_logits(
                result.mask_input[0].astype(np.int16))
            accepted_index = 1
            skipped_index = 1
            rejected_index = 1
            for component in result.stats.get("lumen_point_components", []):
                x, y, width, height = component["bbox_xywh_256"]
                accepted = bool(component["accepted"])
                selected = bool(component.get("selected", accepted))
                if selected:
                    color = (0, 240, 0)
                    label = f"L{accepted_index}"
                    accepted_index += 1
                elif accepted:
                    color = (255, 170, 0)
                    label = f"S{skipped_index}"
                    skipped_index += 1
                else:
                    color = (255, 0, 0)
                    label = f"R{rejected_index}"
                    rejected_index += 1
                cv2.rectangle(
                    lumen_overlay,
                    (x, y),
                    (x + width - 1, y + height - 1),
                    color,
                    thickness=1,
                )
                px, py = component["point_xy_256"]
                cv2.circle(lumen_overlay, (px, py), 4, (255, 255, 255), 1)
                cv2.circle(lumen_overlay, (px, py), 2, color, -1)
                cv2.putText(
                    lumen_overlay,
                    label,
                    (px + 4, max(8, py - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.28,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            self._save_rgb(
                "step3_26_weighted_lumen_point_overlay_256.png",
                lumen_overlay,
            )

        dapi_lumen_candidate = result.debug.get("dapi_lumen_candidate_mask")
        dapi_lumen_accepted = result.debug.get("dapi_lumen_accepted_mask")
        if dapi_lumen_candidate is not None and dapi_lumen_accepted is not None:
            dapi_lumen_wall = result.debug.get("dapi_lumen_wall_mask")
            dapi_lumen_near_wall = result.debug.get("dapi_lumen_near_wall_mask")
            dapi_lumen_protected = result.debug.get(
                "dapi_lumen_protected_prompt_mask")
            self._save_gray(
                "step3_27_weighted_dapi_lumen_candidates.png",
                dapi_lumen_candidate.astype(np.uint8) * 255,
            )
            self._save_gray(
                "step3_28_weighted_dapi_lumen_accepted.png",
                dapi_lumen_accepted.astype(np.uint8) * 255,
            )
            if dapi_lumen_wall is not None:
                self._save_gray(
                    "step3_29_weighted_dapi_lumen_seg_marker_wall.png",
                    dapi_lumen_wall.astype(np.uint8) * 255,
                )
            if dapi_lumen_near_wall is not None:
                self._save_gray(
                    "step3_30_weighted_dapi_lumen_near_wall.png",
                    dapi_lumen_near_wall.astype(np.uint8) * 255,
                )
            if dapi_lumen_protected is not None:
                self._save_gray(
                    "step3_31_weighted_dapi_lumen_protected_prompt.png",
                    dapi_lumen_protected.astype(np.uint8) * 255,
                )
            dapi_intensity = result.debug.get("dapi_intensity")
            if dapi_intensity is not None and dapi_intensity.any():
                dab_overlay = cv2.cvtColor(
                    dapi_intensity.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            else:
                dab_overlay = tile_np[:, :, :3].copy()
            selected_index = 1
            rejected_index = 1
            for component in result.stats.get("dapi_lumen_components", []):
                x, y, width, height = component["bbox_xywh"]
                accepted = bool(component["accepted"])
                if accepted:
                    color = (0, 240, 0)
                    label = f"DL{selected_index}"
                    selected_index += 1
                else:
                    color = (255, 0, 0)
                    label = f"R{rejected_index}"
                    rejected_index += 1
                cv2.rectangle(
                    dab_overlay,
                    (x, y),
                    (x + width - 1, y + height - 1),
                    color,
                    thickness=1,
                )
                px, py = component["point_xy"]
                cv2.circle(dab_overlay, (px, py), 5, (255, 255, 255), 1)
                cv2.circle(dab_overlay, (px, py), 3, color, -1)
                cv2.putText(
                    dab_overlay,
                    label,
                    (px + 6, max(10, py - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            self._save_rgb(
                "step3_32_weighted_dapi_lumen_overlay.png",
                dab_overlay,
            )

        points_overlay = tile_np[:, :, :3].copy()
        for index, component in enumerate(result.point_components, start=1):
            x, y = component["point_xy"]
            kind = component.get("kind", "strong")
            fill = (
                (0, 255, 160) if kind == "dapi_lumen"
                else (80, 255, 255) if kind == "lumen"
                else (255, 255, 0)
            )
            cv2.circle(points_overlay, (x, y), 6, (0, 0, 0), 3)
            cv2.circle(points_overlay, (x, y), 5, fill, -1)
            cv2.putText(
                points_overlay, str(index), (x + 7, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                fill, 1, cv2.LINE_AA)
        self._save_rgb("step3_33_weighted_positive_points.png", points_overlay)

        metadata = dict(result.stats)
        metadata["positive_points"] = result.point_components
        metadata["strong_positive_points"] = [
            component for component in result.point_components
            if component.get("kind", "strong") == "strong"
        ]
        metadata["lumen_positive_points"] = [
            component for component in result.point_components
            if component.get("kind") == "lumen"
        ]
        metadata["dapi_lumen_positive_points"] = [
            component for component in result.point_components
            if component.get("kind") == "dapi_lumen"
        ]
        with open(self._path("step3_34_weighted_prompt_summary.json"), "w",
                  encoding="utf-8") as output:
            json.dump(metadata, output, indent=2, ensure_ascii=False)

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
        self._remove_prefixed_pngs("step5_merged_")
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
