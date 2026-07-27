#!/usr/bin/env python3
"""
cell/main.py -- CD34 WSI Pipeline entry point (Producer-Consumer Architecture).

Architecture
============

  Producer Thread (DeepLIIF batch -> WeightedPrompt)
        |
        |  put()
        v
  +----------------+
  |   Bucket       |  queue capacity = bucket_capacity x deepliif_bs
  |  ####......... |  <- put() blocks when full (backpressure)
  +-------+--------+
          |  get()
          v
  Consumer Thread (SAM2 batch -> MergeMasks -> Save)

Tile filtering uses ROI JSON (crop_region + roi_polygon).
cv2.fillPoly rasterizes the polygon at tile-grid resolution for O(1) lookup.

Usage
=====
    python -m cell.main \\
        --wsi-path /path/to/slide.ndpi \\
        --output-dir ./sample_output \\
        --device cuda:0 \\
        --roi-json /path/to/roi.json \\
        --deepliif-batch-size 4 \\
        --sam2-batch-size 32 \\
        --bucket-capacity 100
"""

import argparse
import csv
import json
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Thread, Event, Lock
from typing import Optional

import numpy as np
import torch

from cell.device import prepare_device
from cell.utils import (Bucket, BucketItem, StickyProgress,
                        load_roi_json, enumerate_tiles_in_roi,
                        enumerate_debug_region_tiles,
                        apply_crop_region_slice, generate_metrics_plots)
from cell.deepliif import DeepLIIFProcessor
from cell.segmentation_backend import create_segmentation_backend
from cell.postprocess import PostProcessor


def save_sam2_merge_diff(tile_np: np.ndarray,
                         sam_mask: np.ndarray,
                         merged_mask: np.ndarray,
                         cells_info: list,
                         output_path: str) -> str:
    """Save notebook-style SAM2 raw-vs-merged difference visualization."""
    import cv2

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
                    0.5, (255, 0, 0), 3)
        cv2.putText(diff_vis, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)

    cv2.imwrite(output_path, cv2.cvtColor(diff_vis, cv2.COLOR_RGB2BGR))
    return output_path


def _parse_debug_region_um(region_um: str, mpp: float) -> tuple[list[list[float]],
                                                                list[float],
                                                                tuple[int, int, int, int]]:
    """Parse four µm coordinate pairs and return points, µm bbox, level-0 bbox."""
    parts = [p.strip() for p in region_um.split(",") if p.strip()]
    if len(parts) != 8:
        raise ValueError("--debug-region-um requires 8 numbers: "
                         "x1,y1,x2,y2,x3,y3,x4,y4")
    if mpp <= 0:
        raise ValueError("WSI has no mpp metadata; cannot use --debug-region-um")

    nums = [float(p) for p in parts]
    points_um = [[nums[i], nums[i + 1]] for i in range(0, 8, 2)]
    xs_um = [p[0] for p in points_um]
    ys_um = [p[1] for p in points_um]
    bbox_um = [min(xs_um), min(ys_um), max(xs_um), max(ys_um)]

    bbox_level0 = (
        int(round(bbox_um[0] / mpp)),
        int(round(bbox_um[1] / mpp)),
        int(round(bbox_um[2] / mpp)),
        int(round(bbox_um[3] / mpp)),
    )
    return points_um, bbox_um, bbox_level0


def _write_debug_region_outputs(args, tiles: list[dict], metadata: dict) -> None:
    """Write selected tile table and region metadata."""
    out_dir = os.path.join(args.output_dir, "debug_region")
    os.makedirs(out_dir, exist_ok=True)
    stale_stitched_outputs = (
        "08_stitched_deepliif_seg.png",
        "09_stitched_deepliif_marker.png",
        "10_stitched_seg_positive.png",
        "11_stitched_combined_positive.png",
        "12_stitched_positive_regions.png",
        "13_stitched_owner_map.png",
        "stitched_deepliif_metadata.json",
    )
    for name in stale_stitched_outputs:
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            os.remove(path)

    csv_path = os.path.join(out_dir, "selected_tiles.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row", "col", "x", "y", "x_level0", "y_level0",
                "actual_w", "actual_h", "role",
            ],
        )
        writer.writeheader()
        for tile in tiles:
            writer.writerow({
                "row": tile["row"],
                "col": tile["col"],
                "x": tile["x"],
                "y": tile["y"],
                "x_level0": tile["x_level0"],
                "y_level0": tile["y_level0"],
                "actual_w": tile["actual_w"],
                "actual_h": tile["actual_h"],
                "role": tile.get("debug_role", ""),
            })

    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Debug region metadata: {meta_path}")
    print(f"  Debug region tiles:    {csv_path}")


def _seg_positive_pixel_count(seg_np: np.ndarray, seg_thresh: int) -> int:
    """Count Seg-positive pixels using the same rule as the debug curve."""
    from cell.debug_vis import compute_seg_positive_r_histogram

    counts, _ = compute_seg_positive_r_histogram(seg_np, seg_thresh)
    return int(counts.sum())


# ============================================================================
# 1. CLI Arguments
# ============================================================================

DEFAULT_CONFIG_PATH = "config/cell_main.json"


def _read_cli_config(path: str, parser: argparse.ArgumentParser,
                     *, required: bool) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        if required:
            parser.error(f"--config file not found: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as exc:
        parser.error(f"invalid JSON in --config {path}: {exc}")
    if "args" in config:
        config = config["args"]
    if not isinstance(config, dict):
        parser.error("--config must contain a JSON object")
    return config


def _apply_config_defaults(parser: argparse.ArgumentParser,
                           config: dict) -> None:
    valid_dests = {
        action.dest for action in parser._actions
        if action.dest not in {"help", argparse.SUPPRESS}
    }
    defaults = {}
    for key, value in config.items():
        dest = str(key).lstrip("-").replace("-", "_")
        if dest not in valid_dests:
            parser.error(f"unknown --config key: {key}")
        defaults[dest] = value
    if defaults:
        parser.set_defaults(**defaults)


def parse_args(argv: Optional[list[str]] = None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG_PATH,
        help="JSON config file. Defaults to config/cell_main.json when it "
             "exists; command-line flags override config values.")
    bootstrap_args, _ = bootstrap.parse_known_args(argv)

    p = argparse.ArgumentParser(
        description="CD34 Pipeline -- batch producer-consumer",
        parents=[bootstrap],
    )

    # -- WSI input --
    p.add_argument("--wsi-path", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="./sample_output")

    # -- ROI JSON --
    p.add_argument("--roi-json", type=str, default=None,
                   help="JSON file with crop_region and roi_polygon "
                        "(level-0 pixel coords).")
    p.add_argument("--crop-region-slice", type=str, default=None,
                   help="Run only part of crop_region for quick checks. "
                        "Format: top:1/4, top:0.25, top:25%%, bottom:1/4, "
                        "left:1/4, right:1/4, or top:1/4,left:1/3.")

    # -- Model paths --
    p.add_argument("--deepliif-model-dir", type=str,
                   default="./data/models/deepliif/")
    p.add_argument("--sam-checkpoint", type=str,
                   default="./data/models/sam2/sam2.1_hiera_large.pt")
    p.add_argument("--sam-config", type=str,
                   default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument(
        "--sam-backend", type=str, default="sam2", choices=["sam2"],
        help="Prompt-driven segmentation backend (default: sam2).")

    # -- Device --
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Device for ALL models: cpu, cuda, cuda:N, or N (default: cuda:0)")

    # -- Processing parameters --
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--target-magnification", type=float, default=40.0)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--resolution", type=str, default="40x",
                   choices=["10x", "20x", "40x"])
    p.add_argument(
        "--seg-thresh", type=int, default=120,
        help="Fixed foreground threshold applied as R+B > value on the "
             "DeepLIIF Seg image; it is not computed per image (default: 120).")
    p.add_argument(
        "--sam-prompt-mode", type=str, default="weighted-points",
        choices=["weighted-points"],
        help="Only supported SAM2 prompt strategy: Seg/Marker -5..5 dense "
             "mask plus positive points.")
    p.add_argument(
        "--weighted-marker-thresh", type=int, default=None,
        help="Fixed Marker threshold for weighted-points mode. If omitted, "
             "use two-stage 3-class Multi-Otsu and keep the upper middle "
             "class plus high class; masks use marker > threshold.")
    p.add_argument(
        "--weighted-marker-max", type=int, default=None,
        help="Marker intensity mapped to logit 5. If omitted, use each tile's "
             "Marker maximum, matching the current single-image experiment.")
    p.add_argument(
        "--weighted-dab-filter", action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the original RGB tile's normalized HED-DAB intensity to "
             "suppress low-DAB weighted-prompt pixels before SAM2. DAPI-lumen "
             "rescue can protect shallow-DAB Seg/Marker walls (default: true).")
    p.add_argument(
        "--weighted-dab-strong-support", action=argparse.BooleanOptionalAction,
        default=True,
        help="After DAB filtering, add strong original-tile DAB pixels back "
             "into the weighted SAM2 mask prompt as graded logits near "
             "existing Seg/Marker support (default: true).")
    p.add_argument(
        "--weighted-dab-strong-support-neighborhood-kernel",
        type=int, default=21,
        help="Odd-pixel dilation kernel defining how far DAB strong support "
             "may extend from existing Seg/Marker prompt support. Use 1 to "
             "only upgrade existing support (default: 21).")
    p.add_argument(
        "--weighted-dab-min-intensity", type=int, default=160,
        help="Minimum normalized DAB intensity retained by "
             "--weighted-dab-filter (0-255, default: 160).")
    p.add_argument(
        "--weighted-dab-normalization-percentile", type=float, default=99.5,
        help="Percentile used to normalize the original tile's DAB channel "
             "to 0-255 before applying --weighted-dab-min-intensity "
             "(default: 99.5).")
    p.add_argument(
        "--weighted-dab-hsv-brown-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require normalized HED-DAB retained pixels to also look brown in "
             "the original tile HSV color space before they can support the "
             "weighted prompt (default: true).")
    p.add_argument(
        "--weighted-dab-hsv-brown-hue-min", type=int, default=0,
        help="Minimum OpenCV HSV hue treated as DAB-brown confirmation "
             "(0-179, default: 0).")
    p.add_argument(
        "--weighted-dab-hsv-brown-hue-max", type=int, default=35,
        help="Maximum OpenCV HSV hue treated as DAB-brown confirmation "
             "(0-179, default: 35).")
    p.add_argument(
        "--weighted-dab-hsv-brown-saturation-min", type=int, default=30,
        help="Minimum OpenCV HSV saturation for DAB-brown confirmation "
             "(0-255, default: 30).")
    p.add_argument(
        "--weighted-dab-hsv-brown-value-min", type=int, default=20,
        help="Minimum OpenCV HSV value for DAB-brown confirmation "
             "(0-255, default: 20).")
    p.add_argument(
        "--weighted-dab-hsv-brown-white-value-min", type=int, default=245,
        help="HSV value above which low-saturation pixels are treated as "
             "near-white, not DAB-brown (0-255, default: 245).")
    p.add_argument(
        "--weighted-dab-hsv-brown-white-saturation-max",
        type=int, default=25,
        help="Maximum OpenCV HSV saturation for near-white pixels excluded "
             "from DAB-brown confirmation (0-255, default: 25).")
    p.add_argument(
        "--weighted-dab-hsv-brown-exclude-seg-blue",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude DeepLIIF Seg blue-negative support from the final DAB "
             "keep mask (default: true).")
    p.add_argument(
        "--weighted-dab-hsv-brown-seg-blue-dilate-kernel",
        type=int, default=3,
        help="Odd-pixel dilation kernel applied to Seg blue-negative support "
             "before excluding it from the DAB keep mask; use 1 to disable "
             "dilation (default: 3).")
    p.add_argument(
        "--weighted-dapi-lumen-dark-max", type=int, default=15,
        help="Maximum DeepLIIF DAPI grayscale intensity treated as no-nucleus "
             "lumen candidate during DAB filtering (default: 15).")
    p.add_argument(
        "--weighted-dapi-lumen-support-logit-min", type=int, default=1,
        help="Minimum Seg/Marker fused logit treated as wall support around "
             "DAPI-dark lumen candidates (default: 1).")
    p.add_argument(
        "--weighted-dapi-lumen-wall-closing-kernel", type=int, default=5,
        help="Morphological closing kernel applied to Seg/Marker support before "
             "testing DAPI-dark lumen enclosure (default: 5).")
    p.add_argument(
        "--weighted-repair-kernel", type=int, default=5,
        help="Morphological close kernel for repairing broken weighted-prompt "
             "support before lumen filling (default: 5).")
    p.add_argument(
        "--weighted-repair-iterations", type=int, default=1,
        help="Morphological close iterations for weighted prompt repair "
             "(default: 1).")
    p.add_argument(
        "--weighted-repair-logit", type=int, default=1,
        help="Logit assigned to repaired gaps in the weighted prompt "
             "(default: 1).")
    p.add_argument(
        "--weighted-lumen-logit", type=int, default=1,
        help="Logit assigned to enclosed lumen/hole pixels in the weighted "
             "prompt (default: 1).")
    p.add_argument(
        "--weighted-uncertain-iterations", type=int, default=1,
        help="Dilation iterations used to add logit-0 uncertainty around the "
             "filled weighted prompt (default: 1).")
    p.add_argument(
        "--weighted-artifact-filter", action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable diffuse weak/mid-positive artifact suppression.")
    p.add_argument(
        "--weighted-artifact-min-area", type=int, default=700,
        help="Minimum positive area considered by the artifact filter.")
    p.add_argument(
        "--weighted-small-fragment-filter",
        action=argparse.BooleanOptionalAction, default=True,
        help="Remove small weak components, including attached logit-0 pixels.")
    p.add_argument(
        "--weighted-small-fragment-max-area", type=int, default=100,
        help="Maximum area removed as a weak fragment (default: 100).")
    p.add_argument(
        "--weighted-isolated-fragment-filter",
        action=argparse.BooleanOptionalAction, default=True,
        help="Remove small isolated prompt components, including strong "
             "components, when they are not near a larger structure.")
    p.add_argument(
        "--weighted-isolated-fragment-max-area", type=int, default=200,
        help="Maximum area removed as an isolated fragment (default: 200).")
    p.add_argument(
        "--weighted-isolated-fragment-gap", type=int, default=8,
        help="Maximum distance in pixels to a larger structure that keeps a "
             "small component from being removed (default: 8).")
    p.add_argument(
        "--weighted-isolated-fragment-neighbor-min-area",
        type=int, default=700,
        help="Minimum area treated as nearby large support for the isolated "
             "fragment filter (default: 700).")
    p.add_argument(
        "--weighted-point-min-area", type=int, default=20,
        help="Minimum logit-5 component area used as a positive point.")
    p.add_argument(
        "--weighted-max-positive-points", type=int, default=30,
        help="Maximum positive points per tile; 0 keeps all (default: 30).")
    p.add_argument(
        "--weighted-lumen-points", action=argparse.BooleanOptionalAction,
        default=True,
        help="Add conservative positive point prompts inside automatically "
             "detected enclosed lumen candidates from the 256x256 weighted "
             "mask_input (default: true).")
    p.add_argument(
        "--weighted-lumen-point-support-logit-min", type=int, default=1,
        help="Minimum low-res mask_input logit treated as wall support for "
             "lumen-point detection (default: 1).")
    p.add_argument(
        "--weighted-lumen-point-closing-kernel", type=int, default=7,
        help="Morphological closing kernel on 256x256 support before finding "
             "lumen-point candidates (default: 7).")
    p.add_argument(
        "--weighted-lumen-point-min-area", type=int, default=8,
        help="Minimum 256x256 lumen candidate area to receive a point "
             "(default: 8).")
    p.add_argument(
        "--weighted-lumen-point-max-area", type=int, default=1200,
        help="Maximum 256x256 lumen candidate area to receive a point "
             "(default: 1200).")
    p.add_argument(
        "--weighted-lumen-point-ring-kernel", type=int, default=5,
        help="Kernel used to measure surrounding wall support around each "
             "lumen candidate (default: 5).")
    p.add_argument(
        "--weighted-lumen-point-min-wall-ratio", type=float, default=0.40,
        help="Minimum fraction of the candidate ring covered by original "
             "256x256 support before adding a lumen point (default: 0.40).")
    p.add_argument(
        "--weighted-lumen-point-fill-logit", type=int, default=2,
        help="Weak logit used to fill selected lumen-point candidates in the "
             "weighted mask prompt; use -5 to disable fill while keeping "
             "lumen points (default: 2).")
    p.add_argument(
        "--weighted-max-lumen-points", type=int, default=3,
        help="Maximum automatic lumen points per tile; 0 keeps all "
             "(default: 3).")
    p.add_argument(
        "--weighted-dab-lumen-fill", action=argparse.BooleanOptionalAction,
        default=True,
        help="During DAB filtering, use DAPI-dark no-nucleus candidates "
             "surrounded by Seg/Marker support to protect shallow-DAB walls "
             "and fill lumens, including candidates clipped by tile borders "
             "(default: true).")
    p.add_argument(
        "--weighted-dab-lumen-wall-min-intensity", type=int, default=160,
        help="Legacy DAB-wall threshold kept for compatibility with older "
             "debug helpers (default: 160).")
    p.add_argument(
        "--weighted-dab-lumen-interior-max-intensity", type=int, default=90,
        help="Legacy DAB-dark interior threshold kept for compatibility "
             "(default: 90).")
    p.add_argument(
        "--weighted-dab-lumen-near-wall-kernel", type=int, default=21,
        help="Kernel used to keep DAPI-dark lumen candidates near Seg/Marker "
             "walls (default: 21).")
    p.add_argument(
        "--weighted-dab-lumen-ring-kernel", type=int, default=9,
        help="Kernel used to measure Seg/Marker wall/border support around lumen "
             "candidates (default: 9).")
    p.add_argument(
        "--weighted-dab-lumen-min-area", type=int, default=80,
        help="Minimum high-resolution DAB lumen candidate area (default: 80).")
    p.add_argument(
        "--weighted-dab-lumen-max-area", type=int, default=8000,
        help="Maximum high-resolution DAB lumen candidate area (default: 8000).")
    p.add_argument(
        "--weighted-dab-lumen-min-wall-ratio", type=float, default=0.18,
        help="Minimum fraction of candidate ring covered by Seg/Marker wall "
             "(default: 0.18).")
    p.add_argument(
        "--weighted-dab-lumen-min-boundary-ratio", type=float, default=0.45,
        help="Minimum fraction of candidate ring covered by Seg/Marker wall "
             "plus tile border for non-border candidates (default: 0.45).")
    p.add_argument(
        "--weighted-dab-lumen-min-border-boundary-ratio", type=float,
        default=0.22,
        help="Minimum Seg/Marker wall plus tile-border support ratio for "
             "lumen candidates clipped by tile borders (default: 0.22).")
    p.add_argument(
        "--weighted-dab-lumen-macro-closing-kernel", type=int, default=31,
        help="Large-scale Seg/Marker wall closing kernel used to accept visually "
             "enclosed lumen candidates with incomplete local rings "
             "(default: 31).")
    p.add_argument(
        "--weighted-dab-lumen-macro-min-overlap", type=float, default=0.50,
        help="Minimum fraction of a dark candidate that must overlap a "
             "large-scale closed Seg/Marker-wall hole before macro support can "
             "bypass the local boundary-ratio test (default: 0.50).")
    p.add_argument(
        "--weighted-dab-lumen-macro-min-wall-ratio", type=float, default=0.30,
        help="Minimum large-scale context ring fraction covered by Seg/Marker wall "
             "for macro lumen support (default: 0.30).")
    p.add_argument(
        "--weighted-dab-lumen-white-interior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Legacy near-white RGB lumen switch kept for config compatibility; "
             "the main flow now uses DeepLIIF DAPI-dark candidates.")
    p.add_argument(
        "--weighted-dab-lumen-white-value-min", type=int, default=210,
        help="Fallback HSV value for near-white lumen interiors when the "
             "post-peak Multi-Otsu threshold cannot be computed "
             "(default: 210).")
    p.add_argument(
        "--weighted-dab-lumen-white-saturation-max", type=float, default=0.18,
        help="Maximum HSV saturation for near-white lumen interiors "
             "(default: 0.18).")
    p.add_argument(
        "--weighted-dab-lumen-white-channel-delta-max", type=int, default=35,
        help="Maximum RGB max-min channel difference for near-white lumen "
             "interiors (default: 35).")
    p.add_argument(
        "--weighted-dab-lumen-max-aspect-ratio", type=float, default=8.0,
        help="Maximum candidate bounding-box aspect ratio before treating a "
             "lumen candidate as an elongated artifact (default: 8.0).")
    p.add_argument(
        "--weighted-max-dab-lumen-points", type=int, default=3,
        help="Maximum DAB-intensity lumen points per tile; 0 keeps all "
             "(default: 3).")
    p.add_argument(
        "--fill-sam-holes", action=argparse.BooleanOptionalAction,
        default=False,
        help="After SAM2 and connected-component merge, fill background holes "
             "fully enclosed inside each instance label. This includes lumens "
             "in the final mask/GeoJSON but leaves open background unchanged "
             "(default: false).")
    p.add_argument("--stitch-mode", type=str, default="center-valid",
                   choices=[
                       "center-valid", "center-valid-raw", "overlap-merge"],
                   help="How final tile masks are reconstructed. "
                        "center-valid keeps each tile's owner/center region "
                        "but uses uncropped overlap masks for conservative "
                        "cross-tile identity matching; center-valid-raw "
                        "exports owner/center regions without cross-tile "
                        "matching; overlap-merge exports the previous "
                        "full-overlap geometry "
                        "(default: center-valid).")

    # -- Batch sizes --
    p.add_argument("--deepliif-batch-size", type=int, default=64)
    p.add_argument("--sam2-batch-size", type=int, default=32)

    # -- Bucket --
    p.add_argument("--bucket-capacity", type=int, default=500,
                   help="Bucket iterations per refill cycle (default: 500)")

    # -- Tile prefetch workers --
    p.add_argument("--prefetch-workers", type=int, default=4)

    # -- Targeted debug --
    p.add_argument("--tile-index", type=str, default=None,
                   help="Process a single tile ROW,COL (debug mode)")
    p.add_argument("--tile-um", type=str, default=None,
                   help="Process a single tile by µm coordinate X_UM,Y_UM "
                        "(level-0 space). Resolved to ROW,COL via mpp + "
                        "WSIReader tile grid. Mutually exclusive with --tile-index.")
    p.add_argument("--debug-region-um", type=str, default=None,
                   help="Run full pipeline only on original-grid tiles that "
                        "intersect the bbox of four µm points, plus one-ring "
                        "neighbor tiles. Format: x1,y1,x2,y2,x3,y3,x4,y4.")

    # -- Smoke test limit --
    p.add_argument("--max-tiles", type=int, default=0,
                   help="Limit total tiles for smoke testing (0 = no limit)")

    # -- Cache switches --
    p.add_argument("--cache-deepliif", action="store_true",
                   help="Save DeepLIIF results to <output-dir>/cache/deepliif/")
    p.add_argument("--cache-sam2", action="store_true",
                   help="Save SAM2 raw masks to <output-dir>/cache/sam2/")
    p.add_argument("--reuse-sam2-cache", type=str, default=None,
                   help="Read SAM2 raw masks from this cache/sam2 directory "
                        "instead of running SAM2 inference")

    # -- GeoJSON export --
    p.add_argument("--skip-reconstruction", action="store_true",
                   help="Skip GeoJSON export after mask generation")
    p.add_argument("--geojson-simplify", type=float, default=0,
                   help="GeoJSON polygon simplify ratio (0=off, default: 0)")
    p.add_argument("--contour-tolerance", type=float, default=0.5,
                   help="Douglas-Peucker contour tolerance in pixels (default: 0.5)")

    config = _read_cli_config(
        bootstrap_args.config,
        p,
        required=bootstrap_args.config != DEFAULT_CONFIG_PATH,
    )
    _apply_config_defaults(p, config)

    args = p.parse_args(argv)
    if args.wsi_path is None:
        p.error("--wsi-path is required; set it in config/cell_main.json "
                "or pass --wsi-path")
    if args.roi_json is None:
        p.error("--roi-json is required; set it in config/cell_main.json "
                "or pass --roi-json")
    if (args.weighted_marker_thresh is not None
            and not 0 <= args.weighted_marker_thresh <= 255):
        p.error("--weighted-marker-thresh must be between 0 and 255")
    if (args.weighted_marker_max is not None
            and not 0 <= args.weighted_marker_max <= 255):
        p.error("--weighted-marker-max must be between 0 and 255")
    if not 0 <= args.weighted_dab_min_intensity <= 255:
        p.error("--weighted-dab-min-intensity must be between 0 and 255")
    if args.weighted_dab_strong_support_neighborhood_kernel < 1:
        p.error("--weighted-dab-strong-support-neighborhood-kernel must be >= 1")
    if not 0 < args.weighted_dab_normalization_percentile <= 100:
        p.error("--weighted-dab-normalization-percentile must be in (0, 100]")
    if not 0 <= args.weighted_dab_hsv_brown_hue_min <= 179:
        p.error("--weighted-dab-hsv-brown-hue-min must be between 0 and 179")
    if not 0 <= args.weighted_dab_hsv_brown_hue_max <= 179:
        p.error("--weighted-dab-hsv-brown-hue-max must be between 0 and 179")
    if not 0 <= args.weighted_dab_hsv_brown_saturation_min <= 255:
        p.error("--weighted-dab-hsv-brown-saturation-min must be between 0 and 255")
    if not 0 <= args.weighted_dab_hsv_brown_value_min <= 255:
        p.error("--weighted-dab-hsv-brown-value-min must be between 0 and 255")
    if not 0 <= args.weighted_dab_hsv_brown_white_value_min <= 255:
        p.error("--weighted-dab-hsv-brown-white-value-min must be between 0 and 255")
    if not 0 <= args.weighted_dab_hsv_brown_white_saturation_max <= 255:
        p.error("--weighted-dab-hsv-brown-white-saturation-max must be between 0 and 255")
    if args.weighted_dab_hsv_brown_seg_blue_dilate_kernel < 1:
        p.error("--weighted-dab-hsv-brown-seg-blue-dilate-kernel must be >= 1")
    if not 0 <= args.weighted_dapi_lumen_dark_max <= 255:
        p.error("--weighted-dapi-lumen-dark-max must be between 0 and 255")
    if not -5 <= args.weighted_dapi_lumen_support_logit_min <= 5:
        p.error("--weighted-dapi-lumen-support-logit-min must be between -5 and 5")
    if args.weighted_dapi_lumen_wall_closing_kernel < 1:
        p.error("--weighted-dapi-lumen-wall-closing-kernel must be >= 1")
    if args.weighted_repair_kernel < 1:
        p.error("--weighted-repair-kernel must be >= 1")
    if args.weighted_repair_iterations < 0:
        p.error("--weighted-repair-iterations must be >= 0")
    if not -5 <= args.weighted_repair_logit <= 5:
        p.error("--weighted-repair-logit must be between -5 and 5")
    if not -5 <= args.weighted_lumen_logit <= 5:
        p.error("--weighted-lumen-logit must be between -5 and 5")
    if args.weighted_uncertain_iterations < 0:
        p.error("--weighted-uncertain-iterations must be >= 0")
    if args.weighted_isolated_fragment_max_area < 1:
        p.error("--weighted-isolated-fragment-max-area must be >= 1")
    if args.weighted_isolated_fragment_gap < 0:
        p.error("--weighted-isolated-fragment-gap must be >= 0")
    if args.weighted_isolated_fragment_neighbor_min_area < 1:
        p.error("--weighted-isolated-fragment-neighbor-min-area must be >= 1")
    if not -5 <= args.weighted_lumen_point_support_logit_min <= 5:
        p.error("--weighted-lumen-point-support-logit-min must be between -5 and 5")
    if args.weighted_lumen_point_closing_kernel < 1:
        p.error("--weighted-lumen-point-closing-kernel must be >= 1")
    if args.weighted_lumen_point_min_area < 1:
        p.error("--weighted-lumen-point-min-area must be >= 1")
    if args.weighted_lumen_point_max_area < 1:
        p.error("--weighted-lumen-point-max-area must be >= 1")
    if args.weighted_lumen_point_min_area > args.weighted_lumen_point_max_area:
        p.error("--weighted-lumen-point-min-area cannot exceed "
                "--weighted-lumen-point-max-area")
    if args.weighted_lumen_point_ring_kernel < 1:
        p.error("--weighted-lumen-point-ring-kernel must be >= 1")
    if not 0 <= args.weighted_lumen_point_min_wall_ratio <= 1:
        p.error("--weighted-lumen-point-min-wall-ratio must be in [0, 1]")
    if not -5 <= args.weighted_lumen_point_fill_logit <= 5:
        p.error("--weighted-lumen-point-fill-logit must be between -5 and 5")
    if args.weighted_max_lumen_points < 0:
        p.error("--weighted-max-lumen-points must be >= 0")
    if not 0 <= args.weighted_dab_lumen_wall_min_intensity <= 255:
        p.error("--weighted-dab-lumen-wall-min-intensity must be between 0 and 255")
    if not 0 <= args.weighted_dab_lumen_interior_max_intensity <= 255:
        p.error("--weighted-dab-lumen-interior-max-intensity must be between 0 and 255")
    if args.weighted_dab_lumen_near_wall_kernel < 1:
        p.error("--weighted-dab-lumen-near-wall-kernel must be >= 1")
    if args.weighted_dab_lumen_ring_kernel < 1:
        p.error("--weighted-dab-lumen-ring-kernel must be >= 1")
    if args.weighted_dab_lumen_min_area < 1:
        p.error("--weighted-dab-lumen-min-area must be >= 1")
    if args.weighted_dab_lumen_max_area < 1:
        p.error("--weighted-dab-lumen-max-area must be >= 1")
    if args.weighted_dab_lumen_min_area > args.weighted_dab_lumen_max_area:
        p.error("--weighted-dab-lumen-min-area cannot exceed "
                "--weighted-dab-lumen-max-area")
    if not 0 <= args.weighted_dab_lumen_min_wall_ratio <= 1:
        p.error("--weighted-dab-lumen-min-wall-ratio must be in [0, 1]")
    if not 0 <= args.weighted_dab_lumen_min_boundary_ratio <= 1:
        p.error("--weighted-dab-lumen-min-boundary-ratio must be in [0, 1]")
    if not 0 <= args.weighted_dab_lumen_min_border_boundary_ratio <= 1:
        p.error("--weighted-dab-lumen-min-border-boundary-ratio must be in [0, 1]")
    if args.weighted_dab_lumen_macro_closing_kernel < 1:
        p.error("--weighted-dab-lumen-macro-closing-kernel must be >= 1")
    if not 0 <= args.weighted_dab_lumen_macro_min_overlap <= 1:
        p.error("--weighted-dab-lumen-macro-min-overlap must be in [0, 1]")
    if not 0 <= args.weighted_dab_lumen_macro_min_wall_ratio <= 1:
        p.error("--weighted-dab-lumen-macro-min-wall-ratio must be in [0, 1]")
    if not 0 <= args.weighted_dab_lumen_white_value_min <= 255:
        p.error("--weighted-dab-lumen-white-value-min must be between 0 and 255")
    if not 0 <= args.weighted_dab_lumen_white_saturation_max <= 1:
        p.error("--weighted-dab-lumen-white-saturation-max must be in [0, 1]")
    if not 0 <= args.weighted_dab_lumen_white_channel_delta_max <= 255:
        p.error("--weighted-dab-lumen-white-channel-delta-max must be between "
                "0 and 255")
    if args.weighted_dab_lumen_max_aspect_ratio < 1:
        p.error("--weighted-dab-lumen-max-aspect-ratio must be >= 1")
    if args.weighted_max_dab_lumen_points < 0:
        p.error("--weighted-max-dab-lumen-points must be >= 0")
    return args


def _build_weighted_prompt_config(args):
    from cd34_pipeline.sam2_wrapper.weighted_prompt import WeightedPromptConfig

    return WeightedPromptConfig(
        seg_thresh=args.seg_thresh,
        marker_thresh=args.weighted_marker_thresh,
        marker_max=args.weighted_marker_max,
        enable_dab_filter=args.weighted_dab_filter,
        enable_dab_strong_support=args.weighted_dab_strong_support,
        dab_strong_support_neighborhood_kernel=(
            args.weighted_dab_strong_support_neighborhood_kernel),
        dab_min_intensity=args.weighted_dab_min_intensity,
        dab_normalization_percentile=(
            args.weighted_dab_normalization_percentile),
        enable_dab_hsv_brown_filter=(
            args.weighted_dab_hsv_brown_filter),
        dab_hsv_brown_hue_min=(
            args.weighted_dab_hsv_brown_hue_min),
        dab_hsv_brown_hue_max=(
            args.weighted_dab_hsv_brown_hue_max),
        dab_hsv_brown_saturation_min=(
            args.weighted_dab_hsv_brown_saturation_min),
        dab_hsv_brown_value_min=(
            args.weighted_dab_hsv_brown_value_min),
        dab_hsv_brown_white_value_min=(
            args.weighted_dab_hsv_brown_white_value_min),
        dab_hsv_brown_white_saturation_max=(
            args.weighted_dab_hsv_brown_white_saturation_max),
        dab_hsv_brown_exclude_seg_blue=(
            args.weighted_dab_hsv_brown_exclude_seg_blue),
        dab_hsv_brown_seg_blue_dilate_kernel=(
            args.weighted_dab_hsv_brown_seg_blue_dilate_kernel),
        dapi_lumen_dark_max=args.weighted_dapi_lumen_dark_max,
        dapi_lumen_support_logit_min=(
            args.weighted_dapi_lumen_support_logit_min),
        dapi_lumen_wall_closing_kernel=(
            args.weighted_dapi_lumen_wall_closing_kernel),
        repair_kernel=args.weighted_repair_kernel,
        repair_iterations=args.weighted_repair_iterations,
        repair_logit=args.weighted_repair_logit,
        lumen_logit=args.weighted_lumen_logit,
        uncertain_iterations=args.weighted_uncertain_iterations,
        enable_artifact_filter=args.weighted_artifact_filter,
        artifact_min_area=args.weighted_artifact_min_area,
        enable_small_fragment_filter=args.weighted_small_fragment_filter,
        small_fragment_max_area=args.weighted_small_fragment_max_area,
        enable_isolated_fragment_filter=(
            args.weighted_isolated_fragment_filter),
        isolated_fragment_max_area=(
            args.weighted_isolated_fragment_max_area),
        isolated_fragment_min_gap=args.weighted_isolated_fragment_gap,
        isolated_fragment_neighbor_min_area=(
            args.weighted_isolated_fragment_neighbor_min_area),
        point_min_area=args.weighted_point_min_area,
        max_positive_points=args.weighted_max_positive_points,
        enable_lumen_points=args.weighted_lumen_points,
        lumen_point_support_logit_min=(
            args.weighted_lumen_point_support_logit_min),
        lumen_point_closing_kernel=args.weighted_lumen_point_closing_kernel,
        lumen_point_min_area=args.weighted_lumen_point_min_area,
        lumen_point_max_area=args.weighted_lumen_point_max_area,
        lumen_point_ring_kernel=args.weighted_lumen_point_ring_kernel,
        lumen_point_min_wall_ratio=args.weighted_lumen_point_min_wall_ratio,
        lumen_point_fill_logit=args.weighted_lumen_point_fill_logit,
        max_lumen_points=args.weighted_max_lumen_points,
        enable_dab_lumen_fill=args.weighted_dab_lumen_fill,
        dab_lumen_wall_min_intensity=(
            args.weighted_dab_lumen_wall_min_intensity),
        dab_lumen_interior_max_intensity=(
            args.weighted_dab_lumen_interior_max_intensity),
        dab_lumen_near_wall_kernel=args.weighted_dab_lumen_near_wall_kernel,
        dab_lumen_ring_kernel=args.weighted_dab_lumen_ring_kernel,
        dab_lumen_min_area=args.weighted_dab_lumen_min_area,
        dab_lumen_max_area=args.weighted_dab_lumen_max_area,
        dab_lumen_min_wall_ratio=args.weighted_dab_lumen_min_wall_ratio,
        dab_lumen_min_boundary_ratio=(
            args.weighted_dab_lumen_min_boundary_ratio),
        dab_lumen_min_border_boundary_ratio=(
            args.weighted_dab_lumen_min_border_boundary_ratio),
        dab_lumen_macro_closing_kernel=(
            args.weighted_dab_lumen_macro_closing_kernel),
        dab_lumen_macro_min_overlap=(
            args.weighted_dab_lumen_macro_min_overlap),
        dab_lumen_macro_min_wall_ratio=(
            args.weighted_dab_lumen_macro_min_wall_ratio),
        dab_lumen_use_white_interior=(
            args.weighted_dab_lumen_white_interior),
        dab_lumen_white_value_min=(
            args.weighted_dab_lumen_white_value_min),
        dab_lumen_white_saturation_max=(
            args.weighted_dab_lumen_white_saturation_max),
        dab_lumen_white_channel_delta_max=(
            args.weighted_dab_lumen_white_channel_delta_max),
        dab_lumen_max_aspect_ratio=(
            args.weighted_dab_lumen_max_aspect_ratio),
        max_dab_lumen_points=args.weighted_max_dab_lumen_points,
    )


def _write_debug_region_stitched_deepliif(args,
                                          records: list[dict]) -> None:
    """Write stitched DeepLIIF global debug images for --debug-region-um."""
    if args.debug_region_um is None:
        return
    if not records:
        print("  [debug-region] No DeepLIIF records available for stitched "
              "global debug images.")
        return
    try:
        from cell.center_valid_stitching import (
            write_center_valid_debug_outputs,
        )
        metadata = write_center_valid_debug_outputs(
            records,
            args.output_dir,
            tile_size=args.tile_size,
            overlap=args.overlap,
            seg_thresh=args.seg_thresh,
            marker_thresh=args.weighted_marker_thresh,
            marker_percentile_factor=0.8,
            morphology_kernel=11,
            min_area=0,
        )
        print("  Debug region stitched DeepLIIF outputs: "
              f"{metadata.get('source_tile_count', len(records))} tile(s)")
    except Exception as exc:
        print(f"  [debug-region] Failed to write stitched DeepLIIF outputs: "
              f"{exc}")


# ============================================================================
# 2. Producer -- DeepLIIF (batch) + Cell Extraction
# ============================================================================

class Producer:
    """
    Reads ROI-filtered tiles from WSI, runs DeepLIIF on them,
    extracts cells, and puts SAM2-ready items into the bucket.
    """

    def __init__(self, wsi_reader, all_tiles: list[dict], args,
                 bucket: Bucket, done_event: Event, stats: dict,
                 progress: Optional[StickyProgress] = None):
        self.wsi_reader = wsi_reader
        self.all_tiles = all_tiles
        self.args = args
        self.bucket = bucket
        self.done_event = done_event
        self.stats = stats
        self.progress = progress
        self.debug_deepliif_records: list[dict] = []
        self._debug_record_lock = Lock()

    def _store_debug_deepliif_record(self, tile_info: dict, tile_name: str,
                                     seg_np: np.ndarray,
                                     marker_np: np.ndarray) -> None:
        """Keep DeepLIIF outputs for debug-region stitched global images."""
        if self.args.debug_region_um is None:
            return
        record = {
            "tile_info": dict(tile_info),
            "tile_name": tile_name,
            "seg_np": seg_np.copy(),
            "marker_np": marker_np.copy(),
        }
        with self._debug_record_lock:
            self.debug_deepliif_records.append(record)

    def run(self):
        try:
            self._run_impl()
        except Exception as e:
            import traceback
            print(f"\n[Producer] FATAL: {e}")
            traceback.print_exc()
        finally:
            self.done_event.set()

    @staticmethod
    def prefetch_tiles(wsi_reader, tiles: list[dict],
                       num_workers: int = 4) -> list:
        """Read tiles from WSI in parallel using a thread pool."""
        images = [None] * len(tiles)

        def _read_one(idx):
            images[idx] = wsi_reader.read_tile(tiles[idx])

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futs = [pool.submit(_read_one, i) for i in range(len(tiles))]
            for f in futs:
                f.result()
        return images

    def _extract_one(self, deepliif, tile_info, tile_np, dl_result):
        """Build the weighted prompt for one tile and enqueue it."""
        try:
            seg_img = dl_result.get('Seg')
            marker_img = dl_result.get('Marker')
            dapi_img = dl_result.get('DAPI')
            if seg_img is None or marker_img is None:
                return False

            seg_np = np.array(seg_img)
            marker_np = np.array(marker_img)
            dapi_np = np.array(dapi_img) if dapi_img is not None else None
            tile_name = self.wsi_reader.get_tile_filename(tile_info)
            deepliif.cache_result(tile_name, seg_np, marker_np, dapi_np)
            self._store_debug_deepliif_record(
                tile_info, tile_name, seg_np, marker_np)

            dbg = None
            seg_positive_pixels = None
            weighted_config = None
            if self.args.debug_region_um is not None:
                from cell.debug_vis import DebugVisualizer
                dbg = DebugVisualizer(self.args.output_dir, tile_name)
                dbg.step1_original(tile_np)
                dbg.step2_deepliif(dl_result)
                seg_summary = dbg.step2_seg_positive_r_intensity(
                    seg_np, self.args.seg_thresh)
                seg_positive_pixels = int(seg_summary["positive_pixel_count"])
                if seg_positive_pixels == 0:
                    dbg.clear_downstream_outputs()
                    print(f"  [skip] {tile_name}: Seg-positive pixels=0; "
                          "skip weighted prompt/SAM2")
                    return False
                weighted_config = _build_weighted_prompt_config(self.args)
                dbg.step2_marker_intensity(
                    marker_np,
                    marker_thresh=self.args.weighted_marker_thresh,
                )
                dbg.step2_dab_intensity(
                    tile_np,
                    seg_np,
                    weighted_config,
                )
                if dapi_np is not None:
                    dbg.step2_dapi_dark_intensity(
                        dapi_np,
                        dark_max=self.args.weighted_dapi_lumen_dark_max,
                    )

            if seg_positive_pixels is None:
                seg_positive_pixels = _seg_positive_pixel_count(
                    seg_np, self.args.seg_thresh)
                if seg_positive_pixels == 0:
                    return False

            from cd34_pipeline.sam2_wrapper.weighted_prompt import (
                build_weighted_prompt,
            )

            if weighted_config is None:
                weighted_config = _build_weighted_prompt_config(self.args)
            prompt = build_weighted_prompt(
                seg_np,
                marker_np,
                weighted_config,
                tile_rgb=tile_np,
                dapi=dapi_np,
            )
            if prompt.stats["final_nonnegative_px"] == 0:
                return False
            if dbg is not None:
                dbg.step3_weighted_prompt(tile_np, prompt)

            item = BucketItem(
                tile_np=tile_np,
                positive_cells_info=[],
                tile_info=tile_info,
                tile_name=tile_name,
                mask_input=prompt.mask_input,
                point_coords=prompt.point_coords,
                point_labels=prompt.point_labels,
                prompt_stats=prompt.stats,
                prompt_debug_dir=(dbg.dir if dbg is not None else None),
            )
            self.bucket.put(item)
            return True
        except Exception as e:
            import traceback
            print(f"[Producer] Weighted prompt error: {e}")
            traceback.print_exc()
            return False

    def _run_impl(self):
        args = self.args

        # -- Init processors --
        deepliif = DeepLIIFProcessor(
            model_dir=args.deepliif_model_dir,
            device=args.device,
            cache_dir=(os.path.join(args.output_dir, "cache", "deepliif")
                       if args.cache_deepliif else None),
        )
        extract_pool = ThreadPoolExecutor(
            max_workers=min(os.cpu_count() or 8, 16),
            thread_name_prefix="WeightedPrompt")

        all_tiles = self.all_tiles
        total_tiles = len(all_tiles)
        chunk_size = args.bucket_capacity
        deepliif_batch_size = args.deepliif_batch_size
        num_prefetch_workers = args.prefetch_workers

        print(f"[Producer] Tiles to process: {total_tiles}")
        print(f"[Producer] Chunk size per refill cycle: {chunk_size} tiles")

        produced = 0
        skipped = 0
        refill_count = 0
        tile_cursor = 0
        t0 = time.time()

        # -- Prefetch first chunk in background --
        prefetch_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ChunkPrefetch")
        first_chunk_end = min(chunk_size, total_tiles)
        first_chunk = all_tiles[0:first_chunk_end]
        prefetch_future: Optional[Future] = prefetch_executor.submit(
            self.prefetch_tiles, self.wsi_reader,
            first_chunk, num_prefetch_workers)
        print(f"[Producer] Prefetching first chunk ({len(first_chunk)} tiles)...")

        while tile_cursor < total_tiles:
            chunk_end = min(tile_cursor + chunk_size, total_tiles)
            chunk_tiles = all_tiles[tile_cursor:chunk_end]
            tile_cursor = chunk_end

            # -- Get prefetched images --
            t_prefetch = time.time()
            if prefetch_future is not None:
                prefetched_images = prefetch_future.result()
                prefetch_future = None
            else:
                prefetched_images = self.prefetch_tiles(
                    self.wsi_reader, chunk_tiles, num_prefetch_workers)
            prefetch_dt = time.time() - t_prefetch

            print(f"[Producer] Chunk [{chunk_end}/{total_tiles}]: "
                  f"{len(chunk_tiles)} tiles prefetched ({prefetch_dt:.1f}s)")

            round_produced_before = produced
            if self.progress:
                self.progress.update(tile_cursor=chunk_end,
                                     bucket_level=self.bucket.qsize(),
                                     round_chunk_in=len(chunk_tiles),
                                     round_filter_out=len(chunk_tiles),
                                     round_produced_out=0)

            # -- Prefetch NEXT chunk (parallel with DeepLIIF) --
            if tile_cursor < total_tiles:
                next_end = min(tile_cursor + chunk_size, total_tiles)
                prefetch_future = prefetch_executor.submit(
                    self.prefetch_tiles, self.wsi_reader,
                    all_tiles[tile_cursor:next_end], num_prefetch_workers)

            # -- DeepLIIF + weighted prompt construction on all tiles in chunk --
            # Prompt construction runs in a thread pool, overlapping with next
            # DeepLIIF GPU batch to hide CPU-bound preprocessing latency.
            prev_extract_futs: list[Future] = []
            batch_cursor = 0
            while batch_cursor < len(chunk_tiles):
                batch_end = min(
                    batch_cursor + deepliif_batch_size, len(chunk_tiles))
                batch_tiles = chunk_tiles[batch_cursor:batch_end]
                batch_images = prefetched_images[batch_cursor:batch_end]
                batch_cursor = batch_end

                tile_nps = [np.array(p) for p in batch_images]

                t_dl = time.time()
                deepliif_results = deepliif.process_batch(
                    batch_images, batch_size=deepliif_batch_size,
                    resolution=args.resolution)
                dl_dt = time.time() - t_dl

                # Collect previous batch's extraction results
                # (ran on CPU threads concurrently with this DeepLIIF batch)
                for fut in prev_extract_futs:
                    if fut.result():
                        produced += 1
                    else:
                        skipped += 1
                prev_extract_futs.clear()

                # Submit current batch's extraction to thread pool
                # (will run on CPU concurrently with the *next* DeepLIIF batch)
                for tile_info, tile_np, dl_result in zip(
                    batch_tiles, tile_nps, deepliif_results
                ):
                    fut = extract_pool.submit(
                        self._extract_one, deepliif, tile_info, tile_np,
                        dl_result)
                    prev_extract_futs.append(fut)

                if self.progress:
                    self.progress.update(deepliif_dt=dl_dt,
                                         produced=produced,
                                         bucket_level=self.bucket.qsize(),
                                         round_produced_out=produced - round_produced_before)

            # Collect last batch's extraction results
            for fut in prev_extract_futs:
                if fut.result():
                    produced += 1
                else:
                    skipped += 1

            del prefetched_images

            print(f"[Producer] Refill #{refill_count + 1} done -- "
                  f"produced={produced} skipped={skipped} "
                  f"bucket={self.bucket.qsize()}/{self.bucket.capacity}")

            refill_count += 1

        extract_pool.shutdown(wait=True)

        deepliif.shutdown()
        prefetch_executor.shutdown(wait=False)

        producer_time = time.time() - t0
        self.stats['producer_time'] = producer_time
        self.stats['total_tiles'] = total_tiles
        self.stats['produced'] = produced
        self.stats['skipped_no_cells'] = skipped
        self.stats['refill_count'] = refill_count
        print(f"[Producer] All done in {producer_time:.1f}s -- "
              f"{produced} items produced, {skipped} skipped, "
              f"{refill_count} refills")


# ============================================================================
# 3. Consumer -- SAM2 (batch) + PostProcess
# ============================================================================

class Consumer:
    """
    Takes SAM2-ready items from the bucket, runs SAM2 batch segmentation,
    merges masks via PostProcessor, and stores results in memory.
    """

    def __init__(self, args, bucket: Bucket,
                 producer_done: Event, stats: dict,
                 progress: Optional[StickyProgress] = None,
                 tile_records: list[dict] | None = None):
        self.args = args
        self.bucket = bucket
        self.producer_done = producer_done
        self.stats = stats
        self.progress = progress
        self.tile_records = tile_records
        self.postprocessor = None

    def run(self):
        try:
            self._run_impl()
        except Exception as e:
            import traceback
            print(f"\n[Consumer] FATAL: {e}")
            traceback.print_exc()

    def _run_impl(self):
        args = self.args

        # -- Init processors --
        sam2 = create_segmentation_backend(
            args.sam_backend,
            config=args.sam_config,
            checkpoint=args.sam_checkpoint,
            device=args.device,
            batch_size=args.sam2_batch_size,
            cache_dir=(os.path.join(args.output_dir, "cache", "sam2")
                       if args.cache_sam2 else None),
            reuse_cache_dir=args.reuse_sam2_cache,
        )
        postprocessor = PostProcessor(
            output_dir=args.output_dir,
            min_area=0,
            tile_size=args.tile_size,
            overlap=args.overlap,
            stitch_mode=args.stitch_mode,
            tile_records=self.tile_records,
            debug_region_metadata=getattr(args, "debug_region_metadata", None),
            debug_region_tiles=getattr(args, "debug_region_tiles", None),
            fill_sam_holes=args.fill_sam_holes,
        )
        self.postprocessor = postprocessor

        consumed = 0
        masks_queued = 0
        tile_batch_size = args.sam2_batch_size
        t0 = time.time()

        post_pool = ThreadPoolExecutor(
            max_workers=min(os.cpu_count() or 4, 8),
            thread_name_prefix="PostProcess")
        prev_post_futs: list[Future] = []

        while True:
            # ── 批量收集 tile ──
            items: list[BucketItem] = []

            # 阻塞等待第一个 item（或超时退出）
            try:
                items.append(self.bucket.get(timeout=0.5))
            except queue.Empty:
                if self.producer_done.is_set() and self.bucket.empty():
                    break
                continue

            # 非阻塞收集剩余 items，凑够 tile_batch_size
            while len(items) < tile_batch_size:
                try:
                    items.append(self.bucket.get(timeout=0))
                except queue.Empty:
                    break

            # ── 多图批量 SAM2 推理 ──
            t_sam2 = time.time()
            batch_results = sam2.segment_batch(items)
            sam2_dt = time.time() - t_sam2

            # Collect previous batch's post-processing results
            # (ran on CPU threads concurrently with this SAM2 batch)
            for fut in prev_post_futs:
                try:
                    if fut.result():
                        masks_queued += 1
                except Exception as e:
                    print(f"[Consumer] Post-process error: {e}")
            prev_post_futs.clear()

            # Submit current batch's post-processing to thread pool
            # (will run on CPU concurrently with the *next* SAM2 batch)
            for item, (sam_mask, scores) in zip(items, batch_results):
                fut = post_pool.submit(
                    postprocessor.merge_and_process,
                    sam_mask, scores, item.positive_cells_info, item.tile_name,
                    item.tile_np, item.tile_info)
                prev_post_futs.append(fut)

            consumed += len(items)
            del items

            if self.bucket.qsize() == 0 and not self.producer_done.is_set():
                print("\033[32m[Consumer] Bucket empty (0), waiting for refill\033[0m")

            if self.progress:
                self.progress.update(
                    consumed=consumed,
                    bucket_level=self.bucket.qsize(),
                    flushed=postprocessor.saved_count,
                    sam2_dt=sam2_dt)

            if consumed % 10 < tile_batch_size:
                print(f"[Consumer] {consumed} consumed, "
                      f"{masks_queued} queued, {postprocessor.saved_count} flushed, "
                      f"bucket={self.bucket.qsize()}/{self.bucket.capacity}")

        # Collect remaining post-processing results
        for fut in prev_post_futs:
            try:
                if fut.result():
                    masks_queued += 1
            except Exception as e:
                print(f"[Consumer] Post-process error: {e}")
        post_pool.shutdown(wait=True)

        sam2.shutdown()
        postprocessor.shutdown()

        sam2_time = time.time() - t0
        self.stats['sam2_time'] = sam2_time
        self.stats['consumed'] = consumed
        self.stats['masks_saved'] = postprocessor.saved_count
        print(f"[Consumer] SAM2 done in {sam2_time:.1f}s -- "
              f"{consumed} consumed, {postprocessor.saved_count} masks saved")


# ============================================================================
# 4. Single-tile debug mode
# ============================================================================

def _resolve_center_tile(args, wsi_reader, all_tiles) -> Optional[dict]:
    """Resolve --tile-index / --tile-um to a single tile_info dict."""
    if args.tile_um is not None:
        parts = args.tile_um.split(",")
        x_um, y_um = float(parts[0]), float(parts[1])
        mpp = wsi_reader.mpp
        if mpp <= 0:
            print("ERROR: WSI has no mpp metadata; cannot use --tile-um.")
            return None
        px = int(round(x_um / mpp))
        py = int(round(y_um / mpp))
        print(f"\n[Single-tile debug] µm=({x_um},{y_um}), mpp={mpp:.4f} "
              f"→ level-0 px=({px},{py})")
        downsample = wsi_reader.level_downsample
        for t in all_tiles:
            x0 = t['x_level0']
            y0 = t['y_level0']
            x1 = x0 + int(round(t['actual_w'] * downsample))
            y1 = y0 + int(round(t['actual_h'] * downsample))
            if x0 <= px < x1 and y0 <= py < y1:
                return t
        print(f"ERROR: No ROI tile covers µm coord ({x_um},{y_um}). "
              f"Point may lie outside crop_region / roi_polygon.")
        return None

    parts = args.tile_index.split(",")
    tile_row, tile_col = int(parts[0]), int(parts[1])
    print(f"\n[Single-tile debug] Tile ({tile_row},{tile_col}) on {args.device}")
    for t in all_tiles:
        if t['row'] == tile_row and t['col'] == tile_col:
            return t
    print(f"ERROR: Tile ({tile_row},{tile_col}) not found in ROI grid.")
    return None


def _process_one_tile_debug(args, wsi_reader, target_tile,
                            deepliif, sam2, postprocessor) -> None:
    """Run the single-tile flow for one tile, reusing pre-loaded models."""
    from cd34_pipeline.sam2_wrapper.inference import merge_connected_masks

    tile_pil = wsi_reader.read_tile(target_tile)
    tile_np = np.array(tile_pil)
    tile_name = wsi_reader.get_tile_filename(target_tile)
    stem = os.path.splitext(tile_name)[0]
    print(f"\n[debug] Tile ({target_tile['row']},{target_tile['col']}) "
          f"-- shape {tile_np.shape}")

    # -- DeepLIIF --
    t0 = time.time()
    results = deepliif.process_batch([tile_pil], batch_size=1,
                                     resolution=args.resolution)
    print(f"  DeepLIIF: {time.time() - t0:.3f}s")
    dl = results[0]
    seg_np = np.array(dl['Seg'])
    marker_np = np.array(dl['Marker'])
    dapi_img = dl.get('DAPI')
    dapi_np = np.array(dapi_img) if dapi_img is not None else None

    seg_positive_pixels = _seg_positive_pixel_count(seg_np, args.seg_thresh)
    print(f"  Seg-positive pixels: {seg_positive_pixels}")
    if seg_positive_pixels == 0:
        print("  No Seg-positive pixels found; skipping weighted prompt/SAM2.")
        return

    t0 = time.time()
    from cd34_pipeline.sam2_wrapper.weighted_prompt import (
        build_weighted_prompt,
    )
    weighted_config = _build_weighted_prompt_config(args)
    prompt = build_weighted_prompt(
        seg_np,
        marker_np,
        weighted_config,
        tile_rgb=tile_np,
        dapi=dapi_np,
    )
    print(f"  Weighted prompt: {time.time() - t0:.3f}s -- "
          f"{prompt.stats['final_nonnegative_px']} active pixels, "
          f"{len(prompt.point_coords)} points")
    if prompt.stats["final_nonnegative_px"] == 0:
        print("  No weighted prompt support found.")
        return
    cells_info = []
    item = BucketItem(
        tile_np=tile_np,
        positive_cells_info=[],
        tile_info=target_tile,
        tile_name=tile_name,
        mask_input=prompt.mask_input,
        point_coords=prompt.point_coords,
        point_labels=prompt.point_labels,
        prompt_stats=prompt.stats,
    )
    t0 = time.time()
    sam_mask, scores = sam2.segment_batch([item])[0]
    print(f"  SAM2: {time.time() - t0:.3f}s -- {len(scores)} kept")

    merged_mask, _, _, _ = merge_connected_masks(
        sam_mask, scores, cells_info, min_area=0,
    )
    save_sam2_merge_diff(
        tile_np, sam_mask, merged_mask, cells_info,
        os.path.join(args.output_dir, "debug", f"{stem}_sam2_merge_diff.png"),
    )

    # -- Post-process (npy + per-tile bookkeeping) --
    t0 = time.time()
    saved = postprocessor.merge_and_process(
        sam_mask, scores, cells_info, tile_name, tile_info=target_tile)
    print(f"  Merge+Process: {time.time() - t0:.3f}s -- saved={saved}")


def run_single_tile_debug(args):
    """Single-tile debug flow (shares one model load)."""
    from cd34_pipeline.io.wsi_reader import WSIReader

    wsi_reader = WSIReader(
        args.wsi_path,
        tile_size=args.tile_size,
        target_magnification=args.target_magnification,
        overlap=args.overlap,
    )

    # Enumerate tiles using the same ROI grid the pipeline uses, so the
    # returned row/col matches what the producer-consumer path would yield.
    roi_data = load_roi_json(args.roi_json)
    crop_region = apply_crop_region_slice(
        roi_data['crop_region'], args.crop_region_slice)
    all_tiles = enumerate_tiles_in_roi(
        crop_region=crop_region,
        roi_polygon=roi_data['roi_polygon'],
        tile_size=args.tile_size,
        overlap=args.overlap,
        level_downsample=wsi_reader.level_downsample,
    )

    center = _resolve_center_tile(args, wsi_reader, all_tiles)
    if center is None:
        wsi_reader.close()
        return

    tiles_to_run = [center]
    print(f"[debug] Will process {len(tiles_to_run)} tile(s).")

    # -- Load models once --
    deepliif = DeepLIIFProcessor(args.deepliif_model_dir, args.device)
    sam2 = create_segmentation_backend(
        args.sam_backend,
        config=args.sam_config,
        checkpoint=args.sam_checkpoint,
        device=args.device,
        batch_size=args.sam2_batch_size,
        reuse_cache_dir=args.reuse_sam2_cache,
    )
    postprocessor = PostProcessor(
        output_dir=args.output_dir, min_area=0,
        tile_size=args.tile_size, overlap=args.overlap,
        stitch_mode=args.stitch_mode,
        tile_records=tiles_to_run,
        fill_sam_holes=args.fill_sam_holes,
    )

    try:
        for t in tiles_to_run:
            _process_one_tile_debug(
                args, wsi_reader, t,
                deepliif, sam2, postprocessor,
            )
    finally:
        postprocessor.shutdown()
        wsi_reader.close()
    print("  Done.")


# ============================================================================
# 5. Main -- orchestrate producer & consumer
# ============================================================================

def main():
    args = parse_args()
    args.device = prepare_device(args.device)

    if args.device.startswith("cuda"):
        print(f"[Main] All models (DeepLIIF, SAM2) will run on {args.device}")

    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.output_dir, exist_ok=True)

    # -- Debug mode validation --
    debug_modes = [
        args.tile_index is not None,
        args.tile_um is not None,
        args.debug_region_um is not None,
    ]
    if sum(debug_modes) > 1:
        print("ERROR: --tile-index, --tile-um, and --debug-region-um are mutually exclusive.")
        return
    # -- Single-tile debug shortcut --
    if args.tile_index is not None or args.tile_um is not None:
        run_single_tile_debug(args)
        return

    # -- Compute effective queue capacity --
    queue_capacity = args.bucket_capacity * args.deepliif_batch_size
    print(f"\n{'='*60}")
    print("CD34 BATCH PIPELINE -- PRODUCER-CONSUMER (ROI polygon mode)")
    print(f"{'='*60}")
    print(f"  WSI:               {args.wsi_path}")
    print(f"  Output:            {args.output_dir}")
    print(f"  Device:            {args.device}")
    print(f"  SAM backend:       {args.sam_backend}")
    print(f"  ROI JSON:          {args.roi_json}")
    print(f"  DeepLIIF batch_size: {args.deepliif_batch_size}")
    print(f"  SAM2 batch_size:   {args.sam2_batch_size}")
    print(f"  Bucket capacity:   {args.bucket_capacity} iterations "
          f"(queue={queue_capacity} items)")
    print(f"  Cache DeepLIIF:    {'ON' if args.cache_deepliif else 'OFF'}")
    print(f"  Cache SAM2:        {'ON' if args.cache_sam2 else 'OFF'}")
    print(f"  Reuse SAM2 cache:  {args.reuse_sam2_cache or 'OFF'}")
    print(f"  SAM prompt mode:   {args.sam_prompt_mode}")
    print(f"{'='*60}\n")

    # -- Load ROI JSON --
    roi_data = load_roi_json(args.roi_json)
    crop_region = apply_crop_region_slice(
        roi_data['crop_region'], args.crop_region_slice)
    roi_polygon = roi_data['roi_polygon']

    # -- Open WSI --
    from cd34_pipeline.io.wsi_reader import WSIReader
    wsi_reader = WSIReader(
        args.wsi_path,
        tile_size=args.tile_size,
        target_magnification=args.target_magnification,
        overlap=args.overlap,
    )

    # -- Enumerate tiles within ROI polygon (no full WSI scan) --
    debug_region_metadata = None
    if args.debug_region_um is not None:
        try:
            points_um, bbox_um, bbox_level0 = _parse_debug_region_um(
                args.debug_region_um, wsi_reader.mpp)
        except ValueError as e:
            print(f"ERROR: {e}")
            wsi_reader.close()
            return

        print("\n[Debug region]")
        print(f"  points_um: {points_um}")
        print(f"  bbox_um:   {bbox_um}")
        print(f"  bbox_px:   {bbox_level0}")

        all_tiles, debug_counts = enumerate_debug_region_tiles(
            crop_region=crop_region,
            roi_polygon=roi_polygon,
            bbox_level0=bbox_level0,
            tile_size=args.tile_size,
            overlap=args.overlap,
            level_downsample=wsi_reader.level_downsample,
            neighbor_radius=1,
        )
        debug_region_metadata = {
            "debug_region_um": points_um,
            "debug_bbox_um": bbox_um,
            "debug_bbox_level0": list(bbox_level0),
            "mpp": wsi_reader.mpp,
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "stride": args.tile_size - args.overlap,
            "level_downsample": wsi_reader.level_downsample,
            "crop_origin_level0": [crop_region["x"], crop_region["y"]],
            "neighbor_radius": 1,
            "result_clipping": "none",
            "sam_prompt_mode": args.sam_prompt_mode,
            **debug_counts,
        }
    else:
        all_tiles = enumerate_tiles_in_roi(
            crop_region=crop_region,
            roi_polygon=roi_polygon,
            tile_size=args.tile_size,
            overlap=args.overlap,
            level_downsample=wsi_reader.level_downsample,
        )

    if args.max_tiles > 0:
        all_tiles = all_tiles[:args.max_tiles]
        print(f"  Limited to {len(all_tiles)} tiles (--max-tiles)")

    print(f"  Tiles to process: {len(all_tiles)}")

    if debug_region_metadata is not None:
        debug_region_metadata["selected_tile_count_after_max_tiles"] = len(all_tiles)
        _write_debug_region_outputs(args, all_tiles, debug_region_metadata)
        args.debug_region_metadata = debug_region_metadata
        args.debug_region_tiles = all_tiles

    if not all_tiles:
        print("[Main] No tiles to process. Exiting.")
        wsi_reader.close()
        return

    # -- Create bucket --
    bucket = Bucket(capacity=queue_capacity)
    producer_done = Event()
    stats: dict = {}

    # -- Progress bar --
    progress = StickyProgress(
        total_tiles=len(all_tiles),
        bucket_capacity=queue_capacity,
    )

    # -- Launch threads --
    total_start = time.time()

    producer = Producer(wsi_reader, all_tiles, args, bucket,
                        producer_done, stats, progress)
    consumer = Consumer(args, bucket, producer_done, stats, progress,
                        tile_records=all_tiles)

    producer_thread = Thread(target=producer.run, name="Producer", daemon=True)
    consumer_thread = Thread(target=consumer.run, name="Consumer", daemon=True)

    print("[Main] Starting producer and consumer threads...")
    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

    if args.debug_region_um is not None:
        _write_debug_region_stitched_deepliif(
            args, producer.debug_deepliif_records)

    if args.debug_region_um is not None and consumer.postprocessor is not None:
        consumer.postprocessor.write_debug_region_tile_artifacts()

    # -- Close progress bar & cleanup --
    progress.close()
    wsi_reader.close()
    total_elapsed = time.time() - total_start

    # -- Generate metrics plots --
    saved_plots = generate_metrics_plots(progress.snapshots, args.output_dir)

    # -- Summary --
    masks_saved = stats.get('masks_saved', 0)
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  ROI polygon tiles:   {stats.get('total_tiles', '?')}")
    print(f"  Produced items:      {stats.get('produced', '?')}")
    print(f"  Skipped (no prompt): {stats.get('skipped_no_cells', '?')}")
    print(f"  Consumed items:      {stats.get('consumed', '?')}")
    print(f"  Masks saved:         {masks_saved}")
    print(f"  Refill cycles:       {stats.get('refill_count', '?')}")
    print(f"  Stitch mode:         {args.stitch_mode}")
    print(f"  SAM prompt mode:     {args.sam_prompt_mode}")
    print(f"{'='*60}")
    print(f"  Producer time:       {stats.get('producer_time', 0):.1f}s "
          f"(DeepLIIF + WeightedPrompt)")
    print(f"  SAM2 time:           {stats.get('sam2_time', 0):.1f}s")
    print(f"  Total time:          {total_elapsed:.1f}s")
    print(f"  Output:              {args.output_dir}")
    if saved_plots:
        print(f"  Metrics plots:       {len(saved_plots)} files")
    print(f"{'='*60}")

    # -- GeoJSON export (using in-memory data from Consumer) --
    if not args.skip_reconstruction and masks_saved > 0:
        pp = consumer.postprocessor
        if pp is not None:
            t_geo = time.time()
            geojson_path = pp.export_geojson(
                wsi_path=args.wsi_path,
                tile_size=args.tile_size,
                overlap=args.overlap,
                simplify=args.geojson_simplify,
                contour_tolerance=args.contour_tolerance,
                min_area=0,
                level_downsample=wsi_reader.level_downsample,
                crop_origin=(crop_region['x'], crop_region['y']),
            )
            if geojson_path:
                print(f"\n  GeoJSON export: {time.time() - t_geo:.1f}s")
                print(f"  GeoJSON:         {geojson_path}")
        else:
            print("\n  No PostProcessor available, skipping GeoJSON export.")
    elif masks_saved == 0:
        print("\n  No masks produced, skipping GeoJSON export.")
    else:
        print("\n  Skipping GeoJSON export (--skip-reconstruction).")


if __name__ == "__main__":
    main()
