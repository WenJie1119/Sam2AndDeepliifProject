#!/usr/bin/env python3
"""Run a two-tile Seg/Marker weighted-prompt SAM2 test."""

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


DEFAULT_TILES = (
    "debug_output/debug_marker_keep31/debug_vis/tile_39_13_4992_14976",
    "debug_output/debug_marker_keep31/debug_vis/tile_39_12_4608_14976",
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


def colorize_logits(logits: np.ndarray) -> np.ndarray:
    vis = np.zeros((*logits.shape, 3), dtype=np.uint8)
    for value, color in COLORS.items():
        vis[logits == value] = color
    return vis


def overlay_mask(image: np.ndarray, mask: np.ndarray,
                 color: tuple[int, int, int] = (255, 0, 0),
                 alpha: float = 0.55) -> np.ndarray:
    overlay = image.copy()
    pixels = mask.astype(bool)
    if pixels.any():
        color_arr = np.array(color, dtype=np.float32)
        overlay[pixels] = (
            overlay[pixels].astype(np.float32) * (1.0 - alpha)
            + color_arr * alpha
        ).clip(0, 255).astype(np.uint8)
    return overlay


def seg_logits(seg: np.ndarray, seg_thresh: int,
               foreground_green_max: int) -> tuple[np.ndarray, dict]:
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


def marker_logits(marker: np.ndarray, marker_thresh: int,
                  marker_max: int) -> tuple[np.ndarray, dict]:
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
    flood = (mask.astype(np.uint8) * 255)
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


def build_weighted_prompt(
        seg: np.ndarray,
        marker: np.ndarray,
        args: argparse.Namespace,
        marker_max: int) -> tuple[np.ndarray, np.ndarray, dict]:
    seg_logit, seg_stats = seg_logits(
        seg, args.seg_thresh, args.foreground_green_max)
    marker_logit, marker_stats = marker_logits(
        marker, args.marker_thresh, marker_max)

    raw = np.maximum(seg_logit, marker_logit).astype(np.int16)
    support = raw > 0

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (args.repair_kernel, args.repair_kernel))
    closed = cv2.morphologyEx(
        support.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel,
        iterations=args.repair_iterations).astype(bool)
    repair = closed & ~support

    filled = fill_holes(closed)
    lumen = filled & ~closed

    band_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(
        filled.astype(np.uint8), band_kernel,
        iterations=args.uncertain_iterations).astype(bool)
    uncertain = dilated & ~filled

    final = raw.copy()
    final[repair & (final < args.repair_logit)] = args.repair_logit
    final[lumen & (final < args.lumen_logit)] = args.lumen_logit
    final[uncertain & (final < 0)] = 0
    final = np.clip(final, -5, 5).astype(np.int16)

    mask_input = max_pool_lowres(final, args.target_size)[None, :, :]

    stats = {
        **seg_stats,
        **marker_stats,
        "raw_positive_support_px": int(support.sum()),
        "repair_px": int(repair.sum()),
        "lumen_px": int(lumen.sum()),
        "uncertain_band_px": int(uncertain.sum()),
        "final_nonnegative_px": int((final >= 0).sum()),
    }
    for value in (-5, 0, 1, 2, 3, 4, 5):
        stats[f"final_logit_{value}_px"] = int((final == value).sum())
    return final, mask_input.astype(np.float32), stats


def old_binary_prompt(old_mask_path: Path,
                      target_size: int) -> tuple[np.ndarray, np.ndarray]:
    old_mask = read_gray(old_mask_path)
    old_mask = cv2.resize(
        old_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    logits = np.where(old_mask > 0, 5.0, -5.0).astype(np.float32)
    return old_mask, logits[None, :, :]


def save_prompt_outputs(tile_out: Path, image: np.ndarray,
                        logits_512: np.ndarray,
                        mask_input: np.ndarray,
                        stats: dict) -> None:
    heat = colorize_logits(logits_512)
    active = logits_512 >= 0
    overlay = image.copy()
    overlay[active] = (
        overlay[active].astype(np.float32) * 0.45
        + heat[active].astype(np.float32) * 0.55
    ).clip(0, 255).astype(np.uint8)

    np.save(tile_out / "weighted_prompt_logits_512.npy",
            logits_512.astype(np.float32))
    np.save(tile_out / "weighted_prompt_mask_input_256.npy", mask_input)
    save_rgb(tile_out / "weighted_prompt_heatmap_512.png", heat)
    save_rgb(tile_out / "weighted_prompt_overlay.png", overlay)
    save_rgb(
        tile_out / "weighted_prompt_mask_input_256.png",
        colorize_logits(mask_input[0].astype(np.int16)),
    )
    with open(tile_out / "weighted_prompt_stats.json", "w",
              encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def run_sam2_prompt(predictor, image: np.ndarray, mask_input: np.ndarray,
                    output_dir: Path, label: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    masks, scores, low_res_masks = predictor.predict(
        mask_input=mask_input,
        multimask_output=True,
    )
    scores = np.asarray(scores, dtype=np.float32)
    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx].astype(bool)

    for idx, mask in enumerate(masks):
        mask_bool = mask.astype(bool)
        save_gray(output_dir / f"{label}_candidate_{idx}_mask.png",
                  mask_bool.astype(np.uint8) * 255)
        save_rgb(output_dir / f"{label}_candidate_{idx}_overlay.png",
                 overlay_mask(image, mask_bool))

    save_gray(output_dir / f"{label}_best_mask.png",
              best_mask.astype(np.uint8) * 255)
    save_rgb(output_dir / f"{label}_best_overlay.png",
             overlay_mask(image, best_mask, color=(255, 0, 0), alpha=0.55))
    np.save(output_dir / f"{label}_low_res_masks.npy", low_res_masks)

    summary = {
        "label": label,
        "scores": [float(score) for score in scores],
        "best_idx": best_idx,
        "best_score": float(scores[best_idx]),
        "candidate_areas": [int(mask.astype(bool).sum()) for mask in masks],
        "best_area": int(best_mask.sum()),
    }
    with open(output_dir / f"{label}_sam2_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def copy_inputs(tile_dir: Path, tile_out: Path) -> None:
    tile_out.mkdir(parents=True, exist_ok=True)
    copies = {
        "step1_original.png": "original.png",
        "step2_01_deepliif_Seg.png": "deepliif_seg.png",
        "step2_02_deepliif_Marker.png": "deepliif_marker.png",
        "step3_08_sam2_prompt_02_mask_256.png": "old_binary_prompt_256.png",
    }
    for src_name, dst_name in copies.items():
        shutil.copy2(tile_dir / src_name, tile_out / dst_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path,
                        default=Path("debug_output/debug_marker_keep31/"
                                     "seg_marker_weighted_sam2_test"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sam-config",
                        default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam-checkpoint", type=Path,
                        default=Path("data/models/sam2/"
                                     "sam2.1_hiera_large.pt"))
    parser.add_argument("--seg-thresh", type=int, default=120)
    parser.add_argument("--foreground-green-max", type=int, default=80)
    parser.add_argument("--marker-thresh", type=int, default=100)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--repair-kernel", type=int, default=5)
    parser.add_argument("--repair-iterations", type=int, default=1)
    parser.add_argument("--repair-logit", type=int, default=1)
    parser.add_argument("--lumen-logit", type=int, default=1)
    parser.add_argument("--uncertain-iterations", type=int, default=1)
    parser.add_argument("--tile-dir", type=Path, action="append",
                        default=[Path(p) for p in DEFAULT_TILES])
    parser.add_argument("--skip-old-binary-sam2", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tile_dirs = [path.resolve() for path in args.tile_dir]
    args.output_root.mkdir(parents=True, exist_ok=True)

    marker_images = [
        read_gray(tile_dir / "step2_02_deepliif_Marker.png")
        for tile_dir in tile_dirs
    ]
    marker_max = max(int(marker.max()) for marker in marker_images)
    marker_max = max(marker_max, args.marker_thresh + 1)

    predictor = load_sam2(
        args.sam_config, str(args.sam_checkpoint), args.device)

    all_summaries = []
    for tile_dir in tile_dirs:
        tile_out = args.output_root / tile_dir.name
        copy_inputs(tile_dir, tile_out)

        image = read_rgb(tile_dir / "step1_original.png")
        seg = read_rgb(tile_dir / "step2_01_deepliif_Seg.png")
        marker = read_gray(tile_dir / "step2_02_deepliif_Marker.png")
        prompt_logits, weighted_input, prompt_stats = build_weighted_prompt(
            seg, marker, args, marker_max)
        prompt_stats.update({
            "tile": tile_dir.name,
            "scheme": "seg_marker_weighted_prompt",
            "sam_config": args.sam_config,
            "sam_checkpoint": str(args.sam_checkpoint),
            "device": args.device,
        })
        save_prompt_outputs(tile_out, image, prompt_logits, weighted_input,
                            prompt_stats)

        predictor.set_image(image)
        tile_summary = {
            "tile": tile_dir.name,
            "weighted": run_sam2_prompt(
                predictor, image, weighted_input,
                tile_out / "sam2_weighted", "weighted"),
        }

        if not args.skip_old_binary_sam2:
            old_mask, old_input = old_binary_prompt(
                tile_dir / "step3_08_sam2_prompt_02_mask_256.png",
                args.target_size)
            np.save(tile_out / "old_binary_prompt_mask_input_256.npy",
                    old_input)
            save_gray(tile_out / "old_binary_prompt_256_resized.png",
                      old_mask)
            tile_summary["old_binary"] = run_sam2_prompt(
                predictor, image, old_input,
                tile_out / "sam2_old_binary", "old_binary")

        all_summaries.append(tile_summary)

    with open(args.output_root / "run_summary.json", "w",
              encoding="utf-8") as f:
        json.dump({
            "marker_max_used_for_all_tiles": int(marker_max),
            "tiles": all_summaries,
        }, f, indent=2, ensure_ascii=False)
    print(args.output_root)


if __name__ == "__main__":
    main()
