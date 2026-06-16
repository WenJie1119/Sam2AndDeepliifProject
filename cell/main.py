#!/usr/bin/env python3
"""
cell/main.py -- CD34 WSI Pipeline entry point (Producer-Consumer Architecture).

Architecture
============

  Producer Thread (DeepLIIF batch -> CellExtract)
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
from threading import Thread, Event
from typing import Optional

import numpy as np
import torch

from cell.device import prepare_device
from cell.utils import (Bucket, BucketItem, StickyProgress,
                        load_roi_json, enumerate_tiles_in_roi,
                        enumerate_debug_region_tiles,
                        apply_crop_region_slice, generate_metrics_plots)
from cell.deepliif import DeepLIIFProcessor
from cell.segmentation import CellSegmentor
from cell.sam2 import SAM2Processor
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


# ============================================================================
# 1. CLI Arguments
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="CD34 Pipeline -- batch producer-consumer")

    # -- WSI input --
    p.add_argument("--wsi-path", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="./sample_output")

    # -- ROI JSON --
    p.add_argument("--roi-json", type=str, required=True,
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

    # -- Device --
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Device for ALL models: cpu, cuda, cuda:N, or N (default: cuda:0)")

    # -- Processing parameters --
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--target-magnification", type=float, default=40.0)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--resolution", type=str, default="40x",
                   choices=["10x", "20x", "40x"])
    p.add_argument("--seg-thresh", type=int, default=120)
    p.add_argument("--marker-thresh", type=int, default=None)
    p.add_argument("--marker-percentile-factor", type=float, default=0.9,
                   help="Factor used by automatic marker thresholding when "
                        "--marker-thresh is not set. The threshold is "
                        "min + (max - min) * factor over the marker "
                        "0.1%%-99.9%% range (default: 0.9).")
    p.add_argument("--morphology-kernel", type=int, default=11)
    p.add_argument("--min-mask-area", type=int, default=50)

    # -- Batch sizes --
    p.add_argument("--deepliif-batch-size", type=int, default=64)
    p.add_argument("--sam2-batch-size", type=int, default=32)

    # -- Bucket --
    p.add_argument("--bucket-capacity", type=int, default=500,
                   help="Bucket iterations per refill cycle (default: 500)")

    # -- Tile prefetch workers --
    p.add_argument("--prefetch-workers", type=int, default=4)

    # -- Single-tile debug --
    p.add_argument("--tile-index", type=str, default=None,
                   help="Process a single tile ROW,COL (debug mode)")
    p.add_argument("--tile-um", type=str, default=None,
                   help="Process a single tile by µm coordinate X_UM,Y_UM "
                        "(level-0 space). Resolved to ROW,COL via mpp + "
                        "WSIReader tile grid. Mutually exclusive with --tile-index.")
    p.add_argument("--debug-vis", action="store_true",
                   help="Save step-by-step visualization for each debugged tile "
                        "to {output-dir}/debug_vis/{tile_name}/. "
                        "Used by --tile-index/--tile-um; --debug-region-um "
                        "always saves tile debug artefacts.")
    p.add_argument("--debug-3x3", action="store_true",
                   help="Run the debug flow on 9 tiles centered on the input "
                        "coordinate instead of just one. Requires --debug-vis.")
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

    return p.parse_args()


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

    def _extract_one(self, segmentor, deepliif, tile_info, tile_np, dl_result):
        """Extract cells from one tile and enqueue to bucket. (Thread-pool worker)"""
        try:
            seg_img = dl_result.get('Seg')
            marker_img = dl_result.get('Marker')
            if seg_img is None or marker_img is None:
                return False

            seg_np = np.array(seg_img)
            marker_np = np.array(marker_img)
            tile_name = self.wsi_reader.get_tile_filename(tile_info)
            deepliif.cache_result(tile_name, seg_np, marker_np)

            dbg = None
            if self.args.debug_region_um is not None:
                from cell.debug_vis import DebugVisualizer
                dbg = DebugVisualizer(self.args.output_dir, tile_name)
                dbg.step1_original(tile_np)
                dbg.step2_deepliif(dl_result)

            if dbg is not None:
                tile_segmentor = CellSegmentor(
                    seg_thresh=self.args.seg_thresh,
                    marker_thresh=self.args.marker_thresh,
                    marker_percentile_factor=(
                        self.args.marker_percentile_factor),
                    morphology_kernel=self.args.morphology_kernel,
                    min_area=self.args.min_mask_area,
                )
            else:
                tile_segmentor = segmentor

            positive_cells_info, clusters = tile_segmentor.extract(
                seg_np, marker_np)

            if dbg is not None and positive_cells_info:
                dbg.step3_connected_region(
                    tile_np, seg_np, marker_np,
                    positive_cells_info,
                    seg_thresh=self.args.seg_thresh,
                    marker_thresh=self.args.marker_thresh,
                    marker_percentile_factor=(
                        self.args.marker_percentile_factor),
                    morphology_kernel=self.args.morphology_kernel,
                )

            if dbg is not None:
                dbg.step3_sam2_prompt(tile_np, positive_cells_info)

            if not positive_cells_info or not clusters:
                return False

            item = BucketItem(
                tile_np=tile_np,
                clusters=clusters,
                positive_cells_info=positive_cells_info,
                tile_info=tile_info,
                tile_name=tile_name,
            )
            self.bucket.put(item)
            return True
        except Exception as e:
            import traceback
            print(f"[Producer] Cell extraction error: {e}")
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
        segmentor = CellSegmentor(
            seg_thresh=args.seg_thresh,
            marker_thresh=args.marker_thresh,
            marker_percentile_factor=args.marker_percentile_factor,
            morphology_kernel=args.morphology_kernel,
            min_area=args.min_mask_area,
        )
        extract_pool = ThreadPoolExecutor(
            max_workers=min(os.cpu_count() or 8, 16),
            thread_name_prefix="CellExtract")

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
        debug_stitch_records = (
            [] if args.debug_region_um is not None else None
        )

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

            # -- DeepLIIF + cell extraction on all tiles in chunk --
            # Cell extraction runs in thread pool, overlapping with next
            # DeepLIIF GPU batch to hide CPU-bound extraction latency.
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

                if debug_stitch_records is not None:
                    for tile_info, tile_np, dl_result in zip(
                        batch_tiles, tile_nps, deepliif_results
                    ):
                        seg_img = dl_result.get('Seg')
                        marker_img = dl_result.get('Marker')
                        if seg_img is None or marker_img is None:
                            continue
                        debug_stitch_records.append({
                            'tile_info': tile_info,
                            'tile_name': self.wsi_reader.get_tile_filename(
                                tile_info),
                            'tile_np': tile_np,
                            'seg_np': np.array(seg_img),
                            'marker_np': np.array(marker_img),
                        })

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
                        self._extract_one, segmentor, deepliif,
                        tile_info, tile_np, dl_result)
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

        if debug_stitch_records is not None:
            try:
                from cell.center_valid_stitching import (
                    write_center_valid_debug_outputs,
                )

                stitched_meta = write_center_valid_debug_outputs(
                    debug_stitch_records,
                    output_root=args.output_dir,
                    tile_size=args.tile_size,
                    overlap=args.overlap,
                    seg_thresh=args.seg_thresh,
                    marker_thresh=args.marker_thresh,
                    marker_percentile_factor=args.marker_percentile_factor,
                    morphology_kernel=args.morphology_kernel,
                    min_area=args.min_mask_area,
                )
                self.stats['debug_stitched_regions'] = (
                    stitched_meta.get('positive_region_count', 0)
                )
                print("[Producer] Wrote center-valid stitched DeepLIIF "
                      f"debug outputs ({stitched_meta['source_tile_count']} "
                      "tiles, "
                      f"{stitched_meta['positive_region_count']} regions)")
            except Exception as e:
                import traceback
                print(f"[Producer] Center-valid stitching debug failed: {e}")
                traceback.print_exc()

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
                 progress: Optional[StickyProgress] = None):
        self.args = args
        self.bucket = bucket
        self.producer_done = producer_done
        self.stats = stats
        self.progress = progress
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
        sam2 = SAM2Processor(
            config=args.sam_config,
            checkpoint=args.sam_checkpoint,
            device=args.device,
            batch_size=args.sam2_batch_size,
            min_area=args.min_mask_area,
            cache_dir=(os.path.join(args.output_dir, "cache", "sam2")
                       if args.cache_sam2 else None),
            reuse_cache_dir=args.reuse_sam2_cache,
        )
        postprocessor = PostProcessor(
            output_dir=args.output_dir,
            min_area=200,
            tile_size=args.tile_size,
            overlap=args.overlap,
            debug_region_metadata=getattr(args, "debug_region_metadata", None),
            debug_region_tiles=getattr(args, "debug_region_tiles", None),
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


def _expand_to_3x3(all_tiles: list, center: dict) -> list:
    """Return up to 9 tiles centered on `center`, in row-major order.
    Tiles missing from the ROI grid are silently skipped (with a warning)."""
    tiles_by_rc = {(t['row'], t['col']): t for t in all_tiles}
    r0, c0 = center['row'], center['col']
    neighborhood = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rc = (r0 + dr, c0 + dc)
            if rc in tiles_by_rc:
                neighborhood.append(tiles_by_rc[rc])
            else:
                print(f"  [debug-3x3] skip missing neighbor ({rc[0]},{rc[1]})")
    return neighborhood


def _process_one_tile_debug(args, wsi_reader, target_tile,
                            deepliif, segmentor, sam2, postprocessor) -> None:
    """Run the single-tile flow for one tile, reusing pre-loaded models.

    If args.debug_vis is True, step-by-step artefacts are written to
    {output-dir}/debug_vis/{tile_stem}/.
    """
    import cv2 as _cv2
    from cd34_pipeline.sam2_wrapper.inference import merge_connected_masks

    tile_pil = wsi_reader.read_tile(target_tile)
    tile_np = np.array(tile_pil)
    tile_name = wsi_reader.get_tile_filename(target_tile)
    stem = os.path.splitext(tile_name)[0]
    print(f"\n[debug] Tile ({target_tile['row']},{target_tile['col']}) "
          f"-- shape {tile_np.shape}")

    dbg = None
    if args.debug_vis:
        from cell.debug_vis import DebugVisualizer
        dbg = DebugVisualizer(args.output_dir, tile_name)
        dbg.step1_original(tile_np)

    # -- DeepLIIF --
    t0 = time.time()
    results = deepliif.process_batch([tile_pil], batch_size=1,
                                     resolution=args.resolution)
    print(f"  DeepLIIF: {time.time() - t0:.3f}s")
    dl = results[0]
    seg_np = np.array(dl['Seg'])
    marker_np = np.array(dl['Marker'])
    if dbg is not None:
        dbg.step2_deepliif(dl)

    # -- Cell extraction --
    t0 = time.time()
    cells_info, clusters = segmentor.extract(seg_np, marker_np)
    print(f"  Cell extraction: {time.time() - t0:.3f}s -- "
          f"{len(cells_info)} regions")

    if dbg is not None and cells_info:
        dbg.step3_connected_region(
            tile_np, seg_np, marker_np,
            cells_info,
            seg_thresh=args.seg_thresh,
            marker_thresh=args.marker_thresh,
            marker_percentile_factor=args.marker_percentile_factor,
            morphology_kernel=args.morphology_kernel,
        )

    if dbg is not None:
        dbg.step3_sam2_prompt(tile_np, cells_info)

    if not cells_info:
        print("  No positive cells found.")
        return

    # -- SAM2 --
    t0 = time.time()
    sam_mask, scores = sam2.segment(
        tile_np, clusters,
        tile_name=tile_name,
        positive_cells_info=cells_info,
        debug_dir=(dbg.sam2_steps_parent() if dbg is not None else None),
    )
    print(f"  SAM2: {time.time() - t0:.3f}s -- {len(scores)} kept")
    if dbg is not None:
        dbg.step4_sam2_raw(tile_np, sam_mask)

    merged_mask, _, _, _ = merge_connected_masks(
        sam_mask, scores, cells_info, min_area=200,
    )
    if dbg is not None:
        dbg.step5_merged(tile_np, merged_mask)
        dbg.step7_sam2_merge_diff(tile_np, sam_mask, merged_mask, cells_info)
    else:
        # Non-debug-vis path: keep the legacy single-file merge-diff artefact.
        save_sam2_merge_diff(
            tile_np, sam_mask, merged_mask, cells_info,
            os.path.join(args.output_dir, "debug", f"{stem}_sam2_merge_diff.png"),
        )

    # -- Post-process (npy + per-tile bookkeeping) --
    t0 = time.time()
    saved = postprocessor.merge_and_process(sam_mask, scores, cells_info, tile_name)
    print(f"  Merge+Process: {time.time() - t0:.3f}s -- saved={saved}")


def run_single_tile_debug(args):
    """Single-tile / 3x3-neighborhood debug flow (shares one model load)."""
    from cd34_pipeline.io.wsi_reader import WSIReader

    if args.debug_3x3 and not args.debug_vis:
        print("ERROR: --debug-3x3 requires --debug-vis.")
        return

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

    tiles_to_run = (_expand_to_3x3(all_tiles, center)
                    if args.debug_3x3 else [center])
    print(f"[debug] Will process {len(tiles_to_run)} tile(s).")

    # -- Load models once --
    deepliif = DeepLIIFProcessor(args.deepliif_model_dir, args.device)
    segmentor = CellSegmentor(
        seg_thresh=args.seg_thresh,
        marker_thresh=args.marker_thresh,
        marker_percentile_factor=args.marker_percentile_factor,
        morphology_kernel=args.morphology_kernel,
        min_area=args.min_mask_area,
    )
    sam2 = SAM2Processor(
        config=args.sam_config,
        checkpoint=args.sam_checkpoint,
        device=args.device,
        batch_size=args.sam2_batch_size,
        min_area=args.min_mask_area,
        reuse_cache_dir=args.reuse_sam2_cache,
    )
    postprocessor = PostProcessor(
        output_dir=args.output_dir, min_area=200,
        tile_size=args.tile_size, overlap=args.overlap,
    )

    try:
        for t in tiles_to_run:
            _process_one_tile_debug(
                args, wsi_reader, t,
                deepliif, segmentor, sam2, postprocessor,
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
    if args.debug_region_um is not None and args.debug_3x3:
        print("ERROR: --debug-3x3 is only valid with --tile-index or --tile-um.")
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
    print(f"  ROI JSON:          {args.roi_json}")
    print(f"  DeepLIIF batch_size: {args.deepliif_batch_size}")
    print(f"  SAM2 batch_size:   {args.sam2_batch_size}")
    print(f"  Bucket capacity:   {args.bucket_capacity} iterations "
          f"(queue={queue_capacity} items)")
    print(f"  Cache DeepLIIF:    {'ON' if args.cache_deepliif else 'OFF'}")
    print(f"  Cache SAM2:        {'ON' if args.cache_sam2 else 'OFF'}")
    print(f"  Reuse SAM2 cache:  {args.reuse_sam2_cache or 'OFF'}")
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
    consumer = Consumer(args, bucket, producer_done, stats, progress)

    producer_thread = Thread(target=producer.run, name="Producer", daemon=True)
    consumer_thread = Thread(target=consumer.run, name="Consumer", daemon=True)

    print("[Main] Starting producer and consumer threads...")
    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

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
    print(f"  Skipped (no cells):  {stats.get('skipped_no_cells', '?')}")
    print(f"  Consumed items:      {stats.get('consumed', '?')}")
    print(f"  Masks saved:         {masks_saved}")
    print(f"  Refill cycles:       {stats.get('refill_count', '?')}")
    print(f"{'='*60}")
    print(f"  Producer time:       {stats.get('producer_time', 0):.1f}s "
          f"(DeepLIIF + CellExtract)")
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
                min_area=args.min_mask_area,
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
