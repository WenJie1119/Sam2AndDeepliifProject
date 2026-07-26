"""
cell/utils.py -- Shared infrastructure for the CD34 pipeline.

Contents:
  - Bucket: fixed-capacity queue with backpressure
  - BucketItem: minimal payload for SAM2 consumer
  - StickyProgress: APT-style terminal progress bar
  - load_roi_csv / filter_tiles_by_roi: ROI filtering utilities
"""

import csv
import json
import math
import os
import queue
import shutil
import sys
import time
import threading
from dataclasses import dataclass
from threading import Thread
from typing import Optional

import cv2
import numpy as np


# ============================================================================
# Bucket -- fixed-capacity queue with backpressure
# ============================================================================

class Bucket:
    """
    Fixed-size bucket backed by a bounded queue.

    - put() blocks when the bucket is full  (backpressure on producer)
    - get() blocks when the bucket is empty  (consumer waits)
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._queue: queue.Queue = queue.Queue(maxsize=capacity)

    def put(self, item) -> None:
        """Put an item into the bucket.  Blocks if bucket is full."""
        self._queue.put(item)

    def get(self, timeout: float = 0.5):
        """Take an item.  Blocks up to *timeout* seconds if empty."""
        return self._queue.get(timeout=timeout)

    # -- Inspectors --

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def full(self) -> bool:
        return self._queue.full()


# ============================================================================
# BucketItem -- minimal payload for the SAM2 consumer
# ============================================================================

@dataclass
class BucketItem:
    """Minimal weighted-prompt payload for the SAM2 consumer."""
    tile_np: np.ndarray         # RGB image (H, W, 3)
    positive_cells_info: list   # cell dicts for merge_connected_masks
    tile_info: dict             # WSI tile position info
    tile_name: str              # e.g. "tile_5_12_5632_2048"
    mask_input: Optional[np.ndarray] = None
    point_coords: Optional[np.ndarray] = None
    point_labels: Optional[np.ndarray] = None
    prompt_stats: Optional[dict] = None
    prompt_debug_dir: Optional[str] = None


# ============================================================================
# AsyncSaver -- multi-worker background disk writer
# ============================================================================

class AsyncSaver:
    """Multi-worker background saver that writes numpy arrays to disk
    without blocking the caller's GPU pipeline.

    Features:
      - Multiple worker threads for parallel I/O
      - Backpressure via bounded queue (blocks producer when full)
      - Directory creation cached to avoid redundant syscalls
    """

    def __init__(self, num_workers: int = 2, maxsize: int = 200):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._threads = [
            threading.Thread(target=self._worker, daemon=True,
                             name=f"AsyncSaver-{i}")
            for i in range(num_workers)
        ]
        for t in self._threads:
            t.start()
        self._saved = 0
        self._lock = threading.Lock()
        self._dirs_created: set = set()

    @property
    def saved(self) -> int:
        with self._lock:
            return self._saved

    def submit(self, data, path: str,
               allow_pickle: bool = False) -> None:
        """Queue data for async disk write.
        Blocks if internal queue is full (backpressure)."""
        self._queue.put((data, path, allow_pickle))

    def shutdown(self) -> None:
        """Signal all workers to exit and wait for pending writes."""
        for _ in self._threads:
            self._queue.put(None)
        for t in self._threads:
            t.join()

    def _ensure_dir(self, path: str) -> None:
        dir_path = os.path.dirname(path)
        if dir_path and dir_path not in self._dirs_created:
            os.makedirs(dir_path, exist_ok=True)
            self._dirs_created.add(dir_path)

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            data, path, allow_pickle = item
            self._ensure_dir(path)
            np.save(path, data, allow_pickle=allow_pickle)
            with self._lock:
                self._saved += 1


# ============================================================================
# StickyProgress -- APT-style progress bar pinned to terminal bottom
# ============================================================================

class StickyProgress:
    """
    APT-style sticky progress bar pinned to the last terminal line.

    Uses ANSI scroll regions: the top (rows-1) lines scroll normally,
    the last line is reserved for the progress bar.  A background thread
    redraws the bar every 0.5 s.
    """

    def __init__(self, total_tiles: int, bucket_capacity: int):
        self.total_tiles = total_tiles
        self.bucket_capacity = bucket_capacity
        self._consumed = 0
        self._tile_cursor = 0
        self._bucket_level = 0
        self._flushed = 0
        self._produced = 0
        # Latest round funnel: chunk_in → produced_out
        self._last_chunk_in = 0
        self._last_filter_out = 0
        self._last_produced_out = 0
        # Phase cumulative timing
        self._prefetch_time = 0.0
        self._prefetch_calls = 0
        self._deepliif_time = 0.0
        self._deepliif_calls = 0
        self._sam2_time = 0.0
        self._sam2_calls = 0
        self._lock = threading.Lock()
        self._t0 = time.time()
        self._stop = threading.Event()
        self._active = False
        self._snapshots: list[dict] = []

        if not sys.stdout.isatty():
            return

        try:
            cols, rows = shutil.get_terminal_size()
            sys.stdout.write(f"\033[1;{rows - 1}r")
            sys.stdout.write(f"\033[{rows - 1};1H")
            sys.stdout.flush()
            self._active = True
            self._thread = Thread(target=self._loop, daemon=True,
                                  name="ProgressBar")
            self._thread.start()
        except Exception:
            pass

    # -- Public API --

    def update(self, *, consumed=None, tile_cursor=None,
               bucket_level=None, flushed=None, produced=None,
               deepliif_dt=None, sam2_dt=None,
               round_chunk_in=None, round_filter_out=None,
               round_produced_out=None):
        if not self._active:
            return
        with self._lock:
            if consumed is not None:
                self._consumed = consumed
            if tile_cursor is not None:
                self._tile_cursor = tile_cursor
            if bucket_level is not None:
                self._bucket_level = bucket_level
            if flushed is not None:
                self._flushed = flushed
            if produced is not None:
                self._produced = produced
            if deepliif_dt is not None:
                self._deepliif_time += deepliif_dt
                self._deepliif_calls += 1
            if sam2_dt is not None:
                self._sam2_time += sam2_dt
                self._sam2_calls += 1
            if round_chunk_in is not None:
                self._last_chunk_in = round_chunk_in
            if round_filter_out is not None:
                self._last_filter_out = round_filter_out
            if round_produced_out is not None:
                self._last_produced_out = round_produced_out

    def close(self):
        if not self._active:
            return
        self._stop.set()
        self._thread.join(timeout=2)
        try:
            cols, rows = shutil.get_terminal_size()
            sys.stdout.write(f"\033[{rows};1H\033[2K")
            sys.stdout.write(f"\033[1;{rows}r")
            sys.stdout.write(f"\033[{rows};1H")
            sys.stdout.flush()
        except Exception:
            pass
        self._active = False

    @property
    def snapshots(self) -> list[dict]:
        """Return collected time-series snapshots."""
        return self._snapshots

    # -- Internal --

    def _loop(self):
        while not self._stop.wait(0.5):
            self._draw()
        self._draw()

    def _draw(self):
        try:
            cols, rows = shutil.get_terminal_size()
        except Exception:
            return

        with self._lock:
            consumed = self._consumed
            tcur = self._tile_cursor
            blevel = self._bucket_level
            produced = self._produced
            dl_avg = (self._deepliif_time / self._deepliif_calls
                      if self._deepliif_calls else 0)
            sam2_avg = (self._sam2_time / self._sam2_calls
                        if self._sam2_calls else 0)
            dl_total = self._deepliif_time
            sam2_total = self._sam2_time
            chunk_in = self._last_chunk_in
            prod_out = self._last_produced_out

        total = self.total_tiles
        bcap = self.bucket_capacity
        elapsed = time.time() - self._t0

        pct = tcur / total * 100 if total > 0 else 0
        dl_speed = produced / dl_total if dl_total > 0 else 0
        sam2_speed = consumed / sam2_total if sam2_total > 0 else 0
        in_flight = max(0, produced - consumed - blevel)

        if tcur < total:
            tile_rate = tcur / elapsed if elapsed > 0 else 0
            eta_s = (total - tcur) / tile_rate if tile_rate > 0 else 0
        else:
            eta_s = blevel / sam2_speed if sam2_speed > 0 else 0

        eta_str = self._fmt_time(eta_s)
        el_str = self._fmt_time(elapsed)

        # Bottleneck marker: '*' on the phase with highest cumulative time
        phase_times: dict[str, float] = {
            'DL': dl_total, 'S': sam2_total}
        bottleneck = (max(phase_times, key=lambda k: phase_times[k])
                      if any(v > 0 for v in phase_times.values()) else '')
        def bn(tag):
            return '*' if bottleneck == tag else ''

        # Latest round funnel: e.g. "500→65"
        funnel = (f"{chunk_in}\u2192{prod_out}"
                  if chunk_in > 0 else "")

        if cols >= 150:
            suffix = (f" {pct:5.1f}%  "
                      f"Tile {tcur}/{total} | "
                      f"DL {produced} {dl_avg:.1f}s{bn('DL')} "
                      f"{dl_speed:.1f}/s | "
                      f"SAM2 {consumed} {sam2_avg:.1f}s{bn('S')} "
                      f"{sam2_speed:.1f}/s | "
                      f"bkt {blevel}/{bcap} fly {in_flight} | "
                      f"round {funnel} | "
                      f"{el_str} ETA {eta_str}")
        else:
            suffix = (f" {pct:4.0f}%  "
                      f"T:{tcur}/{total} | "
                      f"DL:{produced} {dl_avg:.1f}s{bn('DL')} "
                      f"{dl_speed:.1f}/s | "
                      f"S:{consumed} {sam2_avg:.1f}s{bn('S')} "
                      f"{sam2_speed:.1f}/s | "
                      f"bkt:{blevel} fly:{in_flight} | "
                      f"{funnel} | "
                      f"{el_str} ETA {eta_str}")

        bar_w = max(10, cols - len(suffix) - 4)
        filled = int(bar_w * pct / 100)
        bar = '\u2588' * filled + '\u2591' * (bar_w - filled)

        line = f" [{bar}]{suffix}"
        line = line[:cols]

        sys.stdout.write(
            f"\033[s\033[{rows};1H\033[2K\033[7m{line}\033[0m\033[u")
        sys.stdout.flush()

        self._snapshots.append({
            'elapsed': elapsed, 'pct': pct,
            'tile_cursor': tcur, 'produced': produced,
            'consumed': consumed, 'blevel': blevel,
            'in_flight': in_flight,
            'dl_avg': dl_avg, 'sam2_avg': sam2_avg,
            'dl_speed': dl_speed, 'sam2_speed': sam2_speed,
        })

    @staticmethod
    def _fmt_time(secs):
        s = int(secs)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ============================================================================
# ROI CSV utilities
# ============================================================================

def load_roi_csv(csv_path: str) -> list[dict]:
    """
    Load ROI rectangles from a CSV file.

    Expected CSV format (header required):
        filename,x,y,width,height

    Returns:
        list of dict: [{x, y, w, h}, ...] in level-0 coordinates
    """
    rois = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rois.append({
                'x': int(row['x']),
                'y': int(row['y']),
                'w': int(row['width']),
                'h': int(row['height']),
            })
    print(f"  Loaded {len(rois)} ROI(s) from {csv_path}")
    for i, r in enumerate(rois):
        print(f"    ROI[{i}]: x={r['x']} y={r['y']} "
              f"w={r['w']} h={r['h']}")
    return rois


def filter_tiles_by_roi(tiles: list[dict], rois: list[dict],
                        level_downsample: float) -> list[dict]:
    """
    Keep only tiles that overlap with at least one ROI rectangle.

    Both tile coordinates and ROI coordinates are in level-0 pixel space.
    """
    if not rois:
        return tiles

    roi_bounds = [(r['x'], r['y'], r['x'] + r['w'], r['y'] + r['h'])
                  for r in rois]

    filtered = []
    for t in tiles:
        tx1 = t['x_level0']
        ty1 = t['y_level0']
        tx2 = tx1 + int(t['actual_w'] * level_downsample)
        ty2 = ty1 + int(t['actual_h'] * level_downsample)

        for rx1, ry1, rx2, ry2 in roi_bounds:
            if tx1 < rx2 and tx2 > rx1 and ty1 < ry2 and ty2 > ry1:
                filtered.append(t)
                break

    return filtered


# ============================================================================
# ROI JSON utilities -- polygon-based tile filtering
# ============================================================================

def load_roi_json(json_path: str) -> dict:
    """
    Load ROI JSON file containing crop_region and roi_polygon.

    Expected JSON format:
        {
          "crop_region": {"x": int, "y": int, "width": int, "height": int, "level": int},
          "roi_polygon": [[x1,y1], [x2,y2], ...]
        }

    Returns:
        dict with keys 'crop_region' and 'roi_polygon'
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    crop = data['crop_region']
    polygon = data['roi_polygon']

    print(f"  Loaded ROI JSON from {json_path}")
    print(f"    crop_region: x={crop['x']} y={crop['y']} "
          f"w={crop['width']} h={crop['height']} level={crop['level']}")
    print(f"    roi_polygon: {len(polygon)} vertices")
    return data


def _parse_region_fraction(value: str) -> float:
    """Parse a fraction from 1/4, 0.25, or 25%."""
    value = value.strip()
    if not value:
        raise ValueError("empty fraction")

    if value.endswith("%"):
        fraction = float(value[:-1]) / 100.0
    elif "/" in value:
        numerator, denominator = value.split("/", 1)
        fraction = float(numerator) / float(denominator)
    else:
        fraction = float(value)

    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {value!r}")
    return fraction


def apply_crop_region_slice(crop_region: dict, slice_spec: str | None) -> dict:
    """
    Return a smaller crop_region for quick partial runs.

    slice_spec format:
        top:1/4, bottom:0.25, left:25%, right:1/2
        top:1/4,left:1/3

    Coordinates stay in level-0 pixel space. The input dict is not modified.
    """
    if not slice_spec:
        return dict(crop_region)

    sliced = dict(crop_region)
    original = dict(crop_region)
    seen_axes = set()

    parts = [part.strip() for part in slice_spec.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError(
            "--crop-region-slice must look like top:1/4, "
            "top:1/4,left:1/3, or bottom:0.25,right:25%"
        )

    for part in parts:
        try:
            anchor, raw_fraction = part.split(":", 1)
        except ValueError as exc:
            raise ValueError(
                "--crop-region-slice must look like top:1/4, "
                "top:1/4,left:1/3, or bottom:0.25,right:25%"
            ) from exc

        anchor = anchor.strip().lower()
        if anchor not in {"top", "bottom", "left", "right"}:
            raise ValueError(
                "--crop-region-slice anchor must be one of: "
                "top, bottom, left, right"
            )

        axis = "vertical" if anchor in {"top", "bottom"} else "horizontal"
        if axis in seen_axes:
            raise ValueError(
                "--crop-region-slice can include at most one vertical slice "
                "(top/bottom) and one horizontal slice (left/right)"
            )
        seen_axes.add(axis)

        fraction = _parse_region_fraction(raw_fraction)

        if anchor in {"top", "bottom"}:
            new_h = max(1, min(crop_region['height'],
                               int(round(crop_region['height'] * fraction))))
            if anchor == "bottom":
                sliced['y'] = crop_region['y'] + crop_region['height'] - new_h
            sliced['height'] = new_h
        else:
            new_w = max(1, min(crop_region['width'],
                               int(round(crop_region['width'] * fraction))))
            if anchor == "right":
                sliced['x'] = crop_region['x'] + crop_region['width'] - new_w
            sliced['width'] = new_w

    print(f"  Applied crop_region slice: {slice_spec}")
    print(f"    original: x={original['x']} y={original['y']} "
          f"w={original['width']} h={original['height']}")
    print(f"    sliced:   x={sliced['x']} y={sliced['y']} "
          f"w={sliced['width']} h={sliced['height']}")
    return sliced


def enumerate_tiles_in_roi(crop_region: dict, roi_polygon: list,
                           tile_size: int = 512, overlap: int = 128,
                           level_downsample: float = 1.0) -> list[dict]:
    """
    Enumerate tiles within crop_region that intersect roi_polygon.

    Uses cv2.fillPoly at tile-grid resolution for O(1) per-tile lookup.

    Args:
        crop_region: dict with x, y, width, height (level-0 pixel coords)
        roi_polygon: list of [x, y] vertices (level-0 pixel coords)
        tile_size: tile size in pixels at target level
        overlap: overlap between adjacent tiles
        level_downsample: downsample factor from level-0 to target level

    Returns:
        list of tile info dicts compatible with WSIReader.read_tile()
    """
    stride = tile_size - overlap
    crop_x = crop_region['x']
    crop_y = crop_region['y']
    crop_w = crop_region['width']
    crop_h = crop_region['height']

    # Tile grid dimensions (in target-level pixel space)
    crop_w_level = int(crop_w / level_downsample)
    crop_h_level = int(crop_h / level_downsample)
    n_cols = math.ceil(crop_w_level / stride)
    n_rows = math.ceil(crop_h_level / stride)

    print(f"    Tile grid: {n_cols} cols x {n_rows} rows = {n_cols * n_rows} candidates")

    # Convert roi_polygon (level-0 pixels) → tile-grid coordinates
    # grid_col = (px - crop_x) / (stride * level_downsample)
    # grid_row = (py - crop_y) / (stride * level_downsample)
    stride_level0 = stride * level_downsample
    grid_poly = np.array([
        [
            (px - crop_x) / stride_level0,
            (py - crop_y) / stride_level0,
        ]
        for px, py in roi_polygon
    ], dtype=np.float32)

    # Rasterize polygon onto tile grid
    grid_poly_int = np.round(grid_poly).astype(np.int32)
    mask = np.zeros((n_rows, n_cols), dtype=np.uint8)
    cv2.fillPoly(mask, [grid_poly_int], 1)

    # Generate tile dicts only for mask=1 positions
    roi_positions = np.argwhere(mask == 1)  # (N, 2) -> [row, col]
    tiles = []
    for row, col in roi_positions:
        x_level = col * stride
        y_level = row * stride
        actual_w = min(tile_size, crop_w_level - x_level)
        actual_h = min(tile_size, crop_h_level - y_level)
        if actual_w <= 0 or actual_h <= 0:
            continue

        x_level0 = crop_x + int(x_level * level_downsample)
        y_level0 = crop_y + int(y_level * level_downsample)

        tiles.append({
            'row': int(row),
            'col': int(col),
            'x': x_level,
            'y': y_level,
            'x_level0': x_level0,
            'y_level0': y_level0,
            'actual_w': actual_w,
            'actual_h': actual_h,
        })

    roi_area = int(mask.sum())
    print(f"    ROI polygon mask: {roi_area}/{n_cols * n_rows} tiles inside "
          f"({roi_area / (n_cols * n_rows) * 100:.1f}%)")
    print(f"    Tiles to process: {len(tiles)}")
    return tiles


def enumerate_debug_region_tiles(crop_region: dict, roi_polygon: list,
                                 bbox_level0: tuple[int, int, int, int],
                                 tile_size: int = 512, overlap: int = 128,
                                 level_downsample: float = 1.0,
                                 neighbor_radius: int = 1) -> tuple[list[dict], dict]:
    """
    Enumerate original-pipeline-grid tiles intersecting a debug bbox.

    Unlike enumerate_tiles_in_roi(), this does not scan the whole ROI grid.
    It derives a small row/col window from bbox_level0, marks tiles that
    intersect the bbox as "core", adds one-ring neighbors, then applies the
    same tile-grid ROI polygon mask used by the main pipeline.
    """
    stride = tile_size - overlap
    crop_x = crop_region['x']
    crop_y = crop_region['y']
    crop_w = crop_region['width']
    crop_h = crop_region['height']

    crop_w_level = int(crop_w / level_downsample)
    crop_h_level = int(crop_h / level_downsample)
    n_cols = math.ceil(crop_w_level / stride)
    n_rows = math.ceil(crop_h_level / stride)
    if n_cols <= 0 or n_rows <= 0:
        return [], {
            'core_tile_count': 0,
            'neighbor_tile_count': 0,
            'selected_tile_count': 0,
        }

    min_x, min_y, max_x, max_y = bbox_level0
    if min_x > max_x:
        min_x, max_x = max_x, min_x
    if min_y > max_y:
        min_y, max_y = max_y, min_y

    stride_level0 = stride * level_downsample
    tile_size_level0 = tile_size * level_downsample

    col_start = int(math.floor((min_x - crop_x - tile_size_level0) / stride_level0)) - 1
    col_end = int(math.ceil((max_x - crop_x) / stride_level0)) + 1
    row_start = int(math.floor((min_y - crop_y - tile_size_level0) / stride_level0)) - 1
    row_end = int(math.ceil((max_y - crop_y) / stride_level0)) + 1

    col_start = max(0, col_start)
    row_start = max(0, row_start)
    col_end = min(n_cols - 1, col_end)
    row_end = min(n_rows - 1, row_end)

    def make_tile(row: int, col: int) -> Optional[dict]:
        x_level = col * stride
        y_level = row * stride
        actual_w = min(tile_size, crop_w_level - x_level)
        actual_h = min(tile_size, crop_h_level - y_level)
        if actual_w <= 0 or actual_h <= 0:
            return None
        return {
            'row': int(row),
            'col': int(col),
            'x': x_level,
            'y': y_level,
            'x_level0': crop_x + int(x_level * level_downsample),
            'y_level0': crop_y + int(y_level * level_downsample),
            'actual_w': actual_w,
            'actual_h': actual_h,
        }

    def intersects(tile: dict) -> bool:
        x0 = tile['x_level0']
        y0 = tile['y_level0']
        x1 = x0 + int(round(tile['actual_w'] * level_downsample))
        y1 = y0 + int(round(tile['actual_h'] * level_downsample))
        return x0 < max_x and x1 > min_x and y0 < max_y and y1 > min_y

    core: set[tuple[int, int]] = set()
    for row in range(row_start, row_end + 1):
        for col in range(col_start, col_end + 1):
            tile = make_tile(row, col)
            if tile is not None and intersects(tile):
                core.add((row, col))

    roles: dict[tuple[int, int], str] = {}
    for rc in core:
        roles[rc] = 'core'
        row, col = rc
        for dr in range(-neighbor_radius, neighbor_radius + 1):
            for dc in range(-neighbor_radius, neighbor_radius + 1):
                nr = row + dr
                nc = col + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    roles.setdefault((nr, nc), 'neighbor')

    if not roles:
        print("    Debug region: no grid tiles intersect bbox before ROI filtering")
        return [], {
            'core_tile_count': 0,
            'neighbor_tile_count': 0,
            'selected_tile_count': 0,
        }

    min_row = min(r for r, _ in roles)
    max_row = max(r for r, _ in roles)
    min_col = min(c for _, c in roles)
    max_col = max(c for _, c in roles)

    # Same ROI rasterization as enumerate_tiles_in_roi(), but clipped to the
    # candidate debug window to avoid building a full-slide tile-grid mask.
    grid_poly_int = np.round(np.array([
        [
            (px - crop_x) / stride_level0,
            (py - crop_y) / stride_level0,
        ]
        for px, py in roi_polygon
    ], dtype=np.float32)).astype(np.int32)
    local_poly = grid_poly_int - np.array([min_col, min_row], dtype=np.int32)
    roi_mask = np.zeros((max_row - min_row + 1, max_col - min_col + 1),
                        dtype=np.uint8)
    cv2.fillPoly(roi_mask, [local_poly], 1)

    tiles = []
    for row, col in sorted(roles):
        if roi_mask[row - min_row, col - min_col] != 1:
            continue
        tile = make_tile(row, col)
        if tile is None:
            continue
        tile['debug_role'] = roles[(row, col)]
        tiles.append(tile)

    core_count = sum(1 for t in tiles if t['debug_role'] == 'core')
    neighbor_count = sum(1 for t in tiles if t['debug_role'] == 'neighbor')
    print(f"    Debug region candidates: {len(roles)} tiles "
          f"({len(core)} core before ROI filter)")
    print(f"    Debug region selected: {len(tiles)} tiles "
          f"({core_count} core, {neighbor_count} neighbor)")

    return tiles, {
        'core_tile_count': core_count,
        'neighbor_tile_count': neighbor_count,
        'selected_tile_count': len(tiles),
        'candidate_tile_count': len(roles),
        'core_before_roi_count': len(core),
        'candidate_window': {
            'row_min': min_row,
            'row_max': max_row,
            'col_min': min_col,
            'col_max': max_col,
        },
    }


# ============================================================================
# Metrics plotting
# ============================================================================

def generate_metrics_plots(snapshots: list[dict], output_dir: str) -> list[str]:
    """
    Generate time-series metrics plots from pipeline run snapshots.

    Returns list of saved file paths (empty if no data).
    """
    if len(snapshots) < 2:
        print("[Metrics] Not enough data points for plots, skipping.")
        return []

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Extract columns
    t = [s['elapsed'] / 60.0 for s in snapshots]  # minutes
    produced = [s['produced'] for s in snapshots]
    consumed = [s['consumed'] for s in snapshots]
    blevel = [s['blevel'] for s in snapshots]
    dl_speed = [s['dl_speed'] for s in snapshots]
    sam2_speed = [s['sam2_speed'] for s in snapshots]
    dl_avg = [s['dl_avg'] for s in snapshots]
    sam2_avg = [s['sam2_avg'] for s in snapshots]
    pct = [s['pct'] for s in snapshots]
    in_flight = [s['in_flight'] for s in snapshots]

    saved = []

    # -- Plot 1: Throughput --
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, produced, label='DL produced', linewidth=1.5)
    ax.plot(t, consumed, label='SAM2 consumed', linewidth=1.5)
    ax.plot(t, blevel, label='Bucket level', linewidth=1.2, linestyle='--')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Items')
    ax.set_title('Pipeline Throughput')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(output_dir, 'metrics_throughput.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # -- Plot 2: Speed --
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, dl_speed, label='DeepLIIF speed', linewidth=1.5)
    ax.plot(t, sam2_speed, label='SAM2 speed', linewidth=1.5)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Items / s (GPU time)')
    ax.set_title('Processing Speed')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(output_dir, 'metrics_speed.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # -- Plot 3: Batch Timing --
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, dl_avg, label='DeepLIIF avg', linewidth=1.5)
    ax.plot(t, sam2_avg, label='SAM2 avg', linewidth=1.5)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Seconds / batch')
    ax.set_title('Average Batch Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(output_dir, 'metrics_batch_timing.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # -- Plot 4: Progress + In-flight (dual Y axis) --
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_pct = '#1f77b4'
    color_fly = '#ff7f0e'
    ax1.plot(t, pct, color=color_pct, label='Progress %', linewidth=1.5)
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Progress %', color=color_pct)
    ax1.tick_params(axis='y', labelcolor=color_pct)
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    ax2.plot(t, in_flight, color=color_fly, label='In-flight', linewidth=1.2)
    ax2.set_ylabel('In-flight items', color=color_fly)
    ax2.tick_params(axis='y', labelcolor=color_fly)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.set_title('Progress and In-Flight Items')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(output_dir, 'metrics_progress.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    print(f"[Metrics] Saved {len(saved)} plots to {output_dir}")
    return saved
