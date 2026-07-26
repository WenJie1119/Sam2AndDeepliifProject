"""
cell/postprocess.py -- Post-processing: mask merging + in-memory storage.

Instead of writing intermediate .npy files, this module:
  1. Extracts tile-level polygons (Pass 1) immediately during merge_and_process()
  2. Keeps full masks in memory for cross-tile overlap matching (Pass 2)
  3. Passes both to export_geojson() directly -- zero disk I/O for masks.
"""

import os
import threading

import cv2
import numpy as np

from cd34_pipeline.io.tile_reconstruction import (
    parse_tile_filename,
    _extract_tile_polygons,
)


def _fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    """Fill only holes fully enclosed by a binary mask."""
    if not mask.any():
        return mask

    src = mask.astype(np.uint8, copy=False)
    padded = np.pad(src, 1, mode="constant", constant_values=0)
    exterior = padded.copy()
    cv2.floodFill(exterior, None, (0, 0), 1)
    holes = exterior[1:-1, 1:-1] == 0
    return mask | holes


def _fill_instance_holes(instance_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Fill enclosed background holes inside each instance label."""
    if instance_mask is None or instance_mask.size == 0:
        return instance_mask, 0

    result = instance_mask.copy()
    original = instance_mask
    filled_px = 0
    for inst_id in np.unique(original):
        if inst_id == 0:
            continue
        region = original == inst_id
        filled = _fill_binary_holes(region)
        holes = filled & ~region & (result == 0)
        if holes.any():
            result[holes] = inst_id
            filled_px += int(holes.sum())
    return result, filled_px


class PostProcessor:
    """Merges SAM2 masks into instance labels, extracts polygons in memory,
    then exports GeoJSON without intermediate .npy files."""

    def __init__(self, output_dir: str, min_area: int = 0,
                 tile_size: int = 512, overlap: int = 128,
                 stitch_mode: str = "center-valid",
                 tile_records: list[dict] | None = None,
                 debug_region_metadata: dict | None = None,
                 debug_region_tiles: list[dict] | None = None,
                 fill_sam_holes: bool = False):
        self.output_dir = output_dir
        self.min_area = min_area
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.stitch_mode = stitch_mode.replace("_", "-")
        if self.stitch_mode not in {
                "center-valid", "center-valid-raw", "overlap-merge"}:
            raise ValueError(
                "stitch_mode must be 'center-valid', 'center-valid-raw', "
                "or 'overlap-merge'")
        self.tile_keys = (
            {(int(t["row"]), int(t["col"])) for t in tile_records}
            if tile_records is not None else None
        )
        self.debug_region_metadata = debug_region_metadata
        self.debug_region_tiles = debug_region_tiles or []
        self.fill_sam_holes = fill_sam_holes
        self.debug_region_dir = (
            os.path.join(output_dir, "debug_region")
            if debug_region_metadata is not None else None
        )

        # Pass 1 result: {(row, col, inst_id): [Polygon, ...]}
        self.poly_map: dict = {}
        # Stitching masks for Pass 2: {(row, col): ndarray}.  In center-valid
        # mode these stay uncropped so overlap evidence can still link tiles.
        self.masks: dict = {}
        # Region debug tiles: {(row, col): {'tile_np', 'sam_mask', 'merged_mask'}}
        self.debug_tiles: dict = {}
        self._debug_tile_artifacts_written = False

        self._processed = 0
        self._lock = threading.Lock()

    @property
    def saved_count(self) -> int:
        return self._processed

    def merge_and_process(self, sam_mask: np.ndarray, scores: list,
                          positive_cells_info: list,
                          tile_name: str,
                          tile_np: np.ndarray | None = None,
                          tile_info: dict | None = None) -> bool:
        """
        Merge SAM2 masks, extract polygons and store mask in memory.

        Args:
            sam_mask: raw SAM2 mask array
            scores: score list from SAM2
            positive_cells_info: optional region metadata; weighted-points
                passes an empty list.
            tile_name: tile identifier (e.g. "tile_5_12_5632_2048")

        Returns:
            True if a non-empty mask was processed.
        """
        from cd34_pipeline.sam2_wrapper.inference import merge_connected_masks

        debug_enabled = (
            self.debug_region_metadata is not None and tile_np is not None
        )
        if debug_enabled:
            from cell.debug_vis import DebugVisualizer
            dbg = DebugVisualizer(self.output_dir, tile_name)
            dbg.step4_sam2_raw(tile_np, sam_mask)
            dbg.clear_step5_merge_outputs()
            merged, _, _, _ = merge_connected_masks(
                sam_mask, scores, positive_cells_info,
                min_area=self.min_area,
                debug_dir=dbg.dir,
                debug_prefix="step5",
                original_image=tile_np,
            )
            if self.fill_sam_holes:
                merged, _ = _fill_instance_holes(merged)
            dbg.step5_merged(tile_np, merged)
            dbg.step7_sam2_merge_diff(
                tile_np, sam_mask, merged, positive_cells_info)
        else:
            merged, _, _, _ = merge_connected_masks(
                sam_mask, scores, positive_cells_info,
                min_area=self.min_area,
            )
            if self.fill_sam_holes:
                merged, _ = _fill_instance_holes(merged)

        if np.max(merged) > 0:
            max_id = int(np.max(merged))
            if max_id <= 255:
                merged = merged.astype(np.uint8)
            elif max_id <= 65535:
                merged = merged.astype(np.uint16)
            else:
                merged = merged.astype(np.uint32)

            # Parse tile position from name
            parsed = parse_tile_filename(tile_name + ".npy")
            if parsed is None:
                return False
            row, col, x_off, y_off = parsed

            # Compute global offset
            if x_off is not None:
                gx, gy = x_off, y_off
            else:
                gx = col * self.stride
                gy = row * self.stride

            stitch_mask = merged
            export_mask = self._apply_center_valid_crop(
                merged, row, col, tile_info)
            if debug_enabled:
                self._store_debug_tile(tile_name, tile_info, tile_np,
                                       sam_mask, export_mask)
            if np.max(export_mask) == 0:
                return False

            # -- Pass 1: extract polygons (CPU-intensive, no lock needed) --
            tile_polys = _extract_tile_polygons(export_mask, gx, gy)

            # -- Thread-safe update of shared state --
            with self._lock:
                for inst_id, poly in tile_polys:
                    gid = (row, col, inst_id)
                    if gid not in self.poly_map:
                        self.poly_map[gid] = []
                    self.poly_map[gid].append(poly)

                # -- Store mask for Pass 2 overlap matching --
                self.masks[(row, col)] = stitch_mask

                self._processed += 1
            return True
        return False

    def _apply_center_valid_crop(self, mask: np.ndarray, row: int, col: int,
                                 tile_info: dict | None) -> np.ndarray:
        """Keep only the owner/center-valid part of a tile mask."""
        if (self.stitch_mode not in {"center-valid", "center-valid-raw"}
                or self.overlap <= 0):
            return mask

        h, w = mask.shape[:2]
        if h == 0 or w == 0:
            return mask

        half_left = self.overlap // 2
        half_top = self.overlap // 2
        half_right = self.overlap - half_left
        half_bottom = self.overlap - half_top

        def _has_tile(r: int, c: int) -> bool:
            return self.tile_keys is None or (r, c) in self.tile_keys

        crop_left = half_left if _has_tile(row, col - 1) else 0
        crop_top = half_top if _has_tile(row - 1, col) else 0
        crop_right = half_right if _has_tile(row, col + 1) else 0
        crop_bottom = half_bottom if _has_tile(row + 1, col) else 0

        x0 = min(crop_left, w)
        y0 = min(crop_top, h)
        x1 = min(self.tile_size - crop_right, w)
        y1 = min(self.tile_size - crop_bottom, h)

        if tile_info is not None:
            actual_w = int(tile_info.get("actual_w", w))
            actual_h = int(tile_info.get("actual_h", h))
            x1 = min(x1, actual_w, w)
            y1 = min(y1, actual_h, h)

        if x1 <= x0 or y1 <= y0:
            return np.zeros_like(mask)

        center_mask = np.zeros_like(mask)
        center_mask[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        return center_mask

    def _store_debug_tile(self, tile_name: str, tile_info: dict | None,
                          tile_np: np.ndarray, sam_mask: np.ndarray,
                          merged_mask: np.ndarray) -> None:
        """Keep per-tile arrays for region-level debug mosaics."""
        parsed = parse_tile_filename(tile_name + ".npy")
        if parsed is None:
            return
        row, col, x_off, y_off = parsed
        info = dict(tile_info or {})
        info.setdefault("row", row)
        info.setdefault("col", col)
        if x_off is not None:
            info.setdefault("x", x_off)
            info.setdefault("y", y_off)
        with self._lock:
            self.debug_tiles[(row, col)] = {
                "tile_info": info,
                "tile_np": tile_np.copy(),
                "sam_mask": sam_mask.copy(),
                "merged_mask": merged_mask.copy(),
            }

    def shutdown(self) -> None:
        """No-op (no async I/O to flush)."""
        pass

    def write_debug_region_tile_artifacts(self) -> None:
        """Write raw/tile-merged region mosaics once, if debug data exists."""
        if self.debug_region_dir is None:
            return
        if self._debug_tile_artifacts_written:
            return
        from cell.region_debug import write_tile_region_artifacts
        write_tile_region_artifacts(
            self.output_dir,
            self.debug_region_metadata,
            self.debug_region_tiles,
            self.debug_tiles,
        )
        self._debug_tile_artifacts_written = True

    # -- GeoJSON export (runs after all tiles are processed) --

    def export_geojson(self, wsi_path: str, tile_size: int, overlap: int,
                       simplify: float = 0, contour_tolerance: float = 0.5,
                       min_area: int = 50,
                       level_downsample: float = 1.0,
                       crop_origin: tuple[int, int] | None = None) -> str | None:
        """
        Export in-memory polygons to QuPath-compatible GeoJSON.

        Pass 1 already done during merge_and_process().
        Pass 2: overlap-region pixel matching -> Union-Find cross-tile merge
        Pass 3: unary_union -> simplify -> GeoJSON features

        Returns:
            geojson_path on success, None if no masks exist.
        """
        from cd34_pipeline.io.tile_reconstruction import export_geojson
        from cd34_pipeline.io.file_io import compute_geojson_statistics

        if not self.poly_map:
            print("[PostProcess] No mask data in memory, skipping GeoJSON export.")
            return None

        wsi_stem = os.path.splitext(os.path.basename(wsi_path))[0]
        geojson_path = os.path.join(self.output_dir, f"{wsi_stem}.geojson")
        stride = tile_size - overlap

        print(f"\n[PostProcess] Exporting GeoJSON for QuPath (in-memory mode)...")

        self.write_debug_region_tile_artifacts()

        export_geojson(
            tile_dir=None,
            output_path=geojson_path,
            tile_size=tile_size,
            stride=stride,
            simplify=simplify,
            contour_tolerance=contour_tolerance,
            min_area=min_area,
            poly_map=self.poly_map,
            masks=self.masks,
            level_downsample=level_downsample,
            crop_origin=crop_origin,
            debug_dir=self.debug_region_dir,
            merge_mode=self.stitch_mode,
        )

        if self.debug_region_dir is not None:
            from cell.region_debug import write_geojson_region_artifacts
            write_geojson_region_artifacts(
                self.output_dir,
                self.debug_region_metadata,
                self.debug_region_tiles,
                self.debug_tiles,
                geojson_path,
            )

        # Free memory after export
        self.poly_map.clear()
        self.masks.clear()
        self.debug_tiles.clear()

        print(f"\n[PostProcess] Computing GeoJSON statistics...")
        geojson_stats = compute_geojson_statistics(geojson_path, self.output_dir)
        print(f"  Region count:  {geojson_stats['count']}")
        print(f"  Area mean:     {geojson_stats['area_mean']:.2f} px^2")
        print(f"  Area std:      {geojson_stats['area_std']:.2f} px^2")

        return geojson_path
