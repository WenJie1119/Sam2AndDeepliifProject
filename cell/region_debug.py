"""
Region-level debug visualization for --debug-region-um.

This module only renders artifacts from already computed pipeline state.  It
does not participate in segmentation, tile merging, or GeoJSON stitching.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

import cv2
import numpy as np


REGION_ORIGINAL_MOSAIC = "01_region_original_mosaic.png"
REGION_SAM2_RAW_MOSAIC = "02_region_sam2_raw_mosaic.png"
REGION_TILE_MERGED_MOSAIC = "03_region_tile_merged_mosaic.png"
REGION_TILE_MERGE_DIFF = "04_region_tile_merge_diff.png"
REGION_GEOJSON_OVERLAY = "05_region_geojson_overlay.png"
REGION_TILE_VS_GEOJSON_DIFF = "06_region_tile_vs_geojson_diff.png"
REGION_OVERLAP_MATCHES = "07_region_overlap_matches.png"

LEGACY_REGION_PNGS = (
    "region_original_mosaic.png",
    "region_sam2_raw_mosaic.png",
    "region_tile_merged_mosaic.png",
    "region_tile_merge_diff.png",
    "region_geojson_overlay.png",
    "region_tile_vs_geojson_diff.png",
    "region_overlap_matches.png",
)


def _save_rgb(path: str, image_rgb: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def _remove_legacy_region_pngs(out_dir: str) -> None:
    for name in LEGACY_REGION_PNGS:
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            os.remove(path)


def _stable_color(seed: int) -> np.ndarray:
    return np.array([
        50 + (seed * 67) % 206,
        50 + (seed * 137) % 206,
        50 + (seed * 221) % 206,
    ], dtype=np.uint8)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _tile_name(tile: dict) -> str:
    return f"tile_{tile['row']}_{tile['col']}_{tile['x']}_{tile['y']}"


def _load_debug_original(output_root: str, tile: dict) -> Optional[np.ndarray]:
    path = os.path.join(
        output_root,
        "debug_vis",
        _tile_name(tile),
        "step1_original.png",
    )
    if not os.path.exists(path):
        return None
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _bbox_target(metadata: dict) -> Optional[list[float]]:
    bbox = metadata.get("debug_bbox_level0")
    crop_origin = metadata.get("crop_origin_level0")
    downsample = float(metadata.get("level_downsample", 1.0))
    if not bbox or not crop_origin or downsample <= 0:
        return None
    ox, oy = crop_origin
    return [
        (float(bbox[0]) - float(ox)) / downsample,
        (float(bbox[1]) - float(oy)) / downsample,
        (float(bbox[2]) - float(ox)) / downsample,
        (float(bbox[3]) - float(oy)) / downsample,
    ]


def _poly_points_target(metadata: dict) -> Optional[np.ndarray]:
    points_um = metadata.get("debug_region_um")
    crop_origin = metadata.get("crop_origin_level0")
    mpp = float(metadata.get("mpp", 0.0))
    downsample = float(metadata.get("level_downsample", 1.0))
    if not points_um or not crop_origin or mpp <= 0 or downsample <= 0:
        return None
    ox, oy = crop_origin
    pts = []
    for x_um, y_um in points_um:
        x0 = float(x_um) / mpp
        y0 = float(y_um) / mpp
        pts.append([(x0 - float(ox)) / downsample,
                    (y0 - float(oy)) / downsample])
    return np.array(pts, dtype=np.float32)


def _prepare_records(output_root: str,
                     selected_tiles: list[dict],
                     debug_tiles: dict[tuple[int, int], dict],
                     tile_size: int) -> list[dict]:
    records = []
    for tile in selected_tiles:
        key = (int(tile["row"]), int(tile["col"]))
        debug_record = debug_tiles.get(key, {})

        image = debug_record.get("tile_np")
        if image is None:
            image = _load_debug_original(output_root, tile)
        if image is None:
            image = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = image[:, :, :3]

        raw_mask = debug_record.get("sam_mask")
        merged_mask = debug_record.get("merged_mask")
        if raw_mask is None:
            raw_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        if merged_mask is None:
            merged_mask = np.zeros(image.shape[:2], dtype=np.uint16)

        records.append({
            "tile": tile,
            "tile_name": _tile_name(tile),
            "image": image,
            "sam_mask": raw_mask,
            "merged_mask": merged_mask,
        })
    return records


def _compute_canvas(records: list[dict], metadata: dict,
                    tile_size: int, max_side: int) -> dict:
    min_x = min(float(r["tile"]["x"]) for r in records)
    min_y = min(float(r["tile"]["y"]) for r in records)
    max_x = max(float(r["tile"]["x"]) + tile_size for r in records)
    max_y = max(float(r["tile"]["y"]) + tile_size for r in records)

    bbox = _bbox_target(metadata)
    if bbox is not None:
        min_x = min(min_x, bbox[0])
        min_y = min(min_y, bbox[1])
        max_x = max(max_x, bbox[2])
        max_y = max(max_y, bbox[3])

    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)
    scale = min(1.0, float(max_side) / max(width, height))
    canvas_w = max(1, int(math.ceil(width * scale)))
    canvas_h = max(1, int(math.ceil(height * scale)))

    return {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "scale": scale,
        "width": canvas_w,
        "height": canvas_h,
    }


def _to_canvas_xy(x: float, y: float, canvas: dict) -> tuple[int, int]:
    sx = int(round((float(x) - canvas["min_x"]) * canvas["scale"]))
    sy = int(round((float(y) - canvas["min_y"]) * canvas["scale"]))
    return sx, sy


def _resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return image
    h, w = image.shape[:2]
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)


def _resize_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return mask
    h, w = mask.shape[:2]
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    return cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)


def _paste_patch(canvas_img: np.ndarray, patch: np.ndarray,
                 x0: int, y0: int) -> tuple[slice, slice, np.ndarray]:
    h, w = patch.shape[:2]
    x1 = max(0, x0)
    y1 = max(0, y0)
    x2 = min(canvas_img.shape[1], x0 + w)
    y2 = min(canvas_img.shape[0], y0 + h)
    if x2 <= x1 or y2 <= y1:
        empty = patch[0:0, 0:0]
        return slice(0, 0), slice(0, 0), empty
    px1 = x1 - x0
    py1 = y1 - y0
    px2 = px1 + (x2 - x1)
    py2 = py1 + (y2 - y1)
    return slice(y1, y2), slice(x1, x2), patch[py1:py2, px1:px2]


def _draw_region_guides(image: np.ndarray, records: list[dict],
                        metadata: dict, canvas: dict,
                        tile_size: int) -> None:
    scale = canvas["scale"]
    font_scale = max(0.35, min(0.55, scale * 0.5))

    for record in records:
        tile = record["tile"]
        x0, y0 = _to_canvas_xy(tile["x"], tile["y"], canvas)
        x1, y1 = _to_canvas_xy(float(tile["x"]) + tile_size,
                               float(tile["y"]) + tile_size,
                               canvas)
        role = tile.get("debug_role", "")
        color = (255, 180, 0) if role == "core" else (80, 170, 255)
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 1)
        label = f"r{tile['row']} c{tile['col']}"
        cv2.putText(image, label, (x0 + 3, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    color, 1, cv2.LINE_AA)

    bbox = _bbox_target(metadata)
    if bbox is not None:
        x0, y0 = _to_canvas_xy(bbox[0], bbox[1], canvas)
        x1, y1 = _to_canvas_xy(bbox[2], bbox[3], canvas)
        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 255), 2)

    poly = _poly_points_target(metadata)
    if poly is not None and len(poly) >= 3:
        pts = np.array([_to_canvas_xy(x, y, canvas) for x, y in poly],
                       dtype=np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=(255, 255, 0),
                      thickness=2, lineType=cv2.LINE_AA)


def _build_original_canvas(records: list[dict], metadata: dict,
                           canvas: dict, tile_size: int,
                           draw_guides: bool = True) -> np.ndarray:
    out = np.full((canvas["height"], canvas["width"], 3), 255, dtype=np.uint8)
    for record in records:
        tile = record["tile"]
        patch = _resize_image(record["image"], canvas["scale"])
        x0, y0 = _to_canvas_xy(tile["x"], tile["y"], canvas)
        ys, xs, src = _paste_patch(out, patch, x0, y0)
        if src.size:
            out[ys, xs] = src
    if draw_guides:
        _draw_region_guides(out, records, metadata, canvas, tile_size)
    return out


def _overlay_instances(base: np.ndarray, records: list[dict],
                       canvas: dict, mask_key: str) -> np.ndarray:
    out = base.copy()
    for record in records:
        tile = record["tile"]
        mask = _resize_mask(record[mask_key], canvas["scale"])
        if mask.size == 0 or int(mask.max()) == 0:
            continue
        x0, y0 = _to_canvas_xy(tile["x"], tile["y"], canvas)
        ys, xs, src_mask = _paste_patch(out, mask, x0, y0)
        if src_mask.size == 0:
            continue
        dest = out[ys, xs]
        for inst_id in np.unique(src_mask):
            inst_id = int(inst_id)
            if inst_id <= 0:
                continue
            pixels = src_mask == inst_id
            seed = ((int(tile["row"]) + 1) * 1000003
                    + (int(tile["col"]) + 1) * 9176
                    + inst_id)
            color = _stable_color(seed).astype(np.float32)
            dest[pixels] = (
                dest[pixels].astype(np.float32) * 0.45 + color * 0.55
            ).clip(0, 255).astype(np.uint8)
    return out


def _build_binary_canvas(records: list[dict], canvas: dict,
                         mask_key: str) -> np.ndarray:
    out = np.zeros((canvas["height"], canvas["width"]), dtype=np.uint8)
    for record in records:
        tile = record["tile"]
        mask = (_resize_mask(record[mask_key], canvas["scale"]) > 0)
        x0, y0 = _to_canvas_xy(tile["x"], tile["y"], canvas)
        ys, xs, src = _paste_patch(out, mask.astype(np.uint8), x0, y0)
        if src.size:
            out[ys, xs] = np.maximum(out[ys, xs], src)
    return out


def _build_mask_diff(base: np.ndarray, raw_binary: np.ndarray,
                     merged_binary: np.ndarray) -> np.ndarray:
    out = base.copy()
    kept = raw_binary.astype(bool) & merged_binary.astype(bool)
    removed = raw_binary.astype(bool) & ~merged_binary.astype(bool)
    added = ~raw_binary.astype(bool) & merged_binary.astype(bool)

    out[kept] = (out[kept].astype(np.float32) * 0.45
                 + np.array([0, 220, 80], dtype=np.float32) * 0.55)
    out[removed] = [255, 0, 0]
    out[added] = [0, 120, 255]
    return out.astype(np.uint8)


def write_tile_region_artifacts(output_root: str,
                                metadata: dict,
                                selected_tiles: list[dict],
                                debug_tiles: dict[tuple[int, int], dict],
                                max_side: int = 4096) -> dict:
    """Write region mosaics before GeoJSON export."""
    if not selected_tiles:
        return {}

    out_dir = os.path.join(output_root, "debug_region")
    os.makedirs(out_dir, exist_ok=True)
    _remove_legacy_region_pngs(out_dir)

    tile_size = int(metadata.get("tile_size", 512))
    records = _prepare_records(output_root, selected_tiles, debug_tiles,
                               tile_size)
    canvas = _compute_canvas(records, metadata, tile_size, max_side)

    original = _build_original_canvas(records, metadata, canvas, tile_size)
    raw_overlay = _overlay_instances(original, records, canvas, "sam_mask")
    merged_overlay = _overlay_instances(original, records, canvas, "merged_mask")
    raw_binary = _build_binary_canvas(records, canvas, "sam_mask")
    merged_binary = _build_binary_canvas(records, canvas, "merged_mask")
    diff = _build_mask_diff(original, raw_binary, merged_binary)

    _save_rgb(os.path.join(out_dir, REGION_ORIGINAL_MOSAIC), original)
    _save_rgb(os.path.join(out_dir, REGION_SAM2_RAW_MOSAIC), raw_overlay)
    _save_rgb(os.path.join(out_dir, REGION_TILE_MERGED_MOSAIC),
              merged_overlay)
    _save_rgb(os.path.join(out_dir, REGION_TILE_MERGE_DIFF), diff)

    raw_instance_count = sum(
        len([i for i in np.unique(r["sam_mask"]) if int(i) > 0])
        for r in records
    )
    merged_instance_count = sum(
        len([i for i in np.unique(r["merged_mask"]) if int(i) > 0])
        for r in records
    )
    summary = {
        "tile_count": len(selected_tiles),
        "debug_tiles_with_sam": len(debug_tiles),
        "raw_tile_instances": raw_instance_count,
        "tile_merged_instances": merged_instance_count,
        "canvas": {
            "min_x": canvas["min_x"],
            "min_y": canvas["min_y"],
            "max_x": canvas["max_x"],
            "max_y": canvas["max_y"],
            "scale": canvas["scale"],
            "width": canvas["width"],
            "height": canvas["height"],
        },
        "artifacts": {
            "region_original_mosaic": REGION_ORIGINAL_MOSAIC,
            "region_sam2_raw_mosaic": REGION_SAM2_RAW_MOSAIC,
            "region_tile_merged_mosaic": REGION_TILE_MERGED_MOSAIC,
            "region_tile_merge_diff": REGION_TILE_MERGE_DIFF,
        },
    }
    _write_summary(out_dir, summary)
    return summary


def _load_summary(out_dir: str) -> dict:
    path = os.path.join(out_dir, "region_summary.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_summary(out_dir: str, summary: dict) -> None:
    with open(os.path.join(out_dir, "region_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2)


def _load_geojson_features(geojson_path: str) -> list[dict]:
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        return data.get("features", [])
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and data.get("type") == "Feature":
        return [data]
    return []


def _iter_feature_polygons(feature: dict) -> list[list[list[list[float]]]]:
    geom = feature.get("geometry") or {}
    gtype = geom.get("type")
    coords = geom.get("coordinates") or []
    if gtype == "Polygon":
        return [coords] if coords else []
    if gtype == "MultiPolygon":
        return [poly for poly in coords if poly]
    return []


def _geojson_ring_to_canvas(ring: list[list[float]], metadata: dict,
                            canvas: dict) -> Optional[np.ndarray]:
    crop_origin = metadata.get("crop_origin_level0") or [0, 0]
    downsample = float(metadata.get("level_downsample", 1.0))
    if downsample <= 0:
        return None
    ox, oy = crop_origin
    pts = []
    for point in ring:
        if len(point) < 2:
            continue
        target_x = (float(point[0]) - float(ox)) / downsample
        target_y = (float(point[1]) - float(oy)) / downsample
        pts.append(_to_canvas_xy(target_x, target_y, canvas))
    if len(pts) < 3:
        return None
    return np.array(pts, dtype=np.int32)


def _ring_bbox_target(ring: list[list[float]], metadata: dict) -> Optional[list[float]]:
    crop_origin = metadata.get("crop_origin_level0") or [0, 0]
    downsample = float(metadata.get("level_downsample", 1.0))
    if downsample <= 0 or not ring:
        return None
    ox, oy = crop_origin
    xs = [(float(p[0]) - float(ox)) / downsample for p in ring if len(p) >= 2]
    ys = [(float(p[1]) - float(oy)) / downsample for p in ring if len(p) >= 2]
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _bbox_intersects(a: list[float], b: list[float]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _rasterize_geojson(features: list[dict], metadata: dict,
                       canvas: dict) -> tuple[np.ndarray, np.ndarray, int]:
    fill = np.zeros((canvas["height"], canvas["width"]), dtype=np.uint8)
    outline = np.zeros((canvas["height"], canvas["width"], 3), dtype=np.uint8)
    extent_bbox = [canvas["min_x"], canvas["min_y"],
                   canvas["max_x"], canvas["max_y"]]
    in_extent = 0

    for feature in features:
        feature_in_extent = False
        for polygon in _iter_feature_polygons(feature):
            if not polygon:
                continue
            exterior = polygon[0]
            target_bbox = _ring_bbox_target(exterior, metadata)
            if target_bbox is None or not _bbox_intersects(target_bbox, extent_bbox):
                continue
            feature_in_extent = True
            exterior_pts = _geojson_ring_to_canvas(exterior, metadata, canvas)
            if exterior_pts is None:
                continue
            cv2.fillPoly(fill, [exterior_pts], 1)
            cv2.polylines(outline, [exterior_pts], True, (255, 64, 64), 1,
                          lineType=cv2.LINE_AA)
            hole_pts = []
            for hole in polygon[1:]:
                pts = _geojson_ring_to_canvas(hole, metadata, canvas)
                if pts is not None:
                    hole_pts.append(pts)
                    cv2.polylines(outline, [pts], True, (255, 64, 64), 1,
                                  lineType=cv2.LINE_AA)
            if hole_pts:
                cv2.fillPoly(fill, hole_pts, 0)
        if feature_in_extent:
            in_extent += 1
    return fill, outline, in_extent


def _draw_stitch_matches(base: np.ndarray, metadata: dict,
                         canvas: dict, matches_path: str) -> tuple[np.ndarray, int]:
    out = base.copy()
    if not os.path.exists(matches_path):
        return out, 0
    with open(matches_path, "r", encoding="utf-8") as f:
        matches = json.load(f)
    accepted = 0
    for rec in matches:
        if not rec.get("unioned"):
            continue
        a = rec.get("centroid_a")
        b = rec.get("centroid_b")
        if not a or not b:
            continue
        p1 = _to_canvas_xy(a[0], a[1], canvas)
        p2 = _to_canvas_xy(b[0], b[1], canvas)
        cv2.line(out, p1, p2, (255, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(out, p1, 3, (255, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, p2, 3, (255, 255, 0), -1, lineType=cv2.LINE_AA)
        accepted += 1
    return out, accepted


def write_geojson_region_artifacts(output_root: str,
                                   metadata: dict,
                                   selected_tiles: list[dict],
                                   debug_tiles: dict[tuple[int, int], dict],
                                   geojson_path: str,
                                   max_side: int = 4096) -> dict:
    """Write final GeoJSON overlay and tile-vs-GeoJSON diff."""
    if not selected_tiles or not os.path.exists(geojson_path):
        return {}

    out_dir = os.path.join(output_root, "debug_region")
    os.makedirs(out_dir, exist_ok=True)
    _remove_legacy_region_pngs(out_dir)

    tile_size = int(metadata.get("tile_size", 512))
    records = _prepare_records(output_root, selected_tiles, debug_tiles,
                               tile_size)
    canvas = _compute_canvas(records, metadata, tile_size, max_side)
    original = _build_original_canvas(records, metadata, canvas, tile_size)
    merged_binary = _build_binary_canvas(records, canvas, "merged_mask")

    features = _load_geojson_features(geojson_path)
    geo_binary, geo_outline, features_in_extent = _rasterize_geojson(
        features, metadata, canvas)

    geo_overlay = original.copy()
    geo_pixels = geo_binary.astype(bool)
    geo_overlay[geo_pixels] = (
        geo_overlay[geo_pixels].astype(np.float32) * 0.65
        + np.array([255, 80, 80], dtype=np.float32) * 0.35
    ).clip(0, 255).astype(np.uint8)
    outline_pixels = np.any(geo_outline > 0, axis=2)
    geo_overlay[outline_pixels] = geo_outline[outline_pixels]

    tile_only = merged_binary.astype(bool) & ~geo_binary.astype(bool)
    geo_only = geo_binary.astype(bool) & ~merged_binary.astype(bool)
    both = merged_binary.astype(bool) & geo_binary.astype(bool)
    diff = original.copy()
    diff[both] = (
        diff[both].astype(np.float32) * 0.45
        + np.array([0, 220, 80], dtype=np.float32) * 0.55
    ).clip(0, 255).astype(np.uint8)
    diff[tile_only] = [255, 0, 0]
    diff[geo_only] = [0, 120, 255]

    matches_path = os.path.join(out_dir, "stitch_matches.json")
    matches_overlay, accepted_count = _draw_stitch_matches(
        original, metadata, canvas, matches_path)

    _save_rgb(os.path.join(out_dir, REGION_GEOJSON_OVERLAY),
              geo_overlay)
    _save_rgb(os.path.join(out_dir, REGION_TILE_VS_GEOJSON_DIFF),
              diff)
    _save_rgb(os.path.join(out_dir, REGION_OVERLAP_MATCHES),
              matches_overlay)

    summary = _load_summary(out_dir)
    summary.update({
        "final_geojson_features": len(features),
        "final_geojson_features_in_region": features_in_extent,
        "accepted_stitch_matches_drawn": accepted_count,
        "artifacts": {
            **summary.get("artifacts", {}),
            "region_geojson_overlay": REGION_GEOJSON_OVERLAY,
            "region_tile_vs_geojson_diff": REGION_TILE_VS_GEOJSON_DIFF,
            "region_overlap_matches": REGION_OVERLAP_MATCHES,
        },
    })
    _write_summary(out_dir, summary)
    return summary
