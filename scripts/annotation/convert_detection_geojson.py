#!/usr/bin/env python3
"""Convert detection GeoJSON polygons into simplified QuPath annotations.

This is a post-processing tool. It reads an existing GeoJSON file, keeps the
source file unchanged, simplifies polygon vertices with adaptive tolerances,
and writes a separate annotation GeoJSON plus an aggregate summary.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

try:
    from shapely.geometry import MultiPolygon, Polygon, mapping, shape
    from shapely.geometry.base import BaseGeometry
except ImportError as exc:  # pragma: no cover - depends on the runtime environment
    raise SystemExit(
        "Shapely is required. Run this script in the CD34MVrecognition "
        "environment, for example:\n"
        "  conda run -n CD34MVrecognition python "
        "scripts/annotation/convert_detection_geojson.py --help"
    ) from exc


DEFAULT_LOW_TOLERANCES = (2.0, 1.5, 1.0)
DEFAULT_MEDIUM_TOLERANCES = (4.0, 3.0, 2.0)
DEFAULT_HIGH_TOLERANCES = (6.0, 4.0, 3.0)


@dataclass(frozen=True)
class SimplificationConfig:
    small_area: float = 100.0
    small_vertex_count: int = 12
    medium_area: float = 1000.0
    large_area: float = 10000.0
    low_tolerances: tuple[float, ...] = DEFAULT_LOW_TOLERANCES
    medium_tolerances: tuple[float, ...] = DEFAULT_MEDIUM_TOLERANCES
    high_tolerances: tuple[float, ...] = DEFAULT_HIGH_TOLERANCES
    thinness_fraction: float = 0.25
    max_area_error: float = 0.05
    min_iou: float = 0.95
    max_centroid_shift: float = 2.0
    min_ring_vertices: int = 4
    source_object_type: str = "detection"
    target_object_type: str = "annotation"
    convert_all: bool = False


@dataclass
class FeatureResult:
    feature: dict[str, Any]
    status: str
    coordinates_before: int
    coordinates_after: int
    tolerance: float | None = None


@dataclass
class ConversionStats:
    features_total: int = 0
    features_converted: int = 0
    features_simplified: int = 0
    features_kept_small: int = 0
    features_kept_quality_fallback: int = 0
    features_invalid_geometry: int = 0
    features_unsupported_geometry: int = 0
    features_skipped_object_type: int = 0
    coordinates_before: int = 0
    coordinates_after: int = 0
    tolerance_counts: Counter[str] = field(default_factory=Counter)

    def add(self, result: FeatureResult) -> None:
        self.features_total += 1
        self.coordinates_before += result.coordinates_before
        self.coordinates_after += result.coordinates_after

        if result.status != "skipped_object_type":
            self.features_converted += 1
        if result.status == "simplified":
            self.features_simplified += 1
        elif result.status == "kept_small":
            self.features_kept_small += 1
        elif result.status == "quality_fallback":
            self.features_kept_quality_fallback += 1
        elif result.status == "invalid_geometry":
            self.features_invalid_geometry += 1
        elif result.status == "unsupported_geometry":
            self.features_unsupported_geometry += 1
        elif result.status == "skipped_object_type":
            self.features_skipped_object_type += 1

        if result.tolerance is not None:
            self.tolerance_counts[f"{result.tolerance:g}"] += 1

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tolerance_counts"] = dict(
            sorted(self.tolerance_counts.items(), key=lambda item: float(item[0]))
        )
        before = self.coordinates_before
        data["coordinate_reduction_fraction"] = (
            1.0 - self.coordinates_after / before if before else 0.0
        )
        return data


def parse_tolerances(value: str) -> tuple[float, ...]:
    try:
        tolerances = tuple(
            sorted(
                {float(item.strip()) for item in value.split(",") if item.strip()},
                reverse=True,
            )
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid tolerance list: {value!r}"
        ) from exc
    if not tolerances or any(value <= 0 for value in tolerances):
        raise argparse.ArgumentTypeError(
            "Tolerances must be a comma-separated list of positive numbers"
        )
    return tolerances


def count_coordinates(geometry: dict[str, Any] | None) -> int:
    if not geometry:
        return 0
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        return sum(len(ring) for ring in coordinates)
    if geometry_type == "MultiPolygon":
        return sum(len(ring) for polygon in coordinates for ring in polygon)
    return 0


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    return []


def _ring_vertex_counts(geometry: BaseGeometry) -> list[list[int]]:
    counts: list[list[int]] = []
    for polygon in _polygon_parts(geometry):
        rings = [polygon.exterior, *polygon.interiors]
        counts.append([max(0, len(ring.coords) - 1) for ring in rings])
    return counts


def _topology_signature(geometry: BaseGeometry) -> tuple[int, tuple[int, ...]]:
    parts = _polygon_parts(geometry)
    return len(parts), tuple(len(polygon.interiors) for polygon in parts)


def _candidate_tolerances(
    geometry: BaseGeometry, config: SimplificationConfig
) -> tuple[float, ...]:
    area = geometry.area
    if area < config.medium_area:
        base = config.low_tolerances
    elif area < config.large_area:
        base = config.medium_tolerances
    else:
        base = config.high_tolerances

    if geometry.length <= 0:
        return ()
    estimated_thickness = 2.0 * area / geometry.length
    thinness_limit = config.thinness_fraction * estimated_thickness
    if thinness_limit <= 0:
        return ()

    return tuple(
        sorted(
            {
                round(min(tolerance, thinness_limit), 6)
                for tolerance in base
                if tolerance > 0
            },
            reverse=True,
        )
    )


def _has_enough_vertices(
    original: BaseGeometry,
    candidate: BaseGeometry,
    minimum: int,
) -> bool:
    original_counts = _ring_vertex_counts(original)
    candidate_counts = _ring_vertex_counts(candidate)
    if len(original_counts) != len(candidate_counts):
        return False

    for original_part, candidate_part in zip(original_counts, candidate_counts):
        if len(original_part) != len(candidate_part):
            return False
        for original_count, candidate_count in zip(original_part, candidate_part):
            required = min(original_count, minimum)
            if candidate_count < required:
                return False
    return True


def _passes_quality_checks(
    original: BaseGeometry,
    candidate: BaseGeometry,
    config: SimplificationConfig,
) -> bool:
    if candidate.is_empty or not candidate.is_valid:
        return False
    if _topology_signature(candidate) != _topology_signature(original):
        return False
    if not _has_enough_vertices(original, candidate, config.min_ring_vertices):
        return False

    original_area = original.area
    if original_area <= 0:
        return False
    area_error = abs(candidate.area - original_area) / original_area
    if area_error > config.max_area_error:
        return False

    union_area = original.union(candidate).area
    if union_area <= 0:
        return False
    iou = original.intersection(candidate).area / union_area
    if iou < config.min_iou:
        return False

    centroid_shift = original.centroid.distance(candidate.centroid)
    return centroid_shift <= config.max_centroid_shift


def _normalize_number(value: float) -> int | float:
    if math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def _normalize_coordinates(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        if value and all(isinstance(item, (int, float)) for item in value):
            return [_normalize_number(float(item)) for item in value]
        return [_normalize_coordinates(item) for item in value]
    return value


def geometry_to_geojson(geometry: BaseGeometry) -> dict[str, Any]:
    geojson = mapping(geometry)
    return {
        "type": geojson["type"],
        "coordinates": _normalize_coordinates(geojson["coordinates"]),
    }


def convert_feature(
    feature: dict[str, Any],
    config: SimplificationConfig,
) -> FeatureResult:
    converted = copy.deepcopy(feature)
    geometry_data = converted.get("geometry")
    before = count_coordinates(geometry_data)

    properties = converted.get("properties")
    if not isinstance(properties, dict):
        properties = {}
        converted["properties"] = properties

    source_type = properties.get("objectType")
    if not config.convert_all and source_type != config.source_object_type:
        return FeatureResult(converted, "skipped_object_type", before, before)

    properties["objectType"] = config.target_object_type

    if not isinstance(geometry_data, dict) or geometry_data.get("type") not in {
        "Polygon",
        "MultiPolygon",
    }:
        return FeatureResult(converted, "unsupported_geometry", before, before)

    try:
        original = shape(geometry_data)
    except Exception:
        return FeatureResult(converted, "invalid_geometry", before, before)

    if original.is_empty or not original.is_valid or original.area <= 0:
        return FeatureResult(converted, "invalid_geometry", before, before)

    if original.area < config.small_area or before <= config.small_vertex_count:
        return FeatureResult(converted, "kept_small", before, before)

    for tolerance in _candidate_tolerances(original, config):
        candidate = original.simplify(tolerance, preserve_topology=True)
        if not _passes_quality_checks(original, candidate, config):
            continue

        candidate_geojson = geometry_to_geojson(candidate)
        after = count_coordinates(candidate_geojson)
        if after >= before:
            continue

        converted["geometry"] = candidate_geojson
        return FeatureResult(
            converted,
            "simplified",
            before,
            after,
            tolerance=tolerance,
        )

    return FeatureResult(converted, "quality_fallback", before, before)


def _extract_features(document: Any) -> tuple[list[dict[str, Any]], str]:
    if isinstance(document, list):
        return document, "feature_array"
    if (
        isinstance(document, dict)
        and document.get("type") == "FeatureCollection"
        and isinstance(document.get("features"), list)
    ):
        return document["features"], "feature_collection"
    raise ValueError(
        "Input must be a GeoJSON Feature array or a FeatureCollection"
    )


def convert_document(
    document: Any,
    config: SimplificationConfig,
    *,
    progress_every: int = 0,
) -> tuple[Any, ConversionStats]:
    features, document_type = _extract_features(document)
    converted_features: list[dict[str, Any]] = []
    stats = ConversionStats()
    started = time.monotonic()

    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError(f"Item {index} is not a valid GeoJSON Feature")
        result = convert_feature(feature, config)
        converted_features.append(result.feature)
        stats.add(result)

        if progress_every and index % progress_every == 0:
            reduction = (
                1.0 - stats.coordinates_after / stats.coordinates_before
                if stats.coordinates_before
                else 0.0
            )
            elapsed = time.monotonic() - started
            print(
                f"  Processed {index:,}/{len(features):,} features; "
                f"coordinates reduced {reduction:.1%}; {elapsed:.1f}s",
                flush=True,
            )

    if document_type == "feature_array":
        return converted_features, stats

    converted_document = copy.deepcopy(document)
    converted_document["features"] = converted_features
    return converted_document, stats


def _atomic_json_dump(data: Any, path: Path, *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            if compact:
                json.dump(data, handle, ensure_ascii=True, separators=(",", ":"))
            else:
                json.dump(data, handle, ensure_ascii=True, indent=2)
                handle.write("\n")
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _config_to_dict(config: SimplificationConfig) -> dict[str, Any]:
    data = asdict(config)
    for key in ("low_tolerances", "medium_tolerances", "high_tolerances"):
        data[key] = list(data[key])
    return data


def convert_file(
    input_path: Path,
    output_path: Path,
    config: SimplificationConfig,
    *,
    summary_path: Path | None,
    force: bool,
    progress_every: int,
) -> dict[str, Any]:
    input_resolved = input_path.resolve()
    output_resolved = output_path.resolve()
    if input_resolved == output_resolved:
        raise ValueError("Output path must differ from the input path")
    if summary_path:
        summary_resolved = summary_path.resolve()
        if summary_resolved in {input_resolved, output_resolved}:
            raise ValueError(
                "Summary path must differ from the input and output paths"
            )
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --force to replace it."
        )
    if summary_path and summary_path.exists() and not force:
        raise FileExistsError(
            f"Summary already exists: {summary_path}. Use --force to replace it."
        )

    started = time.monotonic()
    with input_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)

    converted_document, stats = convert_document(
        document,
        config,
        progress_every=progress_every,
    )
    _atomic_json_dump(converted_document, output_path, compact=True)

    elapsed = time.monotonic() - started
    summary = {
        "input": str(input_resolved),
        "output": str(output_resolved),
        "elapsed_seconds": round(elapsed, 3),
        "input_size_bytes": input_path.stat().st_size,
        "output_size_bytes": output_path.stat().st_size,
        "config": _config_to_dict(config),
        "stats": stats.to_dict(),
    }
    if summary_path:
        _atomic_json_dump(summary, summary_path, compact=False)
    return summary


def _default_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_annotation_simplified.geojson")


def _default_summary(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_summary.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simplify an existing detection GeoJSON and convert its polygon "
            "features into editable QuPath annotations."
        )
    )
    parser.add_argument("input", type=Path, help="Existing detection GeoJSON")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path (default: <input>_annotation_simplified.geojson)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Summary JSON path (default: beside the output)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not write the aggregate summary JSON",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output and summary",
    )
    parser.add_argument("--small-area", type=float, default=100.0)
    parser.add_argument("--small-vertex-count", type=int, default=12)
    parser.add_argument("--medium-area", type=float, default=1000.0)
    parser.add_argument("--large-area", type=float, default=10000.0)
    parser.add_argument(
        "--low-tolerances",
        type=parse_tolerances,
        default=DEFAULT_LOW_TOLERANCES,
        metavar="PX_LIST",
        help="Tolerances for area < medium-area (default: 2,1.5,1)",
    )
    parser.add_argument(
        "--medium-tolerances",
        type=parse_tolerances,
        default=DEFAULT_MEDIUM_TOLERANCES,
        metavar="PX_LIST",
        help="Tolerances for medium-area <= area < large-area (default: 4,3,2)",
    )
    parser.add_argument(
        "--high-tolerances",
        type=parse_tolerances,
        default=DEFAULT_HIGH_TOLERANCES,
        metavar="PX_LIST",
        help="Tolerances for area >= large-area (default: 6,4,3)",
    )
    parser.add_argument("--thinness-fraction", type=float, default=0.25)
    parser.add_argument("--max-area-error", type=float, default=0.05)
    parser.add_argument("--min-iou", type=float, default=0.95)
    parser.add_argument("--max-centroid-shift", type=float, default=2.0)
    parser.add_argument("--min-ring-vertices", type=int, default=4)
    parser.add_argument("--source-object-type", default="detection")
    parser.add_argument("--target-object-type", default="annotation")
    parser.add_argument(
        "--convert-all",
        action="store_true",
        help="Convert polygon features regardless of their current objectType",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=2000,
        metavar="N",
        help="Print progress every N features; 0 disables it (default: 2000)",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.small_area < 0:
        parser.error("--small-area must be non-negative")
    if args.small_vertex_count < 4:
        parser.error("--small-vertex-count must be at least 4")
    if not args.small_area <= args.medium_area <= args.large_area:
        parser.error("Area thresholds must satisfy small <= medium <= large")
    if not 0 < args.thinness_fraction <= 1:
        parser.error("--thinness-fraction must be in (0, 1]")
    if not 0 <= args.max_area_error < 1:
        parser.error("--max-area-error must be in [0, 1)")
    if not 0 < args.min_iou <= 1:
        parser.error("--min-iou must be in (0, 1]")
    if args.max_centroid_shift < 0:
        parser.error("--max-centroid-shift must be non-negative")
    if args.min_ring_vertices < 3:
        parser.error("--min-ring-vertices must be at least 3")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")


def _print_summary(summary: dict[str, Any], summary_path: Path | None) -> None:
    stats = summary["stats"]
    reduction = stats["coordinate_reduction_fraction"]
    input_mb = summary["input_size_bytes"] / (1024 * 1024)
    output_mb = summary["output_size_bytes"] / (1024 * 1024)

    print("\nConversion complete")
    print(f"  Features:             {stats['features_total']:,}")
    print(f"  Converted:            {stats['features_converted']:,}")
    print(f"  Simplified:           {stats['features_simplified']:,}")
    print(f"  Kept small:           {stats['features_kept_small']:,}")
    print(
        f"  Quality fallback:     "
        f"{stats['features_kept_quality_fallback']:,}"
    )
    print(f"  Coordinates before:   {stats['coordinates_before']:,}")
    print(f"  Coordinates after:    {stats['coordinates_after']:,}")
    print(f"  Coordinate reduction: {reduction:.1%}")
    print(f"  File size:            {input_mb:.1f} MB -> {output_mb:.1f} MB")
    print(f"  Output:               {summary['output']}")
    if summary_path:
        print(f"  Summary:              {summary_path.resolve()}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    output_path = args.output or _default_output(args.input)
    summary_path = None
    if not args.no_summary:
        summary_path = args.summary or _default_summary(output_path)

    config = SimplificationConfig(
        small_area=args.small_area,
        small_vertex_count=args.small_vertex_count,
        medium_area=args.medium_area,
        large_area=args.large_area,
        low_tolerances=args.low_tolerances,
        medium_tolerances=args.medium_tolerances,
        high_tolerances=args.high_tolerances,
        thinness_fraction=args.thinness_fraction,
        max_area_error=args.max_area_error,
        min_iou=args.min_iou,
        max_centroid_shift=args.max_centroid_shift,
        min_ring_vertices=args.min_ring_vertices,
        source_object_type=args.source_object_type,
        target_object_type=args.target_object_type,
        convert_all=args.convert_all,
    )

    try:
        summary = convert_file(
            args.input,
            output_path,
            config,
            summary_path=summary_path,
            force=args.force,
            progress_every=args.progress_every,
        )
    except (FileNotFoundError, FileExistsError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Error: {exc}\n")

    _print_summary(summary, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
