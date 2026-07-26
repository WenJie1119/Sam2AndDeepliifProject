import json
import tempfile
import unittest
from pathlib import Path

from shapely.geometry import shape

from scripts.annotation.convert_detection_geojson import (
    SimplificationConfig,
    convert_document,
    convert_file,
)


def _feature(ring, object_type="detection"):
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {
            "objectType": object_type,
            "classification": {"name": "CD34+", "color": [200, 50, 50]},
        },
    }


def _dense_rectangle(width=100, height=40):
    ring = []
    ring.extend([[x, 0] for x in range(width + 1)])
    ring.extend([[width, y] for y in range(1, height + 1)])
    ring.extend([[x, height] for x in range(width - 1, -1, -1)])
    ring.extend([[0, y] for y in range(height - 1, 0, -1)])
    ring.append(ring[0])
    return ring


class ConvertDetectionGeojsonTest(unittest.TestCase):
    def test_simplifies_dense_polygon_and_converts_to_annotation(self):
        original_feature = _feature(_dense_rectangle())
        original_geometry = shape(original_feature["geometry"])

        converted, stats = convert_document(
            [original_feature],
            SimplificationConfig(),
        )

        output_feature = converted[0]
        output_geometry = shape(output_feature["geometry"])
        self.assertEqual(
            output_feature["properties"]["objectType"],
            "annotation",
        )
        self.assertEqual(stats.features_simplified, 1)
        self.assertLess(stats.coordinates_after, stats.coordinates_before)
        self.assertAlmostEqual(output_geometry.area, original_geometry.area)
        self.assertTrue(output_geometry.equals(original_geometry))

    def test_keeps_small_geometry_but_still_converts_object_type(self):
        feature = _feature([[0, 0], [5, 0], [5, 5], [0, 5], [0, 0]])
        original_coordinates = feature["geometry"]["coordinates"]

        converted, stats = convert_document([feature], SimplificationConfig())

        self.assertEqual(
            converted[0]["geometry"]["coordinates"],
            original_coordinates,
        )
        self.assertEqual(
            converted[0]["properties"]["objectType"],
            "annotation",
        )
        self.assertEqual(stats.features_kept_small, 1)
        self.assertEqual(stats.features_simplified, 0)

    def test_preserves_feature_collection_and_skips_existing_annotation(self):
        existing_annotation = _feature(
            [[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]],
            object_type="annotation",
        )
        document = {
            "type": "FeatureCollection",
            "name": "mixed",
            "features": [existing_annotation],
        }

        converted, stats = convert_document(document, SimplificationConfig())

        self.assertEqual(converted["type"], "FeatureCollection")
        self.assertEqual(converted["name"], "mixed")
        self.assertEqual(converted["features"][0], existing_annotation)
        self.assertEqual(stats.features_skipped_object_type, 1)
        self.assertEqual(stats.features_converted, 0)

    def test_convert_file_refuses_to_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temp_path = Path(temporary_directory)
            input_path = temp_path / "input.geojson"
            output_path = temp_path / "output.geojson"
            input_path.write_text(
                json.dumps([_feature(_dense_rectangle())]),
                encoding="utf-8",
            )
            output_path.write_text("existing", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                convert_file(
                    input_path,
                    output_path,
                    SimplificationConfig(),
                    summary_path=None,
                    force=False,
                    progress_every=0,
                )

    def test_convert_file_never_allows_output_to_replace_input(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "input.geojson"
            input_path.write_text(
                json.dumps([_feature(_dense_rectangle())]),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                convert_file(
                    input_path,
                    input_path,
                    SimplificationConfig(),
                    summary_path=None,
                    force=True,
                    progress_every=0,
                )


if __name__ == "__main__":
    unittest.main()
