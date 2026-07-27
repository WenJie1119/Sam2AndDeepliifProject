import unittest
from unittest.mock import patch

from cell.segmentation_backend import (
    SegmentationBackend,
    create_segmentation_backend,
)


class _FakeProcessor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def segment_batch(self, items):
        return []

    def shutdown(self):
        return None


class SegmentationBackendTests(unittest.TestCase):
    def test_sam2_factory_preserves_constructor_arguments(self):
        with patch("cell.sam2.SAM2Processor", _FakeProcessor):
            backend = create_segmentation_backend(
                "sam2",
                config="sam.yaml",
                checkpoint="sam.pt",
                device="cuda:0",
                batch_size=8,
                cache_dir="cache",
                reuse_cache_dir=None,
            )

        self.assertIsInstance(backend, SegmentationBackend)
        self.assertEqual(backend.kwargs["config"], "sam.yaml")
        self.assertEqual(backend.kwargs["checkpoint"], "sam.pt")
        self.assertEqual(backend.kwargs["batch_size"], 8)

    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            create_segmentation_backend(
                "unknown",
                config="sam.yaml",
                checkpoint="sam.pt",
                device="cpu",
                batch_size=1,
            )


if __name__ == "__main__":
    unittest.main()
