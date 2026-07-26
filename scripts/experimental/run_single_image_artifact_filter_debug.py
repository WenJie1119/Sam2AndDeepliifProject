#!/usr/bin/env python3
"""Run the artifact-filtered single-image debug pipeline.

This entry point keeps the artifact-filter experiment in a separate output
folder while reusing the full step-by-step pipeline implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

from run_single_image_pipeline_debug import main as run_pipeline


DEFAULT_OUTPUT_DIR = Path(
    "debug_output/single_image_artifact_filter_pipeline_test"
)


def has_arg(name: str) -> bool:
    prefix = f"{name}="
    return any(arg == name or arg.startswith(prefix) for arg in sys.argv[1:])


if __name__ == "__main__":
    if not has_arg("--output-dir"):
        sys.argv.extend(["--output-dir", str(DEFAULT_OUTPUT_DIR)])
    run_pipeline()
