"""
cell -- CD34 WSI Pipeline (modular OOP architecture).

Modules:
    deepliif     - DeepLIIF Seg/Marker inference
    segmentation - Cell region extraction
    sam2         - SAM2 batch segmentation
    postprocess  - Mask merging + async saving
    utils        - Bucket, BucketItem, StickyProgress, ROI utilities
    main         - CLI entry point + Producer/Consumer orchestration

Usage:
    python -m cell.main --wsi-path /path/to/slide.ndpi --output-dir ./output
"""
