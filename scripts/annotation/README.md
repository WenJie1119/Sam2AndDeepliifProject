# Detection GeoJSON to QuPath annotation

`convert_detection_geojson.py` post-processes an existing GeoJSON. It does not
rerun model inference and never overwrites the source file.

Run it in the project environment:

```bash
conda run -n CD34MVrecognition python \
  scripts/annotation/convert_detection_geojson.py \
  "debug_output/debug_region_0702_top1_4_01/DC2200155 A3 CD34.geojson"
```

The default outputs are:

```text
DC2200155 A3 CD34_annotation_simplified.geojson
DC2200155 A3 CD34_annotation_simplified_summary.json
```

The script:

- converts `properties.objectType` from `detection` to `annotation`;
- keeps polygons below `100 px^2` or with at most 12 coordinates unchanged;
- uses area-dependent RDP tolerances for other polygons;
- limits the tolerance for thin polygons;
- accepts a simplification only when topology, area, IoU, centroid, and minimum
  vertex checks pass;
- falls back to the original geometry when no candidate passes.

Use `--help` to see all adjustable thresholds. For example, a more conservative
run can use:

```bash
conda run -n CD34MVrecognition python \
  scripts/annotation/convert_detection_geojson.py \
  INPUT.geojson \
  --low-tolerances 1.5,1,0.5 \
  --medium-tolerances 3,2,1 \
  --high-tolerances 4,3,2 \
  --min-iou 0.97
```
