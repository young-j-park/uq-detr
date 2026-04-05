# Metrics Overview

UQ-DETR provides calibration and reliability metrics designed for object detection.

## Object-Level Calibration Error (OCE)

The key contribution of our paper. OCE evaluates calibration by aggregating predictions **per ground-truth object** rather than per prediction. This design penalizes both:

1. **Retaining suppressed predictions** (secondary predictions with artificially low confidence)
2. **Missing ground-truth objects** (overly aggressive filtering)

This makes OCE suitable for jointly evaluating a model **and** its post-processing scheme.

See [OCE](oce.md) for details.

## D-ECE and LA-ECE

Standard detection calibration metrics from the literature:

- **D-ECE** bins detections by confidence and measures the gap between confidence and precision. See [D-ECE](dece.md).
- **LA-ECE** extends D-ECE with per-class computation and IoU-weighted accuracy. See [LA-ECE](laece.md).

Both support two TP assignment strategies via the `tp_criterion` parameter:

| `tp_criterion` | Matching | Each GT matched... |
|-----------------|----------|---------------------|
| `"independent"` | Non-exclusive | Possibly multiple times |
| `"greedy"` | COCO-style | At most once |

## LRP

Localization Recall Precision combines false positives, false negatives, and localization error into a single score. See [LRP](lrp.md).

## ContrastiveConf

Image-level reliability estimation by contrasting positive and negative prediction confidence. See [ContrastiveConf](contrastive-conf.md).
