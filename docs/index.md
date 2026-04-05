# UQ-DETR

**Uncertainty Quantification in Detection Transformers: Object-Level Calibration and Image-Level Reliability**

A lightweight Python toolkit for evaluating calibration and reliability of object detectors.

- **Model-agnostic**: Works with DETR, DINO, RT-DETR, Faster R-CNN, YOLO, or any custom detector.
- **Minimal dependencies**: Only `numpy` and `scipy`.
- **Pip-installable**: `pip install uq-detr`.

## What's in the package?

| Category | Functions |
|----------|-----------|
| **Calibration metrics** | `oce`, `dece`, `laece`, `lrp` |
| **Image-level reliability** | `contrastive_conf` |
| **Post-processing** | `select` (threshold, top-k, NMS) |
| **Matching** | `hungarian_match`, `compute_iou_matrix`, `compute_giou_matrix` |
| **Utilities** | `box_convert`, `Detections`, `GroundTruth` |

## Paper

> Park, Sobolewski, and Azizan. "Uncertainty Quantification in Detection Transformers: Object-Level Calibration and Image-Level Reliability." *IEEE Trans. Pattern Analysis and Machine Intelligence*, 2025.
> [arXiv:2412.01782](https://arxiv.org/abs/2412.01782)
