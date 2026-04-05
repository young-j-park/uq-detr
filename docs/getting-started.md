# Getting Started

## Installation

```bash
pip install uq-detr
```

Or install from source for development:

```bash
git clone https://github.com/azizanlab/uq-detr.git
cd uq-detr
pip install -e ".[dev]"
```

## Basic Usage

All metrics take the same inputs: a list of `Detections` and a list of `GroundTruth`, one per image.

```python
import numpy as np
import uq_detr
from uq_detr import Detections, GroundTruth

# Predictions for one image
detections = Detections(
    boxes=np.array([[10, 20, 100, 120]]),   # (N, 4) xyxy format
    scores=np.array([[0.05, 0.90, 0.05]]),  # (N, C) class probabilities
    labels=np.array([1]),                    # (N,) predicted class
)

# Ground truth for the same image
gt = GroundTruth(
    boxes=np.array([[12, 18, 98, 118]]),    # (M, 4) xyxy format
    labels=np.array([1]),                    # (M,) class labels
)

# Compute metrics (pass lists — one element per image)
oce_result = uq_detr.oce([detections], [gt])
dece_result = uq_detr.dece([detections], [gt], tp_criterion="greedy")

print(f"OCE:   {oce_result.score:.4f}")
print(f"D-ECE: {dece_result.score:.4f}")
```

## Input Formats

### `Detections`

| Field | Shape | Description |
|-------|-------|-------------|
| `boxes` | `(N, 4)` | Bounding boxes in `xyxy` format (absolute pixel coordinates) |
| `scores` | `(N, C)` or `(N,)` | Class probabilities or max-confidence scores |
| `labels` | `(N,)` | Predicted class index per detection |

When `scores` is `(N, C)`, OCE uses the exact multi-class Brier score. When `scores` is `(N,)` (max confidence only), OCE falls back to a binary approximation.

### `GroundTruth`

| Field | Shape | Description |
|-------|-------|-------------|
| `boxes` | `(M, 4)` | Bounding boxes in `xyxy` format (absolute pixel coordinates) |
| `labels` | `(M,)` | Class index per object |

## TP Criterion for D-ECE and LA-ECE

D-ECE and LA-ECE require you to specify how true positives are assigned via the `tp_criterion` parameter:

- **`"independent"`**: Each detection independently checks if *any* GT object has IoU above threshold with matching class. Multiple detections may match the same GT.
- **`"greedy"`**: COCO-style matching. Detections are processed by descending confidence; each GT is matched at most once.

```python
# Independent matching
uq_detr.dece(dets, gts, tp_criterion="independent")

# COCO-style greedy matching
uq_detr.dece(dets, gts, tp_criterion="greedy")
```

## Converting Box Formats

If your model outputs boxes in a different format, use `box_convert`:

```python
from uq_detr import box_convert

# HuggingFace DETR: normalized cxcywh -> absolute xyxy
boxes_xyxy = box_convert(pred_boxes, "cxcywh", "xyxy", image_size=(H, W))
```

Supported formats: `"xyxy"`, `"xywh"`, `"cxcywh"`.
