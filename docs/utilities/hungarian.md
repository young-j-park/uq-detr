# Hungarian Matching

**`uq_detr.hungarian_match(pred_logits, pred_boxes, gt_labels, gt_boxes, ...)`**

A numpy reimplementation of the Hungarian matcher used in DETR variants (Deformable-DETR, DINO, etc.). Computes optimal one-to-one assignment between predictions and ground-truth objects by minimizing a combined cost of classification, L1 bbox, and GIoU.

## Usage

```python
from uq_detr import hungarian_match

pred_idx, gt_idx = hungarian_match(
    pred_logits,   # (Q, C) raw logits, pre-sigmoid
    pred_boxes,    # (Q, 4) cxcywh, normalized [0, 1]
    gt_labels,     # (N,) class indices
    gt_boxes,      # (N, 4) cxcywh, normalized [0, 1]
)
# pred_idx: which predictions are matched (length N)
# gt_idx:   which GT objects they match to (length N)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pred_logits` | `np.ndarray` | required | `(Q, C)` raw logits (pre-sigmoid) |
| `pred_boxes` | `np.ndarray` | required | `(Q, 4)` boxes in cxcywh normalized format |
| `gt_labels` | `np.ndarray` | required | `(N,)` ground-truth class indices |
| `gt_boxes` | `np.ndarray` | required | `(N, 4)` boxes in cxcywh normalized format |
| `cost_class` | `float` | `2.0` | Weight for focal classification cost |
| `cost_bbox` | `float` | `5.0` | Weight for L1 bounding box cost |
| `cost_giou` | `float` | `2.0` | Weight for GIoU cost |
| `alpha` | `float` | `0.25` | Focal loss alpha |
| `gamma` | `float` | `2.0` | Focal loss gamma |

!!! note "Input format"
    Both `pred_boxes` and `gt_boxes` must be in **cxcywh normalized** format (center-x, center-y, width, height, all in [0, 1]). This matches the raw DETR output format before any conversion.

## Cost Function

The total cost matrix is:

$$
C = w_\text{cls} \cdot C_\text{focal} + w_\text{bbox} \cdot C_\text{L1} + w_\text{giou} \cdot C_\text{GIoU}
$$

- **Focal cost**: Uses sigmoid activation on logits, then computes focal loss cost per (prediction, target) pair.
- **L1 cost**: Manhattan distance between box coordinates.
- **GIoU cost**: Negative Generalized IoU between boxes.

The assignment is solved via `scipy.optimize.linear_sum_assignment`.

## Default Weights

The default weights (`cost_class=2.0, cost_bbox=5.0, cost_giou=2.0`) match the defaults used in Deformable-DETR and DINO. Adjust if your model uses different training hyperparameters.

## Identifying Optimal Positives and Negatives

```python
import numpy as np
from uq_detr import hungarian_match, Detections

pred_idx, gt_idx = hungarian_match(logits, pred_boxes, gt_labels, gt_boxes)

# Optimal positive set (matched to GT)
matched_set = set(pred_idx.tolist())

# Optimal negative set (everything else)
all_idx = set(range(len(logits)))
negative_set = all_idx - matched_set
```

## Related Functions

- `uq_detr.compute_iou_matrix(boxes_a, boxes_b)` --- pairwise IoU between xyxy boxes
- `uq_detr.compute_giou_matrix(boxes_a, boxes_b)` --- pairwise GIoU between xyxy boxes
