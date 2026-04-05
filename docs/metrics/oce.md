# Object-level Calibration Error (OCE)

**`uq_detr.oce(detections, ground_truths, ...)`**

OCE is the key metric introduced in our paper. Unlike prediction-level metrics (D-ECE, LA-ECE), OCE aggregates predictions **per ground-truth object**, making it suitable for jointly evaluating a model and its post-processing scheme.

## Definition

For each ground-truth object $y_{l,j}$ with class $c_{l,j}$, OCE computes the Brier score between the aggregated prediction and a one-hot target:

$$
\text{OCE}_\tau = \frac{1}{|\mathcal{Y}|} \sum_{y_{l,j} \in \mathcal{Y}} \text{Brier}_\tau(\hat{S}_\theta(x_l),\; y_{l,j})
$$

$$
\text{Brier}_\tau(\hat{S}_\theta(x_l),\; y_{l,j}) = \sum_{c=1}^{C} \left( \mathbf{1}(c = c_{l,j}) - \bar{p}_{l,j}^{(\tau)}(c) \right)^2
$$

where $\bar{p}$ is the average predicted distribution over all detections with $\text{IoU} \geq \tau$ to the ground-truth object. If no detection overlaps, the Brier score defaults to 1.0 (maximum penalty).

The final OCE averages over IoU thresholds $\tau \in \{0.5, 0.75\}$.

## Why OCE over D-ECE?

OCE addresses two pitfalls of prediction-level metrics:

1. **D-ECE favors discarding predictions.** Since D-ECE doesn't penalize missed objects, setting a very high confidence threshold yields artificially low D-ECE.

2. **AP favors retaining everything.** AP doesn't penalize low-confidence false positives, so the optimal AP threshold is near 0.

OCE's U-shaped curve naturally identifies the sweet spot --- the threshold that best recovers the well-calibrated primary predictions.

## Usage

```python
import uq_detr

result = uq_detr.oce(detections, ground_truths)
print(result.score)        # aggregated OCE
print(result.per_element)  # per-object Brier scores
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detections` | `list[Detections]` | required | Post-processed predictions per image |
| `ground_truths` | `list[GroundTruth]` | required | Ground-truth annotations per image |
| `iou_thresholds` | `list[float]` | `[0.5, 0.75]` | IoU thresholds; final OCE is their average |
| `aggregation` | `str` | `"mean"` | How to aggregate predictions matched to the same object |

### Aggregation Methods

| Value | Description |
|-------|-------------|
| `"mean"` | Average predicted distribution over all matched detections |
| `"max_iou"` | Use only the detection with the highest IoU |
| `"iou_weighted"` | IoU-weighted average of matched detections |

All three are highly correlated (see paper, Section V-C4).

## Binary Approximation

When `scores` is 1-D (max-confidence only), OCE uses a binary Brier approximation:

- Correct class ($\hat{c} = c^*$): $\text{Brier} = 2(1-p)^2$
- Wrong class ($\hat{c} \neq c^*$): $\text{Brier} = 2p^2$

This enables OCE computation with outputs from frameworks like [supervision](https://github.com/roboflow/supervision) that only provide max-confidence scores.

## Threshold Sweep Example

```python
from uq_detr import select
import numpy as np

for thr in np.arange(0.1, 0.9, 0.1):
    filtered = [select(q, method="threshold", param=thr) for q in all_queries]
    score = uq_detr.oce(filtered, ground_truths).score
    print(f"threshold={thr:.1f}  OCE={score:.4f}")
```
