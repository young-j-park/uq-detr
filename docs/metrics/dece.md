# Detection Expected Calibration Error (D-ECE)

**`uq_detr.dece(detections, ground_truths, *, tp_criterion, ...)`**

D-ECE measures the gap between a detector's confidence and its precision, using binned calibration.

## Definition

Detections are grouped into $J$ bins by confidence score. For each bin $j$:

$$
\text{D-ECE} = \sum_{j=1}^{J} \frac{|\hat{D}_j|}{|\hat{D}|} \left| \bar{p}_j - \text{precision}(j) \right|
$$

where $\bar{p}_j$ is the average confidence in bin $j$, and $\text{precision}(j)$ is the fraction of true positives in that bin.

A detection is a **true positive** if it has IoU above a threshold $\tau$ with a ground-truth object of the same class.

## TP Criterion

D-ECE requires you to specify how TP/FP labels are assigned:

```python
# Each detection independently checks any GT (non-exclusive)
uq_detr.dece(dets, gts, tp_criterion="independent")

# COCO-style: sorted by confidence, each GT matched at most once
uq_detr.dece(dets, gts, tp_criterion="greedy")
```

| `tp_criterion` | Matching | Multiple dets can match same GT? |
|-----------------|----------|----------------------------------|
| `"independent"` | Non-exclusive | Yes |
| `"greedy"` | COCO-style exclusive | No |

!!! note
    This parameter is **required** --- there is no default. This is intentional: the choice affects the metric value and users should be aware of which they are using.

## Usage

```python
import uq_detr

result = uq_detr.dece(
    detections, ground_truths,
    tp_criterion="greedy",
    iou_threshold=0.5,
    n_bins=25,
)
print(result.score)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detections` | `list[Detections]` | required | Predictions per image |
| `ground_truths` | `list[GroundTruth]` | required | Annotations per image |
| `tp_criterion` | `str` | **required** | `"independent"` or `"greedy"` |
| `iou_threshold` | `float` | `0.5` | IoU threshold for TP assignment |
| `n_bins` | `int` | `25` | Number of calibration bins |

## References

- Kuppers et al., "Multivariate confidence calibration for object detection", CVPR Workshops 2020.
- Kuzucu et al., "On calibration of object detectors: Pitfalls, evaluation and baselines", ECCV 2025.
