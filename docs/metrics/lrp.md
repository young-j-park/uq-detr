# Localization Recall Precision (LRP)

**`uq_detr.lrp(detections, ground_truths, ...)`**

LRP combines false positives, false negatives, and localization error into a single performance score.

## Definition

$$
\text{LRP} = \frac{\text{FP} + \text{FN} + \sum_{i \in \text{TP}} \frac{1 - \text{IoU}_i}{1 - \tau}}{\text{TP} + \text{FP} + \text{FN}}
$$

where $\tau$ is the IoU threshold and the localization error term penalizes TPs that are not perfectly localized.

LRP = 0 means perfect detection (all objects found, no false positives, perfect localization). LRP = 1 means complete failure.

## Usage

```python
import uq_detr

result = uq_detr.lrp(detections, ground_truths, iou_threshold=0.5)
print(result.score)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detections` | `list[Detections]` | required | Predictions per image |
| `ground_truths` | `list[GroundTruth]` | required | Annotations per image |
| `iou_threshold` | `float` | `0.5` | IoU threshold for TP assignment |

!!! note
    LRP uses COCO-style exclusive greedy matching internally (detections sorted by descending confidence, each GT matched at most once).

## References

- Oksuz et al., "Localization Recall Precision (LRP): A New Performance Metric for Object Detection", ECCV 2018.
