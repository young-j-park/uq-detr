# Label-Aware Expected Calibration Error (LA-ECE)

**`uq_detr.laece(detections, ground_truths, *, tp_criterion, ...)`**

LA-ECE extends D-ECE by (1) accounting for localization quality and (2) computing ECE per class before averaging.

## Definition

For each class $c$, binned ECE is computed where the accuracy of a TP detection is its IoU with the matched GT object (rather than binary 1/0). The final LA-ECE averages across classes:

$$
\text{LA-ECE} = \frac{1}{K} \sum_{c=1}^{K} \sum_{j=1}^{J} \frac{|\hat{D}_j^c|}{|\hat{D}^c|} \left| \bar{p}_j^c - \text{acc}^c(j) \right|
$$

where $\text{acc}^c(j)$ is the average IoU of true positives in bin $j$ for class $c$ (0 for false positives).

## TP Criterion

Same as D-ECE --- see [D-ECE: TP Criterion](dece.md#tp-criterion).

```python
uq_detr.laece(dets, gts, tp_criterion="greedy")
uq_detr.laece(dets, gts, tp_criterion="independent")
```

## Usage

```python
import uq_detr

result = uq_detr.laece(
    detections, ground_truths,
    tp_criterion="greedy",
    iou_threshold=0.5,
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

- Oksuz et al., "Towards building self-aware object detectors via reliable uncertainty quantification and calibration", CVPR 2023.
