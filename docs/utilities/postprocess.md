# Post-Processing

**`uq_detr.select(detections, method, param)`**

Select a subset of detections via common post-processing strategies. This is the bridge between raw model output (hundreds of queries) and the filtered set used for evaluation.

## Methods

| Method | `param` meaning | Description |
|--------|-----------------|-------------|
| `"threshold"` | Confidence threshold | Keep detections with max confidence > `param` |
| `"topk"` | Number of detections | Keep the top-k by max confidence |
| `"nms"` | IoU threshold | Non-maximum suppression with the given IoU threshold |

## Usage

```python
from uq_detr import select

# Confidence thresholding
filtered = select(all_queries, method="threshold", param=0.3)

# Top-k selection (e.g., DINO uses top-300 out of 900)
filtered = select(all_queries, method="topk", param=300)

# Non-maximum suppression
filtered = select(all_queries, method="nms", param=0.5)
```

## Comparing Post-Processing Strategies

A key use case: sweep configurations and use OCE to find the best one.

```python
import uq_detr
from uq_detr import select
import numpy as np

# Threshold sweep
for thr in np.arange(0.1, 0.9, 0.1):
    filtered = [select(q, method="threshold", param=thr) for q in all_queries]
    print(f"thr={thr:.1f}  OCE={uq_detr.oce(filtered, gts).score:.4f}")

# Top-k sweep
for k in [10, 50, 100, 300]:
    filtered = [select(q, method="topk", param=k) for q in all_queries]
    print(f"top-{k}  OCE={uq_detr.oce(filtered, gts).score:.4f}")
```
