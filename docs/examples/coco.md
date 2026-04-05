# Example: Evaluating from Saved Results

If you've already run inference and saved predictions (pickle, JSON, or COCO format), you can evaluate calibration without re-running the model.

## From a Pickle File (DETR-style)

This is the format used in our paper's experiments:

```python
import pickle
import numpy as np
from scipy.special import expit  # sigmoid

import uq_detr
from uq_detr import Detections, GroundTruth, box_convert, select

# Load saved inference results
with open("infer_results.pkl", "rb") as f:
    raw_data = pickle.load(f)

# Convert to uq_detr format
all_queries = []
ground_truths = []

for sample in raw_data:
    h, w = sample["orig_size"]
    scores = expit(sample["pred_logits"])   # sigmoid activation

    all_queries.append(Detections.from_cxcywh(
        sample["pred_boxes"], scores, image_size=(h, w)
    ))

    if len(sample["boxes"]) > 0:
        gt = GroundTruth.from_cxcywh(
            sample["boxes"], sample["labels"], image_size=(h, w)
        )
    else:
        gt = GroundTruth(
            boxes=np.zeros((0, 4)), labels=np.zeros(0, dtype=int)
        )
    ground_truths.append(gt)

# Evaluate
filtered = [select(q, method="threshold", param=0.3) for q in all_queries]
print("OCE:   ", uq_detr.oce(filtered, ground_truths).score)
print("D-ECE: ", uq_detr.dece(filtered, ground_truths, tp_criterion="independent").score)
print("LA-ECE:", uq_detr.laece(filtered, ground_truths, tp_criterion="independent").score)
```

## From COCO JSON Format

If you have predictions in standard COCO results format:

```python
import json
import numpy as np
from pycocotools.coco import COCO

import uq_detr
from uq_detr import Detections, GroundTruth

# Load COCO annotations and predictions
coco_gt = COCO("instances_val2017.json")
with open("predictions.json") as f:
    coco_preds = json.load(f)

# Group predictions by image
from collections import defaultdict
preds_by_image = defaultdict(list)
for p in coco_preds:
    preds_by_image[p["image_id"]].append(p)

# Convert
all_detections = []
all_ground_truths = []

for img_id in coco_gt.getImgIds():
    # Ground truth
    ann_ids = coco_gt.getAnnIds(imgIds=img_id)
    anns = coco_gt.loadAnns(ann_ids)
    gt_boxes = np.array([a["bbox"] for a in anns])  # xywh
    gt_labels = np.array([a["category_id"] for a in anns])

    if len(gt_boxes) > 0:
        gt = GroundTruth.from_xywh(gt_boxes, gt_labels)
    else:
        gt = GroundTruth(boxes=np.zeros((0, 4)), labels=np.zeros(0, dtype=int))
    all_ground_truths.append(gt)

    # Predictions (COCO result format: xywh boxes, scalar scores)
    preds = preds_by_image.get(img_id, [])
    if preds:
        pred_boxes = np.array([p["bbox"] for p in preds])
        pred_scores = np.array([p["score"] for p in preds])
        pred_labels = np.array([p["category_id"] for p in preds])
        det = Detections.from_xywh(pred_boxes, pred_scores, labels=pred_labels)
    else:
        det = Detections(
            boxes=np.zeros((0, 4)), scores=np.zeros(0), labels=np.zeros(0, dtype=int)
        )
    all_detections.append(det)

# Evaluate
print("OCE:   ", uq_detr.oce(all_detections, all_ground_truths).score)
print("D-ECE: ", uq_detr.dece(all_detections, all_ground_truths, tp_criterion="greedy").score)
```

!!! note
    COCO JSON results only contain max-confidence scores, not full class distributions. OCE will use the binary approximation.
