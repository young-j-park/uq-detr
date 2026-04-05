# Example: HuggingFace DETR Models

Evaluate calibration of any HuggingFace DETR-family model on a dataset.

## Setup

```bash
pip install uq-detr transformers torch torchvision
```

## End-to-End: DETR on COCO

```python
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

import uq_detr
from uq_detr import Detections, GroundTruth, select

# ── Load model ──
model_name = "facebook/detr-resnet-50"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForObjectDetection.from_pretrained(model_name)
model.eval()

# ── Run inference on your dataset ──
all_queries = []
ground_truths = []

for image, annotation in dataset:  # your COCO-style dataset
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Raw DETR outputs (all queries, before post-processing)
    logits = outputs.logits.sigmoid().squeeze(0).numpy()   # (Q, C)
    boxes = outputs.pred_boxes.squeeze(0).numpy()           # (Q, 4) cxcywh normalized
    H, W = image.height, image.width

    all_queries.append(
        Detections.from_cxcywh(boxes, logits, image_size=(H, W))
    )
    ground_truths.append(
        GroundTruth(boxes=annotation["boxes"], labels=annotation["labels"])
    )

# ── Evaluate with different post-processing ──
for thr in [0.1, 0.3, 0.5, 0.7]:
    filtered = [select(q, method="threshold", param=thr) for q in all_queries]
    oce = uq_detr.oce(filtered, ground_truths).score
    dece = uq_detr.dece(filtered, ground_truths, tp_criterion="greedy").score
    print(f"thr={thr:.1f}  OCE={oce:.4f}  D-ECE={dece:.4f}")
```

## Notes on HuggingFace DETR Outputs

`model(**inputs)` returns the **raw, unfiltered output** from all object queries:

- **`outputs.logits`**: shape `(batch, Q, C)`, **pre-sigmoid** raw logits. Apply `.sigmoid()` to get class probabilities.
- **`outputs.pred_boxes`**: shape `(batch, Q, 4)`, **normalized cxcywh** coordinates in [0, 1]. Use `Detections.from_cxcywh()` with `image_size` to convert to absolute xyxy.

No filtering or post-processing is applied --- you get all Q queries (DETR: 100, Deformable-DETR: 300, DINO: 900). This is exactly what you want for OCE evaluation: pass the full query set and let `uq_detr.select()` handle post-processing.

For RT-DETR and other DETR variants on HuggingFace, the same pattern applies --- only the model name changes.

## Using the HuggingFace Post-Processor

If you prefer to use HuggingFace's built-in post-processing:

```python
# HuggingFace post-processes into xyxy absolute coords
results = processor.post_process_object_detection(
    outputs, threshold=0.3, target_sizes=[(H, W)]
)[0]

det = Detections(
    boxes=results["boxes"].numpy(),    # already xyxy absolute
    scores=results["scores"].numpy(),  # (N,) max-confidence
    labels=results["labels"].numpy(),  # (N,) class indices
)
```

!!! warning
    HuggingFace's post-processor returns max-confidence scores `(N,)`, not full class distributions `(N, C)`. OCE will use the binary approximation in this case. For exact OCE, use the raw `outputs.logits` as shown above.
