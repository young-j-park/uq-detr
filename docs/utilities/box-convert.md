# Box Format Conversion

**`uq_detr.box_convert(boxes, from_fmt, to_fmt, image_size=None)`**

Convert bounding boxes between common formats.

## Supported Formats

| Format | Coordinates | Description |
|--------|-------------|-------------|
| `"xyxy"` | `(x_min, y_min, x_max, y_max)` | Two corners. Default in torchvision, supervision, Detectron2. |
| `"xywh"` | `(x_min, y_min, width, height)` | Corner + size. Used in COCO annotations. |
| `"cxcywh"` | `(center_x, center_y, width, height)` | Center + size. Used in DETR model outputs. |

## Usage

```python
from uq_detr import box_convert

# DETR output (normalized cxcywh) -> absolute xyxy
boxes_xyxy = box_convert(pred_boxes, "cxcywh", "xyxy", image_size=(H, W))

# COCO annotations (absolute xywh) -> xyxy
boxes_xyxy = box_convert(coco_boxes, "xywh", "xyxy")

# Any format roundtrip
boxes_cx = box_convert(boxes_xyxy, "xyxy", "cxcywh")
boxes_rt = box_convert(boxes_cx, "cxcywh", "xyxy")  # identical to original
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `boxes` | `np.ndarray` | required | `(N, 4)` bounding boxes |
| `from_fmt` | `str` | required | Source format: `"xyxy"`, `"xywh"`, or `"cxcywh"` |
| `to_fmt` | `str` | required | Target format |
| `image_size` | `(int, int)` | `None` | `(height, width)` for denormalizing [0, 1] coords |

## Normalized vs Absolute Coordinates

DETR models output boxes in **normalized** [0, 1] coordinates. Pass `image_size=(H, W)` to convert to absolute pixel coordinates:

```python
# Normalized [0, 1] -> absolute pixels
boxes_abs = box_convert(norm_boxes, "cxcywh", "xyxy", image_size=(480, 640))
```

If `image_size` is `None`, boxes are assumed to already be in absolute coordinates.

## Convenience Constructors

Instead of calling `box_convert` manually, use the class methods on `Detections` and `GroundTruth`:

```python
from uq_detr import Detections, GroundTruth

# DETR output
det = Detections.from_cxcywh(pred_boxes, scores, image_size=(H, W))

# COCO annotations
gt = GroundTruth.from_xywh(coco_boxes, labels)

# Already xyxy
det = Detections(boxes=boxes_xyxy, scores=scores)
```
