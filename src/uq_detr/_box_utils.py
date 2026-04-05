from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


def box_convert(
    boxes: np.ndarray,
    from_fmt: str,
    to_fmt: str,
    image_size: Optional[Union[Tuple[int, int], np.ndarray]] = None,
) -> np.ndarray:
    """Convert bounding boxes between formats.

    Supported formats:
        - ``"xyxy"``: ``(x_min, y_min, x_max, y_max)``
        - ``"xywh"``: ``(x_min, y_min, width, height)``
        - ``"cxcywh"``: ``(center_x, center_y, width, height)``

    Args:
        boxes: Array of shape ``(N, 4)``.
        from_fmt: Source format.
        to_fmt: Target format.
        image_size: ``(height, width)`` for denormalizing normalized
            coordinates (values in [0, 1]). If ``None``, boxes are assumed
            to be in absolute pixel coordinates.

    Returns:
        Converted boxes of shape ``(N, 4)``.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim == 1:
        boxes = boxes[np.newaxis, :]

    if image_size is not None:
        h, w = image_size
        scale = np.array([w, h, w, h], dtype=np.float64)
        boxes = boxes * scale

    if from_fmt == to_fmt:
        return boxes

    # Convert to xyxy first
    if from_fmt == "xyxy":
        xyxy = boxes
    elif from_fmt == "xywh":
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0]
        xyxy[:, 1] = boxes[:, 1]
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]
    elif from_fmt == "cxcywh":
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    else:
        raise ValueError(f"Unknown source format: {from_fmt!r}")

    # Convert from xyxy to target
    if to_fmt == "xyxy":
        return xyxy
    elif to_fmt == "xywh":
        out = np.empty_like(xyxy)
        out[:, 0] = xyxy[:, 0]
        out[:, 1] = xyxy[:, 1]
        out[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        out[:, 3] = xyxy[:, 3] - xyxy[:, 1]
        return out
    elif to_fmt == "cxcywh":
        out = np.empty_like(xyxy)
        out[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
        out[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
        out[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        out[:, 3] = xyxy[:, 3] - xyxy[:, 1]
        return out
    else:
        raise ValueError(f"Unknown target format: {to_fmt!r}")
