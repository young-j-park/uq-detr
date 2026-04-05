"""Post-processing methods for selecting positive predictions."""

from __future__ import annotations

import numpy as np

from uq_detr._types import Detections


def select(
    detections: Detections,
    method: str = "threshold",
    param: float = 0.3,
) -> Detections:
    """Select a subset of detections via post-processing.

    Args:
        detections: Full set of predictions for one image.
        method: Selection method:
            - ``"threshold"``: Keep detections with max confidence > param.
            - ``"topk"``: Keep top-k detections by max confidence.
            - ``"nms"``: Non-maximum suppression with IoU threshold = param.
        param: Method-specific parameter (threshold value, k, or NMS IoU
            threshold).

    Returns:
        Filtered :class:`Detections`.
    """
    if detections.num_detections == 0:
        return detections

    if method == "threshold":
        mask = detections.max_confidence > param
        return _filter(detections, mask)

    elif method == "topk":
        k = min(int(param), detections.num_detections)
        top_indices = np.argsort(-detections.max_confidence)[:k]
        return _index(detections, top_indices)

    elif method == "nms":
        keep = _nms(detections.boxes, detections.max_confidence, param)
        return _index(detections, keep)

    else:
        raise ValueError(f"Unknown method: {method!r}")


def _filter(dets: Detections, mask: np.ndarray) -> Detections:
    return Detections(
        boxes=dets.boxes[mask],
        scores=dets.scores[mask],
        labels=dets.labels[mask],
    )


def _index(dets: Detections, indices: np.ndarray) -> Detections:
    return Detections(
        boxes=dets.boxes[indices],
        scores=dets.scores[indices],
        labels=dets.labels[indices],
    )


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(-scores)
    keep = []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    suppressed = np.zeros(len(boxes), dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)

        xx1 = np.maximum(x1[idx], x1[order])
        yy1 = np.maximum(y1[idx], y1[order])
        xx2 = np.minimum(x2[idx], x2[order])
        yy2 = np.minimum(y2[idx], y2[order])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[idx] + areas[order] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        for j, o in enumerate(order):
            if iou[j] > iou_threshold:
                suppressed[o] = True

    return np.array(keep, dtype=np.int64)
