"""Matching utilities: IoU, GIoU, and Hungarian matcher for DETR variants."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# IoU / GIoU
# ---------------------------------------------------------------------------

def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of xyxy boxes.

    Args:
        boxes_a: Shape ``(M, 4)`` in xyxy format.
        boxes_b: Shape ``(N, 4)`` in xyxy format.

    Returns:
        IoU matrix of shape ``(M, N)``.
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float64)
    boxes_b = np.asarray(boxes_b, dtype=np.float64)

    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection

    return np.where(union > 0, intersection / union, 0.0)


def compute_giou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise Generalized IoU between two sets of xyxy boxes.

    Reference: https://giou.stanford.edu/

    Args:
        boxes_a: Shape ``(M, 4)`` in xyxy format.
        boxes_b: Shape ``(N, 4)`` in xyxy format.

    Returns:
        GIoU matrix of shape ``(M, N)``, values in [-1, 1].
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float64)
    boxes_b = np.asarray(boxes_b, dtype=np.float64)

    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    # Intersection
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    iou = np.where(union > 0, intersection / union, 0.0)

    # Enclosing box
    ex1 = np.minimum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    ey1 = np.minimum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    ex2 = np.maximum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    ey2 = np.maximum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    enclosing_area = np.maximum(0, ex2 - ex1) * np.maximum(0, ey2 - ey1)

    return iou - np.where(enclosing_area > 0, (enclosing_area - union) / enclosing_area, 0.0)


# ---------------------------------------------------------------------------
# Hungarian Matcher (numpy-only, compatible with DETR variants)
# ---------------------------------------------------------------------------

def hungarian_match(
    pred_logits: np.ndarray,
    pred_boxes: np.ndarray,
    gt_labels: np.ndarray,
    gt_boxes: np.ndarray,
    cost_class: float = 2.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian matching between predictions and ground-truth objects.

    This is a numpy reimplementation of the matching used in Deformable-DETR,
    DINO, and other DETR variants. It computes a cost matrix using focal
    classification cost, L1 bbox cost, and GIoU cost, then solves the
    linear sum assignment problem.

    Args:
        pred_logits: Raw logits of shape ``(Q, C)`` (pre-sigmoid).
        pred_boxes: Predicted boxes of shape ``(Q, 4)`` in cxcywh
            normalized format.
        gt_labels: Ground-truth class indices of shape ``(N,)``.
        gt_boxes: Ground-truth boxes of shape ``(N, 4)`` in cxcywh
            normalized format.
        cost_class: Weight for the focal classification cost.
        cost_bbox: Weight for the L1 bounding box cost.
        cost_giou: Weight for the GIoU cost.
        alpha: Focal loss alpha parameter.
        gamma: Focal loss gamma parameter.

    Returns:
        Tuple of ``(pred_indices, gt_indices)`` — matched prediction and
        ground-truth indices, each of shape ``(M,)`` where M = N
        (number of GT objects).
    """
    if len(gt_labels) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    Q = pred_logits.shape[0]

    # Sigmoid activation
    out_prob = 1.0 / (1.0 + np.exp(-pred_logits.astype(np.float64)))  # (Q, C)

    # --- Classification cost (focal) ---
    neg_cost = (1 - alpha) * (out_prob ** gamma) * (-np.log(1 - out_prob + 1e-8))
    pos_cost = alpha * ((1 - out_prob) ** gamma) * (-np.log(out_prob + 1e-8))
    # Select columns for target classes: (Q, N)
    cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]

    # --- L1 bbox cost ---
    # (Q, N) via broadcasting: |pred_i - gt_j| summed over 4 coords
    bbox_cost = np.abs(
        pred_boxes[:, None, :].astype(np.float64) - gt_boxes[None, :, :].astype(np.float64)
    ).sum(axis=2)

    # --- GIoU cost ---
    from uq_detr._box_utils import box_convert as _box_convert
    pred_xyxy = _box_convert(pred_boxes, "cxcywh", "xyxy")
    gt_xyxy = _box_convert(gt_boxes, "cxcywh", "xyxy")
    giou_cost = -compute_giou_matrix(pred_xyxy, gt_xyxy)

    # --- Total cost ---
    C = cost_class * cls_cost + cost_bbox * bbox_cost + cost_giou * giou_cost

    pred_idx, gt_idx = linear_sum_assignment(C)
    return pred_idx.astype(np.int64), gt_idx.astype(np.int64)


# ---------------------------------------------------------------------------
# Greedy detection-to-GT matching (for D-ECE / LA-ECE / LRP)
# ---------------------------------------------------------------------------

def match_detections_to_gt(
    iou_matrix: np.ndarray,
    det_labels: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Greedy matching of detections to ground-truth objects.

    Each detection is matched to the GT object with the highest IoU
    above the threshold, with class agreement required. Each GT object
    can be matched at most once. Detections are processed in order
    (caller should sort by confidence if desired).

    Args:
        iou_matrix: Shape ``(M_gt, N_det)``.
        det_labels: Shape ``(N_det,)``.
        gt_labels: Shape ``(M_gt,)``.
        iou_threshold: Minimum IoU for a valid match.

    Returns:
        Array of shape ``(N_det,)`` with the matched GT index for each
        detection, or ``-1`` if unmatched.
    """
    n_det = iou_matrix.shape[1]
    matched_gt = np.full(n_det, -1, dtype=np.int64)
    gt_used = set()

    for det_idx in range(n_det):
        ious = iou_matrix[:, det_idx].copy()
        class_mask = gt_labels != det_labels[det_idx]
        ious[class_mask] = 0.0
        for used_idx in gt_used:
            ious[used_idx] = 0.0

        best_gt = np.argmax(ious)
        if ious[best_gt] >= iou_threshold:
            matched_gt[det_idx] = best_gt
            gt_used.add(int(best_gt))

    return matched_gt
