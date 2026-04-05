"""Localization Recall Precision (LRP).

Combines localization, recall, and precision into a single score.

Reference: Oksuz et al., "Localization Recall Precision (LRP): A New
Performance Metric for Object Detection", ECCV 2018.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from uq_detr._matching import compute_iou_matrix, match_detections_to_gt
from uq_detr._types import Detections, GroundTruth, MetricResult


def lrp(
    detections: Sequence[Detections],
    ground_truths: Sequence[GroundTruth],
    iou_threshold: float = 0.5,
) -> MetricResult:
    """Compute Localization Recall Precision (LRP).

    LRP = (FP + FN + sum of localization errors) / (TP + FP + FN),
    where localization error for a TP is ``(1 - IoU) / (1 - tau)``.

    Args:
        detections: Post-processed predictions per image.
        ground_truths: Ground-truth annotations per image.
        iou_threshold: IoU threshold for TP/FP assignment.

    Returns:
        :class:`MetricResult` with ``score`` (the LRP value).
    """
    n_tp = 0
    n_fp = 0
    n_fn = 0
    loc_error_sum = 0.0

    for dets, gt in zip(detections, ground_truths):
        if dets.num_detections == 0:
            n_fn += gt.num_objects
            continue

        confs = dets.max_confidence
        order = np.argsort(-confs)

        if gt.num_objects == 0:
            n_fp += dets.num_detections
            continue

        iou_mat = compute_iou_matrix(gt.boxes, dets.boxes)
        reordered_iou = iou_mat[:, order]
        reordered_labels = dets.labels[order]

        matched = match_detections_to_gt(
            reordered_iou, reordered_labels, gt.labels, iou_threshold
        )

        gt_matched = set()
        for det_idx, gt_idx in enumerate(matched):
            if gt_idx >= 0:
                n_tp += 1
                gt_matched.add(int(gt_idx))
                iou_val = reordered_iou[gt_idx, det_idx]
                loc_error_sum += (1 - iou_val) / (1 - iou_threshold) if iou_threshold < 1.0 else 0.0
            else:
                n_fp += 1

        n_fn += gt.num_objects - len(gt_matched)

    denom = n_tp + n_fp + n_fn
    if denom == 0:
        return MetricResult(score=0.0)

    score = (n_fp + n_fn + loc_error_sum) / denom
    return MetricResult(score=float(score))
