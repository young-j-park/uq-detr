"""Label-Aware Expected Calibration Error (LA-ECE).

Extends D-ECE by computing ECE per class and averaging, accounting for
localization quality in the accuracy definition.

Reference: Oksuz et al., "Towards building self-aware object detectors
via reliable uncertainty quantification and calibration", CVPR 2023.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from uq_detr._matching import compute_iou_matrix, match_detections_to_gt
from uq_detr._types import Detections, GroundTruth, MetricResult
from uq_detr.metrics.dece import _binned_ece, _VALID_TP_CRITERIA


def laece(
    detections: Sequence[Detections],
    ground_truths: Sequence[GroundTruth],
    *,
    tp_criterion: str,
    iou_threshold: float = 0.5,
    n_bins: int = 25,
) -> MetricResult:
    """Compute Label-Aware Expected Calibration Error (LA-ECE).

    Accuracy for a TP detection is defined as the IoU with its matched
    GT object (rather than binary 1/0 as in D-ECE). ECE is computed
    per class and then averaged.

    Args:
        detections: Post-processed predictions per image.
        ground_truths: Ground-truth annotations per image.
        iou_threshold: IoU threshold for TP/FP assignment.
        n_bins: Number of calibration bins.
        tp_criterion: How to assign TP/FP labels. **Required.** See
            :func:`dece` for details. One of ``"independent"`` or
            ``"greedy"``.

    Returns:
        :class:`MetricResult` with ``score`` (the LA-ECE value).
    """
    if tp_criterion not in _VALID_TP_CRITERIA:
        raise ValueError(
            f"tp_criterion must be one of {_VALID_TP_CRITERIA}, got {tp_criterion!r}"
        )

    all_confs = []
    all_accs = []
    all_labels = []

    for dets, gt in zip(detections, ground_truths):
        if dets.num_detections == 0:
            continue

        confs = dets.max_confidence

        if gt.num_objects == 0:
            all_confs.append(confs)
            all_accs.append(np.zeros(len(confs)))
            all_labels.append(dets.labels)
            continue

        iou_mat = compute_iou_matrix(gt.boxes, dets.boxes)  # (M_gt, N_det)

        if tp_criterion == "greedy":
            order = np.argsort(-confs)
            reordered_iou = iou_mat[:, order]
            reordered_labels = dets.labels[order]

            matched = match_detections_to_gt(
                reordered_iou, reordered_labels, gt.labels, iou_threshold
            )
            accs = np.zeros(len(order))
            for det_idx, gt_idx in enumerate(matched):
                if gt_idx >= 0:
                    accs[det_idx] = reordered_iou[gt_idx, det_idx]

            all_confs.append(confs[order])
            all_accs.append(accs)
            all_labels.append(dets.labels[order])
        else:
            accs = _independent_tp_iou(
                iou_mat, dets.labels, gt.labels, iou_threshold
            )
            all_confs.append(confs)
            all_accs.append(accs)
            all_labels.append(dets.labels)

    if not all_confs:
        return MetricResult(score=0.0)

    confs = np.concatenate(all_confs)
    accs = np.concatenate(all_accs)
    labels = np.concatenate(all_labels)

    # Per-class ECE
    unique_classes = np.unique(labels)
    class_eces = []
    for cls in unique_classes:
        mask = labels == cls
        if mask.sum() == 0:
            continue
        cls_ece = _binned_ece(confs[mask], accs[mask], n_bins)
        class_eces.append(cls_ece)

    score = float(np.mean(class_eces)) if class_eces else 0.0
    return MetricResult(score=score)


def _independent_tp_iou(
    iou_matrix: np.ndarray,
    det_labels: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Each detection picks the best IoU among class-matching GTs."""
    n_det = iou_matrix.shape[1]
    accs = np.zeros(n_det)
    for det_idx in range(n_det):
        det_cls = det_labels[det_idx]
        best_iou = 0.0
        for gt_idx in range(iou_matrix.shape[0]):
            if (iou_matrix[gt_idx, det_idx] > iou_threshold
                    and gt_labels[gt_idx] == det_cls):
                best_iou = max(best_iou, iou_matrix[gt_idx, det_idx])
        accs[det_idx] = best_iou if best_iou > iou_threshold else 0.0
    return accs
