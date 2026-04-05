"""Detection Expected Calibration Error (D-ECE).

Measures the gap between predicted confidence and precision for
object detections, using binned calibration.

Reference: Kuppers et al., "Multivariate confidence calibration for
object detection", CVPR Workshops 2020.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from uq_detr._matching import compute_iou_matrix, match_detections_to_gt
from uq_detr._types import Detections, GroundTruth, MetricResult

_VALID_TP_CRITERIA = ("independent", "greedy")


def dece(
    detections: Sequence[Detections],
    ground_truths: Sequence[GroundTruth],
    *,
    tp_criterion: str,
    iou_threshold: float = 0.5,
    n_bins: int = 25,
) -> MetricResult:
    """Compute Detection Expected Calibration Error (D-ECE).

    For each detection, precision is defined as 1 if the detection is
    a true positive (correct class + IoU above threshold) and 0 otherwise.

    Args:
        detections: Post-processed predictions per image.
        ground_truths: Ground-truth annotations per image.
        iou_threshold: IoU threshold for TP/FP assignment.
        n_bins: Number of calibration bins.
        tp_criterion: How to assign TP/FP labels. **Required.**

            - ``"independent"``: each detection independently checks if
              any GT has IoU above threshold with matching class. Multiple
              detections may match the same GT object.
            - ``"greedy"``: COCO-style exclusive matching. Detections are
              processed by descending confidence; each GT is matched at
              most once.

    Returns:
        :class:`MetricResult` with ``score`` (the D-ECE value).
    """
    if tp_criterion not in _VALID_TP_CRITERIA:
        raise ValueError(
            f"tp_criterion must be one of {_VALID_TP_CRITERIA}, got {tp_criterion!r}"
        )

    all_confs = []
    all_tps = []

    for dets, gt in zip(detections, ground_truths):
        if dets.num_detections == 0:
            continue

        confs = dets.max_confidence

        if gt.num_objects == 0:
            all_confs.append(confs)
            all_tps.append(np.zeros(len(confs)))
            continue

        iou_mat = compute_iou_matrix(gt.boxes, dets.boxes)  # (M_gt, N_det)

        if tp_criterion == "greedy":
            order = np.argsort(-confs)
            reordered_iou = iou_mat[:, order]
            reordered_labels = dets.labels[order]
            matched = match_detections_to_gt(
                reordered_iou, reordered_labels, gt.labels, iou_threshold
            )
            tps = (matched >= 0).astype(np.float64)
            all_confs.append(confs[order])
            all_tps.append(tps)
        else:
            tps = _independent_tp(iou_mat, dets.labels, gt.labels, iou_threshold)
            all_confs.append(confs)
            all_tps.append(tps)

    if not all_confs:
        return MetricResult(score=0.0)

    confs = np.concatenate(all_confs)
    tps = np.concatenate(all_tps)

    return MetricResult(score=_binned_ece(confs, tps, n_bins))


def _independent_tp(
    iou_matrix: np.ndarray,
    det_labels: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Each detection independently checks if any GT has IoU above
    threshold with matching class."""
    n_det = iou_matrix.shape[1]
    tps = np.zeros(n_det)
    for det_idx in range(n_det):
        det_cls = det_labels[det_idx]
        for gt_idx in range(iou_matrix.shape[0]):
            if (iou_matrix[gt_idx, det_idx] > iou_threshold
                    and gt_labels[gt_idx] == det_cls):
                tps[det_idx] = 1.0
                break
    return tps


def _binned_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int) -> float:
    """Compute binned ECE."""
    if len(confidences) == 0:
        return 0.0

    bin_edges = np.linspace(confidences.min(), confidences.max(), n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges, right=True)
    bin_indices = np.clip(bin_indices, 1, n_bins) - 1

    total = len(confidences)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        count = mask.sum()
        if count > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            ece += (count / total) * abs(avg_conf - avg_acc)

    return float(ece)
