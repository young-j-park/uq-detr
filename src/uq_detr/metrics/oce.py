"""Object-level Calibration Error (OCE).

Evaluates calibration by aggregating predictions per ground-truth object
rather than per prediction. Penalizes both retaining suppressed predictions
and missing ground-truth objects.

Reference: Park et al., "Uncertainty Quantification in Detection Transformers:
Object-Level Calibration and Image-Level Reliability", IEEE TPAMI.
"""

from __future__ import annotations

import warnings
from typing import List, Sequence

import numpy as np

from uq_detr._matching import compute_iou_matrix
from uq_detr._types import Detections, GroundTruth, MetricResult


def _brier_full(
    matched_scores: np.ndarray,
    gt_class: int,
    aggregation: str,
    iou_values: np.ndarray,
) -> float:
    """Brier score using full class distributions."""
    n_matched, n_classes = matched_scores.shape
    one_hot = np.zeros(n_classes)
    one_hot[gt_class] = 1.0

    if aggregation == "mean":
        avg_pred = matched_scores.mean(axis=0)
    elif aggregation == "max_iou":
        best_idx = np.argmax(iou_values)
        avg_pred = matched_scores[best_idx]
    elif aggregation == "iou_weighted":
        weights = iou_values / iou_values.sum()
        avg_pred = (weights[:, None] * matched_scores).sum(axis=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation!r}")

    return float(np.sum((one_hot - avg_pred) ** 2))


def _brier_binary(
    matched_confs: np.ndarray,
    matched_labels: np.ndarray,
    gt_class: int,
    aggregation: str,
    iou_values: np.ndarray,
) -> float:
    """Approximate Brier score using only max confidence + predicted label.

    Uses binary reduction: collapse C classes into
    {predicted_class, everything_else}.
    """
    if aggregation == "mean":
        avg_conf = matched_confs.mean()
        # Majority-vote label
        label_counts = np.bincount(matched_labels.astype(int))
        avg_label = int(np.argmax(label_counts))
    elif aggregation == "max_iou":
        best_idx = np.argmax(iou_values)
        avg_conf = matched_confs[best_idx]
        avg_label = int(matched_labels[best_idx])
    elif aggregation == "iou_weighted":
        weights = iou_values / iou_values.sum()
        avg_conf = float(np.dot(weights, matched_confs))
        best_idx = np.argmax(iou_values)
        avg_label = int(matched_labels[best_idx])
    else:
        raise ValueError(f"Unknown aggregation: {aggregation!r}")

    p = avg_conf
    if avg_label == gt_class:
        return float(2 * (1 - p) ** 2)
    else:
        return float(2 * p ** 2)


def oce(
    detections: Sequence[Detections],
    ground_truths: Sequence[GroundTruth],
    iou_thresholds: Sequence[float] = (0.5, 0.75),
    aggregation: str = "mean",
) -> MetricResult:
    """Compute Object-level Calibration Error (OCE).

    OCE measures calibration by averaging the Brier score per ground-truth
    object. For each GT object, predictions with IoU above the threshold
    are aggregated and compared against the one-hot class target.

    When ``detections[i].scores`` has shape ``(N, C)`` (full class
    distributions), the exact multi-class Brier score is used. When
    scores are 1-D (max confidence only), a binary approximation is used.

    Args:
        detections: Post-processed predictions per image.
        ground_truths: Ground-truth annotations per image.
        iou_thresholds: IoU thresholds for matching. The final OCE is the
            average over thresholds.
        aggregation: How to aggregate predictions matched to the same GT
            object. One of ``"mean"``, ``"max_iou"``, ``"iou_weighted"``.

    Returns:
        :class:`MetricResult` with ``score`` (the OCE value) and
        ``per_element`` (per-object Brier scores).
    """
    if len(detections) != len(ground_truths):
        raise ValueError(
            f"Number of detections ({len(detections)}) != "
            f"ground truths ({len(ground_truths)})"
        )

    use_full = detections[0].has_class_distribution if len(detections) > 0 else True
    if not use_full:
        warnings.warn(
            "Scores are 1-D (max confidence only). Using binary Brier "
            "approximation for OCE. Pass full class distributions (N, C) "
            "for exact computation.",
            stacklevel=2,
        )

    per_threshold_scores: List[List[float]] = []

    for tau in iou_thresholds:
        brier_scores: List[float] = []

        for dets, gt in zip(detections, ground_truths):
            if gt.num_objects == 0:
                continue

            if dets.num_detections == 0:
                brier_scores.extend([1.0] * gt.num_objects)
                continue

            iou_mat = compute_iou_matrix(gt.boxes, dets.boxes)  # (M_gt, N_det)

            for obj_idx in range(gt.num_objects):
                gt_class = int(gt.labels[obj_idx])
                match_mask = iou_mat[obj_idx] >= tau

                if not match_mask.any():
                    brier_scores.append(1.0)
                    continue

                iou_vals = iou_mat[obj_idx, match_mask]

                if use_full:
                    matched_scores = dets.scores[match_mask]
                    brier = _brier_full(
                        matched_scores, gt_class, aggregation, iou_vals
                    )
                else:
                    matched_confs = dets.max_confidence[match_mask]
                    matched_labels = dets.labels[match_mask]
                    brier = _brier_binary(
                        matched_confs, matched_labels, gt_class,
                        aggregation, iou_vals,
                    )

                brier_scores.append(brier)

        per_threshold_scores.append(brier_scores)

    # Average across thresholds
    all_per_object = np.array(
        [np.mean(s) for s in zip(*per_threshold_scores)]
    ) if per_threshold_scores and per_threshold_scores[0] else np.array([])

    # Final score: average across thresholds then across objects
    threshold_means = [
        np.mean(s) if s else 1.0 for s in per_threshold_scores
    ]
    score = float(np.mean(threshold_means))

    return MetricResult(score=score, per_element=all_per_object)
