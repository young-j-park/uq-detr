"""Image-level reliability estimation via contrastive confidence.

Quantifies per-image reliability by contrasting the average foreground
confidence of positive predictions against negative predictions.

Reference: Park et al., "Uncertainty Quantification in Detection Transformers:
Object-Level Calibration and Image-Level Reliability", IEEE TPAMI.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.stats import pearsonr

from uq_detr._types import Detections
from uq_detr.postprocess import select


def _split_conf(
    all_queries: Sequence[Detections],
    method: str,
    param: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-image Conf+ and Conf-."""
    conf_pos_list = []
    conf_neg_list = []

    for queries in all_queries:
        positives = select(queries, method=method, param=param)

        if positives.num_detections == 0:
            conf_pos_list.append(0.0)
            conf_neg_list.append(
                float(queries.max_confidence.mean())
                if queries.num_detections > 0 else 0.0
            )
            continue

        conf_pos_list.append(float(positives.max_confidence.mean()))

        # Negative set
        if method == "threshold":
            neg_mask = queries.max_confidence <= param
        elif method == "topk":
            k = min(int(param), queries.num_detections)
            top_indices = set(np.argsort(-queries.max_confidence)[:k].tolist())
            neg_mask = np.array([
                i not in top_indices for i in range(queries.num_detections)
            ])
        else:
            neg_mask = np.ones(queries.num_detections, dtype=bool)
            neg_mask[np.argsort(-queries.max_confidence)[:positives.num_detections]] = False

        neg_confs = queries.max_confidence[neg_mask]
        conf_neg_list.append(float(neg_confs.mean()) if len(neg_confs) > 0 else 0.0)

    return np.array(conf_pos_list), np.array(conf_neg_list)


def contrastive_conf(
    all_queries: Sequence[Detections],
    method: str = "threshold",
    param: float = 0.3,
    lambda_: float = 5.0,
) -> np.ndarray:
    """Compute per-image ContrastiveConf reliability scores.

    For each image, splits predictions into positive and negative sets
    using the specified post-processing method, then computes:

    ``ContrastiveConf = Conf+(x) - lambda * Conf-(x)``

    where ``Conf+`` and ``Conf-`` are the average max-foreground-confidence
    of the positive and negative sets, respectively.

    Args:
        all_queries: Full set of predictions (all queries) per image.
            Typically the raw DETR output before post-processing.
        method: Post-processing method for splitting into positive/negative.
        param: Post-processing parameter.
        lambda_: Scaling factor for the negative confidence term.
            Recommended range: 5.0-10.0. Use :func:`fit_lambda` to find
            the optimal value on a validation set.

    Returns:
        Array of shape ``(N_images,)`` with per-image reliability scores.
    """
    conf_pos, conf_neg = _split_conf(all_queries, method, param)
    return conf_pos - lambda_ * conf_neg


def fit_lambda(
    all_queries: Sequence[Detections],
    reliability: np.ndarray,
    method: str = "threshold",
    param: float = 0.3,
    lambda_range: Optional[Sequence[float]] = None,
) -> tuple[float, float]:
    """Find the optimal scaling factor lambda on a validation set.

    Sweeps over lambda values and returns the one that maximizes the
    Pearson correlation between ContrastiveConf and a user-provided
    per-image reliability measure (e.g., per-image AP).

    Args:
        all_queries: Full set of predictions (all queries) per image.
        reliability: Per-image reliability scores, shape ``(N_images,)``.
            For example, per-image AP computed via pycocotools, or
            negated per-image OCE.
        method: Post-processing method for splitting pos/neg.
        param: Post-processing parameter.
        lambda_range: Lambda values to search over. Defaults to
            ``[0, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]``.

    Returns:
        Tuple of ``(best_lambda, best_pcc)`` --- the lambda value that
        maximizes Pearson correlation, and the corresponding PCC.

    Example:
        ```python
        # On validation set
        best_lam, best_pcc = uq_detr.fit_lambda(
            val_queries, val_per_image_ap,
            method="threshold", param=0.3,
        )
        print(f"Best lambda: {best_lam} (PCC={best_pcc:.4f})")

        # Apply on test set
        test_scores = uq_detr.contrastive_conf(
            test_queries, method="threshold", param=0.3, lambda_=best_lam,
        )
        ```
    """
    if lambda_range is None:
        lambda_range = [
            0, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20
        ]

    reliability = np.asarray(reliability)
    valid = ~np.isnan(reliability)

    if valid.sum() < 3:
        raise ValueError("Need at least 3 valid (non-NaN) reliability values.")

    conf_pos, conf_neg = _split_conf(all_queries, method, param)

    best_lambda = 0.0
    best_pcc = -np.inf

    for lam in lambda_range:
        scores = conf_pos - lam * conf_neg
        pcc, _ = pearsonr(scores[valid], reliability[valid])
        if pcc > best_pcc:
            best_pcc = pcc
            best_lambda = lam

    return float(best_lambda), float(best_pcc)
