from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np


@dataclass
class Detections:
    """Predictions for a single image.

    Boxes must be in **xyxy format with absolute pixel coordinates**
    (i.e., ``[x_min, y_min, x_max, y_max]``). If your model outputs a
    different format, use :func:`Detections.from_cxcywh` or
    :func:`uq_detr.box_convert` to convert.

    At minimum, provide ``boxes`` and one of the following:

    - ``scores`` with shape ``(N, C)``: full class distributions.
      ``labels`` is inferred via argmax if not provided.
    - ``scores`` with shape ``(N,)`` and ``labels``: max-confidence
      scores with predicted class indices.

    Attributes:
        boxes: Bounding boxes in **xyxy** format, **absolute pixel
            coordinates**. Shape ``(N, 4)``.
        scores: Class probability distributions ``(N, C)`` or
            max-confidence scores ``(N,)``.
        labels: Predicted class indices ``(N,)``. Optional when
            ``scores`` is ``(N, C)``.
    """

    boxes: np.ndarray
    scores: np.ndarray
    labels: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float64)
        self.scores = np.asarray(self.scores, dtype=np.float64)

        if self.labels is not None:
            self.labels = np.asarray(self.labels, dtype=np.int64)
        elif self.scores.ndim == 2 and self.scores.shape[1] > 1:
            self.labels = self.scores.argmax(axis=1)
        else:
            raise ValueError(
                "labels must be provided when scores is 1-D (max-confidence only)."
            )

    @classmethod
    def from_cxcywh(
        cls,
        boxes: np.ndarray,
        scores: np.ndarray,
        image_size: Tuple[int, int],
        labels: Optional[np.ndarray] = None,
        normalized: bool = True,
    ) -> "Detections":
        """Create from center-format boxes (typical DETR output).

        Args:
            boxes: ``(N, 4)`` in ``(cx, cy, w, h)`` format.
            scores: ``(N, C)`` or ``(N,)``.
            image_size: ``(height, width)`` of the image.
            labels: ``(N,)`` predicted class indices. Optional when
                ``scores`` is ``(N, C)``.
            normalized: If ``True`` (default), boxes are in [0, 1]
                normalized coordinates (as in HuggingFace / DETR).
                If ``False``, boxes are already in absolute pixels.
        """
        from uq_detr._box_utils import box_convert
        boxes_xyxy = box_convert(
            boxes, "cxcywh", "xyxy",
            image_size=image_size if normalized else None,
        )
        return cls(boxes=boxes_xyxy, scores=scores, labels=labels)

    @classmethod
    def from_xywh(
        cls,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        image_size: Optional[Tuple[int, int]] = None,
        normalized: bool = False,
    ) -> "Detections":
        """Create from ``(x_min, y_min, width, height)`` boxes (COCO format).

        Args:
            boxes: ``(N, 4)`` in ``(x, y, w, h)`` format.
            scores: ``(N, C)`` or ``(N,)``.
            labels: ``(N,)`` predicted class indices. Optional when
                ``scores`` is ``(N, C)``.
            image_size: ``(height, width)``. Required if ``normalized``
                is ``True``.
            normalized: If ``True``, boxes are in [0, 1].
        """
        from uq_detr._box_utils import box_convert
        boxes_xyxy = box_convert(
            boxes, "xywh", "xyxy",
            image_size=image_size if normalized else None,
        )
        return cls(boxes=boxes_xyxy, scores=scores, labels=labels)

    @property
    def num_detections(self) -> int:
        return len(self.boxes)

    @property
    def has_class_distribution(self) -> bool:
        """Whether scores contain full class distributions (N, C)."""
        return self.scores.ndim == 2 and self.scores.shape[1] > 1

    @property
    def max_confidence(self) -> np.ndarray:
        """Max confidence per detection. Shape ``(N,)``."""
        if self.scores.ndim == 1:
            return self.scores
        return self.scores.max(axis=1)


@dataclass
class GroundTruth:
    """Ground-truth annotations for a single image.

    Boxes must be in **xyxy format with absolute pixel coordinates**.
    Use :func:`GroundTruth.from_cxcywh` or :func:`GroundTruth.from_xywh`
    if your annotations use a different format.

    Attributes:
        boxes: Bounding boxes in **xyxy** format, **absolute pixel
            coordinates**. Shape ``(M, 4)``.
        labels: Class indices. Shape ``(M,)``.
    """

    boxes: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float64)
        self.labels = np.asarray(self.labels, dtype=np.int64)

    @classmethod
    def from_cxcywh(
        cls,
        boxes: np.ndarray,
        labels: np.ndarray,
        image_size: Tuple[int, int],
        normalized: bool = True,
    ) -> "GroundTruth":
        """Create from center-format boxes.

        Args:
            boxes: ``(M, 4)`` in ``(cx, cy, w, h)`` format.
            labels: ``(M,)`` class indices.
            image_size: ``(height, width)`` of the image.
            normalized: If ``True`` (default), boxes are in [0, 1].
        """
        from uq_detr._box_utils import box_convert
        boxes_xyxy = box_convert(
            boxes, "cxcywh", "xyxy",
            image_size=image_size if normalized else None,
        )
        return cls(boxes=boxes_xyxy, labels=labels)

    @classmethod
    def from_xywh(
        cls,
        boxes: np.ndarray,
        labels: np.ndarray,
        image_size: Optional[Tuple[int, int]] = None,
        normalized: bool = False,
    ) -> "GroundTruth":
        """Create from ``(x_min, y_min, width, height)`` boxes (COCO format).

        Args:
            boxes: ``(M, 4)`` in ``(x, y, w, h)`` format.
            labels: ``(M,)`` class indices.
            image_size: ``(height, width)``. Required if ``normalized``
                is ``True``.
            normalized: If ``True``, boxes are in [0, 1].
        """
        from uq_detr._box_utils import box_convert
        boxes_xyxy = box_convert(
            boxes, "xywh", "xyxy",
            image_size=image_size if normalized else None,
        )
        return cls(boxes=boxes_xyxy, labels=labels)

    @property
    def num_objects(self) -> int:
        return len(self.boxes)


@dataclass
class MetricResult:
    """Result container for a calibration metric.

    Attributes:
        score: The aggregated metric value.
        per_element: Per-element scores (per-object for OCE,
            per-detection for D-ECE/LA-ECE, etc.). ``None`` if not applicable.
    """

    score: float
    per_element: Optional[np.ndarray] = None
