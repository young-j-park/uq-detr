"""UQ-DETR: Uncertainty Quantification in Detection Transformers.

A pip-installable toolkit for evaluating calibration and image-level
reliability of object detection models (DETRs and beyond).

Reference:
    Park, Sobolewski, and Azizan. "Uncertainty Quantification in Detection
    Transformers: Object-Level Calibration and Image-Level Reliability."
    IEEE Trans. Pattern Analysis and Machine Intelligence.
    https://arxiv.org/abs/2412.01782
"""

from uq_detr._box_utils import box_convert
from uq_detr._matching import compute_giou_matrix, compute_iou_matrix, hungarian_match
from uq_detr._types import Detections, GroundTruth, MetricResult
from uq_detr.imreli import contrastive_conf, fit_lambda
from uq_detr.metrics import dece, laece, lrp, oce
from uq_detr.postprocess import select

__version__ = "0.1.0"

__all__ = [
    # Data types
    "Detections",
    "GroundTruth",
    "MetricResult",
    # Metrics
    "oce",
    "dece",
    "laece",
    "lrp",
    # Image-level reliability
    "contrastive_conf",
    "fit_lambda",
    # Post-processing
    "select",
    # Matching
    "hungarian_match",
    "compute_iou_matrix",
    "compute_giou_matrix",
    # Utilities
    "box_convert",
]
