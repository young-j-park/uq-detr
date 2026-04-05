"""Unit tests for uq_detr metrics using synthetic data."""

import numpy as np
import pytest

import uq_detr
from uq_detr import Detections, GroundTruth, box_convert, select


# ---------------------------------------------------------------------------
# Fixtures: synthetic detection scenarios
# ---------------------------------------------------------------------------

def _make_gt(n_objects=3):
    """Ground truth with n non-overlapping objects, classes 0..n-1."""
    boxes = np.array([
        [100 * i, 100, 100 * i + 80, 180] for i in range(n_objects)
    ], dtype=np.float64)
    labels = np.arange(n_objects)
    return GroundTruth(boxes=boxes, labels=labels)


def _perfect_detections(gt: GroundTruth, n_classes=5):
    """Detections that perfectly match GT with high confidence."""
    n = gt.num_objects
    scores = np.full((n, n_classes), 0.01)
    for i in range(n):
        scores[i, gt.labels[i]] = 0.95
    return Detections(boxes=gt.boxes.copy(), scores=scores, labels=gt.labels.copy())


def _bad_detections(gt: GroundTruth, n_classes=5):
    """Detections on correct boxes but wrong classes, high confidence."""
    n = gt.num_objects
    wrong_labels = (gt.labels + 1) % n_classes
    scores = np.full((n, n_classes), 0.01)
    for i in range(n):
        scores[i, wrong_labels[i]] = 0.90
    return Detections(boxes=gt.boxes.copy(), scores=scores, labels=wrong_labels)


def _no_detections(n_classes=5):
    """Empty detection set."""
    return Detections(
        boxes=np.zeros((0, 4)),
        scores=np.zeros((0, n_classes)),
        labels=np.zeros(0, dtype=np.int64),
    )


def _scalar_detections(gt: GroundTruth):
    """Detections with scalar (1-D) confidence only."""
    n = gt.num_objects
    return Detections(
        boxes=gt.boxes.copy(),
        scores=np.array([0.9] * n),
        labels=gt.labels.copy(),
    )


# ---------------------------------------------------------------------------
# Tests: OCE
# ---------------------------------------------------------------------------

class TestOCE:
    def test_perfect_detections_low_oce(self):
        gt = _make_gt(3)
        dets = _perfect_detections(gt)
        result = uq_detr.oce([dets], [gt])
        assert result.score < 0.1, f"Perfect detections should have low OCE, got {result.score}"

    def test_no_detections_oce_is_one(self):
        gt = _make_gt(3)
        dets = _no_detections()
        result = uq_detr.oce([dets], [gt])
        assert result.score == 1.0, f"No detections should give OCE=1.0, got {result.score}"

    def test_wrong_class_high_oce(self):
        gt = _make_gt(3)
        dets = _bad_detections(gt)
        result = uq_detr.oce([dets], [gt])
        assert result.score > 0.5, f"Wrong-class detections should have high OCE, got {result.score}"

    def test_binary_approx_oce(self):
        gt = _make_gt(3)
        dets = _scalar_detections(gt)
        with pytest.warns(UserWarning, match="binary Brier"):
            result = uq_detr.oce([dets], [gt])
        # Correct class, p=0.9: binary brier = 2*(1-0.9)^2 = 0.02
        assert result.score < 0.1

    def test_aggregation_max_iou(self):
        gt = _make_gt(1)
        dets = _perfect_detections(gt)
        result = uq_detr.oce([dets], [gt], aggregation="max_iou")
        assert result.score < 0.1

    def test_aggregation_iou_weighted(self):
        gt = _make_gt(1)
        dets = _perfect_detections(gt)
        result = uq_detr.oce([dets], [gt], aggregation="iou_weighted")
        assert result.score < 0.1

    def test_multiple_images(self):
        gts = [_make_gt(2), _make_gt(3)]
        dets = [_perfect_detections(gts[0]), _perfect_detections(gts[1])]
        result = uq_detr.oce(dets, gts)
        assert result.score < 0.1


# ---------------------------------------------------------------------------
# Tests: D-ECE
# ---------------------------------------------------------------------------

class TestDECE:
    def test_perfect_detections_low_dece(self):
        gt = _make_gt(3)
        dets = _perfect_detections(gt)
        result = uq_detr.dece([dets], [gt], tp_criterion="independent")
        assert result.score < 0.2

    def test_no_detections(self):
        gt = _make_gt(3)
        dets = _no_detections()
        result = uq_detr.dece([dets], [gt], tp_criterion="independent")
        assert result.score == 0.0

    def test_all_false_positives(self):
        gt = _make_gt(3)
        # Detections far from GT
        fp_boxes = np.array([[900, 900, 950, 950]] * 3, dtype=np.float64)
        dets = Detections(
            boxes=fp_boxes,
            scores=np.array([0.9, 0.8, 0.7]),
            labels=np.array([0, 1, 2]),
        )
        result = uq_detr.dece([dets], [gt], tp_criterion="independent")
        # High confidence but all FP → high ECE
        assert result.score > 0.5


# ---------------------------------------------------------------------------
# Tests: LA-ECE
# ---------------------------------------------------------------------------

class TestLaECE:
    def test_perfect_detections(self):
        gt = _make_gt(3)
        dets = _perfect_detections(gt)
        result = uq_detr.laece([dets], [gt], tp_criterion="independent")
        assert isinstance(result.score, float)
        assert result.score < 0.2

    def test_no_detections(self):
        gt = _make_gt(3)
        dets = _no_detections()
        result = uq_detr.laece([dets], [gt], tp_criterion="independent")
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# Tests: LRP
# ---------------------------------------------------------------------------

class TestLRP:
    def test_perfect_detections_low_lrp(self):
        gt = _make_gt(3)
        dets = _perfect_detections(gt)
        result = uq_detr.lrp([dets], [gt])
        assert result.score < 0.1

    def test_no_detections_lrp_is_one(self):
        gt = _make_gt(3)
        dets = _no_detections()
        result = uq_detr.lrp([dets], [gt])
        assert result.score == 1.0

    def test_all_false_positives(self):
        gt = _make_gt(3)
        fp_boxes = np.array([[900, 900, 950, 950]] * 5, dtype=np.float64)
        dets = Detections(
            boxes=fp_boxes,
            scores=np.array([0.9] * 5),
            labels=np.array([0] * 5),
        )
        result = uq_detr.lrp([dets], [gt])
        # 5 FP + 3 FN = 8, denom = 0 + 5 + 3 = 8 → LRP = 1.0
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# Tests: ContrastiveConf
# ---------------------------------------------------------------------------

class TestContrastiveConf:
    def test_basic(self):
        gt = _make_gt(3)
        # Simulate DETR output: 3 good queries + 7 background queries
        n_classes = 5
        good_scores = np.full((3, n_classes), 0.01)
        for i in range(3):
            good_scores[i, i] = 0.9
        bad_scores = np.full((7, n_classes), 0.02)

        all_scores = np.vstack([good_scores, bad_scores])
        all_boxes = np.vstack([gt.boxes, np.random.rand(7, 4) * 100])
        all_labels = np.concatenate([gt.labels, np.zeros(7, dtype=int)])

        queries = Detections(boxes=all_boxes, scores=all_scores, labels=all_labels)
        result = uq_detr.contrastive_conf([queries], method="threshold", param=0.3)
        assert result.shape == (1,)
        assert result[0] > 0  # Positive predictions should dominate


# ---------------------------------------------------------------------------
# Tests: Post-processing
# ---------------------------------------------------------------------------

class TestPostprocess:
    def test_threshold(self):
        gt = _make_gt(3)
        dets = _perfect_detections(gt)
        filtered = select(dets, method="threshold", param=0.5)
        assert filtered.num_detections == 3

    def test_topk(self):
        gt = _make_gt(5)
        dets = _perfect_detections(gt)
        filtered = select(dets, method="topk", param=3)
        assert filtered.num_detections == 3

    def test_nms(self):
        # Two overlapping boxes
        boxes = np.array([
            [0, 0, 100, 100],
            [10, 10, 110, 110],
            [500, 500, 600, 600],
        ], dtype=np.float64)
        scores = np.array([0.9, 0.8, 0.7])
        dets = Detections(boxes=boxes, scores=scores, labels=np.array([0, 0, 1]))
        filtered = select(dets, method="nms", param=0.5)
        assert filtered.num_detections == 2  # One of the overlapping pair removed

    def test_empty(self):
        dets = _no_detections()
        filtered = select(dets, method="threshold", param=0.5)
        assert filtered.num_detections == 0


# ---------------------------------------------------------------------------
# Tests: Box conversion
# ---------------------------------------------------------------------------

class TestBoxConvert:
    def test_cxcywh_to_xyxy(self):
        boxes = np.array([[50, 50, 100, 100]])  # center=(50,50), size=100x100
        result = box_convert(boxes, "cxcywh", "xyxy")
        np.testing.assert_allclose(result, [[0, 0, 100, 100]])

    def test_xyxy_to_cxcywh(self):
        boxes = np.array([[0, 0, 100, 100]])
        result = box_convert(boxes, "xyxy", "cxcywh")
        np.testing.assert_allclose(result, [[50, 50, 100, 100]])

    def test_with_image_size(self):
        # Normalized cxcywh
        boxes = np.array([[0.5, 0.5, 1.0, 1.0]])
        result = box_convert(boxes, "cxcywh", "xyxy", image_size=(480, 640))
        np.testing.assert_allclose(result, [[0, 0, 640, 480]])

    def test_roundtrip(self):
        boxes = np.array([[10, 20, 50, 80]])
        rt = box_convert(box_convert(boxes, "xyxy", "cxcywh"), "cxcywh", "xyxy")
        np.testing.assert_allclose(rt, boxes)


# ---------------------------------------------------------------------------
# Tests: Data types
# ---------------------------------------------------------------------------

class TestTypes:
    def test_detections_properties(self):
        dets = Detections(
            boxes=np.array([[0, 0, 1, 1]]),
            scores=np.array([[0.1, 0.9]]),
            labels=np.array([1]),
        )
        assert dets.num_detections == 1
        assert dets.has_class_distribution is True
        np.testing.assert_allclose(dets.max_confidence, [0.9])

    def test_scalar_scores(self):
        dets = Detections(
            boxes=np.array([[0, 0, 1, 1]]),
            scores=np.array([0.9]),
            labels=np.array([1]),
        )
        assert dets.has_class_distribution is False
        np.testing.assert_allclose(dets.max_confidence, [0.9])

    def test_ground_truth(self):
        gt = GroundTruth(
            boxes=np.array([[0, 0, 1, 1], [2, 2, 3, 3]]),
            labels=np.array([0, 1]),
        )
        assert gt.num_objects == 2
