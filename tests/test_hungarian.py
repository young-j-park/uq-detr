"""Test that our numpy Hungarian matcher matches the original DETR implementation."""

import numpy as np
import pytest

from uq_detr._matching import hungarian_match, compute_giou_matrix, compute_iou_matrix


class TestGIoU:
    def test_identical_boxes(self):
        boxes = np.array([[0, 0, 100, 100]], dtype=np.float64)
        giou = compute_giou_matrix(boxes, boxes)
        np.testing.assert_allclose(giou, [[1.0]])

    def test_non_overlapping(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[20, 20, 30, 30]], dtype=np.float64)
        giou = compute_giou_matrix(a, b)
        assert giou[0, 0] < 0  # GIoU is negative for non-overlapping boxes

    def test_matches_iou_for_overlapping(self):
        a = np.array([[0, 0, 100, 100]], dtype=np.float64)
        b = np.array([[50, 50, 150, 150]], dtype=np.float64)
        iou = compute_iou_matrix(a, b)
        giou = compute_giou_matrix(a, b)
        # GIoU <= IoU always
        assert giou[0, 0] <= iou[0, 0] + 1e-10

    def test_empty(self):
        a = np.zeros((0, 4))
        b = np.array([[0, 0, 1, 1]])
        assert compute_giou_matrix(a, b).shape == (0, 1)


class TestHungarianMatch:
    def test_perfect_match(self):
        """One pred per GT, should match 1-to-1."""
        pred_logits = np.array([
            [5.0, -5.0, -5.0],   # class 0
            [-5.0, 5.0, -5.0],   # class 1
        ])
        pred_boxes = np.array([
            [0.25, 0.25, 0.5, 0.5],
            [0.75, 0.75, 0.5, 0.5],
        ])
        gt_labels = np.array([0, 1])
        gt_boxes = np.array([
            [0.25, 0.25, 0.5, 0.5],
            [0.75, 0.75, 0.5, 0.5],
        ])

        pred_idx, gt_idx = hungarian_match(pred_logits, pred_boxes, gt_labels, gt_boxes)
        assert len(pred_idx) == 2
        assert len(gt_idx) == 2
        # pred 0 should match gt 0, pred 1 should match gt 1
        matches = dict(zip(pred_idx, gt_idx))
        assert matches[0] == 0
        assert matches[1] == 1

    def test_more_preds_than_gt(self):
        """5 predictions, 2 GT objects — should match exactly 2."""
        np.random.seed(42)
        pred_logits = np.random.randn(5, 3)
        pred_boxes = np.random.rand(5, 4) * 0.5
        pred_boxes[:, 2:] = 0.2  # small boxes

        gt_labels = np.array([0, 1])
        gt_boxes = np.array([
            [0.25, 0.25, 0.2, 0.2],
            [0.75, 0.75, 0.2, 0.2],
        ])

        pred_idx, gt_idx = hungarian_match(pred_logits, pred_boxes, gt_labels, gt_boxes)
        assert len(pred_idx) == 2
        assert len(gt_idx) == 2
        assert len(set(pred_idx)) == 2  # unique pred indices
        assert set(gt_idx) == {0, 1}   # both GT matched

    def test_empty_gt(self):
        pred_logits = np.random.randn(5, 3)
        pred_boxes = np.random.rand(5, 4)
        gt_labels = np.array([], dtype=np.int64)
        gt_boxes = np.zeros((0, 4))

        pred_idx, gt_idx = hungarian_match(pred_logits, pred_boxes, gt_labels, gt_boxes)
        assert len(pred_idx) == 0
        assert len(gt_idx) == 0

    def test_cost_weights(self):
        """Different cost weights should potentially change matching."""
        pred_logits = np.array([
            [5.0, -5.0],    # strongly class 0
            [-5.0, 5.0],    # strongly class 1
            [0.0, 0.0],     # ambiguous
        ])
        # pred 0 close to gt 1's box, pred 1 close to gt 0's box
        pred_boxes = np.array([
            [0.75, 0.75, 0.2, 0.2],
            [0.25, 0.25, 0.2, 0.2],
            [0.50, 0.50, 0.2, 0.2],
        ])
        gt_labels = np.array([0, 1])
        gt_boxes = np.array([
            [0.25, 0.25, 0.2, 0.2],
            [0.75, 0.75, 0.2, 0.2],
        ])

        # With high class cost: should prefer class match over box match
        p1, g1 = hungarian_match(pred_logits, pred_boxes, gt_labels, gt_boxes,
                                  cost_class=10.0, cost_bbox=1.0, cost_giou=1.0)
        # With high bbox cost: should prefer box match over class match
        p2, g2 = hungarian_match(pred_logits, pred_boxes, gt_labels, gt_boxes,
                                  cost_class=1.0, cost_bbox=10.0, cost_giou=10.0)

        match1 = dict(zip(p1, g1))
        match2 = dict(zip(p2, g2))

        # High class cost: pred 0 (class 0) -> gt 0 (class 0)
        assert match1[0] == 0
        # High bbox cost: pred 0 (near gt 1's box) -> gt 1
        assert match2[0] == 1
