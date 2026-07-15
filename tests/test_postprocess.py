"""Unit tests for the preprocessing / postprocessing math.

These guard the coordinate-space mapping and NMS logic, which are exactly the
places where a detector can produce plausible-but-wrong boxes.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import letterbox, nms, scale_boxes, xywh2xyxy  # noqa: E402


def test_letterbox_preserves_aspect_ratio():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out, r, (pad_w, pad_h) = letterbox(img, (640, 640))
    assert out.shape[:2] == (640, 640)
    # 640/640 vs 640/480 -> limited by width, r = 1.0
    assert abs(r - 1.0) < 1e-6
    # Vertical padding should account for the height difference.
    assert pad_h > 0 and abs(pad_w) < 1e-6


def test_scale_boxes_roundtrip():
    # A box in letterboxed space maps back into original bounds.
    orig = (480, 640)
    _, r, pad = letterbox(np.zeros((*orig, 3), np.uint8), (640, 640))
    boxes = np.array([[100.0, 100.0, 200.0, 200.0]])
    mapped = scale_boxes(boxes, r, pad, orig)
    assert (mapped[:, 0] >= 0).all() and (mapped[:, 2] <= orig[1]).all()
    assert (mapped[:, 1] >= 0).all() and (mapped[:, 3] <= orig[0]).all()


def test_xywh2xyxy():
    x = np.array([[50.0, 50.0, 20.0, 40.0]])  # cx, cy, w, h
    y = xywh2xyxy(x)
    np.testing.assert_allclose(y, [[40.0, 30.0, 60.0, 70.0]])


def test_nms_suppresses_overlaps():
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],  # heavy overlap with box 0
            [50.0, 50.0, 60.0, 60.0],  # disjoint
        ]
    )
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_thres=0.45)
    assert 0 in keep and 2 in keep
    assert 1 not in keep  # suppressed by the higher-scoring overlapping box


def test_nms_orders_by_score():
    boxes = np.array([[0.0, 0.0, 5.0, 5.0], [100.0, 100.0, 105.0, 105.0]])
    scores = np.array([0.3, 0.95])
    keep = nms(boxes, scores, iou_thres=0.45)
    assert keep[0] == 1  # highest score first
