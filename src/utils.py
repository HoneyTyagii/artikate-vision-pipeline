"""Shared preprocessing / postprocessing helpers for the ONNX inference path.

These mirror what Ultralytics does internally so that FP32 PyTorch results and
exported-ONNX results are comparable. Getting the letterbox + coordinate
un-mapping exactly right is the single most common source of "boxes look
plausible but are subtly wrong" bugs, so it lives in one place with tests.
"""

from __future__ import annotations

import cv2
import numpy as np


def letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize while preserving aspect ratio and pad to `new_shape`.

    Returns the padded image, the scale ratio, and the (left, top) padding so
    that predictions can be mapped back to the original image exactly.
    """
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(dh), int(dh)
    left, right = int(dw), int(dw)
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, r, (left, top)


def preprocess(
    img: np.ndarray, imgsz: int = 640
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """BGR uint8 HWC image -> normalized float32 NCHW tensor + scale/pad meta."""
    lb, r, (pad_w, pad_h) = letterbox(img, (imgsz, imgsz))
    # BGR -> RGB, HWC -> CHW
    blob = lb[:, :, ::-1].transpose(2, 0, 1)
    blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
    blob = blob[None]  # add batch dim
    return blob, r, (pad_w, pad_h)


def scale_boxes(
    boxes: np.ndarray, r: float, pad: tuple[float, float], orig_shape: tuple[int, int]
) -> np.ndarray:
    """Map xyxy boxes from letterboxed space back to the original image."""
    pad_w, pad_h = pad
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes[:, :4] /= r
    h, w = orig_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Center xywh -> corner xyxy."""
    y = np.empty_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.45) -> list[int]:
    """Pure-numpy greedy NMS. Returns kept indices ordered by descending score."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thres]
    return keep


def postprocess(
    output: np.ndarray,
    r: float,
    pad: tuple[float, float],
    orig_shape: tuple[int, int],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a raw YOLOv8 ONNX output tensor into (boxes, scores, class_ids).

    Expects `output` shaped (1, 4 + num_classes, num_anchors) as produced by
    an Ultralytics ONNX export. Returns boxes in original-image xyxy space.
    """
    pred = np.squeeze(output, 0).T  # (num_anchors, 4 + num_classes)
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]

    class_ids = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)

    mask = scores > conf_thres
    boxes_xywh, scores, class_ids = boxes_xywh[mask], scores[mask], class_ids[mask]
    if boxes_xywh.shape[0] == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    boxes = xywh2xyxy(boxes_xywh)
    keep = nms(boxes, scores, iou_thres)
    boxes = scale_boxes(boxes[keep], r, pad, orig_shape)
    return boxes, scores[keep], class_ids[keep]
