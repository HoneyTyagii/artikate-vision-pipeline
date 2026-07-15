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
