"""Run the exported ONNX model on the held-out video clip.

Prints a per-frame log (frame index, detections, confidence, latency) and
optionally writes an annotated video.

Usage:
    python src/video_inference.py --model weights/best.onnx --video clip.mp4 \
        --out results/annotated.mp4
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort

from utils import postprocess, preprocess


def load_session(model_path: str) -> ort.InferenceSession:
    providers = ort.get_available_providers()
    # Prefer CUDA/TensorRT when present, else fall back to CPU.
    preferred = [
        p
        for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
        if p in providers
    ]
    return ort.InferenceSession(model_path, providers=preferred)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ONNX video inference")
    p.add_argument("--model", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--out", default=None, help="Optional annotated video path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    session = load_session(args.model)
    input_name = session.get_inputs()[0].name
    print("Using providers:", session.get_providers())

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    writer = None
    if args.out:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    latencies = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        blob, r, pad = preprocess(frame, args.imgsz)

        t0 = time.perf_counter()
        output = session.run(None, {input_name: blob})[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        boxes, scores, class_ids = postprocess(
            output, r, pad, frame.shape[:2], args.conf, args.iou
        )

        print(
            f"Frame {frame_idx:04d} | dets={len(scores):2d} | "
            f"latency={latency_ms:6.2f} ms"
        )

        if writer is not None:
            for (x1, y1, x2, y2), s, c in zip(boxes, scores, class_ids):
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{int(c)}:{s:.2f}",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            writer.write(frame)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    if latencies:
        arr = np.array(latencies)
        print(
            f"\nFrames: {len(arr)} | mean {arr.mean():.2f} ms | "
            f"p50 {np.percentile(arr, 50):.2f} ms | p95 {np.percentile(arr, 95):.2f} ms | "
            f"~{1000.0 / arr.mean():.1f} FPS"
        )


if __name__ == "__main__":
    main()
