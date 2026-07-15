"""Export a trained YOLOv8 checkpoint to ONNX.

Usage:
    python src/export.py --weights runs/detect/artikate_yolov8n/weights/best.pt \
        --imgsz 640 --opset 12

Notes:
- opset 12 is a safe floor for ONNX Runtime and TensorRT on Jetson.
- We keep dynamic=False and a fixed imgsz so the INT8 calibration and the
  latency benchmark run against a stable input shape.
- simplify=True folds constants so the graph is easier to quantize later.
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLOv8 to ONNX")
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=False,
        simplify=True,
    )
    print("Exported ONNX model to:", path)


if __name__ == "__main__":
    main()
