"""Fine-tune a YOLOv8 detector on the Artikate private inspection dataset.

Usage:
    python src/train.py --data configs/dataset.yaml --model yolov8n.pt \
        --epochs 50 --imgsz 640 --batch 16

Design notes:
- yolov8n is the default because the target is edge deployment (Jetson Orin).
  A nano model keeps latency low; scale up to s/m only if recall is short.
- Results (weights, curves, confusion matrix) land under runs/detect/train*.
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 on the inspection set")
    p.add_argument("--data", default="configs/dataset.yaml")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="'0' for GPU, 'cpu' for CPU")
    p.add_argument("--name", default="artikate_yolov8n")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=15,
        seed=0,
    )
    print("Training complete. Best weights:", results.save_dir)


if __name__ == "__main__":
    main()
