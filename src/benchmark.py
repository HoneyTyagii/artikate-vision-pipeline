"""Benchmark latency across PyTorch FP32, ONNX FP32, and INT8.

INT8 uses onnxruntime's static quantization with a small calibration set drawn
from the validation images. If INT8 tooling is unavailable, benchmark an FP16
ONNX export instead (note which you used in the README).

Usage:
    python src/benchmark.py --pt weights/best.pt --onnx weights/best.onnx \
        --calib-dir dataset/images/val --imgsz 640 --runs 100
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

from utils import preprocess


def bench_onnx(model_path: str, sample: np.ndarray, runs: int, warmup: int = 10) -> dict:
    sess = ort.InferenceSession(model_path, providers=ort.get_available_providers())
    name = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {name: sample})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: sample})
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {"mean_ms": arr.mean(), "p95_ms": np.percentile(arr, 95), "fps": 1000.0 / arr.mean()}

def bench_pytorch(weights: str, sample_img: np.ndarray, imgsz: int, runs: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(weights)
    for _ in range(10):
        model.predict(sample_img, imgsz=imgsz, verbose=False)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(sample_img, imgsz=imgsz, verbose=False)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {"mean_ms": arr.mean(), "p95_ms": np.percentile(arr, 95), "fps": 1000.0 / arr.mean()}


class CalibReader:
    """Static-quantization calibration data reader for onnxruntime."""

    def __init__(self, calib_dir: str, input_name: str, imgsz: int, limit: int = 100):
        self.input_name = input_name
        self.imgsz = imgsz
        files = sorted(glob.glob(os.path.join(calib_dir, "*")))[:limit]
        self.files = iter(files)

    def get_next(self):
        try:
            path = next(self.files)
        except StopIteration:
            return None
        img = cv2.imread(path)
        if img is None:
            return self.get_next()
        blob, _, _ = preprocess(img, self.imgsz)
        return {self.input_name: blob}


def quantize_int8(onnx_path: str, calib_dir: str, imgsz: int) -> str:
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self.r = CalibReader(calib_dir, input_name, imgsz)

        def get_next(self):
            return self.r.get_next()

    out_path = onnx_path.replace(".onnx", "_int8.onnx")
    quantize_static(onnx_path, out_path, _Reader(), weight_type=QuantType.QInt8)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Latency benchmark")
    p.add_argument("--pt", default=None, help="PyTorch weights (.pt) for FP32 baseline")
    p.add_argument("--onnx", required=True)
    p.add_argument("--calib-dir", default=None, help="Val images for INT8 calibration")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--sample", default=None, help="One image to drive the benchmark")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.sample and os.path.exists(args.sample):
        img = cv2.imread(args.sample)
    else:
        img = (np.random.rand(args.imgsz, args.imgsz, 3) * 255).astype(np.uint8)
    blob, _, _ = preprocess(img, args.imgsz)

    print(f"{'Config':<22}{'mean (ms)':>12}{'p95 (ms)':>12}{'FPS':>10}")
    print("-" * 56)

    if args.pt:
        r = bench_pytorch(args.pt, img, args.imgsz, args.runs)
        print(f"{'PyTorch FP32':<22}{r['mean_ms']:>12.2f}{r['p95_ms']:>12.2f}{r['fps']:>10.1f}")

    r = bench_onnx(args.onnx, blob, args.runs)
    print(f"{'ONNX FP32':<22}{r['mean_ms']:>12.2f}{r['p95_ms']:>12.2f}{r['fps']:>10.1f}")

    if args.calib_dir:
        int8_path = quantize_int8(args.onnx, args.calib_dir, args.imgsz)
        r = bench_onnx(int8_path, blob, args.runs)
        print(f"{'ONNX INT8':<22}{r['mean_ms']:>12.2f}{r['p95_ms']:>12.2f}{r['fps']:>10.1f}")
        print(f"\nINT8 model written to: {int8_path}")


if __name__ == "__main__":
    main()
