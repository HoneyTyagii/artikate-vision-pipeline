<div align="center">

# 🔍 Artikate Vision Pipeline

**A production-style computer-vision pipeline that fine-tunes a YOLOv8 detector on a private industrial-inspection dataset, exports to ONNX, runs real-time inference on a held-out clip, and benchmarks FP32 vs INT8 latency -built for edge deployment on NVIDIA Jetson.**

Train → Export → Infer → Benchmark. Every number comes from code you actually run.

<p>
<img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
<img alt="Ultralytics" src="https://img.shields.io/badge/Ultralytics-YOLOv8-0B23A9?logo=yolo&logoColor=white">
<img alt="ONNX" src="https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white">
<img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.9-5C3EE8?logo=opencv&logoColor=white">
<img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.24-013243?logo=numpy&logoColor=white">
<img alt="Jetson" src="https://img.shields.io/badge/Target-Jetson%20Orin-76B900?logo=nvidia&logoColor=white">
<img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

</div>

---

> **Status of numbers:** every metric in the tables below is a placeholder until
> run against the private dataset. Replace `TODO` cells with the real measured
> values before submitting. Do not report numbers you have not measured.

## ✨ Highlights

| | Capability | Description |
|---|---|---|
| 🎯 | **YOLOv8 Fine-tuning** | Adapts a nano/small detector to the private inspection set -nano by default to keep edge latency low |
| 📦 | **ONNX Export** | Exports `best.pt` → `best.onnx` (opset 12, simplified) for portable, framework-free inference |
| ⚡ | **ONNX Runtime Inference** | Runs on the held-out clip with automatic CUDA/TensorRT/CPU provider selection |
| 📊 | **FP32 vs INT8 Benchmark** | Static INT8 quantization with a real calibration set -measures mean/p95 latency and FPS |
| 🧮 | **Exact Coordinate Mapping** | Letterbox + reverse-mapping isolated in `utils.py`, guarded by unit tests |
| 🧾 | **Per-frame Logging** | Emits a CSV of detections, confidence, and latency for every frame |
| 🧪 | **Tested Postprocessing** | Unit tests pin the NMS + coordinate math that silently breaks detectors |

---

## 🚀 Quickstart

### Setup

```bash
git clone https://github.com/HoneyTyagii/artikate-vision-pipeline.git
cd artikate-vision-pipeline

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

Unpack the provided dataset into `dataset/` in YOLO format and set the class
names in `configs/dataset.yaml`.

### Run the pipeline

```bash
# 1. Train
python src/train.py --data configs/dataset.yaml --model yolov8n.pt --epochs 50 --imgsz 640

# 2. Export to ONNX
python src/export.py --weights runs/detect/artikate_yolov8n/weights/best.pt

# 3. Inference on the held-out clip
python src/video_inference.py --model weights/best.onnx --video clip.mp4 --out results/annotated.mp4

# 4. Benchmark FP32 vs INT8
python src/benchmark.py --pt weights/best.pt --onnx weights/best.onnx --calib-dir dataset/images/val

# tests
pytest -q
```

---

## 🗂 Repo layout

| Path | Purpose |
|---|---|
| `configs/dataset.yaml` | Ultralytics data config (edit class names + path) |
| `src/train.py` | Fine-tune YOLOv8 on the inspection set |
| `src/export.py` | Export `best.pt` → `best.onnx` |
| `src/video_inference.py` | ONNX Runtime inference on the held-out clip + CSV log |
| `src/benchmark.py` | FP32 vs INT8 latency + FPS |
| `src/utils.py` | Letterbox / preprocess / NMS / coordinate mapping |
| `tests/test_postprocess.py` | Guards the coordinate + NMS math |
| `ANSWERS.md` | Sections 1, 3, 4 written answers |

---

## 🖥 Hardware

- **CPU:** TODO (e.g. Intel i7-xxxx)
- **GPU:** TODO (or "CPU-only")
- **ONNX Runtime provider used:** TODO (CPUExecutionProvider / CUDAExecutionProvider / TensorRT)

---

## 📈 Measured results

### Validation metrics (`train.py` / `model.val()`)

| Metric        | Value |
|---------------|-------|
| mAP@0.5       | TODO  |
| mAP@0.5:0.95  | TODO  |
| Precision     | TODO  |
| Recall        | TODO  |

### Latency (imgsz 640, batch 1)

| Config       | mean (ms) | p95 (ms) | FPS  |
|--------------|-----------|----------|------|
| PyTorch FP32 | TODO      | TODO     | TODO |
| ONNX FP32    | TODO      | TODO     | TODO |
| ONNX INT8    | TODO      | TODO     | TODO |

### Worst failure cases on the held-out clip

1. TODO - hypothesis: TODO
2. TODO - hypothesis: TODO
3. TODO - hypothesis: TODO

---

## 🎯 Confidence in these numbers on a different machine

TODO - e.g. "Latency was measured CPU-only on ORT's CPUExecutionProvider; the
ranking (INT8 < FP32) should hold across machines, but absolute ms will scale with
core count and clock. GPU/TensorRT numbers would differ substantially and I have
not benchmarked those."
