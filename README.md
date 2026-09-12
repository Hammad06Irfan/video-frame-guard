# Video Frame Guard 

A low-latency, end-to-end computer vision pipeline and C++ inference engine designed to detect, classify, and isolate real-time GPU rendering and display corruptions.

![C++](https://img.shields.io/badge/C++-17-00599C?style=flat&logo=c%2B%2B&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-C%2B%2B_API-005CED?style=flat)
![Throughput](https://img.shields.io/badge/Throughput-680+_FPS-brightgreen?style=flat)
![Latency](https://img.shields.io/badge/Latency-1.46_ms-blue?style=flat)

---

## Motivation

In modern GPU display pipelines and video processing stacks, visual corruptions can introduce subtle yet severe failures:
* **V-Sync / Buffer Swap Desync:** Horizontal tearing across frame slices.
* **Codec & Decompression Degradation:** Coarse macroblocking artifacts.
* **VRAM & PCIe Transmission Errors:** Bit-flip salt-and-pepper noise.

Writing heuristic, hand-crafted rules to detect every possible failure mode is brittle. **Video Frame Guard** addresses this by pairing a lightweight convolutional neural network (**MobileNetV3**) trained in PyTorch with a standalone **C++ inference engine** powered by the **ONNX Runtime C++ API**.

---

## Performance & Benchmarks

Benchmarked on Apple Silicon CPU across 50 execution runs (224x224x3 input resolution, single-frame batch size N=1):

| Metric | Measured Value | Display Budget Impact |
| :--- | :--- | :--- |
| **Inference Latency** | **1.46 ms** | Consumes only **21%** of a 144 Hz display frame window (6.94 ms budget) |
| **Throughput** | **~680 FPS** | Capable of high-framerate stream inspection |
| **Validation Accuracy** | **88.7%** | Fine-tuned across 4 discrete corruption classes |
| **FP32 Model Footprint** | **0.31 MB** | Extremely compact classification head for fast cache residency |

---

## System Architecture

```
[ Synthetic Corruption Pipeline ]
   ├── Screen Tearing Simulation (Horizontal scanline displacement)
   ├── Macroblocking Simulation (Severe codec downsampling & blending)
   └── Bit-Noise Generation (Uniform random VRAM bit-flip errors)
                 │
                 ▼
[ PyTorch MobileNetV3 Transfer Learning ]
   ├── Frozen Feature-Extraction Layers (Preserves edge & texture weights)
   ├── Fine-Tuned 4-Way Classifier Head (Trained with CrossEntropyLoss + Adam)
   └── MPS (Metal) Hardware-Accelerated Training
                 │
                 ▼ (ONNX Graph Serialization - Opset 14)
[ High-Throughput C++ Engine ]
   ├── stb_image Raw RGB Byte Loading
   ├── In-Memory Layout Conversion: Interleaved HWC -> Planar CHW Float Tensors
   └── Multi-Threaded CPU Session with Graph Optimization
```

---

## Systems & Engineering Insights

### 1. In-Memory Planar Layout (`HWC` to `CHW`)
Standard image loaders decode pixel data into contiguous interleaved bytes: `[R0, G0, B0, R1, G1, B1, ...]` (**HWC**).

Deep learning vector engines and SIMD kernels achieve peak throughput when operating on contiguous color channels (**CHW**). Rather than introducing external conversion dependencies, the C++ runtime handles normalization and planar restructuring in a single linear pass directly into pre-allocated tensor memory with zero intermediate buffer copies.

### 2. Quantization Overhead on Sub-Megabyte Graphs
Post-training dynamic **INT8 quantization** was evaluated to compare execution footprint:
* **FP32 Baseline:** `0.31 MB`
* **INT8 Quantized:** `1.69 MB`

**Takeaway:** Because feature extraction layers were frozen, the model graph was already exceptionally lightweight. Introducing runtime quantization parameters, scaling lookups, and dequantization operator wrappers (`QuantizeLinear`/`DequantizeLinear`) added more metadata overhead than the weight reduction saved. For micro-classification heads under 1 MB, **FP32 execution remains the optimal systems choice**.

---

## Repository Structure

```
video-frame-guard/
├── data/
│   ├── raw/                 # Procedural seed base frames
│   └── dataset/             # Organized 80/20 train/val splits
├── models/
│   ├── model.onnx           # Exported FP32 ONNX model (0.31 MB)
│   └── model_int8.onnx      # Quantized model artifact
├── src/
│   ├── generate_data.py     # Procedural visual corruption generator
│   ├── train.py             # Transfer learning pipeline & ONNX exporter
│   └── quantize.py          # Dynamic INT8 quantization script
├── cpp/
│   ├── main.cpp             # Low-latency C++ inference runner
│   ├── CMakeLists.txt       # CMake build system
│   └── stb_image.h          # Lightweight header-only image decoder
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites
* Python 3.10+
* CMake 3.16+ & C++17 compiler
* Homebrew (macOS) with ONNX Runtime:
  ```bash
  brew install onnxruntime
  ```

### 1. Python Pipeline (Data & Model Generation)
```bash
# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Generate synthetic failure dataset
python3 src/generate_data.py

# 2. Train classifier and export models/model.onnx
python3 src/train.py

# 3. (Optional) Run INT8 quantization benchmark
python3 src/quantize.py
```

### 2. Build & Run the C++ Inference Engine
```bash
# Build binary
mkdir -p cpp/build && cd cpp/build
cmake ..
make
cd ../..

# Run inference on any frame
./cpp/build/frame_guard data/dataset/val/torn/frame_0162_torn.png
```

### Example Engine Output
```text
==========================================
  AMD Frame Guard - C++ Inference Engine   
==========================================
[*] Loading Model: models/model.onnx
[*] Target Frame:  data/dataset/val/torn/frame_0162_torn.png
[*] Benchmarking inference over 50 iterations...
  - blocky: 0.128487%
  - clean: 12.6465%
  - noise: 0.0947117%
  - torn: 87.1303%

------------------------------------------
[+] Result: Frame Classified as [torn]
[+] Average Latency: 1.46238 ms
[+] Throughput:      683 FPS
------------------------------------------
```
