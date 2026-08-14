# 02_yolo_prismghost: 20-Class Plant Disease Object Detection

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.7+cu128](https://img.shields.io/badge/PyTorch-2.7%2Bcu128-EE4C2C.svg)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ONNX](https://img.shields.io/badge/ONNX-8.64_MB-005CED.svg)](https://onnx.ai/)

Experiment 2 of the AgriYOLO project: A lightweight, high-performance object detection model for **20-Class Plant Disease Detection** using **YOLOv8 + PrismGhost Backbone** and **Quad-Head Detection ($P_2, P_3, P_4, P_5$)**.

---

## 📁 Repository Organization
```
02_yolo_prismghost/
│
├── README.md                          # Project documentation and benchmarks
│
├── model/
│   ├── yolo_prismghost.yaml           # Quad-head YOLO architecture configuration
│   └── prism_modules.py               # PyTorch PrismGhost module implementation
│
├── code/
│   └── run_exp2_local_gpu.py          # End-to-end Python pipeline script
│
├── notebook/
│   └── final_error_free_model.ipynb   # Interactive step-by-step Jupyter notebook
│
├── results/
│   ├── results.csv                    # Epoch-by-epoch training metrics
│   ├── results.png                    # Training loss and mAP curves
│   ├── confusion_matrix_normalized.png# 20-class normalized confusion matrix
│   ├── F1_curve.png                   # F1-confidence curve
│   ├── PR_curve.png                   # Precision-Recall curve
│   └── weights/                       # Model weights (best.pt & best.onnx)
│
└── metrics/
    └── metrics.json                   # Structured JSON metrics summary
```

---

## 🎯 20 Unified Target Classes
0: `Apple Scab` | 1: `Apple Cedar Rust` | 2: `Apple Healthy` | 3: `Corn Gray Leaf Spot` | 4: `Corn Common Rust` | 5: `Corn Northern Leaf Blight` | 6: `Grape Black Rot` | 7: `Grape Healthy` | 8: `Pepper Bacterial Spot` | 9: `Pepper Healthy` | 10: `Potato Early Blight` | 11: `Potato Late Blight` | 12: `Tomato Bacterial Spot` | 13: `Tomato Early Blight` | 14: `Tomato Late Blight` | 15: `Tomato Leaf Mold` | 16: `Tomato Septoria Leaf Spot` | 17: `Tomato Yellow Leaf Curl Virus` | 18: `Tomato Mosaic Virus` | 19: `Tomato Healthy`

---

## 📊 Benchmark Performance

| Metric | Result |
|---|:---:|
| **Precision ($P$)** | **92.32%** |
| **Recall ($R$)** | **76.90%** |
| **mAP@50** | **84.65%** |
| **mAP@50-95** | **78.76%** |
| **Parameters** | **2.25 M** |
| **GFLOPs** | **10.8** |
| **Inference Latency** | **11.8 ms** / image ($640 \times 640$) |
| **PyTorch Weights (`best.pt`)** | **4.79 MB** |
| **ONNX Export (`best.onnx`)** | **8.64 MB** |

---

## 🚀 Quickstart

```bash
# Clone the repository
git clone https://github.com/<your-username>/02_yolo_prismghost.git
cd 02_yolo_prismghost

# Install dependencies
pip install ultralytics==8.3.40 torch torchvision pillow kaggle opencv-python matplotlib albumentations pyyaml pandas seaborn onnx onnxruntime

# Run training & evaluation
python code/run_exp2_local_gpu.py
```
