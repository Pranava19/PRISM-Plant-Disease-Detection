# PRISM: Plant Disease Detection & Severity Estimation Benchmark

Multi-Experiment Deep Learning Framework for 20-Class Agricultural Pathology Identification.

---

## 🔬 Experiment Architecture Roadmap

- **Experiment 1:** Baseline Stock YOLO11n (Unbalanced Raw Dataset)
- **Experiment 2 (This Sub-module):** Stock YOLO11n + Class-Aware Balanced Dataset (experiments/exp2_balanced/)
- **Experiment 3:** YOLO11n + Coordinate Attention (CA)
- **Experiment 4:** YOLO11n + Cross-Scale Local Spatial Fusion (CSLSF / BiFPN)
- **Experiment 5:** Full PRISM Architecture (Adaptive Multi-Scale Spatial Fusion + Dynamic Feature Routing + Tiny-Lesion P2 Scale Head)

---

## 📁 Repository Structure

``text
PRISM-Plant-Disease-Detection/
├── experiments/
│   └── exp2_balanced/             # Experiment 2 (YOLO11n + Balanced Dataset)
│       ├── weights/
│       │   ├── best.pt            # Peak validation PyTorch checkpoint
│       │   └── best.onnx          # Exported ONNX model graph
│       ├── balance_report.csv     # 20-class Before vs After instance counts
│       ├── per_class_metrics.csv  # 20-class AP@50 and AP@50-95 benchmark table
│       ├── metrics.json           # Cross-experiment comparison metadata
│       ├── EXPERIMENT2_REPORT.md  # Consolidated Technical Report
│       ├── README.md              # Alignment guide and specifications
│       ├── train.py               # Standalone training pipeline
│       ├── evaluate.py            # Standalone evaluation & visualization pipeline
│       └── [Visual plots: curves, AP bar chart, normalized confusion matrix]
│
├── data/
│   └── exp2_balanced.yaml         # Balanced dataset specification (20 classes)
├── scripts/
│   └── balance_dataset.py         # Class-aware oversampling engine
└── requirements.txt               # Pinned Python package dependencies
``

---

## 🚀 Quickstart

``bash
pip install -r requirements.txt

# 1. Generate balanced dataset
python scripts/balance_dataset.py

# 2. Train Experiment 2
python experiments/exp2_balanced/train.py

# 3. Evaluate on unseen test set
python experiments/exp2_balanced/evaluate.py
``
