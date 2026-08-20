# Experiment 2: YOLO11n + Class-Aware Balanced Dataset

**PRISM Project Sub-module (Self-contained)**

---

## 🎯 Objective
Evaluate the impact of **Class-Aware Dataset Balancing** on stock **YOLO11n** for 20-Class Plant Disease Object Detection.

```
PlantVillage + PlantDoc (20 Classes)
                ↓
    Class-Aware Balancing (Oversampling until Max:Min <= 3:1)
                ↓
            YOLO11n (Stock, Pretrained)
                ↓
     P3 / P4 / P5 Detection Heads (Standard unmodified YOLO11 neck/head)
                ↓
     20 Plant-Disease Classes Evaluation
```

> **Note on Architecture:** This experiment uses unmodified, stock **YOLO11n** with standard P3/P4/P5 detection heads (no Coordinate Attention, CSLSF, BiFPN, ALDL, or P2 modifications). Only the training data is balanced.

---

## 📋 Training & Hyperparameter Configuration

| Parameter | Value | Description |
|---|:---:|---|
| **Model** | `yolo11n.pt` | Stock Ultralytics YOLO11 Nano pretrained checkpoint |
| **Dataset Config** | `data/exp2_balanced.yaml` | Balanced train split + untouched validation/test splits |
| **Number of Classes** | `20` | Harmonized PlantVillage + PlantDoc class taxonomy |
| **Epochs** | `25` (or standard `100` if extended) | Training epochs |
| **Batch Size** | `16` | Mini-batch size |
| **Image Resolution** | `640 x 640` | Input spatial dimensions |
| **Optimizer** | `AdamW` | Decoupled Weight Decay Regularized Adam |
| **Base Learning Rate (`lr0`)** | `0.001` | Initial learning rate with Cosine Annealing |
| **Precision** | `AMP` | Automatic Mixed Precision on CUDA GPU |
| **Hardware** | NVIDIA RTX 5050 Laptop GPU | CUDA 12.8 acceleration |

---

## 📁 Experiment Directory Structure
```
experiments/exp2_balanced/
├── README.md                      # Experiment specification and alignment notes
├── balance_report.csv             # Before vs After class instance counts
├── train.py                       # Training pipeline script
├── evaluate.py                    # Evaluation & metrics generation script
├── metrics.json                   # Aggregated metrics for cross-experiment comparison
└── runs/
    ├── weights/
    │   ├── best.pt                # Peak validation PyTorch checkpoint
    │   ├── last.pt                # Final epoch PyTorch checkpoint
    │   └── best.onnx              # Exported ONNX graph
    ├── results.csv                # Epoch-by-epoch training metrics
    ├── results.png                # Loss & mAP progression curves
    ├── confusion_matrix_normalized.png # Normalized 20-class confusion matrix
    ├── F1_curve.png               # F1 score vs confidence threshold
    ├── PR_curve.png               # Precision vs Recall curve
    └── predictions/               # Test split bounding box visual predictions
```

---

## 🚀 How to Run

1. **Balance the Dataset:**
   ```bash
   python scripts/balance_dataset.py
   ```
2. **Train YOLO11n on Balanced Data:**
   ```bash
   python experiments/exp2_balanced/train.py
   ```
3. **Evaluate and Generate Artifacts:**
   ```bash
   python experiments/exp2_balanced/evaluate.py
   ```
