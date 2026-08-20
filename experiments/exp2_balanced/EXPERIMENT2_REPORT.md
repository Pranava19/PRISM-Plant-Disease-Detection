# PRISM Technical Report: Experiment 2 (YOLO11n + Class-Aware Balanced Dataset)

**Project:** PRISM (Plant Disease Detection and Severity Estimation)  
**Module:** `experiments/exp2_balanced/`  
**Author Attribution:** Pranava19 (`pranava194@gmail.com`)  
**Date:** August 2026  

---

## 1. Overview

### 1.1 Architectural Flow Diagram

```text
+--------------------------------------------------------------------------------+
|                     PlantVillage + PlantDoc Merged Dataset                     |
|                               (20 Classes)                                     |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|                       Class-Aware Dataset Balancing                            |
|             (Oversampling minority classes until Max:Min <= 3.0:1)             |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|                           YOLO11n (Stock, Pretrained)                          |
|             (Backbone: Conv, C3k2, SPPF, C2PSA | Neck: Standard PANet)         |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|                       P3 / P4 / P5 Multi-Scale Detection                       |
|           (P3: 80x80 / stride 8, P4: 40x40 / stride 16, P5: 20x20 / stride 32) |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|                    20 Plant-Disease Classes Object Detection                   |
|              (mAP@50: 93.08%, mAP@50-95: 89.06%, Precision: 95.35%)            |
+--------------------------------------------------------------------------------+
```

### 1.2 Purpose and Experimental Scope
Experiment 2 is designed to establish an empirical benchmark isolating the sole impact of **Class-Aware Dataset Balancing** on the stock YOLO11n object detection architecture. Within the overall 5-experiment PRISM research roadmap:
- **Experiment 1:** Baseline Stock YOLO11n on raw, unbalanced dataset.
- **Experiment 2 (This Study):** Stock YOLO11n on class-aware balanced dataset.
- **Experiments 3–5:** Progressive structural enhancements (Coordinate Attention, CSLSF, BiFPN, ALDL, and P2 scale features).

By keeping the model architecture strictly identical to default stock YOLO11n (standard depth/width multipliers, unmodified P3/P4/P5 detection heads, and zero custom modules), Experiment 2 quantifies the performance gains attributable purely to mitigation of long-tail class imbalance.

---

## 2. Dataset and Balancing Methodology

### 2.1 Dataset Composition
The benchmark comprises a harmonized taxonomy merged from the laboratory-controlled **PlantVillage** dataset and the complex in-the-wild **PlantDoc** dataset, covering 20 classes across 6 agricultural crops (Apple, Corn, Grape, Bell Pepper, Potato, Tomato) spanning 16 pathogen conditions and 4 healthy baseline controls.

- **Original Training Images:** 22,614 images
- **Balanced Training Images:** 31,129 images (8,515 augmented images added)
- **Validation Set (Untouched):** 2,826 images
- **Test Set (Untouched):** 2,828 images

### 2.2 Balancing Strategy
The original training split exhibited a severe long-tail imbalance with a maximum-to-minimum instance ratio of 13.49:1 (Tomato Yellow Leaf Curl Virus at 4,896 instances vs. Apple Cedar Rust at 363 instances). 

To eliminate minority class gradient starvation without discarding majority class visual information:
1. **Target Instance Minimum:** The lower bound threshold was set to:
   $$T_{\min} = \left\lceil \frac{\text{Max\_Count}}{3.0} \right\rceil = \left\lceil \frac{4896}{3.0} \right\rceil = 1632$$
2. **Zero Undersampling:** Majority classes exceeding $T_{\min}$ were preserved in full.
3. **Photometric and Spatial Augmentations:** Underrepresented classes were oversampled through label-consistent transformations:
   - Horizontal Flip ($x_c' = 1 - x_c$)
   - Vertical Flip ($y_c' = 1 - y_c$)
   - Brightness perturbation ($\pm 15\%$)
   - Local contrast adjustment ($\pm 15\%$)
   - Color jitter ($\pm 10\%$ with random additive offset)
4. **Partition Integrity:** Validation and test splits were directly cloned without applying augmentations.

### 2.3 Per-Class Distribution Table (`balance_report.csv`)

| Class ID | Class Name | Category | Count Before | Count After | Absolute Added | % Increase |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| 0 | Apple Scab | Fungal | 672 | 1,632 | +960 | +142.86% |
| 1 | Apple Cedar Rust | Fungal | 363 | 1,632 | +1,269 | +349.59% |
| 2 | Apple Healthy | Healthy | 1,530 | 1,632 | +102 | +6.67% |
| 3 | Corn Gray Leaf Spot | Fungal | 465 | 1,632 | +1,167 | +250.97% |
| 4 | Corn Common Rust | Fungal | 1,054 | 1,632 | +578 | +54.84% |
| 5 | Corn Northern Leaf Blight | Fungal | 1,037 | 1,632 | +595 | +57.38% |
| 6 | Grape Black Rot | Fungal | 1,097 | 1,632 | +535 | +48.77% |
| 7 | Grape Healthy | Healthy | 535 | 1,632 | +1,097 | +205.05% |
| 8 | Pepper Bacterial Spot | Bacterial | 1,004 | 1,632 | +628 | +62.55% |
| 9 | Pepper Healthy | Healthy | 1,423 | 1,635 | +212 | +14.90% |
| 10 | Potato Early Blight | Fungal | 1,057 | 1,632 | +575 | +54.40% |
| 11 | Potato Late Blight | Oomycete | 1,000 | 1,634 | +634 | +63.40% |
| 12 | Tomato Bacterial Spot | Bacterial | 1,944 | 1,961 | +17 | +0.87% |
| 13 | Tomato Early Blight | Fungal | 957 | 1,632 | +675 | +70.53% |
| 14 | Tomato Late Blight | Oomycete | 1,736 | 1,736 | 0 | 0.00% |
| 15 | Tomato Leaf Mold | Fungal | 995 | 1,632 | +637 | +64.02% |
| 16 | Tomato Septoria Leaf Spot | Fungal | 1,764 | 1,764 | 0 | 0.00% |
| 17 | Tomato Yellow Leaf Curl Virus | Viral | 4,896 | 4,896 | 0 | 0.00% |
| 18 | Tomato Mosaic Virus | Viral | 489 | 1,633 | +1,144 | +233.95% |
| 19 | Tomato Healthy | Healthy | 1,578 | 1,636 | +58 | +3.68% |

---

## 3. Training Configuration

### 3.1 Hyperparameter Specifications

| Parameter | Value | Justification / Source |
|---|:---:|---|
| Base Model | `yolo11n.pt` | Official Ultralytics COCO pretrained checkpoint for transfer learning |
| Target Classes (`nc`) | 20 | PRISM harmonized taxonomy |
| Input Resolution (`imgsz`) | $640 \times 640$ | Standard spatial scale for YOLO11 |
| Total Epochs | 25 | Full fine-tuning budget allowing learning rate decay completion |
| Batch Size | 16 | Optimized for GPU VRAM stability (2.5 GB peak usage) |
| Optimizer | `AdamW` | Decoupled weight decay ($1 \times 10^{-4}$) for stable convergence |
| Base Learning Rate (`lr0`) | 0.001 | Standard AdamW initial rate |
| LR Scheduler | Cosine Annealing | Decay factor to $0.01 \times \text{lr0}$ over 25 epochs |
| Mixed Precision | AMP (`float16`/`float32`) | Accelerated tensor core execution on CUDA |
| Random Seed | 42 | Deterministic execution parameter |
| Hardware Platform | NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM) | CUDA 12.8 acceleration |
| Total Wall-Clock Time | 130.43 minutes (7,825.8 seconds) | ~313 seconds per epoch |

### 3.2 Architecture Verification
The network configuration corresponds to the unmodified stock `yolo11n.yaml`:

```yaml
scale: 'n'
scales:
  n: [0.50, 0.25, 1024]  # depth=0.50, width=0.25, max_channels=1024

backbone:
  - [-1, 1, Conv, [64, 3, 2]]          # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]         # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2
  - [-1, 1, Conv, [256, 3, 2]]         # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4
  - [-1, 1, Conv, [512, 3, 2]]         # 5-P4/16
  - [-1, 2, C3k2, [512, True]]         # 6
  - [-1, 1, Conv, [1024, 3, 2]]        # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]        # 8
  - [-1, 1, SPPF, [1024, 5]]           # 9
  - [-1, 2, C2PSA, [1024]]             # 10

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]        # 13
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]        # 16 (P3/8 feature map)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]        # 19 (P4/16 feature map)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]        # 22 (P5/32 feature map)
  - [[16, 19, 22], 1, Detect, [nc]]    # Detect heads at P3, P4, P5
```

### 3.3 Training Progression Summary

| Epoch | Train Box Loss | Train Cls Loss | Val Box Loss | Val Cls Loss | Precision | Recall | mAP@50 | mAP@50-95 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.6240 | 1.7088 | 0.6294 | 1.5135 | 65.32% | 61.53% | 66.38% | 57.03% |
| **5** | 0.5043 | 0.7865 | 0.3809 | 0.4529 | 92.56% | 81.31% | 89.16% | 82.86% |
| **10** | 0.4557 | 0.6078 | 0.3162 | 0.3319 | 93.91% | 84.45% | 92.62% | 87.25% |
| **15** | 0.4208 | 0.5013 | 0.3028 | 0.3048 | 94.19% | 86.08% | 93.53% | 88.77% |
| **20** | 0.2464 | 0.2709 | 0.2596 | 0.2557 | 95.81% | 85.86% | 93.83% | 89.21% |
| **25 (Best)** | **0.2292** | **0.2295** | **0.2537** | **0.2476** | **94.80%** | **86.07%** | **93.75%** | **89.55%** |

---

## 4. Benchmark Results & Analysis

### 4.1 Global Test Split Metrics
Evaluated on the unseen test split containing 2,828 images and 3,218 ground-truth annotations:

| Metric | Benchmark Score |
|---|:---:|
| **Precision ($P$)** | **95.35%** |
| **Recall ($R$)** | **84.90%** |
| **F1-Score** | **89.82%** |
| **mAP @ 50 (IoU 0.50)** | **93.08%** |
| **mAP @ 50–95 (COCO metric)** | **89.06%** |
| **GPU Inference Latency** | **2.5 ms** per image |
| **Parameter Count** | **2.59 Million** |
| **Computational Complexity** | **6.5 GFLOPs** |

### 4.2 20-Class Per-Class Average Precision Breakdown

| Class ID | Target Class | Pathogen / Type | AP@50 | AP@50-95 |
|:---:|:---|:---|:---:|:---:|
| 0 | Apple Scab | Fungal | 94.25% | 89.64% |
| 1 | Apple Cedar Rust | Fungal | 83.15% | 70.80% |
| 2 | Apple Healthy | Healthy | 99.50% | 98.53% |
| 3 | Corn Gray Leaf Spot | Fungal | 96.72% | 95.11% |
| 4 | Corn Common Rust | Fungal | 99.48% | 98.37% |
| 5 | Corn Northern Leaf Blight | Fungal | 89.05% | 79.89% |
| 6 | Grape Black Rot | Fungal | 99.49% | 99.49% |
| 7 | Grape Healthy | Healthy | 91.04% | 84.30% |
| 8 | Pepper Bacterial Spot | Bacterial | 87.77% | 82.58% |
| 9 | Pepper Healthy | Healthy | 97.92% | 95.18% |
| 10 | Potato Early Blight | Fungal | 88.42% | 85.43% |
| 11 | Potato Late Blight | Oomycete | 97.60% | 95.46% |
| 12 | Tomato Bacterial Spot | Bacterial | 98.53% | 97.66% |
| 13 | Tomato Early Blight | Fungal | 91.78% | 84.72% |
| 14 | Tomato Late Blight | Oomycete | 99.07% | 96.24% |
| 15 | Tomato Leaf Mold | Fungal | 97.90% | 97.47% |
| 16 | Tomato Septoria Leaf Spot | Fungal | 96.12% | 91.44% |
| 17 | Tomato Yellow Leaf Curl Virus | Viral | 94.51% | 90.03% |
| 18 | Tomato Mosaic Virus | Viral | 66.42% | 61.64% |
| 19 | Tomato Healthy | Healthy | 92.84% | 87.16% |

### 4.3 Outlier Analysis: Tomato Mosaic Virus (`AP@50 = 66.42%`)
While 19 of the 20 classes achieved high precision (exceeding 83% to 99.5% AP@50), *Tomato Mosaic Virus* demonstrated an AP@50 of 66.42% (Precision: 82.1%, Recall: 59.5%). Two primary factors account for this discrepancy:

1. **Visual Symptom Morphology Overlap:** Unlike fungal leaf spots characterized by localized necrotic lesions with distinct circular borders (e.g., Grape Black Rot at 99.49% AP), Tomato Mosaic Virus manifests as diffuse, systemic chlorotic mottling across the leaf lamina. On natural backgrounds in the PlantDoc subset, these diffuse patterns visually cross-confound with *Tomato Yellow Leaf Curl Virus* (4,896 training instances) and early-stage *Tomato Bacterial Spot*.
2. **Pathology Sample Diversity:** Although oversampling expanded the instance count from 489 to 1,633 in training, affine augmentations duplicated existing field specimens. The underlying morphological variation in natural field lighting remained constrained compared to higher-volume classes, impacting recall on the 45 test images (74 ground-truth instances).

---

## 5. Artifact Manifest and Reproducibility

### 5.1 File Manifest

| File Path | Description | Generation Source |
|---|---|---|
| `experiments/exp2_balanced/EXPERIMENT2_REPORT.md` | Consolidated technical report | Documentation build |
| `experiments/exp2_balanced/README.md` | Module configuration & alignment notes | Module spec |
| `experiments/exp2_balanced/balance_report.csv` | Full before/after class instance counts | `scripts/balance_dataset.py` |
| `experiments/exp2_balanced/per_class_metrics.csv` | 20-class AP50 and AP50-95 benchmark table | `evaluate.py` |
| `experiments/exp2_balanced/metrics.json` | Structured cross-experiment comparison metadata | `evaluate.py` |
| `experiments/exp2_balanced/training_validation_curves.png` | Box/Cls loss and mAP progression plots | `evaluate.py` |
| `experiments/exp2_balanced/per_class_ap50.png` | 20-class AP@50 horizontal bar visualization | `evaluate.py` |
| `experiments/exp2_balanced/confusion_matrix_normalized.png` | Normalized test split confusion matrix | `evaluate.py` |
| `experiments/exp2_balanced/train.py` | Standalone YOLO11n training pipeline | Source code |
| `experiments/exp2_balanced/evaluate.py` | Standalone test set evaluation pipeline | Source code |
| `experiments/exp2_balanced/weights/best.pt` | PyTorch checkpoint (5.2 MB) | `train.py` |
| `experiments/exp2_balanced/weights/best.onnx` | Exported ONNX graph (9.95 MB) | `evaluate.py` |
| `data/exp2_balanced.yaml` | Dataset YAML specification | Dataset config |
| `scripts/balance_dataset.py` | Class-aware oversampling engine | Script |

### 5.2 Fresh Clone Execution Instructions

```bash
# 1. Clone repository
git clone https://github.com/Pranava19/PRISM-Plant-Disease-Detection.git
cd PRISM-Plant-Disease-Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Execute class-aware balancing (generates data/balanced/ and balance_report.csv)
python scripts/balance_dataset.py

# 4. Train stock YOLO11n on balanced data (seed=42 hardcoded)
python experiments/exp2_balanced/train.py

# 5. Evaluate best.pt on unseen test set and compile metrics
python experiments/exp2_balanced/evaluate.py
```

### 5.3 Git Provenance
- **Repository:** `https://github.com/Pranava19/PRISM-Plant-Disease-Detection`
- **Branch:** `main`
- **Commit Author & Committer:** `Pranava19 <pranava194@gmail.com>`

---

## 6. Next Steps & Comparative Context

Experiment 2 establishes that mitigating dataset long-tail imbalance alone elevates stock YOLO11n detection accuracy to **93.08% mAP@50** and **89.06% mAP@50-95** while maintaining a lightweight footprint of 2.59M parameters and 2.5 ms GPU latency.

This dataset configuration (`data/exp2_balanced.yaml`) serves as the standardized training baseline for the subsequent architectural experiments in the PRISM suite:
- **Experiment 3 (Coordinate Attention):** Evaluating spatial direction-aware attention mechanisms for subtle lesion localization.
- **Experiment 4 (CSLSF / BiFPN):** Evaluating cross-scale multi-level feature aggregation.
- **Experiment 5 (ALDL / P2 Head):** Evaluating high-resolution tiny-lesion feature detection heads.
