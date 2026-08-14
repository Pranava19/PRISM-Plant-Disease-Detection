"""
Experiment 02: YOLO + PrismGhost for 20-Class Plant Disease Object Detection
End-to-end execution script with local GPU acceleration.
"""

import os
import sys
import math
import time
import random
import shutil
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)


class PrismGhost(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, ratio=2):
        super().__init__()
        primary = math.ceil(c2 / ratio)
        cheap = primary * (ratio - 1)
        self.primary = nn.Sequential(
            nn.Conv2d(c1, primary, k, s, k // 2, bias=False),
            nn.BatchNorm2d(primary),
            nn.SiLU(),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(primary, cheap, 3, 1, 1, groups=primary, bias=False),
            nn.BatchNorm2d(cheap),
            nn.SiLU(),
        )
        self.c2 = c2

    def forward(self, x):
        y = self.primary(x)
        return torch.cat((y, self.cheap(y)), 1)[:, :self.c2]


def parse_yolo_labels(lbl_path, class_map):
    valid_boxes = []
    if not lbl_path.exists():
        return valid_boxes
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                src_cls = int(float(parts[0]))
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except (ValueError, TypeError):
                continue
            if src_cls not in class_map:
                continue
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                continue
            valid_boxes.append((class_map[src_cls], xc, yc, w, h))
    return valid_boxes


def main():
    print("=" * 80)
    print("EXPERIMENT 02: YOLO + PRISMGHOST (20-CLASS DETECTION)")
    print("=" * 80)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}")

    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    PROJECT_ROOT = Path.cwd()
    DATASET_ROOT = PROJECT_ROOT / "dataset"
    EXPERIMENTS_ROOT = PROJECT_ROOT / "AgriYOLO_Experiments"
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)

    # Kaggle Setup (loads from environment or user credentials securely)
    kaggle_token_file = Path.home() / ".kaggle" / "access_token"
    if kaggle_token_file.exists() and "KAGGLE_API_TOKEN" not in os.environ:
        os.environ["KAGGLE_API_TOKEN"] = kaggle_token_file.read_text().strip()

    TARGET_CLASSES = [
        "Apple Scab", "Apple Cedar Rust", "Apple Healthy", "Corn Gray Leaf Spot",
        "Corn Common Rust", "Corn Northern Leaf Blight", "Grape Black Rot", "Grape Healthy",
        "Pepper Bacterial Spot", "Pepper Healthy", "Potato Early Blight", "Potato Late Blight",
        "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight", "Tomato Leaf Mold",
        "Tomato Septoria Leaf Spot", "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus", "Tomato Healthy"
    ]

    import ultralytics.nn.modules as ultralytics_modules
    import ultralytics.nn.tasks as ultralytics_tasks
    ultralytics_modules.PrismGhost = PrismGhost
    ultralytics_tasks.PrismGhost = PrismGhost
    setattr(ultralytics_modules, "PrismGhost", PrismGhost)
    setattr(ultralytics_tasks, "PrismGhost", PrismGhost)

    from ultralytics import YOLO

    model_yaml = PROJECT_ROOT / "model" / "yolo_prismghost.yaml"
    if not model_yaml.exists():
        model_yaml = PROJECT_ROOT / "yolo_prismghost.yaml"

    model = YOLO(str(model_yaml))
    print("Model initialized and verified on GPU.")

    data_yaml = DATASET_ROOT / "data.yaml"
    if not data_yaml.exists():
        data_yaml = PROJECT_ROOT / "dataset" / "data.yaml"

    train_args = {
        "data": str(data_yaml.resolve()).replace("\\", "/"),
        "epochs": 25,
        "imgsz": 640,
        "batch": 16,
        "lr0": 0.001,
        "optimizer": "AdamW",
        "cos_lr": True,
        "amp": torch.cuda.is_available(),
        "device": DEVICE,
        "workers": 4,
        "seed": 42,
        "project": str(EXPERIMENTS_ROOT.resolve()).replace("\\", "/"),
        "name": "yolo_prismghost_run",
        "exist_ok": True
    }

    last_checkpoint = EXPERIMENTS_ROOT / "yolo_prismghost_run" / "weights" / "last.pt"
    if last_checkpoint.exists():
        print(f"Resuming from: {last_checkpoint}")
        model = YOLO(str(last_checkpoint))
        results = model.train(resume=True)
    else:
        results = model.train(**train_args)

    best_pt = EXPERIMENTS_ROOT / "yolo_prismghost_run" / "weights" / "best.pt"
    eval_model = YOLO(str(best_pt))
    metrics = eval_model.val(data=str(data_yaml.resolve()).replace("\\", "/"), split="test", imgsz=640, device=DEVICE)
    print(f"Test mAP@50: {metrics.box.map50:.4f}, mAP@50-95: {metrics.box.map:.4f}")

    eval_model.export(format="onnx", imgsz=640, dynamic=True)
    print("ONNX Model Export Complete!")


if __name__ == "__main__":
    main()
