"""
experiments/exp2_balanced/train.py

Training Pipeline for Experiment 2: YOLO11n on Class-Aware Balanced Dataset.
- Model: Stock YOLO11n (pretrained on COCO, yolo11n.pt).
- Architecture: Unmodified P3/P4/P5 detection heads (C3k2, SPPF, C2PSA).
- Dataset: data/exp2_balanced.yaml (20 classes).
- Output: experiments/exp2_balanced/runs/
"""

import os
import sys
import time
from pathlib import Path
import torch
import numpy as np

# Patch numpy 2.x trapezoid compatibility if needed
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

from ultralytics import YOLO


def train_exp2(
    data_yaml="data/exp2_balanced.yaml",
    epochs=25,
    batch_size=16,
    imgsz=640,
    device=0 if torch.cuda.is_available() else "cpu",
    project_dir="experiments/exp2_balanced/runs",
    run_name="train_exp2"
):
    print("=" * 80)
    print("EXPERIMENT 2: YOLO11n + BALANCED DATASET TRAINING")
    print("=" * 80)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name    : {torch.cuda.get_device_name(0)}")

    data_path = Path(data_yaml).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found at: {data_path}")

    # Load stock YOLO11n pretrained weights
    print("\nLoading stock pretrained YOLO11n model (yolo11n.pt)...")
    model = YOLO("yolo11n.pt")

    train_args = {
        "data": str(data_path).replace("\\", "/"),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "cos_lr": True,
        "amp": torch.cuda.is_available(),
        "device": device,
        "workers": 4,
        "seed": 42,
        "project": str(Path(project_dir).resolve()).replace("\\", "/"),
        "name": run_name,
        "exist_ok": True,
        "verbose": True
    }

    print("\nTraining Configuration:")
    for k, v in train_args.items():
        print(f"  {k:15s}: {v}")

    # Check for resumption
    last_ckpt = Path(project_dir) / run_name / "weights" / "last.pt"
    start_time = time.time()
    if last_ckpt.exists():
        print(f"\nResuming training from checkpoint: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        results = model.train(resume=True)
    else:
        print("\nStarting training from yolo11n.pt...")
        results = model.train(**train_args)

    duration = time.time() - start_time
    print(f"\nTraining completed in {duration / 60:.2f} minutes.")
    best_pt = Path(project_dir) / run_name / "weights" / "best.pt"
    print(f"Best model weights saved at: {best_pt}")

    return results


if __name__ == "__main__":
    train_exp2()
