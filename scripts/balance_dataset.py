"""
scripts/balance_dataset.py

Class-Aware Balancing for Merged PlantVillage + PlantDoc YOLO Dataset.
- Computes per-class instance counts from labels.
- Oversamples underrepresented classes (duplicate + light augmentation: flip, brightness, contrast, slight rotation)
  until max:min class ratio <= 3:1.
- NEVER undersamples majority classes.
- Outputs balanced train split to data/balanced/ (val split is untouched).
- Generates before/after balance report to experiments/exp2_balanced/balance_report.csv.
"""

import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

# Set fixed seed for determinism
random.seed(42)
np.random.seed(42)

TARGET_CLASSES = [
    "Apple Scab", "Apple Cedar Rust", "Apple Healthy", "Corn Gray Leaf Spot",
    "Corn Common Rust", "Corn Northern Leaf Blight", "Grape Black Rot", "Grape Healthy",
    "Pepper Bacterial Spot", "Pepper Healthy", "Potato Early Blight", "Potato Late Blight",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight", "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot", "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus", "Tomato Healthy"
]


def parse_label_file(label_path):
    """Returns list of (class_id, xc, yc, w, h)."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    cid = int(float(parts[0]))
                    xc, yc, w, h = map(float, parts[1:])
                    boxes.append((cid, xc, yc, w, h))
                except (ValueError, TypeError):
                    continue
    return boxes


def write_label_file(label_path, boxes):
    """Writes list of (class_id, xc, yc, w, h) to file."""
    with open(label_path, "w", encoding="utf-8") as f:
        for cid, xc, yc, w, h in boxes:
            f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def apply_light_aug(img_np, boxes, aug_type):
    """
    Applies light augmentation to image and adjusts bounding boxes accordingly.
    aug_type in ['hflip', 'vflip', 'brightness', 'contrast', 'rotate_90']
    """
    h, w = img_np.shape[:2]
    new_boxes = []

    if aug_type == "hflip":
        aug_img = np.fliplr(img_np).copy()
        for cid, xc, yc, bw, bh in boxes:
            new_boxes.append((cid, 1.0 - xc, yc, bw, bh))
    elif aug_type == "vflip":
        aug_img = np.flipud(img_np).copy()
        for cid, xc, yc, bw, bh in boxes:
            new_boxes.append((cid, xc, 1.0 - yc, bw, bh))
    elif aug_type == "brightness":
        pil_img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_aug = enhancer.enhance(factor)
        aug_img = cv2.cvtColor(np.array(pil_aug), cv2.COLOR_RGB2BGR)
        new_boxes = list(boxes)
    elif aug_type == "contrast":
        pil_img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_aug = enhancer.enhance(factor)
        aug_img = cv2.cvtColor(np.array(pil_aug), cv2.COLOR_RGB2BGR)
        new_boxes = list(boxes)
    else:  # combined subtle jitter
        pil_img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        b_enh = ImageEnhance.Brightness(pil_img).enhance(random.uniform(0.85, 1.15))
        c_enh = ImageEnhance.Contrast(b_enh).enhance(random.uniform(0.85, 1.15))
        aug_img = cv2.cvtColor(np.array(c_enh), cv2.COLOR_RGB2BGR)
        new_boxes = list(boxes)

    return aug_img, new_boxes


def balance_dataset(
    source_root=Path("dataset"),
    output_root=Path("data/balanced"),
    report_path=Path("experiments/exp2_balanced/balance_report.csv"),
    max_min_ratio_target=3.0
):
    print("=" * 80)
    print("CLASS-AWARE DATASET BALANCING (YOLO FORMAT)")
    print("=" * 80)

    train_img_dir = source_root / "images" / "train"
    train_lbl_dir = source_root / "labels" / "train"
    val_img_dir = source_root / "images" / "val"
    val_lbl_dir = source_root / "labels" / "val"
    test_img_dir = source_root / "images" / "test"
    test_lbl_dir = source_root / "labels" / "test"

    out_train_img = output_root / "images" / "train"
    out_train_lbl = output_root / "labels" / "train"
    out_val_img = output_root / "images" / "val"
    out_val_lbl = output_root / "labels" / "val"
    out_test_img = output_root / "images" / "test"
    out_test_lbl = output_root / "labels" / "test"

    for p in [out_train_img, out_train_lbl, out_val_img, out_val_lbl, out_test_img, out_test_lbl]:
        p.mkdir(parents=True, exist_ok=True)

    # 1. Copy validation and test sets untouched
    print("\n1. Copying Validation and Test splits (untouched)...")
    for s_img, s_lbl, d_img, d_lbl in [
        (val_img_dir, val_lbl_dir, out_val_img, out_val_lbl),
        (test_img_dir, test_lbl_dir, out_test_img, out_test_lbl)
    ]:
        if s_img.exists():
            for img_file in s_img.glob("*.*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    dst_img = d_img / img_file.name
                    if not dst_img.exists():
                        shutil.copy2(img_file, dst_img)
                    lbl_file = s_lbl / f"{img_file.stem}.txt"
                    if lbl_file.exists():
                        dst_lbl = d_lbl / f"{img_file.stem}.txt"
                        if not dst_lbl.exists():
                            shutil.copy2(lbl_file, dst_lbl)

    # 2. Count class occurrences in train split
    print("\n2. Analyzing train split class frequencies...")
    image_files = [f for f in train_img_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    class_to_images = defaultdict(list)
    initial_counts = Counter()
    image_to_boxes = {}

    for img_file in image_files:
        lbl_file = train_lbl_dir / f"{img_file.stem}.txt"
        boxes = parse_label_file(lbl_file)
        if not boxes:
            continue
        image_to_boxes[img_file] = boxes
        classes_in_img = set()
        for cid, _, _, _, _ in boxes:
            initial_counts[cid] += 1
            classes_in_img.add(cid)
        for cid in classes_in_img:
            class_to_images[cid].append(img_file)

    max_count = max(initial_counts.values())
    min_count = min(initial_counts.values())
    target_min = int(np.ceil(max_count / max_min_ratio_target))

    print(f"Total training images with valid labels: {len(image_to_boxes)}")
    print(f"Initial Instance Counts -> Max: {max_count}, Min: {min_count}, Max:Min Ratio: {max_count/min_count:.2f}:1")
    print(f"Target Minimum Instance Count (for max:min <= {max_min_ratio_target}:1): {target_min}")

    # 3. Copy all original train images and labels first (NEVER undersample)
    print("\n3. Copying original training set...")
    for img_file, boxes in image_to_boxes.items():
        dst_img = out_train_img / img_file.name
        dst_lbl = out_train_lbl / f"{img_file.stem}.txt"
        if not dst_img.exists():
            shutil.copy2(img_file, dst_img)
        if not dst_lbl.exists():
            write_label_file(dst_lbl, boxes)

    # 4. Oversample underrepresented classes with light augmentations
    print("\n4. Oversampling underrepresented classes with light augmentations...")
    balanced_counts = Counter(initial_counts)
    aug_types = ["hflip", "vflip", "brightness", "contrast", "jitter"]
    aug_counter = 0

    # Sort classes ascending by count
    sorted_classes = sorted(range(20), key=lambda c: balanced_counts[c])

    for cid in sorted_classes:
        deficit = target_min - balanced_counts[cid]
        if deficit <= 0:
            continue

        cname = TARGET_CLASSES[cid]
        print(f"  Oversampling class {cid:2d} ({cname}): current count {balanced_counts[cid]}, deficit {deficit}...")

        available_images = class_to_images[cid]
        if not available_images:
            print(f"  Warning: No images found for class {cid}!")
            continue

        img_idx = 0
        while balanced_counts[cid] < target_min:
            src_img = available_images[img_idx % len(available_images)]
            img_idx += 1
            boxes = image_to_boxes[src_img]

            # Read image
            img_np = cv2.imread(str(src_img))
            if img_np is None:
                continue

            aug_type = aug_types[aug_counter % len(aug_types)]
            aug_counter += 1

            aug_img_np, aug_boxes = apply_light_aug(img_np, boxes, aug_type)

            aug_stem = f"{src_img.stem}_aug{aug_counter}_{aug_type}"
            dst_img_path = out_train_img / f"{aug_stem}.jpg"
            dst_lbl_path = out_train_lbl / f"{aug_stem}.txt"

            cv2.imwrite(str(dst_img_path), aug_img_np)
            write_label_file(dst_lbl_path, aug_boxes)

            # Update counts for all classes present in this augmented image
            for box_cid, _, _, _, _ in aug_boxes:
                balanced_counts[box_cid] += 1

    final_max = max(balanced_counts.values())
    final_min = min(balanced_counts.values())
    print(f"\nFinal Instance Counts -> Max: {final_max}, Min: {final_min}, Ratio: {final_max/final_min:.2f}:1")

    # 5. Generate Balance Report CSV
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = []
    for cid in range(20):
        cname = TARGET_CLASSES[cid]
        before = initial_counts[cid]
        after = balanced_counts[cid]
        pct_increase = ((after - before) / before * 100.0) if before > 0 else 0.0
        report_data.append({
            "class_id": cid,
            "class_name": cname,
            "count_before": before,
            "count_after": after,
            "percent_increase": round(pct_increase, 2)
        })

    df_report = pd.DataFrame(report_data)
    df_report.to_csv(report_path, index=False)
    print(f"\nBalance report saved to: {report_path}")
    print("\n--- Summary Table ---")
    print(df_report.to_string(index=False))

    return df_report


if __name__ == "__main__":
    balance_dataset()
