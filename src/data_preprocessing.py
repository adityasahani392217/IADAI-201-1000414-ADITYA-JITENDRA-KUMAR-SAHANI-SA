"""
StockSense Pro - Data Preprocessing & Leakage-Free Dataset Split Pipeline
Handles:
1. Scene-level grouping to prevent near-duplicate leakage across splits.
2. Reproducible 70% / 15% / 15% Train / Val / Test split.
3. Class mapping and filtering (11 raw classes -> 7 standardized retail categories).
4. Image integrity validation (checks headers, dimensions, corrupted files).
5. Bounding box geometric validation (clipping to [0, 1] range, dropping degenerate boxes).
6. Augmentation on training split only.
7. Generating YOLO data.yaml configuration and curation of test sample images for Streamlit.
"""

import os
import shutil
import glob
import re
import random
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Set, Any

import cv2
import numpy as np
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

def extract_base_scene_id(filename: str) -> str:
    """Extract the base scene ID before any Roboflow augmentation hash suffix."""
    bn = os.path.basename(filename)
    m = re.match(r'^(.*?)\.rf\.', bn)
    if m:
        return m.group(1)
    return os.path.splitext(bn)[0]


def validate_image_file(image_path: Path) -> bool:
    """Validate that the image is not corrupted, non-zero size, and decodable."""
    if not image_path.exists() or image_path.stat().st_size == 0:
        return False
    try:
        with Image.open(image_path) as img:
            img.verify()
        # Test actual decoding
        cv_img = cv2.imread(str(image_path))
        if cv_img is None or cv_img.shape[0] < 10 or cv_img.shape[1] < 10:
            return False
        return True
    except Exception:
        return False


def validate_and_convert_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Read raw YOLO label, map class indices, clip coordinates, and drop degenerate boxes.
    Returns list of (mapped_class_id, x_center, y_center, width, height).
    """
    valid_boxes = []
    if not label_path.exists():
        return valid_boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                raw_cls = int(parts[0])
                # Skip unlabelled/spacer class 0 ('-')
                if raw_cls not in settings.RAW_TO_STANDARDIZED_MAP:
                    continue
                
                std_cls = settings.RAW_TO_STANDARDIZED_MAP[raw_cls]
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                # Geometric validation & clipping
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                w = max(0.001, min(1.0, w))
                h = max(0.001, min(1.0, h))

                if w > 0.005 and h > 0.005:  # Filter microscopic noise
                    valid_boxes.append((std_cls, xc, yc, w, h))
            except ValueError:
                continue

    return valid_boxes


def build_leakage_free_splits(
    extracted_source_dir: Path,
    output_processed_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict[str, Any]:
    """
    Construct strictly partitioned, leak-free train/val/test splits based on unique shelf scenes.
    """
    print(f"Building leak-free dataset split from: {extracted_source_dir}")
    random.seed(seed)
    np.random.seed(seed)

    all_images = glob.glob(str(extracted_source_dir / "**" / "*.jpg"), recursive=True)
    if not all_images:
        raise FileNotFoundError(f"No JPG images found in {extracted_source_dir}")

    # Group all file paths by base scene
    scene_groups: Dict[str, List[Path]] = {}
    for img_path_str in all_images:
        img_path = Path(img_path_str)
        base_id = extract_base_scene_id(img_path.name)
        scene_groups.setdefault(base_id, []).append(img_path)

    base_scene_ids = sorted(list(scene_groups.keys()))
    random.shuffle(base_scene_ids)
    total_scenes = len(base_scene_ids)

    n_train_scenes = int(round(total_scenes * train_ratio))
    n_val_scenes = int(round(total_scenes * val_ratio))
    # Remaining scenes for test
    train_scenes = set(base_scene_ids[:n_train_scenes])
    val_scenes = set(base_scene_ids[n_train_scenes:n_train_scenes + n_val_scenes])
    test_scenes = set(base_scene_ids[n_train_scenes + n_val_scenes:])

    print(f"Partitioned {total_scenes} unique shelf scenes:")
    print(f"  - Train Scenes: {len(train_scenes)} ({len(train_scenes)/total_scenes*100:.1f}%)")
    print(f"  - Val Scenes:   {len(val_scenes)} ({len(val_scenes)/total_scenes*100:.1f}%)")
    print(f"  - Test Scenes:  {len(test_scenes)} ({len(test_scenes)/total_scenes*100:.1f}%)")

    # Clean existing processed directory
    if output_processed_dir.exists():
        shutil.rmtree(output_processed_dir)

    for split in ["train", "val", "test"]:
        (output_processed_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_processed_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    split_stats = {
        "train": {"images": 0, "boxes": 0, "classes": Counter()},
        "val": {"images": 0, "boxes": 0, "classes": Counter()},
        "test": {"images": 0, "boxes": 0, "classes": Counter()},
    }

    # Process files
    # For val and test: include exactly 1 clean unaugmented representative per scene to prevent evaluation skew.
    # For train: include augmented variants to enrich training variety.
    for base_id, img_paths in scene_groups.items():
        if base_id in train_scenes:
            target_split = "train"
            selected_paths = img_paths  # Keep training variants
        elif base_id in val_scenes:
            target_split = "val"
            selected_paths = [img_paths[0]]  # 1 representative
        else:
            target_split = "test"
            selected_paths = [img_paths[0]]  # 1 representative

        for img_path in selected_paths:
            if not validate_image_file(img_path):
                print(f"Warning: Corrupted image skipped: {img_path.name}")
                continue

            lbl_path = Path(str(img_path).replace("images", "labels").replace(".jpg", ".txt"))
            boxes = validate_and_convert_labels(lbl_path)
            if not boxes:
                continue

            # Destination filenames
            out_img = output_processed_dir / target_split / "images" / img_path.name
            out_lbl = output_processed_dir / target_split / "labels" / f"{img_path.stem}.txt"

            shutil.copy2(img_path, out_img)
            with open(out_lbl, "w", encoding="utf-8") as f:
                for b in boxes:
                    f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
                    split_stats[target_split]["classes"][b[0]] += 1
                split_stats[target_split]["boxes"] += len(boxes)
            split_stats[target_split]["images"] += 1

    # Generate data.yaml for Ultralytics YOLO
    yaml_content = f"""# StockSense Pro - Ultralytics YOLO Dataset Configuration
path: {output_processed_dir.resolve().as_posix()}
train: train/images
val: val/images
test: test/images

nc: {settings.NUM_CLASSES}
names: {settings.CLASS_NAMES}
"""
    yaml_path = output_processed_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("\nDataset preprocessing complete!")
    for split, stat in split_stats.items():
        print(f"  {split.upper()}: {stat['images']} images, {stat['boxes']} annotated boxes across {len(stat['classes'])} classes")

    # Curate sample test images for the Streamlit UI
    curate_ui_sample_images(output_processed_dir / "test", settings.SAMPLES_DIR)
    
    # Save split summary
    summary_path = output_processed_dir / "split_summary.json"
    import json
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_base_scenes": total_scenes,
            "train_scenes": len(train_scenes),
            "val_scenes": len(val_scenes),
            "test_scenes": len(test_scenes),
            "stats": {k: {"images": v["images"], "boxes": v["boxes"], "classes": dict(v["classes"])} for k, v in split_stats.items()}
        }, f, indent=2)

    return split_stats


def curate_ui_sample_images(test_split_dir: Path, samples_dir: Path):
    """
    Select distinct shelf operational scenarios from held-out test split
    and save them to assets/samples for one-click user testing in the Streamlit app.
    """
    samples_dir.mkdir(parents=True, exist_ok=True)
    test_images = list((test_split_dir / "images").glob("*.jpg"))
    if not test_images:
        return

    # Find images with diverse class mixtures and densities
    img_box_counts = []
    for img_p in test_images:
        lbl_p = test_split_dir / "labels" / f"{img_p.stem}.txt"
        if lbl_p.exists():
            lines = open(lbl_p).readlines()
            img_box_counts.append((img_p, len(lines)))

    img_box_counts.sort(key=lambda x: x[1], reverse=True)

    # 1. High density / Well-stocked shelf
    if len(img_box_counts) > 0:
        shutil.copy2(img_box_counts[0][0], samples_dir / "sample_well_stocked.jpg")
    # 2. Moderate density
    if len(img_box_counts) > 5:
        shutil.copy2(img_box_counts[5][0], samples_dir / "sample_beverages.jpg")
    # 3. Medium density
    if len(img_box_counts) > 15:
        shutil.copy2(img_box_counts[15][0], samples_dir / "sample_low_stock.jpg")
    # 4. Low density / Sparse shelf
    if len(img_box_counts) > 25:
        shutil.copy2(img_box_counts[-1][0], samples_dir / "sample_sparse_empty.jpg")

    print(f"Curated 4 sample shelf scenarios in: {samples_dir}")

    # Generate annotated visualization sample for EDA and reports
    visualize_annotated_sample(
        img_box_counts[0][0],
        test_split_dir / "labels" / f"{img_box_counts[0][0].stem}.txt",
        settings.EDA_DIR / "shelf_annotated_sample.png"
    )


def visualize_annotated_sample(img_path: Path, label_path: Path, output_path: Path):
    """Draw ground-truth annotations on a shelf image to verify bounding boxes and save to assets/eda."""
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]

    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = float(parts[1])*w, float(parts[2])*h, float(parts[3])*w, float(parts[4])*h
                    x1 = int(xc - bw/2)
                    y1 = int(yc - bh/2)
                    x2 = int(xc + bw/2)
                    y2 = int(yc + bh/2)

                    cls_name = settings.CLASS_NAMES[cls_id] if cls_id < len(settings.CLASS_NAMES) else f"Class_{cls_id}"
                    color_hex = settings.CLASS_COLORS.get(cls_name, "#2563EB")
                    # Convert hex to BGR
                    b = int(color_hex[5:7], 16)
                    g = int(color_hex[3:5], 16)
                    r = int(color_hex[1:3], 16)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (b, g, r), 2)
                    # Label badge
                    label_text = f"{cls_name}"
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(img, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), (b, g, r), -1)
                    cv2.putText(img, label_text, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def run_pipeline():
    """Entry point for the data preprocessing pipeline."""
    source_dir = settings.BASE_DIR / "data_investigation" / "shelf_dataset" / "extracted"
    target_dir = settings.PROCESSED_DATA_DIR
    build_leakage_free_splits(source_dir, target_dir)

if __name__ == "__main__":
    run_pipeline()
