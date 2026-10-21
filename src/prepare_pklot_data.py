"""
ParkVision AI - PKLot Dataset Extraction & Preprocessing Pipeline
==================================================================
Extracts images and converts COCO JSON annotations from pklot-dataset.zip
into Ultralytics YOLOv8 format:
  data/pklot/
    images/train/, images/val/
    labels/train/, labels/val/
    pklot.yaml
Classes:
  0: space-empty
  1: space-occupied
"""

import os
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Any

BASE_DIR = Path(__file__).resolve().parent.parent
ZIP_PATH = BASE_DIR / "data" / "pklot" / "pklot-dataset.zip"
OUTPUT_DIR = BASE_DIR / "data" / "pklot"


def prepare_pklot_yolo_dataset(max_train_samples: int = 800, max_val_samples: int = 200):
    """Extracts zip contents and converts COCO annotations to YOLO format."""
    print("=" * 60)
    print("PKLot Dataset Extraction & YOLO Conversion")
    print("=" * 60)

    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Archive not found: {ZIP_PATH}")

    # Create destination directories
    img_train_dir = OUTPUT_DIR / "images" / "train"
    img_val_dir = OUTPUT_DIR / "images" / "val"
    lbl_train_dir = OUTPUT_DIR / "labels" / "train"
    lbl_val_dir = OUTPUT_DIR / "labels" / "val"

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        # Process train split
        _process_split(zf, "train", img_train_dir, lbl_train_dir, max_samples=max_train_samples)
        # Process val split
        _process_split(zf, "valid", img_val_dir, lbl_val_dir, max_samples=max_val_samples)

    # Generate pklot.yaml
    yaml_path = OUTPUT_DIR / "pklot.yaml"
    # Use forward slashes for cross-platform compatibility
    yaml_content = f"""# PKLot Dataset Configuration for YOLOv8 Training
path: {OUTPUT_DIR.resolve().as_posix()}
train: images/train
val: images/val

# Classes
names:
  0: space-empty
  1: space-occupied
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[SUCCESS] Dataset prepared at: {OUTPUT_DIR}")
    print(f"[SUCCESS] YAML configuration created: {yaml_path}")
    return yaml_path


def _process_split(zf: zipfile.ZipFile, split_prefix: str, img_out: Path, lbl_out: Path, max_samples: int):
    coco_json_name = f"{split_prefix}/_annotations.coco.json"
    print(f"Reading {coco_json_name} from archive...")
    coco_data = json.loads(zf.read(coco_json_name).decode('utf-8'))

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])

    print(f"Total available images in {split_prefix}: {len(images)}")
    selected_images = images[:max_samples]

    # Map image_id -> list of annotations
    img_id_to_anns: Dict[int, List[Dict[str, Any]]] = {}
    for ann in annotations:
        iid = ann["image_id"]
        if iid not in img_id_to_anns:
            img_id_to_anns[iid] = []
        img_id_to_anns[iid].append(ann)

    print(f"Extracting {len(selected_images)} images and writing YOLO label files to {img_out.parent}...")
    converted_count = 0
    total_labels_written = 0

    for img_meta in selected_images:
        iid = img_meta["id"]
        file_name = img_meta["file_name"]
        w = float(img_meta["width"])
        h = float(img_meta["height"])

        # Zip entry path
        zip_img_path = f"{split_prefix}/{file_name}"
        target_img_path = img_out / file_name

        # Extract image if not already extracted
        if not target_img_path.exists():
            try:
                img_bytes = zf.read(zip_img_path)
                with open(target_img_path, "wb") as f:
                    f.write(img_bytes)
            except KeyError:
                continue

        # Convert annotations for this image
        anns = img_id_to_anns.get(iid, [])
        yolo_lines = []
        for ann in anns:
            cat_id = ann["category_id"]
            # Category 1: space-empty -> 0, Category 2: space-occupied -> 1
            if cat_id not in (1, 2):
                continue
            yolo_cls = cat_id - 1

            bbox = ann["bbox"]  # [x, y, width, height] in pixels
            bx, by, bw, bh = bbox
            if bw <= 0 or bh <= 0 or w <= 0 or h <= 0:
                continue

            # Convert to normalized center format: [x_center, y_center, w, h]
            xc = (bx + bw / 2.0) / w
            yc = (by + bh / 2.0) / h
            nw = bw / w
            nh = bh / h

            # Clamp coordinates to [0, 1]
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            yolo_lines.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        # Write label file (.txt)
        txt_name = Path(file_name).stem + ".txt"
        target_lbl_path = lbl_out / txt_name
        with open(target_lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")

        total_labels_written += len(yolo_lines)
        converted_count += 1

    print(f"[{split_prefix.upper()}] Extracted {converted_count} images with {total_labels_written} space bounding boxes.")


if __name__ == "__main__":
    prepare_pklot_yolo_dataset()
