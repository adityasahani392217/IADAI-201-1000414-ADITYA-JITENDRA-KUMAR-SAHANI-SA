"""
StockSense Pro - Test Suite for Preprocessing & Leak-Free Splitting
"""

import pytest
import os
import tempfile
from pathlib import Path
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from src.data_preprocessing import (
    extract_base_scene_id,
    validate_image_file,
    validate_and_convert_labels,
    build_leakage_free_splits
)

def test_extract_base_scene_id():
    """Verify hash suffix stripping to identify parent shelf scenes."""
    fname1 = "001_2_jpeg_jpg.rf.6fb48bf297c8512d4ca88f181ba3613a.jpg"
    assert extract_base_scene_id(fname1) == "001_2_jpeg_jpg"

    fname2 = "shelf_aisle_04.jpg"
    assert extract_base_scene_id(fname2) == "shelf_aisle_04"


def test_label_validation_and_mapping(tmp_path):
    """Verify raw class mapping to 7 standardized classes and geometric clamping."""
    lbl_file = tmp_path / "test_label.txt"
    with open(lbl_file, "w") as f:
        # Class 0: should be skipped ('-')
        f.write("0 0.5 0.5 0.2 0.2\n")
        # Class 1: '- Big Bottle' -> should map to 0 ('Bottle')
        f.write("1 0.3 0.4 0.1 0.3\n")
        # Class 5: '- Can' -> should map to 3 ('Can')
        f.write("5 0.7 0.8 0.15 0.2\n")
        # Class 7: '- Snack' -> should map to 5 ('Snack')
        f.write("7 0.2 0.2 0.25 0.25\n")
        # Out of bounds coordinates: should be clamped
        f.write("3 1.2 -0.1 0.3 0.4\n")

    boxes = validate_and_convert_labels(lbl_file)
    assert len(boxes) == 4  # Class 0 was dropped, 4 valid boxes remain

    # Verify mappings
    classes = [b[0] for b in boxes]
    assert classes == [0, 3, 5, 1]  # 1->0 (Bottle), 5->3 (Can), 7->5 (Snack), 3->1 (Box)

    # Verify clamping on 4th box
    box4 = boxes[3]
    assert box4[1] == 1.0  # xc clamped from 1.2 to 1.0
    assert box4[2] == 0.0  # yc clamped from -0.1 to 0.0


def test_image_validation(tmp_path):
    """Verify image validation with valid image, corrupted file, and missing file."""
    # 1. Missing file
    assert validate_image_file(tmp_path / "nonexistent.jpg") is False

    # 2. Zero-byte corrupted file
    zero_file = tmp_path / "zero.jpg"
    zero_file.touch()
    assert validate_image_file(zero_file) is False

    # 3. Valid RGB image
    valid_file = tmp_path / "valid.jpg"
    img = Image.new("RGB", (640, 640), color=(100, 150, 200))
    img.save(valid_file)
    assert validate_image_file(valid_file) is True


def test_processed_dataset_integrity():
    """Verify that processed data.yaml and splits exist and have zero overlap."""
    yaml_path = settings.PROCESSED_DATA_DIR / "data.yaml"
    assert yaml_path.exists()

    train_imgs = {extract_base_scene_id(p.name) for p in (settings.PROCESSED_DATA_DIR / "train" / "images").glob("*.jpg")}
    val_imgs = {extract_base_scene_id(p.name) for p in (settings.PROCESSED_DATA_DIR / "val" / "images").glob("*.jpg")}
    test_imgs = {extract_base_scene_id(p.name) for p in (settings.PROCESSED_DATA_DIR / "test" / "images").glob("*.jpg")}

    # Zero leakage check
    train_val_overlap = train_imgs.intersection(val_imgs)
    train_test_overlap = train_imgs.intersection(test_imgs)
    val_test_overlap = val_imgs.intersection(test_imgs)

    assert len(train_val_overlap) == 0, f"Leakage detected between Train and Val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Leakage detected between Train and Test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Leakage detected between Val and Test: {val_test_overlap}"
