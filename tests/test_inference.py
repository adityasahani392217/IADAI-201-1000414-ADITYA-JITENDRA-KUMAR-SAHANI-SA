"""
StockSense Pro - Test Suite for Inference Engine & Product Detector
"""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from src.inference import StockSenseDetector

@pytest.fixture(scope="module")
def detector():
    """Fixture providing initialized detector instance."""
    # If trained weight exists, use it; otherwise use backbone
    weight_path = settings.DEFAULT_MODEL_PATH if settings.DEFAULT_MODEL_PATH.exists() else (settings.BASE_DIR / settings.PRETRAINED_BACKBONE)
    return StockSenseDetector(model_path=weight_path)


def test_detector_initialization(detector):
    """Verify that detector loads correctly and is ready."""
    assert detector.is_ready() is True


def test_inference_on_synthetic_image(detector):
    """Verify inference pipeline executes without error on a blank/synthetic image."""
    img = np.ones((640, 640, 3), dtype=np.uint8) * 180
    result = detector.detect(img)

    assert "detections" in result
    assert "latency_ms" in result
    assert "category_counts" in result
    assert result["latency_ms"] > 0
    assert result["image_dimensions"] == (640, 640)


def test_inference_on_sample_shelf_image(detector):
    """Verify inference on a real sample shelf image from assets/samples."""
    sample_img = settings.SAMPLES_DIR / "sample_well_stocked.jpg"
    if not sample_img.exists():
        pytest.skip("Sample image not present yet.")

    result = detector.detect(sample_img, conf_threshold=0.25)
    assert "detections" in result
    assert isinstance(result["detections"], list)
    assert result["total_detections"] == len(result["detections"])

    # If any detection was found, test structure
    for det in result["detections"]:
        assert "class_name" in det
        assert "confidence" in det
        assert "bbox" in det
        assert len(det["bbox"]) == 4
        assert det["confidence"] >= 0.25
        assert det["class_name"] in settings.CLASS_NAMES or det["class_name"].startswith("Class_")


def test_confidence_threshold_filtering(detector):
    """Verify that increasing confidence threshold strictly filters low-confidence predictions."""
    sample_img = settings.SAMPLES_DIR / "sample_well_stocked.jpg"
    if not sample_img.exists():
        pytest.skip("Sample image not present yet.")

    low_conf_result = detector.detect(sample_img, conf_threshold=0.15)
    high_conf_result = detector.detect(sample_img, conf_threshold=0.60)

    assert low_conf_result["total_detections"] >= high_conf_result["total_detections"]


def test_invalid_input_error_handling(detector):
    """Verify error handling for missing files and unsupported data types."""
    with pytest.raises(FileNotFoundError):
        detector.detect("non_existent_shelf_path_xyz.jpg")

    with pytest.raises(TypeError):
        detector.detect(12345)
