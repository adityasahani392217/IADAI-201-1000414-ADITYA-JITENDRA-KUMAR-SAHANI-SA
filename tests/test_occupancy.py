"""
Unit Tests - Parking Occupancy Classifier & Conflict Resolution
"""

import pytest
from src.parking_occupancy import ParkingOccupancyClassifier
from config import settings


def test_occupancy_empty_slots():
    """Verify all slots evaluate to available when no vehicles are detected."""
    classifier = ParkingOccupancyClassifier(smoothing_window=1)
    slots = [
        {"id": "A01", "polygon": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.3], [0.1, 0.3]], "section": "A", "type": "Standard"},
        {"id": "A02", "polygon": [[0.3, 0.1], [0.4, 0.1], [0.4, 0.3], [0.3, 0.3]], "section": "A", "type": "Standard"},
    ]
    detections = []
    
    results = classifier.evaluate_occupancy(slots, detections, (1000, 1000))
    assert len(results) == 2
    assert results[0]["status"] == "available"
    assert results[1]["status"] == "available"
    assert results[0]["ioa"] == 0.0


def test_occupancy_single_vehicle_high_ioa():
    """Verify slot evaluates to occupied when a vehicle has high IoA."""
    classifier = ParkingOccupancyClassifier(smoothing_window=1, ioa_threshold=0.35)
    slots = [
        {"id": "A01", "polygon": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]], "section": "A", "type": "Standard"},
    ]
    # Frame size 1000x1000 -> Slot is [100, 100] to [300, 400]
    # Vehicle covering [100, 100, 300, 400] -> IoA = 1.0
    detections = [
        {
            "bbox": [100, 100, 300, 400],
            "confidence": 0.92,
            "class_id": 2,
            "class_name": "Car",
            "bottom_center": [200.0, 400.0]
        }
    ]
    
    results = classifier.evaluate_occupancy(slots, detections, (1000, 1000))
    assert len(results) == 1
    assert results[0]["status"] == "occupied"
    assert results[0]["ioa"] >= 0.90


def test_uncertain_boundary_state():
    """Verify slot evaluates to uncertain when IoA is in ambiguity band."""
    # Threshold = 0.40, uncertain band = [0.24, 0.40]
    classifier = ParkingOccupancyClassifier(smoothing_window=1, ioa_threshold=0.40, uncertain_low_ratio=0.60)
    slots = [
        {"id": "A01", "polygon": [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]], "section": "A", "type": "Standard"}
    ]
    # Frame 1000x1000 -> Slot area = 10,000.
    # Overlap 3,000 -> IoA = 0.30 (between 0.24 and 0.40, without ground contact in slot)
    detections = [
        {
            "bbox": [70, 0, 150, 100],  # Overlap x in [70, 100], y in [0, 100] -> 30 * 100 = 3000
            "confidence": 0.85,
            "class_id": 2,
            "class_name": "Car",
            "bottom_center": [110.0, 100.0]  # Outside slot (x=110 > 100)
        }
    ]
    results = classifier.evaluate_occupancy(slots, detections, (1000, 1000))
    assert results[0]["status"] == "uncertain"
