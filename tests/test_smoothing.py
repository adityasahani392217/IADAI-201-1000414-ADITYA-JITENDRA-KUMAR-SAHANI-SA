"""
Unit Tests - Sliding Window Hysteresis & Temporal Smoothing
"""

import pytest
from src.parking_occupancy import ParkingOccupancyClassifier
from config import settings


def test_hysteresis_single_frame_glitch_resistance():
    """Verify that a single-frame glitch doesn't immediately flip confirmed state."""
    classifier = ParkingOccupancyClassifier(smoothing_window=5)
    slots = [{"id": "S01", "polygon": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]], "section": "A", "type": "Standard"}]
    
    # 5 frames of empty slot -> confirmed available
    for _ in range(5):
        res = classifier.evaluate_occupancy(slots, [], (1000, 1000))
        assert res[0]["status"] == "available"
        
    # Frame 6: Transient occlusion (1 frame occupied)
    veh = [{"bbox": [100, 100, 300, 300], "confidence": 0.9, "class_id": 2, "class_name": "Car", "bottom_center": [200.0, 300.0]}]
    res = classifier.evaluate_occupancy(slots, veh, (1000, 1000))
    # Due to smoothing (only 1 out of 5 frames occupied), smoothed status remains available!
    assert res[0]["status"] == "available"
    assert res[0]["raw_status"] == "occupied"


def test_hysteresis_state_transition():
    """Verify that state stably transitions when majority of window agrees."""
    classifier = ParkingOccupancyClassifier(smoothing_window=5)
    slots = [{"id": "S01", "polygon": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]], "section": "A", "type": "Standard"}]
    
    # Start available
    classifier.evaluate_occupancy(slots, [], (1000, 1000))
    
    veh = [{"bbox": [100, 100, 300, 300], "confidence": 0.9, "class_id": 2, "class_name": "Car", "bottom_center": [200.0, 300.0]}]
    # Feed 3 occupied frames (>= 60% of window of 5)
    classifier.evaluate_occupancy(slots, veh, (1000, 1000))
    classifier.evaluate_occupancy(slots, veh, (1000, 1000))
    res = classifier.evaluate_occupancy(slots, veh, (1000, 1000))
    
    # Now confirmed state should transition to occupied
    assert res[0]["status"] == "occupied"
