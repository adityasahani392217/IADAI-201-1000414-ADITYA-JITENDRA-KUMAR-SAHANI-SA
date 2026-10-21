"""
Unit Tests - Parking Layout Manager & JSON Serialization
"""

import os
import json
import pytest
from pathlib import Path
from src.parking_layout import ParkingLayoutManager
from config import settings


def test_preset_loading():
    """Verify loading outdoor and indoor presets."""
    mgr = ParkingLayoutManager()
    mgr.load_preset("outdoor_lot")
    assert len(mgr.get_slots()) == len(settings.DEFAULT_OUTDOOR_PRESET)
    
    mgr.load_preset("real_commercial")
    assert len(mgr.get_slots()) == len(settings.REAL_COMMERCIAL_PRESET)

    mgr.load_preset("indoor_garage")
    assert len(mgr.get_slots()) == len(settings.INDOOR_GARAGE_PRESET)


def test_add_and_remove_slot():
    """Verify adding and removing slots dynamically."""
    mgr = ParkingLayoutManager()
    mgr.load_preset("outdoor_lot")
    initial_count = len(mgr.get_slots())
    
    # Add new slot
    success = mgr.add_slot("NEW_01", [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]], section="X", slot_type="EV Charging")
    assert success is True
    assert len(mgr.get_slots()) == initial_count + 1
    
    # Prevent duplicate ID
    duplicate = mgr.add_slot("NEW_01", [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]])
    assert duplicate is False
    
    # Remove slot
    removed = mgr.remove_slot("NEW_01")
    assert removed is True
    assert len(mgr.get_slots()) == initial_count


def test_json_export_and_import(tmp_path):
    """Verify exporting layout to JSON and importing it back intact."""
    mgr = ParkingLayoutManager()
    mgr.load_preset("real_commercial")
    
    json_file = tmp_path / "test_layout.json"
    mgr.export_to_json(json_file)
    assert json_file.exists()
    
    # Load back with fresh manager
    mgr_new = ParkingLayoutManager()
    success = mgr_new.load_from_json(json_file)
    assert success is True
    assert len(mgr_new.get_slots()) == len(settings.REAL_COMMERCIAL_PRESET)
    assert mgr_new.get_slots()[0]["id"] == "P01"
