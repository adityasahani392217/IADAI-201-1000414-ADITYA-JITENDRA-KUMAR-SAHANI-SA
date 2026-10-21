"""
StockSense Pro - Test Suite for Inspection History & Store Operations Persistence
"""

import pytest
import os
import json
import tempfile
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from src.history_manager import HistoryManager

@pytest.fixture
def temp_history(tmp_path):
    """Fixture providing an isolated HistoryManager instance with temporary storage."""
    temp_json = tmp_path / "test_history.json"
    return HistoryManager(storage_path=temp_json)


def test_history_initialization(temp_history):
    """Verify that HistoryManager initializes storage and creates seed inspection records."""
    assert temp_history.storage_path.exists()
    records = temp_history.get_all_inspections()
    assert len(records) >= 4  # Initialized with 4 seed shift audits
    assert "inspection_id" in records[0]
    assert "timestamp" in records[0]
    assert "shelf_tag" in records[0]


def test_record_new_inspection(temp_history):
    """Verify recording a new shelf inspection prepends to the log."""
    initial_count = len(temp_history.get_all_inspections())
    dummy_eval = {
        "total_visible_units": 15,
        "active_categories": 2,
        "overall_shelf_health": "OPTIMAL",
        "out_of_stock_categories_count": 0,
        "low_stock_categories_count": 1,
        "items": [
            {"category": "Bottle", "detected_count": 10, "status": settings.STATUS_SUFFICIENT, "recommended_action": "Healthy"},
            {"category": "Can", "detected_count": 5, "status": settings.STATUS_SUFFICIENT, "recommended_action": "Healthy"},
            {"category": "Bread", "detected_count": 0, "status": settings.STATUS_OUT_OF_STOCK, "recommended_action": "Restock"}
        ]
    }

    new_id = temp_history.record_inspection(
        shelf_tag="Aisle 9 - Test Display",
        inventory_result=dummy_eval,
        confidence_threshold=0.35,
        notes="Automated unit test record."
    )

    assert new_id.startswith("INSP-")
    records = temp_history.get_all_inspections()
    assert len(records) == initial_count + 1
    # Most recent record is at top
    assert records[0]["inspection_id"] == new_id
    assert records[0]["shelf_tag"] == "Aisle 9 - Test Display"
    assert records[0]["total_visible_units"] == 15


def test_history_dataframe_and_csv(temp_history):
    """Verify DataFrame structure and CSV export functionality."""
    df = temp_history.get_history_dataframe()
    assert not df.empty
    expected_cols = ["Inspection ID", "Timestamp", "Shelf Location", "Visible Units", "Shelf Health"]
    for col in expected_cols:
        assert col in df.columns

    csv_data = temp_history.export_csv()
    assert isinstance(csv_data, str)
    assert "Inspection ID" in csv_data


def test_recurring_shortages_tally(temp_history):
    """Verify calculation of recurring product shortages across inspections."""
    shortages = temp_history.get_recurring_shortages()
    assert isinstance(shortages, dict)
    for c in settings.CLASS_NAMES:
        assert c in shortages
        assert shortages[c] >= 0
