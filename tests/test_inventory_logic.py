"""
StockSense Pro - Test Suite for Inventory Intelligence Rules
Tests:
- Stock status boundary thresholds (0, 1, 3, 4, 100)
- Configurable threshold updates and validation errors
- Zero-detection handling
- Count aggregation
- Health status computation
- CSV generation
"""

import pytest
import pandas as pd
from io import StringIO

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from src.inventory_logic import InventoryEngine

def test_default_threshold_boundaries():
    """Verify default threshold classifications: 0 -> Out of stock, 1-3 -> Low stock, 4+ -> Sufficient."""
    engine = InventoryEngine()

    # Out of stock (0 instances)
    status_0, sev_0, icon_0, _, _ = engine.determine_status(0)
    assert status_0 == settings.STATUS_OUT_OF_STOCK
    assert sev_0 == "HIGH"
    assert icon_0 == "🔴"

    # Low stock lower bound (1 instance)
    status_1, sev_1, icon_1, _, _ = engine.determine_status(1)
    assert status_1 == settings.STATUS_LOW_STOCK
    assert sev_1 == "MEDIUM"
    assert icon_1 == "🟡"

    # Low stock upper bound (3 instances)
    status_3, sev_3, icon_3, _, _ = engine.determine_status(3)
    assert status_3 == settings.STATUS_LOW_STOCK
    assert sev_3 == "MEDIUM"
    assert icon_3 == "🟡"

    # Sufficient stock lower bound (4 instances)
    status_4, sev_4, icon_4, _, _ = engine.determine_status(4)
    assert status_4 == settings.STATUS_SUFFICIENT
    assert sev_4 == "NORMAL"
    assert icon_4 == "🟢"

    # High quantity (50 instances)
    status_50, sev_50, icon_50, _, _ = engine.determine_status(50)
    assert status_50 == settings.STATUS_SUFFICIENT
    assert sev_50 == "NORMAL"


def test_custom_threshold_update():
    """Verify configurable threshold modification."""
    engine = InventoryEngine()
    engine.update_thresholds(out_of_stock_max=1, low_stock_min=2, low_stock_max=5, sufficient_stock_min=6)

    # 1 is now out of stock under new rule
    status_1, _, _, _, _ = engine.determine_status(1)
    assert status_1 == settings.STATUS_OUT_OF_STOCK

    # 5 is low stock
    status_5, _, _, _, _ = engine.determine_status(5)
    assert status_5 == settings.STATUS_LOW_STOCK

    # 6 is sufficient
    status_6, _, _, _, _ = engine.determine_status(6)
    assert status_6 == settings.STATUS_SUFFICIENT


def test_invalid_threshold_raises_error():
    """Verify that inverted threshold boundaries raise ValueError."""
    engine = InventoryEngine()
    with pytest.raises(ValueError):
        # low_stock_min > low_stock_max
        engine.update_thresholds(out_of_stock_max=0, low_stock_min=5, low_stock_max=2, sufficient_stock_min=6)

    with pytest.raises(ValueError):
        # sufficient_stock_min <= low_stock_max
        engine.update_thresholds(out_of_stock_max=0, low_stock_min=1, low_stock_max=5, sufficient_stock_min=4)


def test_zero_detections_handling():
    """Ensure empty detection list correctly registers 0 counts and CRITICAL shelf health."""
    engine = InventoryEngine()
    result = engine.evaluate_inventory([])

    assert result["total_visible_units"] == 0
    assert result["active_categories"] == 0
    assert result["out_of_stock_categories_count"] == len(settings.CLASS_NAMES)
    assert result["overall_shelf_health"] == "CRITICAL"

    # Check each item
    for it in result["items"]:
        assert it["detected_count"] == 0
        assert it["status"] == settings.STATUS_OUT_OF_STOCK


def test_count_aggregation_and_confidences():
    """Verify aggregation across detections with varied confidences."""
    engine = InventoryEngine()
    detections = [
        {"class_name": "Bottle", "confidence": 0.85, "bbox": [10, 20, 50, 80]},
        {"class_name": "Bottle", "confidence": 0.95, "bbox": [55, 20, 95, 80]},
        {"class_name": "Can", "confidence": 0.70, "bbox": [100, 20, 140, 60]},
        {"class_name": "Can", "confidence": 0.76, "bbox": [145, 20, 185, 60]},
        {"class_name": "Can", "confidence": 0.82, "bbox": [190, 20, 230, 60]},
        {"class_name": "Can", "confidence": 0.88, "bbox": [235, 20, 275, 60]},
        {"class_name": "Snack", "confidence": 0.60, "bbox": [280, 20, 320, 70]},
    ]
    result = engine.evaluate_inventory(detections)

    assert result["total_visible_units"] == 7
    assert result["active_categories"] == 3

    items_map = {it["category"]: it for it in result["items"]}
    assert items_map["Bottle"]["detected_count"] == 2
    assert items_map["Bottle"]["status"] == settings.STATUS_LOW_STOCK
    assert items_map["Bottle"]["mean_confidence"] == 0.90  # (0.85 + 0.95)/2

    assert items_map["Can"]["detected_count"] == 4
    assert items_map["Can"]["status"] == settings.STATUS_SUFFICIENT

    assert items_map["Snack"]["detected_count"] == 1
    assert items_map["Snack"]["status"] == settings.STATUS_LOW_STOCK

    assert items_map["Bread"]["detected_count"] == 0
    assert items_map["Bread"]["status"] == settings.STATUS_OUT_OF_STOCK


def test_csv_export_format():
    """Verify that export_csv outputs valid CSV structure with required headers."""
    engine = InventoryEngine()
    detections = [
        {"class_name": "Bottle", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
        {"class_name": "Box", "confidence": 0.8, "bbox": [10, 10, 20, 20]},
    ]
    result = engine.evaluate_inventory(detections)
    csv_str = engine.export_csv(result)

    assert isinstance(csv_str, str)
    df = pd.read_csv(StringIO(csv_str))
    expected_cols = ["Category", "Detected Facings", "Stock Status", "Alert Level", "Avg Confidence", "Action Directive"]
    for col in expected_cols:
        assert col in df.columns
    assert len(df) == len(settings.CLASS_NAMES)
