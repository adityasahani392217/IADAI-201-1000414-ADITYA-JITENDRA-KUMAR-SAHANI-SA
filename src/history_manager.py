"""
StockSense Pro - Inspection History & Store Operations Persistence Manager
Provides lightweight, reliable JSON-backed storage for retail shelf inspections.
Enables store associates and managers to log inspections, track shelf health trends,
and identify recurring out-of-stock product categories over time.
"""

import os
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

HISTORY_FILE = settings.DATA_DIR / "inspection_history.json"

class HistoryManager:
    """Manages persistence, retrieval, and analytics for shelf inspections."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or HISTORY_FILE
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_storage()

    def _initialize_storage(self):
        """Create storage file with realistic seed data if not present."""
        if not self.storage_path.exists():
            # Seed with 4 realistic prior store inspections from the current shift
            seed_data = [
                {
                    "inspection_id": "INSP-20261021-0815",
                    "timestamp": "2026-10-21 08:15:22",
                    "shelf_tag": "Aisle 3 - Cold Beverages Cooler",
                    "total_visible_units": 28,
                    "active_categories_count": 3,
                    "overall_shelf_health": "OPTIMAL",
                    "out_of_stock_count": 0,
                    "low_stock_count": 1,
                    "model_version": "YOLOv8n-StockSense-v1.0",
                    "confidence_threshold": 0.35,
                    "category_breakdown": {
                        "Bottle": {"count": 18, "status": settings.STATUS_SUFFICIENT},
                        "Can": {"count": 8, "status": settings.STATUS_SUFFICIENT},
                        "Pack": {"count": 2, "status": settings.STATUS_LOW_STOCK}
                    },
                    "notes": "Morning opening inspection. Stock levels healthy; cold drink coolers faced."
                },
                {
                    "inspection_id": "INSP-20261021-0835",
                    "timestamp": "2026-10-21 08:35:40",
                    "shelf_tag": "Aisle 1 - Packaged Breakfast & Cereals",
                    "total_visible_units": 14,
                    "active_categories_count": 2,
                    "overall_shelf_health": "WARNING",
                    "out_of_stock_count": 1,
                    "low_stock_count": 1,
                    "model_version": "YOLOv8n-StockSense-v1.0",
                    "confidence_threshold": 0.35,
                    "category_breakdown": {
                        "Box": {"count": 12, "status": settings.STATUS_SUFFICIENT},
                        "Bread": {"count": 2, "status": settings.STATUS_LOW_STOCK},
                        "Snack": {"count": 0, "status": settings.STATUS_OUT_OF_STOCK}
                    },
                    "notes": "Bread shelf depleted; snack endcap empty after overnight restocking delay."
                },
                {
                    "inspection_id": "INSP-20261021-0905",
                    "timestamp": "2026-10-21 09:05:11",
                    "shelf_tag": "Aisle 6 - Confectionery & Bagged Snacks",
                    "total_visible_units": 36,
                    "active_categories_count": 2,
                    "overall_shelf_health": "OPTIMAL",
                    "out_of_stock_count": 0,
                    "low_stock_count": 0,
                    "model_version": "YOLOv8n-StockSense-v1.0",
                    "confidence_threshold": 0.35,
                    "category_breakdown": {
                        "Snack": {"count": 26, "status": settings.STATUS_SUFFICIENT},
                        "Pack": {"count": 10, "status": settings.STATUS_SUFFICIENT}
                    },
                    "notes": "Mid-morning audit. Excellent facing depth across all snack rows."
                },
                {
                    "inspection_id": "INSP-20261021-0922",
                    "timestamp": "2026-10-21 09:22:45",
                    "shelf_tag": "Aisle 4 - Fresh Produce Display",
                    "total_visible_units": 5,
                    "active_categories_count": 1,
                    "overall_shelf_health": "CRITICAL",
                    "out_of_stock_count": 2,
                    "low_stock_count": 1,
                    "model_version": "YOLOv8n-StockSense-v1.0",
                    "confidence_threshold": 0.35,
                    "category_breakdown": {
                        "Vegetable": {"count": 5, "status": settings.STATUS_SUFFICIENT},
                        "Bread": {"count": 0, "status": settings.STATUS_OUT_OF_STOCK},
                        "Bottle": {"count": 0, "status": settings.STATUS_OUT_OF_STOCK}
                    },
                    "notes": "Urgent attention required. Morning rush depleted side racks."
                }
            ]
            self._save_raw_data(seed_data)

    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load inspection list from JSON file."""
        if not self.storage_path.exists():
            return []
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_raw_data(self, data: List[Dict[str, Any]]):
        """Serialize inspection list to JSON."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def record_inspection(
        self,
        shelf_tag: str,
        inventory_result: Dict[str, Any],
        confidence_threshold: float,
        notes: str = ""
    ) -> str:
        """
        Record a completed shelf inspection into persistent store.
        Returns the assigned inspection ID.
        """
        inspections = self._load_raw_data()
        now = datetime.now()
        insp_id = f"INSP-{now.strftime('%Y%m%d-%H%M%S')}"

        cat_breakdown = {}
        for it in inventory_result.get("items", []):
            cat_breakdown[it["category"]] = {
                "count": it["detected_count"],
                "status": it["status"],
                "recommended_action": it["recommended_action"]
            }

        record = {
            "inspection_id": insp_id,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "shelf_tag": shelf_tag or "General Shelf Facing",
            "total_visible_units": inventory_result.get("total_visible_units", 0),
            "active_categories_count": inventory_result.get("active_categories", 0),
            "overall_shelf_health": inventory_result.get("overall_shelf_health", "UNKNOWN"),
            "out_of_stock_count": inventory_result.get("out_of_stock_categories_count", 0),
            "low_stock_count": inventory_result.get("low_stock_categories_count", 0),
            "model_version": "YOLOv8n-StockSense-v1.0",
            "confidence_threshold": confidence_threshold,
            "category_breakdown": cat_breakdown,
            "notes": notes
        }

        # Insert at top (most recent first)
        inspections.insert(0, record)
        self._save_raw_data(inspections)
        return insp_id

    def get_all_inspections(self) -> List[Dict[str, Any]]:
        """Retrieve all historical inspection records."""
        return self._load_raw_data()

    def get_history_dataframe(self) -> pd.DataFrame:
        """Convert inspection log into a tabular pandas DataFrame."""
        inspections = self._load_raw_data()
        rows = []
        for insp in inspections:
            rows.append({
                "Inspection ID": insp.get("inspection_id"),
                "Timestamp": insp.get("timestamp"),
                "Shelf Location": insp.get("shelf_tag"),
                "Visible Units": insp.get("total_visible_units"),
                "Active Categories": insp.get("active_categories_count"),
                "Shelf Health": insp.get("overall_shelf_health"),
                "Low Stock": insp.get("low_stock_count"),
                "Out of Stock": insp.get("out_of_stock_count"),
                "Notes": insp.get("notes", "")
            })
        return pd.DataFrame(rows)

    def export_csv(self) -> str:
        """Generate CSV string for historical inspection audit export."""
        df = self.get_history_dataframe()
        return df.to_csv(index=False)

    def get_recurring_shortages(self) -> Dict[str, int]:
        """Tally how many times each category has appeared in Low Stock or Out of Stock across all inspections."""
        inspections = self._load_raw_data()
        shortage_counts = {c: 0 for c in settings.CLASS_NAMES}

        for insp in inspections:
            breakdown = insp.get("category_breakdown", {})
            for cname, details in breakdown.items():
                if details.get("status") in [settings.STATUS_OUT_OF_STOCK, settings.STATUS_LOW_STOCK]:
                    shortage_counts[cname] = shortage_counts.get(cname, 0) + 1

        return shortage_counts
