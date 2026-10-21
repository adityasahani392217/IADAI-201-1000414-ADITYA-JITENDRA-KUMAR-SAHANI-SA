"""
StockSense Pro - Inventory Intelligence & Business Rules Engine
Translates detected bounding boxes and instance counts into operational retail decisions.

Rules:
- 0 visible instances: Out of Stock / No Visible Instances Detected (Alert Level: High / Red)
- 1–3 visible instances: Low Visible Stock (Alert Level: Medium / Yellow)
- >=4 visible instances: Sufficient Stock (Alert Level: Normal / Green)

Note: This system audits visible on-shelf inventory in the analyzed image frame,
not total backroom warehouse stock or point-of-sale ERP transactions.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

class InventoryEngine:
    """
    Evaluates detected retail objects, aggregates counts per category,
    applies configurable stock-status heuristics, and provides actionable replenishment directives.
    """

    def __init__(self, thresholds: Optional[settings.InventoryThresholds] = None):
        self.thresholds = thresholds or settings.DEFAULT_THRESHOLDS

    def update_thresholds(self, out_of_stock_max: int, low_stock_min: int, low_stock_max: int, sufficient_stock_min: int):
        """Update configurable stock threshold boundaries."""
        if low_stock_min > low_stock_max:
            raise ValueError("low_stock_min cannot be greater than low_stock_max")
        if sufficient_stock_min <= low_stock_max:
            raise ValueError("sufficient_stock_min must be greater than low_stock_max")

        self.thresholds = settings.InventoryThresholds(
            out_of_stock_max=out_of_stock_max,
            low_stock_min=low_stock_min,
            low_stock_max=low_stock_max,
            sufficient_stock_min=sufficient_stock_min
        )

    def determine_status(self, count: int) -> Tuple[str, str, str, str, str]:
        """
        Classify a visible count into operational stock status.
        Returns: (status_label, alert_severity, icon, color, recommendation)
        """
        if count <= self.thresholds.out_of_stock_max:
            return (
                settings.STATUS_OUT_OF_STOCK,
                "HIGH",
                settings.STATUS_ICONS[settings.STATUS_OUT_OF_STOCK],
                settings.STATUS_COLORS[settings.STATUS_OUT_OF_STOCK],
                settings.STATUS_RECOMMENDATIONS[settings.STATUS_OUT_OF_STOCK]
            )
        elif self.thresholds.low_stock_min <= count <= self.thresholds.low_stock_max:
            return (
                settings.STATUS_LOW_STOCK,
                "MEDIUM",
                settings.STATUS_ICONS[settings.STATUS_LOW_STOCK],
                settings.STATUS_COLORS[settings.STATUS_LOW_STOCK],
                settings.STATUS_RECOMMENDATIONS[settings.STATUS_LOW_STOCK]
            )
        else:
            return (
                settings.STATUS_SUFFICIENT,
                "NORMAL",
                settings.STATUS_ICONS[settings.STATUS_SUFFICIENT],
                settings.STATUS_COLORS[settings.STATUS_SUFFICIENT],
                settings.STATUS_RECOMMENDATIONS[settings.STATUS_SUFFICIENT]
            )

    def evaluate_inventory(
        self,
        detections: List[Dict[str, Any]],
        supported_categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate full shelf inventory from detection list.
        detections: List of dicts with keys ['class_name', 'confidence', 'bbox']
        """
        cats = supported_categories or settings.CLASS_NAMES
        
        # Initialize counts and confidences
        counts = {c: 0 for c in cats}
        confidences = {c: [] for c in cats}

        for det in detections:
            cname = det.get("class_name")
            conf = det.get("confidence", 0.0)
            if cname in counts:
                counts[cname] += 1
                confidences[cname].append(conf)

        items_summary = []
        low_stock_count = 0
        out_of_stock_count = 0
        sufficient_count = 0

        for c in cats:
            cnt = counts[c]
            status_label, severity, icon, color, rec = self.determine_status(cnt)
            mean_conf = float(sum(confidences[c]) / len(confidences[c])) if confidences[c] else 0.0

            if status_label == settings.STATUS_OUT_OF_STOCK:
                out_of_stock_count += 1
            elif status_label == settings.STATUS_LOW_STOCK:
                low_stock_count += 1
            else:
                sufficient_count += 1

            items_summary.append({
                "category": c,
                "detected_count": cnt,
                "status": status_label,
                "severity": severity,
                "icon": icon,
                "color": color,
                "mean_confidence": round(mean_conf, 3),
                "recommended_action": rec
            })

        # Calculate Executive Metrics
        total_visible_units = sum(counts.values())
        active_categories = sum(1 for c, cnt in counts.items() if cnt > 0)
        overall_shelf_health = (
            "CRITICAL" if out_of_stock_count >= 3 or (total_visible_units < 5 and len(cats) >= 5)
            else "WARNING" if (low_stock_count + out_of_stock_count) >= 2
            else "OPTIMAL"
        )

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_visible_units": total_visible_units,
            "active_categories": active_categories,
            "total_supported_categories": len(cats),
            "out_of_stock_categories_count": out_of_stock_count,
            "low_stock_categories_count": low_stock_count,
            "sufficient_categories_count": sufficient_count,
            "overall_shelf_health": overall_shelf_health,
            "items": items_summary,
            "audit_disclaimer": (
                "Disclaimer: StockSense Pro measures visible facings within the submitted camera frame. "
                "Zero detections indicate no visible items in the captured scene and do not represent "
                "warehouse stock exhaustion or ERP inventory records."
            )
        }

    def export_to_dataframe(self, eval_result: Dict[str, Any]) -> pd.DataFrame:
        """Convert inventory evaluation result into a clean pandas DataFrame for UI table and CSV export."""
        rows = []
        for item in eval_result.get("items", []):
            rows.append({
                "Category": item["category"],
                "Detected Facings": item["detected_count"],
                "Stock Status": f"{item['icon']} {item['status']}",
                "Alert Level": item["severity"],
                "Avg Confidence": f"{item['mean_confidence']*100:.1f}%" if item["mean_confidence"] > 0 else "N/A",
                "Action Directive": item["recommended_action"]
            })
        return pd.DataFrame(rows)

    def export_csv(self, eval_result: Dict[str, Any]) -> str:
        """Generate formatted CSV string for reporting and user download."""
        df = self.export_to_dataframe(eval_result)
        return df.to_csv(index=False)



