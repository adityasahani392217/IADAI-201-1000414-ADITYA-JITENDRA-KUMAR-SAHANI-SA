"""
ParkVision AI - Reporting and Data Export Module
Generates formatted CSV audit logs, slot status manifests, and annotated image exports.
"""

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings


class ParkingReportGenerator:
    """
    Exports structured parking audit manifests and session logs.
    """

    @staticmethod
    def export_slots_to_csv(evaluated_slots: List[Dict[str, Any]], output_path: Path) -> Path:
        """Export current space-by-space occupancy manifest to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Metadata header
            writer.writerow(["# ParkVision AI - Space Occupancy Manifest"])
            writer.writerow(["# Generated At", now_str])
            writer.writerow(["# Student", f"{settings.STUDENT_NAME} (ID: {settings.STUDENT_ID})"])
            writer.writerow(["# Institution", settings.SCHOOL_NAME])
            writer.writerow([])
            # Data table
            writer.writerow(["Slot_ID", "Section", "Slot_Type", "Occupancy_Status", "IoA_Overlap", "Confidence", "Vehicle_Type"])
            
            for s in evaluated_slots:
                v_type = s["vehicle"]["class_name"] if s.get("vehicle") else "None"
                writer.writerow([
                    s["id"],
                    s.get("section", s["id"][0]),
                    s.get("type", "Standard"),
                    s["status"].capitalize(),
                    s.get("ioa", 0.0),
                    s.get("confidence", 1.0),
                    v_type
                ])

        return output_path

    @staticmethod
    def export_telemetry_history_to_csv(history_df, output_path: Path) -> Path:
        """Export continuous session time-series telemetry to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        history_df.to_csv(output_path, index=False)
        return output_path

    @staticmethod
    def save_annotated_snapshot(frame: np.ndarray, output_dir: Optional[Path] = None) -> Path:
        """Save an annotated visual frame snapshot with timestamped filename."""
        target_dir = output_dir or (settings.REPORTS_DIR / "snapshots")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"parkvision_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        save_path = target_dir / filename
        cv2.imwrite(str(save_path), frame)
        return save_path


class AuditReportGenerator(ParkingReportGenerator):
    """
    Unified audit report and manifest generator.
    """

    def export_space_manifest(self, evaluated_slots: List[Dict[str, Any]], output_path: Optional[Path] = None) -> Path:
        target = output_path or (settings.REPORTS_DIR / "parkvision_space_manifest.csv")
        return self.export_slots_to_csv(evaluated_slots, target)

    def export_telemetry_history(self, history_records_or_df: Any, output_path: Optional[Path] = None) -> Path:
        target = output_path or (settings.REPORTS_DIR / "parkvision_telemetry_history.csv")
        if isinstance(history_records_or_df, list):
            import pandas as pd
            if history_records_or_df:
                df = pd.DataFrame(history_records_or_df)
            else:
                df = pd.DataFrame(columns=[
                    "timestamp", "total_slots", "occupied_slots", "available_slots", "uncertain_slots", "occupancy_rate_pct"
                ])
        else:
            df = history_records_or_df
        return self.export_telemetry_history_to_csv(df, target)


# Class Aliases
ParkingReportGenerator = AuditReportGenerator
__all__ = ["AuditReportGenerator", "ParkingReportGenerator"]

