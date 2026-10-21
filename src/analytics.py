"""
ParkVision AI - Parking Intelligence & Analytics Engine
Maintains time-series telemetry, session aggregates, peak capacity monitoring,
and slot-level utilization tracking.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings


class ParkingAnalyticsTracker:
    """
    Records and calculates operational metrics over time for parking facilities.
    """

    def __init__(self, max_records: int = 2000):
        self.max_records = max_records
        self.history_records: List[Dict[str, Any]] = []
        self.slot_state_durations: Dict[str, Dict[str, float]] = {}
        self._last_update_time = time.time()

    def record_frame_telemetry(
        self,
        summary: Dict[str, Any],
        evaluated_slots: List[Dict[str, Any]],
        fps: float = 0.0,
        latency_ms: float = 0.0
    ) -> None:
        """Append an instantaneous frame state to the analytics history."""
        now = time.time()
        dt = now - self._last_update_time
        self._last_update_time = now

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": now,
            "total_slots": summary["total_slots"],
            "occupied_slots": summary["occupied_slots"],
            "available_slots": summary["available_slots"],
            "uncertain_slots": summary["uncertain_slots"],
            "occupancy_rate_pct": summary["occupancy_rate_pct"],
            "fps": round(fps, 1),
            "latency_ms": round(latency_ms, 1)
        }

        self.history_records.append(record)
        if len(self.history_records) > self.max_records:
            self.history_records.pop(0)

        # Track per-slot duration
        if dt < 5.0:  # Ignore large pause gaps
            for slot in evaluated_slots:
                sid = slot["id"]
                st = slot["status"]
                if sid not in self.slot_state_durations:
                    self.slot_state_durations[sid] = {"occupied_sec": 0.0, "available_sec": 0.0, "uncertain_sec": 0.0}
                if st == settings.SLOT_STATUS_OCCUPIED:
                    self.slot_state_durations[sid]["occupied_sec"] += dt
                elif st == settings.SLOT_STATUS_AVAILABLE:
                    self.slot_state_durations[sid]["available_sec"] += dt
                else:
                    self.slot_state_durations[sid]["uncertain_sec"] += dt

    def get_history_dataframe(self) -> pd.DataFrame:
        """Return time-series history as a pandas DataFrame."""
        if not self.history_records:
            return pd.DataFrame(columns=[
                "timestamp", "total_slots", "occupied_slots", "available_slots", "uncertain_slots", "occupancy_rate_pct", "fps", "latency_ms"
            ])
        return pd.DataFrame(self.history_records)

    def get_session_kpis(self) -> Dict[str, Any]:
        """Compute aggregate Key Performance Indicators (KPIs) for the current session."""
        if not self.history_records:
            return {
                "peak_occupancy_pct": 0.0,
                "peak_occupied_slots": 0,
                "avg_occupancy_pct": 0.0,
                "avg_fps": 0.0,
                "avg_latency_ms": 0.0,
                "total_frames_recorded": 0
            }

        df = pd.DataFrame(self.history_records)
        return {
            "peak_occupancy_pct": float(df["occupancy_rate_pct"].max()),
            "peak_occupied_slots": int(df["occupied_slots"].max()),
            "avg_occupancy_pct": round(float(df["occupancy_rate_pct"].mean()), 1),
            "avg_fps": round(float(df["fps"].mean()), 1),
            "avg_latency_ms": round(float(df["latency_ms"].mean()), 1),
            "total_frames_recorded": len(df)
        }

    def get_slot_utilization_breakdown(self) -> List[Dict[str, Any]]:
        """Return slot-level utilization percentage."""
        breakdown = []
        for sid, d in sorted(self.slot_state_durations.items()):
            total = d["occupied_sec"] + d["available_sec"] + d["uncertain_sec"]
            occ_pct = (d["occupied_sec"] / max(0.001, total)) * 100.0
            breakdown.append({
                "slot_id": sid,
                "occupied_seconds": round(d["occupied_sec"], 1),
                "available_seconds": round(d["available_sec"], 1),
                "utilization_pct": round(occ_pct, 1)
            })
        return breakdown

    def check_operational_alerts(self, summary: Dict[str, Any]) -> List[Dict[str, str]]:
        """Evaluate operational triggers and return active alerts."""
        alerts = []
        total = summary.get("total_slots", 0)
        occupied = summary.get("occupied_slots", 0)
        uncertain = summary.get("uncertain_slots", 0)
        rate = summary.get("occupancy_rate_pct", 0.0)

        if total > 0:
            if occupied >= total:
                alerts.append({
                    "level": "critical",
                    "title": "Parking Facility at Full Capacity (100%)",
                    "message": "All designated bays are occupied. Divert incoming traffic to secondary parking zones.",
                    "action": "Trigger external digital signage: LOT FULL."
                })
            elif rate >= (settings.DEFAULT_HIGH_CAPACITY_THRESHOLD * 100.0):
                alerts.append({
                    "level": "warning",
                    "title": f"High Capacity Warning ({rate:.0f}%)",
                    "message": f"Parking occupancy exceeded {settings.DEFAULT_HIGH_CAPACITY_THRESHOLD*100:.0f}% threshold. Limited bays remaining.",
                    "action": "Alert ground security staff to manage entrance lanes."
                })

            if uncertain >= 2:
                alerts.append({
                    "level": "info",
                    "title": f"{uncertain} Spaces in Uncertain State",
                    "message": "Detections near bay boundaries are ambiguous. Vehicles may be straddling lines or partially occluded.",
                    "action": "Verify bays via manual patrol or adjust camera angle."
                })

        return alerts
