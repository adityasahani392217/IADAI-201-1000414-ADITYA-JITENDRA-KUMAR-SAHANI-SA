"""
ParkVision AI - Parking Layout Configuration & Management
Handles creating, editing, removing, validating, saving, and loading
parking bay polygon layouts with camera resolution calibration guards.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings


class ParkingLayoutManager:
    """
    Manages parking space geometries, identifiers, and configuration files.
    """

    def __init__(self, presets_dir: Optional[Path] = None):
        self.presets_dir = presets_dir or settings.PRESETS_DIR
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        self.active_layout: List[Dict[str, Any]] = []
        self.camera_metadata: Dict[str, Any] = {
            "camera_id": "default_camera_01",
            "camera_name": "Outdoor Main Entrance",
            "calibrated_resolution": [1920, 1080],
            "created_at": None,
            "version": "1.0"
        }
        # Load default preset on startup
        self.load_preset("outdoor_lot")

    def load_preset(self, preset_name: str = "outdoor_lot") -> List[Dict[str, Any]]:
        """Load a predefined demo layout configuration."""
        if preset_name == "outdoor_lot":
            self.active_layout = [dict(s) for s in settings.DEFAULT_OUTDOOR_PRESET]
            self.camera_metadata["camera_name"] = "Outdoor Parking Lot (North Entrance)"
        elif preset_name == "real_commercial":
            self.active_layout = [dict(s) for s in settings.REAL_COMMERCIAL_PRESET]
            self.camera_metadata["camera_name"] = "Commercial Retail Parking (CCTV-04)"
        elif preset_name == "indoor_garage":
            self.active_layout = [dict(s) for s in settings.INDOOR_GARAGE_PRESET]
            self.camera_metadata["camera_name"] = "Underground Garage (Level B1)"
        else:
            self.active_layout = [dict(s) for s in settings.DEFAULT_OUTDOOR_PRESET]
            self.camera_metadata["camera_name"] = "Outdoor Parking Lot (North Entrance)"

        return self.active_layout

    def add_slot(self, slot_id: str, polygon: List[List[float]], section: str = "A", slot_type: str = "Standard") -> bool:
        """Add a new parking space polygon."""
        if not slot_id or any(s["id"] == slot_id for s in self.active_layout):
            return False
        # Validate polygon points
        if len(polygon) < 3:
            return False

        # Clamp normalized coordinates to [0, 1]
        clamped_poly = [[max(0.0, min(1.0, float(pt[0]))), max(0.0, min(1.0, float(pt[1])))] for pt in polygon]

        self.active_layout.append({
            "id": slot_id,
            "polygon": clamped_poly,
            "section": section,
            "type": slot_type
        })
        return True

    def remove_slot(self, slot_id: str) -> bool:
        """Remove a parking space by ID."""
        initial_len = len(self.active_layout)
        self.active_layout = [s for s in self.active_layout if s["id"] != slot_id]
        return len(self.active_layout) < initial_len

    def update_slot(self, slot_id: str, polygon: Optional[List[List[float]]] = None, slot_type: Optional[str] = None) -> bool:
        """Update existing slot geometry or metadata."""
        for slot in self.active_layout:
            if slot["id"] == slot_id:
                if polygon is not None and len(polygon) >= 3:
                    slot["polygon"] = [[max(0.0, min(1.0, float(pt[0]))), max(0.0, min(1.0, float(pt[1])))] for pt in polygon]
                if slot_type is not None:
                    slot["type"] = slot_type
                return True
        return False

    def validate_resolution(self, current_shape: Tuple[int, int]) -> Optional[str]:
        """
        Check if the incoming video stream resolution matches calibrated resolution.
        Returns a warning string if aspect ratio or resolution diverges significantly.
        """
        h, w = current_shape[:2]
        cal_w, cal_h = self.camera_metadata.get("calibrated_resolution", [w, h])
        
        cal_aspect = cal_w / max(1, cal_h)
        cur_aspect = w / max(1, h)

        if abs(cal_aspect - cur_aspect) > 0.15:
            return (
                f"Camera aspect ratio discrepancy detected! Calibrated: {cal_aspect:.2f}, "
                f"Current: {cur_aspect:.2f}. Bounding polygons may require recalibration."
            )
        return None

    def export_to_json(self, file_path: Path) -> None:
        """Save parking layout and camera metadata to a JSON file."""
        data = {
            "metadata": self.camera_metadata,
            "slots_count": len(self.active_layout),
            "slots": self.active_layout
        }
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_json(self, file_path: Path) -> bool:
        """Load parking layout and camera metadata from a JSON file."""
        if not file_path.exists():
            return False
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.active_layout = data.get("slots", [])
            self.camera_metadata = data.get("metadata", self.camera_metadata)
            return True
        except Exception:
            return False

    def get_slots(self) -> List[Dict[str, Any]]:
        """Return the current active list of parking slots."""
        return self.active_layout
