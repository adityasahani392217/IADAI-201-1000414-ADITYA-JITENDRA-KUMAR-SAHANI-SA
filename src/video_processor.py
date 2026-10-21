"""
ParkVision AI - Video & Image Processing Pipeline
Handles visual annotation overlays, polygon semi-transparent fills,
bounding box rendering, and video frame sequence export.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings


class ParkingVisualizer:
    """
    Renders high-contrast, professional visual overlays on parking lot frames.
    """

    @staticmethod
    def draw_parking_overlay(
        frame: np.ndarray,
        evaluated_slots: List[Dict[str, Any]],
        detected_vehicles: List[Dict[str, Any]],
        show_vehicles: bool = True,
        show_slot_labels: bool = True,
        alpha_fill: float = 0.35
    ) -> np.ndarray:
        """
        Render semi-transparent parking bay polygon masks and vehicle bounding boxes.
        """
        if frame is None or frame.size == 0:
            return frame

        canvas = frame.copy()
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # 1. Draw Parking Bay Polygons
        for slot in evaluated_slots:
            slot_id = slot["id"]
            status = slot.get("status", settings.SLOT_STATUS_AVAILABLE)
            color_bgr = settings.STATUS_COLORS_BGR.get(status, (74, 163, 22))

            # Scale normalized polygon points to pixel dimensions
            raw_pts = slot["polygon"]
            pixel_pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in raw_pts], dtype=np.int32)
            pixel_pts = pixel_pts.reshape((-1, 1, 2))

            # Semi-transparent polygon fill
            cv2.fillPoly(overlay, [pixel_pts], color_bgr)
            # Crisp solid outline
            cv2.polylines(canvas, [pixel_pts], isClosed=True, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)

        # Blend transparent polygons with original frame
        cv2.addWeighted(overlay, alpha_fill, canvas, 1.0 - alpha_fill, 0, canvas)

        # 2. Draw Slot Identifier Badges
        if show_slot_labels:
            for slot in evaluated_slots:
                slot_id = slot["id"]
                status = slot.get("status", settings.SLOT_STATUS_AVAILABLE)
                color_bgr = settings.STATUS_COLORS_BGR.get(status, (74, 163, 22))

                raw_pts = slot["polygon"]
                # Compute centroid of slot polygon for badge placement
                cx = int(np.mean([p[0] * w for p in raw_pts]))
                cy = int(np.mean([p[1] * h for p in raw_pts]))

                badge_text = f"{slot_id}"
                (tw, th), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

                bx1 = cx - tw // 2 - 4
                by1 = cy - th // 2 - 4
                bx2 = cx + tw // 2 + 4
                by2 = cy + th // 2 + 4

                # Background capsule
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color_bgr, -1)
                cv2.putText(
                    canvas,
                    badge_text,
                    (bx1 + 4, by2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        # 3. Draw Vehicle Bounding Boxes
        if show_vehicles and detected_vehicles:
            for v in detected_vehicles:
                x1, y1, x2, y2 = v["bbox"]
                cls_name = v.get("class_name", "Vehicle")
                conf = v.get("conf", 0.0)

                # Slate Gray / Cyan for detected vehicles
                box_color = (255, 180, 50)  # BGR Cyan-Blue
                cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2, lineType=cv2.LINE_AA)

                # Ground contact marker (bottom center)
                bc_x, bc_y = int(v["bottom_center"][0]), int(v["bottom_center"][1])
                cv2.circle(canvas, (bc_x, bc_y), 4, (0, 0, 255), -1)

                label = f"{cls_name} {int(conf * 100)}%"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                cv2.rectangle(canvas, (x1, max(0, y1 - lh - 6)), (x1 + lw + 6, y1), box_color, -1)
                cv2.putText(canvas, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

        return canvas

    @staticmethod
    def compute_occupancy_summary(evaluated_slots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate high-level occupancy counts and capacity percentage."""
        total = len(evaluated_slots)
        occupied = sum(1 for s in evaluated_slots if s["status"] == settings.SLOT_STATUS_OCCUPIED)
        available = sum(1 for s in evaluated_slots if s["status"] == settings.SLOT_STATUS_AVAILABLE)
        uncertain = sum(1 for s in evaluated_slots if s["status"] == settings.SLOT_STATUS_UNCERTAIN)

        occupancy_rate = (occupied / max(1, total)) * 100.0

        return {
            "total_slots": total,
            "occupied_slots": occupied,
            "available_slots": available,
            "uncertain_slots": uncertain,
            "occupancy_rate_pct": round(occupancy_rate, 1),
            "is_near_capacity": occupancy_rate >= (settings.DEFAULT_HIGH_CAPACITY_THRESHOLD * 100.0),
            "is_full": occupied >= total and total > 0
        }
