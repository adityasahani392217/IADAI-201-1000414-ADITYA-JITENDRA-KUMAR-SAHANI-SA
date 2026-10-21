"""
ParkVision AI - Parking Space Occupancy Engine
Part 2 of Computer Vision Pipeline
Implements rigorous geometric Intersection-over-Area (IoA),
ground contact point evaluation, bipartite conflict resolution,
and temporal hysteresis smoothing to eliminate flickering.
"""

from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from shapely.geometry import Polygon, box, Point

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings


class ParkingOccupancyClassifier:
    """
    Evaluates occupancy state of designated parking bay polygons against detected vehicles.
    Includes temporal smoothing queues and hysteresis to prevent flickering.
    """

    def __init__(
        self,
        ioa_threshold: float = settings.DEFAULT_IOA_THRESHOLD,
        uncertain_low_ratio: float = 0.60,
        smoothing_window: int = settings.DEFAULT_SMOOTHING_FRAMES
    ):
        self.ioa_threshold = ioa_threshold
        # Ambiguity band: [ioa_threshold * uncertain_low_ratio, ioa_threshold]
        self.uncertain_threshold = ioa_threshold * uncertain_low_ratio
        self.smoothing_window = smoothing_window
        
        # Temporal history per slot: {slot_id: deque([states], maxlen=window)}
        self.slot_histories: Dict[str, deque] = {}
        # Confirmed stable state per slot: {slot_id: 'available' | 'occupied' | 'uncertain'}
        self.confirmed_states: Dict[str, str] = {}

    def reset_histories(self) -> None:
        """Reset temporal smoothing queues."""
        self.slot_histories.clear()
        self.confirmed_states.clear()

    def evaluate_occupancy(
        self,
        slots: List[Dict[str, Any]],
        detected_vehicles: List[Dict[str, Any]],
        frame_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate occupancy for all configured slots against detected vehicles in the frame.
        
        Args:
            slots: List of slot dicts with 'id', 'polygon' (normalized [0, 1]), etc.
            detected_vehicles: Output list from VehicleDetector.detect_vehicles
            frame_shape: (height, width)
        
        Returns:
            List of evaluated slot dictionaries with updated occupancy state, metrics, and overlaps.
        """
        h, w = frame_shape[:2]
        evaluated_slots = []

        if not slots:
            return evaluated_slots

        # Step 1: Build Shapely geometries for vehicles and slots (in pixel coordinates)
        vehicle_polygons = []
        for v in detected_vehicles:
            bx1, by1, bx2, by2 = v["bbox"]
            v_poly = box(bx1, by1, bx2, by2)
            v_bc = Point(v["bottom_center"])
            vehicle_polygons.append({
                "poly": v_poly,
                "bottom_center": v_bc,
                "data": v
            })

        # Step 2: Compute pairwise overlaps between every slot and every vehicle
        # We compute IoA = Overlap_Area / Slot_Area
        slot_candidate_overlaps: Dict[str, List[Dict[str, Any]]] = {}

        for slot in slots:
            slot_id = slot["id"]
            # Convert normalized polygon coordinates to pixel coordinates
            raw_pts = slot["polygon"]
            pixel_pts = [(p[0] * w, p[1] * h) for p in raw_pts]
            try:
                slot_poly = Polygon(pixel_pts)
                if not slot_poly.is_valid:
                    slot_poly = slot_poly.buffer(0)
            except Exception:
                continue

            slot_area = slot_poly.area
            if slot_area <= 0:
                continue

            overlaps = []
            for v_idx, v_obj in enumerate(vehicle_polygons):
                v_poly = v_obj["poly"]
                v_bc = v_obj["bottom_center"]

                if not slot_poly.intersects(v_poly):
                    continue

                intersection = slot_poly.intersection(v_poly)
                inter_area = intersection.area
                ioa = inter_area / slot_area
                point_in_slot = slot_poly.contains(v_bc)

                overlaps.append({
                    "vehicle_idx": v_idx,
                    "ioa": ioa,
                    "inter_area": inter_area,
                    "point_in_slot": point_in_slot,
                    "vehicle": v_obj["data"]
                })

            # Sort candidate vehicles by highest IoA
            overlaps.sort(key=lambda x: x["ioa"], reverse=True)
            slot_candidate_overlaps[slot_id] = overlaps

        # Step 3: Conflict Resolution (Greedy bipartite matching)
        # Avoid assigning the same vehicle to multiple slots if it only clips adjacent bays
        assigned_vehicles = set()
        slot_raw_decisions = {}

        for slot in slots:
            slot_id = slot["id"]
            candidates = slot_candidate_overlaps.get(slot_id, [])
            
            best_match = None
            for cand in candidates:
                v_idx = cand["vehicle_idx"]
                if v_idx not in assigned_vehicles:
                    best_match = cand
                    break

            if best_match is not None:
                ioa = best_match["ioa"]
                point_in_slot = best_match["point_in_slot"]

                # High IoA or Ground contact point inside slot
                if ioa >= self.ioa_threshold or (point_in_slot and ioa >= self.uncertain_threshold):
                    raw_state = settings.SLOT_STATUS_OCCUPIED
                    assigned_vehicles.add(best_match["vehicle_idx"])
                elif ioa >= self.uncertain_threshold:
                    raw_state = settings.SLOT_STATUS_UNCERTAIN
                else:
                    raw_state = settings.SLOT_STATUS_AVAILABLE
                
                confidence = float(min(1.0, max(0.50, ioa)))
                matching_vehicle = best_match["vehicle"]
            else:
                raw_state = settings.SLOT_STATUS_AVAILABLE
                ioa = 0.0
                confidence = 1.0
                matching_vehicle = None

            slot_raw_decisions[slot_id] = {
                "raw_state": raw_state,
                "ioa": ioa,
                "confidence": confidence,
                "vehicle": matching_vehicle
            }

        # Step 4: Temporal Smoothing & Hysteresis across consecutive frames
        for slot in slots:
            slot_id = slot["id"]
            decision = slot_raw_decisions[slot_id]
            raw_state = decision["raw_state"]

            # Initialize queue if first time seeing this slot
            if slot_id not in self.slot_histories:
                self.slot_histories[slot_id] = deque(maxlen=self.smoothing_window)
                self.confirmed_states[slot_id] = raw_state

            history = self.slot_histories[slot_id]
            history.append(raw_state)

            # Check if all recent frames agree on new state before changing confirmed state
            recent_states = list(history)
            if recent_states.count(raw_state) >= max(2, int(self.smoothing_window * 0.60)):
                smoothed_state = raw_state
                self.confirmed_states[slot_id] = raw_state
            else:
                smoothed_state = self.confirmed_states[slot_id]

            evaluated_slots.append({
                "id": slot_id,
                "polygon": slot["polygon"],
                "section": slot.get("section", slot_id[0]),
                "type": slot.get("type", "Standard"),
                "status": smoothed_state,
                "raw_status": raw_state,
                "ioa": round(decision["ioa"], 4),
                "confidence": round(decision["confidence"], 3),
                "vehicle": decision["vehicle"]
            })

        return evaluated_slots
