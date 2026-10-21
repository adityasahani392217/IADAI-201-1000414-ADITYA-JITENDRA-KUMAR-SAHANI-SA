"""
ParkVision AI - Vehicle Detection Engine
Part 1 of Computer Vision Pipeline
Implements YOLOv8 vehicle detection for cars, trucks, buses, and motorcycles
with hardware-accelerated GPU inference and latency profiling.
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

class VehicleDetector:
    """
    High-performance YOLOv8 vehicle detector.
    Filters raw detections to parking-relevant vehicles (Car, Motorcycle, Bus, Truck).
    """

    def __init__(self, model_name: str = settings.DEFAULT_MODEL_NAME, conf_thresh: float = settings.DEFAULT_CONFIDENCE_THRESHOLD):
        self.conf_thresh = conf_thresh
        self.device = "0" if torch.cuda.is_available() else "cpu"
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        
        # Load YOLO model
        weights_path = settings.MODEL_WEIGHTS_PATH
        if weights_path.exists():
            self.model = YOLO(str(weights_path))
        else:
            self.model = YOLO(model_name)
            # Save downloaded weights to weights directory
            settings.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            try:
                self.model.save(str(weights_path))
            except Exception:
                pass

        # Warmup model
        self._warmup()

    def _warmup(self) -> None:
        """Run a single zero-tensor inference to compile GPU kernels and eliminate cold-start latency."""
        try:
            dummy = np.zeros((384, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, device=self.device, verbose=False, classes=settings.VEHICLE_CLASS_IDS)
        except Exception:
            pass

    def detect_vehicles(
        self,
        frame: np.ndarray,
        conf_thresh: Optional[float] = None,
        iou_thresh: float = 0.45
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Run vehicle detection on a single video frame or image.
        Returns:
            Tuple of (list_of_detected_vehicles, inference_latency_ms)
            Each vehicle dict:
                'bbox': [x1, y1, x2, y2] in pixel coords
                'norm_bbox': [x1, y1, x2, y2] normalized [0, 1]
                'bottom_center': (cx, y2)
                'conf': float confidence
                'class_id': int COCO class id
                'class_name': str ('Car', 'Truck', etc.)
        """
        if frame is None or frame.size == 0:
            return [], 0.0

        h, w = frame.shape[:2]
        effective_conf = conf_thresh if conf_thresh is not None else self.conf_thresh

        start_time = time.perf_counter()

        results = self.model.predict(
            source=frame,
            conf=effective_conf,
            iou=iou_thresh,
            classes=settings.VEHICLE_CLASS_IDS,
            device=self.device,
            verbose=False,
            imgsz=640
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                conf = float(box.conf[0].cpu().item())
                cls_id = int(box.cls[0].cpu().item())
                cls_name = settings.VEHICLE_CLASS_NAMES.get(cls_id, "Vehicle")

                # Bottom-center contact point (ground plane contact point)
                bottom_center = (float((x1 + x2) / 2.0), float(y2))
                norm_bottom_center = (bottom_center[0] / w, bottom_center[1] / h)

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "norm_bbox": [x1 / w, y1 / h, x2 / w, y2 / h],
                    "bottom_center": bottom_center,
                    "norm_bottom_center": norm_bottom_center,
                    "conf": conf,
                    "class_id": cls_id,
                    "class_name": cls_name
                })

        return detections, latency_ms


class PKLotSpaceDetector:
    """
    Direct parking space occupancy detector fine-tuned on the PKLot dataset.
    Detects and classifies parking stalls into 'space-empty' (Available)
    and 'space-occupied' (Occupied).
    """

    def __init__(self, weights_path: Optional[Path] = None, conf_thresh: float = 0.25):
        self.conf_thresh = conf_thresh
        self.device = "0" if torch.cuda.is_available() else "cpu"
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        
        target_path = weights_path or settings.PKLOT_MODEL_WEIGHTS_PATH
        if Path(target_path).exists():
            self.model = YOLO(str(target_path))
        else:
            self.model = YOLO(str(settings.MODEL_WEIGHTS_PATH))

    def detect_spaces(self, frame: np.ndarray, conf_thresh: Optional[float] = None, iou_thresh: float = 0.50):
        """
        Detect and classify parking spaces directly from image.
        Returns list of space dictionaries and latency in ms.
        """
        h, w = frame.shape[:2]
        effective_conf = conf_thresh if conf_thresh is not None else self.conf_thresh
        t0 = time.perf_counter()

        results = self.model.predict(
            source=frame,
            conf=effective_conf,
            iou=iou_thresh,
            device=self.device,
            verbose=False,
            imgsz=640
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        spaces = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for idx, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                conf = float(box.conf[0].cpu().item())
                cls_id = int(box.cls[0].cpu().item())
                
                # Class 0: space-empty -> available
                # Class 1: space-occupied -> occupied
                status = "available" if cls_id == 0 else "occupied"
                cls_name = "space-empty" if cls_id == 0 else "space-occupied"

                spaces.append({
                    "id": f"PK-{idx+1:03d}",
                    "bbox": [x1, y1, x2, y2],
                    "norm_bbox": [x1 / w, y1 / h, x2 / w, y2 / h],
                    "status": status,
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name
                })

        return spaces, latency_ms

