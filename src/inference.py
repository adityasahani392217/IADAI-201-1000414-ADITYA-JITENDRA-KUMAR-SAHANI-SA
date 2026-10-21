"""
StockSense Pro - Real-Time Inference & Detection Engine
Provides high-performance, robust detection on shelf imagery using fine-tuned YOLOv8.
"""

import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

class StockSenseDetector:
    """
    Inference wrapper for the fine-tuned StockSense Pro YOLO model.
    Handles image formatting, model inference, NMS filtering, and latency profiling.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = settings.DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = settings.DEFAULT_IOU_THRESHOLD,
        device: str = "cpu"
    ):
        self.model_path = Path(model_path) if model_path else settings.DEFAULT_MODEL_PATH
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self):
        """Safely load YOLO model weights with clear diagnostic messages."""
        if not self.model_path.exists():
            # If fine-tuned weights are not yet saved, check pretrained backbone fallback
            fallback = settings.BASE_DIR / settings.PRETRAINED_BACKBONE
            if fallback.exists():
                print(f"Notice: Fine-tuned weights not found at {self.model_path}. Loading backbone {fallback} for initialization.")
                self.model = YOLO(str(fallback))
            else:
                print(f"Warning: Model weight file not found at {self.model_path}. Model will be loaded on demand.")
                self.model = None
            return

        try:
            self.model = YOLO(str(self.model_path))
            print(f"StockSense Pro YOLO detector successfully loaded from: {self.model_path}")
        except Exception as e:
            print(f"Error loading model from {self.model_path}: {e}")
            self.model = None

    def is_ready(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self.model is not None

    def preprocess_image(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """Standardize any image input format into a valid BGR numpy array."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image file does not exist: {path}")
            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                raise ValueError(f"Could not decode image at {path}")
            return img_bgr
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.ndim == 3:
                # Assume RGB or BGR
                return image_input
            else:
                raise ValueError(f"Invalid image array shape: {image_input.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def detect(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute product detection and instance counting on shelf image.
        Returns detailed detections, category counts, and inference latency.
        """
        if not self.is_ready():
            raise RuntimeError("StockSense detector is not initialized. Check model weights path.")

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold

        img_bgr = self.preprocess_image(image_input)
        h, w = img_bgr.shape[:2]

        t_start = time.perf_counter()
        results = self.model.predict(
            source=img_bgr,
            conf=conf,
            iou=iou,
            device=self.device,
            imgsz=settings.DEFAULT_IMAGE_SIZE,
            verbose=False
        )
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        detections: List[Dict[str, Any]] = []
        counts = {c: 0 for c in settings.CLASS_NAMES}

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)

                for box, score, cid in zip(xyxy, confs, classes):
                    # Check class name mapping
                    if cid < len(settings.CLASS_NAMES):
                        cname = settings.CLASS_NAMES[cid]
                    elif str(cid) in self.model.names:
                        cname = self.model.names[str(cid)]
                    elif cid in self.model.names:
                        cname = self.model.names[cid]
                    else:
                        cname = f"Class_{cid}"

                    bx1, by1, bx2, by2 = [float(val) for val in box]
                    area = (bx2 - bx1) * (by2 - by1)

                    detections.append({
                        "class_id": int(cid),
                        "class_name": cname,
                        "confidence": float(round(score, 4)),
                        "bbox": [bx1, by1, bx2, by2],
                        "area": float(round(area, 1))
                    })
                    if cname in counts:
                        counts[cname] += 1
                    else:
                        counts[cname] = 1

        return {
            "detections": detections,
            "total_detections": len(detections),
            "category_counts": counts,
            "latency_ms": round(latency_ms, 2),
            "image_dimensions": (w, h),
            "confidence_threshold_used": conf,
            "iou_threshold_used": iou
        }
