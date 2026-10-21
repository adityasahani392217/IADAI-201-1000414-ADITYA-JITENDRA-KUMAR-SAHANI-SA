"""
ParkVision AI - Source Package
==============================
Exposes primary computer vision, layout analysis, analytics, and reporting modules.
"""

from src.vehicle_detector import VehicleDetector, PKLotSpaceDetector
from src.parking_layout import ParkingLayoutManager
from src.parking_occupancy import ParkingOccupancyClassifier
from src.video_processor import VideoAnnotationRenderer, ParkingVisualizer
from src.camera_manager import CameraStreamManager
from src.analytics import ParkingAnalyticsEngine, ParkingAnalyticsTracker
from src.reporting import AuditReportGenerator, ParkingReportGenerator

__all__ = [
    "VehicleDetector",
    "PKLotSpaceDetector",
    "ParkingLayoutManager",
    "ParkingOccupancyClassifier",
    "VideoAnnotationRenderer",
    "ParkingVisualizer",
    "CameraStreamManager",
    "ParkingAnalyticsEngine",
    "ParkingAnalyticsTracker",
    "AuditReportGenerator",
    "ParkingReportGenerator"
]
