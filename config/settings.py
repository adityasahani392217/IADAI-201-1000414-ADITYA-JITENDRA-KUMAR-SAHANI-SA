"""
ParkVision AI - Global Configuration & Settings
CRS Artificial Intelligence Summative Assessment: Machine Learning & Deep Learning

Student: Aditya Jitendra Kumar Sahani
Student ID: 1000414
School: Aspee Nutan Academy
Programme: IBCP Artificial Intelligence / Grade 12
Course: Machine Learning and Deep Learning
"""

from pathlib import Path
from typing import Dict, List, Any

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
PRESETS_DIR = DATA_DIR / "presets"
SAMPLE_MEDIA_DIR = DATA_DIR / "sample_media"
WEIGHTS_DIR = BASE_DIR / "weights"
ASSETS_DIR = BASE_DIR / "assets"
REPORTS_DIR = BASE_DIR / "reports"
TESTS_DIR = BASE_DIR / "tests"
SAMPLES_DIR = ASSETS_DIR / "samples"
EDA_DIR = ASSETS_DIR / "eda"

# Metadata
STUDENT_NAME = "Aditya Jitendra Kumar Sahani"
STUDENT_ID = "1000414"
SCHOOL_NAME = "Aspee Nutan Academy"
PROGRAMME = "IBCP Artificial Intelligence / Grade 12"
COURSE_TITLE = "Machine Learning and Deep Learning"
PROJECT_NAME = "ParkVision AI"
PROJECT_TAGLINE = "Real-Time Smart Parking Intelligence & Space Occupancy Platform"
CANDIDATE_SESSION_NO = "[Candidate Registration / Session Number: To be provided]"

# Vehicle Object Detection Schema (COCO vehicle class mapping)
# COCO class IDs: 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
VEHICLE_CLASS_NAMES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Parking Slot Statuses & Design System Palette (Editorial Light Theme)
SLOT_STATUS_AVAILABLE = "available"
SLOT_STATUS_OCCUPIED = "occupied"
SLOT_STATUS_UNCERTAIN = "uncertain"

STATUS_COLORS_HEX = {
    SLOT_STATUS_AVAILABLE: "#16A34A",  # Emerald Green
    SLOT_STATUS_OCCUPIED: "#DC2626",   # Crimson Red
    SLOT_STATUS_UNCERTAIN: "#D97706",  # Amber Orange
}

STATUS_COLORS_BGR = {
    SLOT_STATUS_AVAILABLE: (74, 163, 22),   # BGR for OpenCV
    SLOT_STATUS_OCCUPIED: (38, 38, 220),    # BGR for OpenCV
    SLOT_STATUS_UNCERTAIN: (6, 119, 217),   # BGR for OpenCV
}

STATUS_BG_HEX = {
    SLOT_STATUS_AVAILABLE: "#F0FDF4",
    SLOT_STATUS_OCCUPIED: "#FEF2F2",
    SLOT_STATUS_UNCERTAIN: "#FFFBEB",
}

STATUS_BADGES = {
    SLOT_STATUS_AVAILABLE: "🟢 Available",
    SLOT_STATUS_OCCUPIED: "🔴 Occupied",
    SLOT_STATUS_UNCERTAIN: "🟡 Uncertain",
}

# Operational Detection Defaults
DEFAULT_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_IOA_THRESHOLD = 0.30       # Intersection over Slot Area
DEFAULT_SMOOTHING_FRAMES = 5       # Temporal window for state confirmation
DEFAULT_HIGH_CAPACITY_THRESHOLD = 0.85 # Alert when 85% capacity reached

# Model Weights
DEFAULT_MODEL_NAME = "yolov8n.pt"
MODEL_WEIGHTS_PATH = WEIGHTS_DIR / "yolov8n.pt"

# Demo Presets: Default 12-Slot Ground Lot Configuration (Normalized [0.0, 1.0])
DEFAULT_OUTDOOR_PRESET = [
    {
        "id": "A01",
        "polygon": [[0.06, 0.40], [0.18, 0.40], [0.16, 0.65], [0.03, 0.65]],
        "section": "A",
        "type": "Standard"
    },
    {
        "id": "A02",
        "polygon": [[0.20, 0.40], [0.32, 0.40], [0.31, 0.65], [0.18, 0.65]],
        "section": "A",
        "type": "Standard"
    },
    {
        "id": "A03",
        "polygon": [[0.34, 0.40], [0.46, 0.40], [0.46, 0.65], [0.33, 0.65]],
        "section": "A",
        "type": "Standard"
    },
    {
        "id": "A04",
        "polygon": [[0.48, 0.40], [0.60, 0.40], [0.61, 0.65], [0.48, 0.65]],
        "section": "A",
        "type": "Standard"
    },
    {
        "id": "A05",
        "polygon": [[0.62, 0.40], [0.74, 0.40], [0.76, 0.65], [0.63, 0.65]],
        "section": "A",
        "type": "Standard"
    },
    {
        "id": "A06",
        "polygon": [[0.76, 0.40], [0.88, 0.40], [0.91, 0.65], [0.78, 0.65]],
        "section": "A",
        "type": "Standard"
    },
    {
        "id": "B01",
        "polygon": [[0.02, 0.70], [0.15, 0.70], [0.13, 0.96], [0.01, 0.96]],
        "section": "B",
        "type": "Standard"
    },
    {
        "id": "B02",
        "polygon": [[0.17, 0.70], [0.30, 0.70], [0.29, 0.96], [0.15, 0.96]],
        "section": "B",
        "type": "Standard"
    },
    {
        "id": "B03",
        "polygon": [[0.32, 0.70], [0.45, 0.70], [0.45, 0.96], [0.31, 0.96]],
        "section": "B",
        "type": "Accessible"
    },
    {
        "id": "B04",
        "polygon": [[0.47, 0.70], [0.60, 0.70], [0.61, 0.96], [0.47, 0.96]],
        "section": "B",
        "type": "EV Charging"
    },
    {
        "id": "B05",
        "polygon": [[0.62, 0.70], [0.75, 0.70], [0.77, 0.96], [0.63, 0.96]],
        "section": "B",
        "type": "Standard"
    },
    {
        "id": "B06",
        "polygon": [[0.77, 0.70], [0.90, 0.70], [0.93, 0.96], [0.79, 0.96]],
        "section": "B",
        "type": "Standard"
    }
]

# Real Commercial Parking Lot Preset (calibrated for real_parking_1.jpg)
REAL_COMMERCIAL_PRESET: List[Dict[str, Any]] = [
    {
        "id": "P01",
        "polygon": [[0.68, 0.42], [0.99, 0.42], [0.99, 0.83], [0.68, 0.83]],
        "section": "A",
        "type": "Standard"
    },
    {
        "id": "P02",
        "polygon": [[0.35, 0.43], [0.69, 0.43], [0.69, 0.72], [0.35, 0.72]],
        "section": "A",
        "type": "Standard"
    },
    {
        "id": "P03",
        "polygon": [[0.01, 0.47], [0.14, 0.47], [0.14, 0.58], [0.01, 0.58]],
        "section": "A",
        "type": "Standard"
    },
    {
        "id": "P04",
        "polygon": [[0.15, 0.47], [0.32, 0.47], [0.32, 0.61], [0.15, 0.61]],
        "section": "A",
        "type": "Standard"
    },
    {
        "id": "P05",
        "polygon": [[0.30, 0.75], [0.65, 0.75], [0.65, 0.98], [0.30, 0.98]],
        "section": "B",
        "type": "EV Charging"
    },
    {
        "id": "P06",
        "polygon": [[0.02, 0.68], [0.28, 0.68], [0.28, 0.95], [0.02, 0.95]],
        "section": "B",
        "type": "Accessible"
    }
]

# Indoor Garage Preset
INDOOR_GARAGE_PRESET: List[Dict[str, Any]] = [
    {"id": "G01", "polygon": [[0.08, 0.45], [0.22, 0.45], [0.20, 0.75], [0.05, 0.75]], "section": "G", "type": "Standard"},
    {"id": "G02", "polygon": [[0.24, 0.45], [0.38, 0.45], [0.37, 0.75], [0.22, 0.75]], "section": "G", "type": "Standard"},
    {"id": "G03", "polygon": [[0.40, 0.45], [0.54, 0.45], [0.54, 0.75], [0.39, 0.75]], "section": "G", "type": "EV Charging"},
    {"id": "G04", "polygon": [[0.56, 0.45], [0.70, 0.45], [0.71, 0.75], [0.56, 0.75]], "section": "G", "type": "Accessible"},
    {"id": "G05", "polygon": [[0.72, 0.45], [0.86, 0.45], [0.88, 0.75], [0.73, 0.75]], "section": "G", "type": "Standard"}
]

