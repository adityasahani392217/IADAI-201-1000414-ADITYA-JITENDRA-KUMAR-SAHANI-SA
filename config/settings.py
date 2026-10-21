"""
StockSense Pro - Configuration and Settings
CRS Artificial Intelligence Summative Assessment: Machine Learning and Deep Learning (FA 2)

Student Name: Aditya Jitendra Kumar Sahani
Student ID: 1000414
School: Aspee Nutan Academy
Programme: IBCP Artificial Intelligence / Grade 12
Candidate Session Number: [Candidate Registration / Session Number: To be provided]
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Base Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
WEIGHTS_DIR = BASE_DIR / "weights"
ASSETS_DIR = BASE_DIR / "assets"
REPORTS_DIR = BASE_DIR / "reports"
TESTS_DIR = BASE_DIR / "tests"
SAMPLES_DIR = ASSETS_DIR / "samples"
EDA_DIR = ASSETS_DIR / "eda"
SCREENSHOTS_DIR = ASSETS_DIR / "screenshots"

# Default Model Weights
DEFAULT_MODEL_PATH = WEIGHTS_DIR / "stocksense_yolov8n.pt"
PRETRAINED_BACKBONE = "yolov8n.pt"

# Student & Academic Metadata
STUDENT_NAME = "Aditya Jitendra Kumar Sahani"
STUDENT_ID = "1000414"
SCHOOL_NAME = "Aspee Nutan Academy"
PROGRAMME = "IBCP Artificial Intelligence / Grade 12"
COURSE_TITLE = "Machine Learning and Deep Learning"
PROJECT_TITLE = "StockSense Pro"
PROJECT_TAGLINE = "AI-Powered Retail Shelf Intelligence & Product Detection System"
CANDIDATE_SESSION_NO = "[Candidate Registration / Session Number: To be provided]"

# Target Retail Product Categories (Standardized 7-Class Schema)
CLASS_NAMES: List[str] = [
    "Bottle",       # 0: Beverages, water, sodas, glass/plastic bottles
    "Box",          # 1: Cartons, cereals, packaged dry goods
    "Bread",        # 2: Bakery loaves, sliced breads, buns
    "Can",          # 3: Tinned goods, soda cans, canned provisions
    "Pack",         # 4: Pouches, sachets, vacuum packs
    "Snack",        # 5: Chips, crisps, biscuits, bagged snacks
    "Vegetable"     # 6: Fresh produce, greens, root vegetables
]

NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Class Visual Styling (High contrast, accessible palette)
CLASS_COLORS: Dict[str, str] = {
    "Bottle": "#2563EB",     # Royal Blue
    "Box": "#7C3AED",        # Violet
    "Bread": "#D97706",      # Amber
    "Can": "#0D9488",        # Teal
    "Pack": "#EA580C",       # Deep Orange
    "Snack": "#DB2777",      # Pink
    "Vegetable": "#16A34A"   # Emerald Green
}

# Raw Dataset Mapping (from 11 Roboflow classes to 7 standardized retail categories)
# Dropping raw class 0 ('-') as background/unlabeled artifact
RAW_TO_STANDARDIZED_MAP: Dict[int, int] = {
    1: 0,   # '- Big Bottle' -> Bottle (0)
    2: 1,   # '- Big Box'    -> Box (1)
    3: 1,   # '- Box'        -> Box (1)
    4: 2,   # '- Bread'      -> Bread (2)
    5: 3,   # '- Can'        -> Can (3)
    6: 4,   # '- Pack'       -> Pack (4)
    7: 5,   # '- Snack'      -> Snack (5)
    8: 6,   # '- Vegetable'  -> Vegetable (6)
    9: 6,   # '- Vegetable 2'-> Vegetable (6)
    10: 0   # 'Bottle'       -> Bottle (0)
}

# Inventory Intelligence Stock Status Thresholds
@dataclass
class InventoryThresholds:
    out_of_stock_max: int = 0      # 0 instances = No visible instances detected
    low_stock_min: int = 1         # 1 instance
    low_stock_max: int = 3         # 3 instances
    sufficient_stock_min: int = 4  # 4+ instances = Healthy facing count

DEFAULT_THRESHOLDS = InventoryThresholds()

# Stock Status Labels & Severity
STATUS_OUT_OF_STOCK = "No Detections / Empty Shelf"
STATUS_LOW_STOCK = "Low Visible Stock"
STATUS_SUFFICIENT = "Sufficient Stock"

STATUS_COLORS = {
    STATUS_OUT_OF_STOCK: "#DC2626",  # Red
    STATUS_LOW_STOCK: "#D97706",     # Amber
    STATUS_SUFFICIENT: "#16A34A"     # Green
}

STATUS_ICONS = {
    STATUS_OUT_OF_STOCK: "🔴",
    STATUS_LOW_STOCK: "🟡",
    STATUS_SUFFICIENT: "🟢"
}

STATUS_RECOMMENDATIONS = {
    STATUS_OUT_OF_STOCK: "Immediate Restock: Zero visible facings on shelf. Audit backroom storage and dispatch replenishment immediately.",
    STATUS_LOW_STOCK: "Replenish Soon: Visible shelf inventory is running low. Queue restock before peak customer footfall.",
    STATUS_SUFFICIENT: "Healthy Stock: Sufficient facing volume detected. Maintain standard rotation and routine facing alignment."
}

# Inference Defaults
DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_IMAGE_SIZE = 640

# Primary and Supplementary Dataset References
PRIMARY_DATASET_NAME = "Retail Product Checkout (RPC) Dataset"
PRIMARY_DATASET_URL = "https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset"
PRIMARY_DATASET_PAPER = "Wei et al., 'RPC: A Large-Scale Retail Product Checkout Dataset', arXiv:1901.07249 (2019)"
PRIMARY_DATASET_LICENSE = "Creative Commons CC-BY-NC-SA 4.0"

SUPPLEMENTARY_DATASET_NAME = "Retail Shelf Detection Benchmark"
SUPPLEMENTARY_DATASET_URL = "https://universe.roboflow.com/kickkung/shelf-detection-for-coco-format/dataset/2"
SUPPLEMENTARY_DATASET_LICENSE = "Creative Commons CC BY 4.0 / MIT"

# Editorial Light Theme UI Colors
THEME = {
    "bg_main": "#F8FAFC",
    "bg_card": "#FFFFFF",
    "text_primary": "#0F172A",
    "text_secondary": "#475569",
    "text_muted": "#94A3B8",
    "border_card": "#E2E8F0",
    "accent_primary": "#2563EB",
    "accent_secondary": "#3B82F6",
    "status_danger": "#DC2626",
    "status_warning": "#D97706",
    "status_success": "#16A34A"
}
