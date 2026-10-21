"""
StockSense Pro - Dataset Analysis and Exploratory Data Analysis (EDA)
Comprehensive investigation of:
1. Primary Mandatory Dataset: Kaggle Retail Product Checkout (RPC) Dataset
2. Supplementary Benchmark: Retail Shelf Detection Dataset
"""

import os
import json
import glob
import re
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

def analyze_rpc_dataset(rpc_json_path: Path) -> Dict[str, Any]:
    """
    Perform deep structural and statistical analysis on the Kaggle RPC Dataset.
    Investigates categories, macro-groups, image dimensions, and annotations.
    """
    if not rpc_json_path.exists():
        raise FileNotFoundError(f"RPC annotation file not found at: {rpc_json_path}")

    with open(rpc_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    categories = data.get("categories", [])
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    # Group 200 fine-grained categories into 17 macro families
    macro_counts = Counter()
    cat_to_macro = {}
    for cat in categories:
        raw_name = cat["name"]
        parts = raw_name.split("_")
        macro = "_".join(parts[1:]) if len(parts) > 1 else raw_name
        cat_to_macro[cat["id"]] = macro
        macro_counts[macro] += 1

    # Count annotation instances per macro family
    macro_instance_counts = Counter()
    bbox_areas = []
    for ann in annotations:
        cid = ann["category_id"]
        macro = cat_to_macro.get(cid, "unknown")
        macro_instance_counts[macro] += 1
        bbox = ann.get("bbox", [])
        if len(bbox) == 4:
            w, h = bbox[2], bbox[3]
            bbox_areas.append(w * h)

    # Resolution statistics
    resolutions = [(img.get("width", 0), img.get("height", 0)) for img in images[:500]]
    unique_resolutions = set(resolutions)

    stats = {
        "dataset_name": settings.PRIMARY_DATASET_NAME,
        "total_categories": len(categories),
        "total_images_in_split": len(images),
        "total_annotations_in_split": len(annotations),
        "macro_categories_count": len(macro_counts),
        "macro_distribution": dict(macro_counts),
        "macro_instance_counts": dict(macro_instance_counts),
        "sample_resolutions": list(unique_resolutions)[:5],
        "mean_bbox_area": float(np.mean(bbox_areas)) if bbox_areas else 0.0,
        "median_bbox_area": float(np.median(bbox_areas)) if bbox_areas else 0.0,
        "camera_viewpoint": "Nadir / Top-down Zenith (Automatic Checkout Tray)",
        "intended_use_case": "Automatic Checkout (ACO) on checkout counter",
        "license": settings.PRIMARY_DATASET_LICENSE
    }
    return stats


def plot_rpc_eda(rpc_stats: Dict[str, Any], output_path: Path):
    """Generate high-resolution chart of RPC macro category instance distribution."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    macro_instances = rpc_stats["macro_instance_counts"]
    sorted_items = sorted(macro_instances.items(), key=lambda x: x[1], reverse=True)
    names = [item[0].replace("_", " ").title() for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    plt.figure(figsize=(10, 6), dpi=300)
    bars = plt.barh(names[::-1], counts[::-1], color="#2563EB", edgecolor="#1E40AF", alpha=0.85)
    plt.title("Kaggle RPC Dataset: Instance Distribution Across 17 Macro-Categories", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Ground Truth Bounding Box Annotations (Validation Split)", fontsize=11)
    plt.ylabel("Macro Product Category", fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    # Add count labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 100, bar.get_y() + bar.get_height()/2, f'{int(width):,}',
                 va='center', ha='left', fontsize=8, color="#0F172A")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def analyze_shelf_dataset(extracted_dir: Path) -> Dict[str, Any]:
    """
    Analyze the supplementary Retail Shelf Detection Benchmark.
    Computes base scene count, leakage potential, bounding box aspect ratios, and class instance counts.
    """
    if not extracted_dir.exists():
        raise FileNotFoundError(f"Shelf dataset directory not found at: {extracted_dir}")

    all_images = glob.glob(str(extracted_dir / "**" / "*.jpg"), recursive=True)
    base_scenes = set()
    scene_to_images = {}

    for img_path in all_images:
        bn = os.path.basename(img_path)
        m = re.match(r'^(.*?)\.rf\.', bn)
        base = m.group(1) if m else os.path.splitext(bn)[0]
        base_scenes.add(base)
        scene_to_images.setdefault(base, []).append(img_path)

    # Count standardized class instances across all base scenes (1 representative per scene)
    standardized_counts = Counter()
    raw_class_counts = Counter()
    aspect_ratios = {c: [] for c in settings.CLASS_NAMES}
    box_areas = []

    for base, paths in scene_to_images.items():
        rep_img_path = paths[0]
        lbl_path = rep_img_path.replace("images", "labels").replace(".jpg", ".txt")
        if not os.path.exists(lbl_path):
            continue
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    raw_cls = int(parts[0])
                    raw_class_counts[raw_cls] += 1
                    if raw_cls in settings.RAW_TO_STANDARDIZED_MAP:
                        std_cls_idx = settings.RAW_TO_STANDARDIZED_MAP[raw_cls]
                        std_cls_name = settings.CLASS_NAMES[std_cls_idx]
                        standardized_counts[std_cls_name] += 1
                        
                        w, h = float(parts[3]), float(parts[4])
                        aspect_ratios[std_cls_name].append(w / max(h, 1e-4))
                        box_areas.append(w * h)

    stats = {
        "dataset_name": settings.SUPPLEMENTARY_DATASET_NAME,
        "total_files": len(all_images),
        "unique_base_scenes": len(base_scenes),
        "mean_augmentations_per_scene": round(len(all_images) / max(len(base_scenes), 1), 2),
        "raw_class_counts": dict(raw_class_counts),
        "standardized_counts": dict(standardized_counts),
        "total_standardized_annotations": sum(standardized_counts.values()),
        "mean_aspect_ratio": {k: float(np.mean(v)) if v else 0.0 for k, v in aspect_ratios.items()},
        "camera_viewpoint": "Frontal Shelf Facing (Multi-tier retail shelving units)",
        "license": settings.SUPPLEMENTARY_DATASET_LICENSE
    }
    return stats, aspect_ratios


def plot_shelf_eda(shelf_stats: Dict[str, Any], aspect_ratios: Dict[str, List[float]], output_dir: Path):
    """Generate professional publication charts for the Retail Shelf Dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Standardized Class Distribution Chart
    plt.figure(figsize=(9, 5), dpi=300)
    std_counts = shelf_stats["standardized_counts"]
    classes = settings.CLASS_NAMES
    counts = [std_counts.get(c, 0) for c in classes]
    colors = [settings.CLASS_COLORS[c] for c in classes]

    bars = plt.bar(classes, counts, color=colors, edgecolor="#0F172A", alpha=0.9, width=0.6)
    plt.title("StockSense Pro: Ground-Truth Product Instances by Standardized Category", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Retail Product Category", fontsize=11, labelpad=8)
    plt.ylabel("Annotated Bounding Boxes (Unaugmented Scenes)", fontsize=11, labelpad=8)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 100, f'{int(height):,}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color="#0F172A")

    plt.ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.savefig(output_dir / "shelf_class_distribution.png", dpi=300)
    plt.close()

    # 2. Aspect Ratio Distributions
    plt.figure(figsize=(10, 5), dpi=300)
    data_to_plot = [aspect_ratios[c] for c in classes if aspect_ratios[c]]
    labels_to_plot = [c for c in classes if aspect_ratios[c]]
    
    bp = plt.boxplot(data_to_plot, patch_artist=True, tick_labels=labels_to_plot,
                     medianprops=dict(color="#0F172A", linewidth=1.5),
                     showfliers=False)
    
    for patch, c in zip(bp['boxes'], labels_to_plot):
        patch.set_facecolor(settings.CLASS_COLORS[c])
        patch.set_alpha(0.7)

    plt.title("Product Bounding Box Aspect Ratio (Width / Height) by Category", fontsize=12, fontweight='bold', pad=15)
    plt.ylabel("Aspect Ratio (w / h)", fontsize=11)
    plt.xlabel("Retail Product Category", fontsize=11)
    plt.axhline(1.0, color="#DC2626", linestyle=":", label="Square (1:1 Ratio)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "shelf_bbox_aspect_ratios.png", dpi=300)
    plt.close()


def plot_comparison_infographic(output_path: Path):
    """
    Creates an architectural comparison infographic contrasting
    Top-Down Checkout (RPC) vs Frontal Retail Shelf (StockSense Pro).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)
    
    # Left: RPC Top-down ACO
    ax1 = axes[0]
    ax1.set_facecolor("#F1F5F9")
    ax1.set_title("Kaggle Primary Dataset: Automatic Checkout (RPC)\nTop-Down Nadir Camera View", fontsize=11, fontweight='bold', pad=12)
    # Draw top-down table
    rect_tray = patches.Rectangle((0.15, 0.15), 0.7, 0.7, linewidth=2, edgecolor="#475569", facecolor="#E2E8F0")
    ax1.add_patch(rect_tray)
    # Scatter items on tray
    for (x, y, w, h, col, label) in [
        (0.25, 0.3, 0.18, 0.15, "#2563EB", "Puffed Food"),
        (0.55, 0.25, 0.15, 0.22, "#D97706", "Drink"),
        (0.35, 0.6, 0.22, 0.18, "#16A34A", "Noodles"),
        (0.65, 0.65, 0.12, 0.12, "#DB2777", "Candy")
    ]:
        box = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor="#0F172A", facecolor=col, alpha=0.8)
        ax1.add_patch(box)
        ax1.text(x + w/2, y + h/2, label, ha="center", va="center", color="white", fontsize=8, fontweight='bold')

    ax1.text(0.5, 0.05, "Zenith View: Flat Checkout Tray\nUnsuitable for Multi-Tier Shelf Inventory", ha="center", fontsize=10, color="#DC2626", fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # Right: StockSense Pro Retail Shelf Frontal View
    ax2 = axes[1]
    ax2.set_facecolor("#F8FAFC")
    ax2.set_title("StockSense Pro Supplementary Benchmark\nFrontal Multi-Tier Retail Shelf View", fontsize=11, fontweight='bold', pad=12)
    
    # Draw shelves
    for y_shelf in [0.2, 0.5, 0.8]:
        ax2.axhline(y_shelf, color="#334155", linewidth=4)
        
    # Tier 1 (Bottles)
    for x in np.linspace(0.1, 0.85, 6):
        box = patches.Rectangle((x, 0.8), 0.08, 0.15, linewidth=1.5, edgecolor="#0F172A", facecolor="#2563EB", alpha=0.85)
        ax2.add_patch(box)
    ax2.text(0.02, 0.87, "Tier 1: Bottles", fontsize=8, fontweight='bold', color="#2563EB")

    # Tier 2 (Cans & Snacks)
    for x in np.linspace(0.1, 0.5, 4):
        box = patches.Rectangle((x, 0.5), 0.09, 0.12, linewidth=1.5, edgecolor="#0F172A", facecolor="#0D9488", alpha=0.85)
        ax2.add_patch(box)
    for x in np.linspace(0.6, 0.85, 3):
        box = patches.Rectangle((x, 0.5), 0.09, 0.14, linewidth=1.5, edgecolor="#0F172A", facecolor="#DB2777", alpha=0.85)
        ax2.add_patch(box)
    ax2.text(0.02, 0.56, "Tier 2: Cans / Snacks", fontsize=8, fontweight='bold', color="#0D9488")

    # Tier 3 (Boxes & Produce)
    for x in np.linspace(0.1, 0.45, 3):
        box = patches.Rectangle((x, 0.2), 0.12, 0.16, linewidth=1.5, edgecolor="#0F172A", facecolor="#7C3AED", alpha=0.85)
        ax2.add_patch(box)
    for x in np.linspace(0.6, 0.85, 3):
        box = patches.Rectangle((x, 0.2), 0.09, 0.12, linewidth=1.5, edgecolor="#0F172A", facecolor="#16A34A", alpha=0.85)
        ax2.add_patch(box)
    ax2.text(0.02, 0.27, "Tier 3: Boxes / Veg", fontsize=8, fontweight='bold', color="#7C3AED")

    ax2.text(0.5, 0.05, "Frontal View: Multi-Tier Shelf Facings\nTarget Domain for In-Store Stock Monitoring", ha="center", fontsize=10, color="#16A34A", fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_full_eda():
    """Execute complete dataset analysis and generate visualization artifacts."""
    print("--- 1. Analyzing Kaggle RPC Dataset ---")
    rpc_path = settings.BASE_DIR / "data_investigation" / "rpc" / "instances_val2019.json"
    rpc_stats = analyze_rpc_dataset(rpc_path)
    print(f"Kaggle RPC Dataset: {rpc_stats['total_categories']} classes, {rpc_stats['total_images_in_split']} images, {rpc_stats['total_annotations_in_split']} annotations")
    plot_rpc_eda(rpc_stats, settings.EDA_DIR / "rpc_macro_distribution.png")

    print("\n--- 2. Analyzing Retail Shelf Dataset ---")
    shelf_extracted = settings.BASE_DIR / "data_investigation" / "shelf_dataset" / "extracted"
    shelf_stats, aspect_ratios = analyze_shelf_dataset(shelf_extracted)
    print(f"Shelf Dataset: {shelf_stats['unique_base_scenes']} unique scenes, {shelf_stats['total_files']} files, {shelf_stats['total_standardized_annotations']} annotations")
    plot_shelf_eda(shelf_stats, aspect_ratios, settings.EDA_DIR)

    print("\n--- 3. Generating Camera Perspective Comparison Diagram ---")
    plot_comparison_infographic(settings.EDA_DIR / "checkout_vs_shelf_comparison.png")

    summary_file = settings.EDA_DIR / "dataset_investigation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "rpc_dataset": rpc_stats,
            "shelf_dataset": shelf_stats
        }, f, indent=2)
    print(f"EDA Analysis completed. Results saved to: {settings.EDA_DIR}")

if __name__ == "__main__":
    run_full_eda()
