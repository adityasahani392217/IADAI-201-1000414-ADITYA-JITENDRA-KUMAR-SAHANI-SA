"""
StockSense Pro - YOLOv8 Model Fine-Tuning Pipeline
Trains YOLOv8 on the leak-free retail shelf dataset, monitors convergence,
and packages weights for deployment.
"""

import os
import shutil
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from ultralytics import YOLO

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

def train_stocksense_yolo(
    data_yaml_path: Optional[Path] = None,
    epochs: int = 7,
    imgsz: int = 416,
    batch_size: int = 16,
    device: str = "cpu",
    output_weights_path: Optional[Path] = None,
    project_run_name: str = "stocksense_run"
) -> Dict[str, Any]:
    """
    Execute fine-tuning of YOLOv8 nano model on retail shelf dataset.
    """
    yaml_file = data_yaml_path or (settings.PROCESSED_DATA_DIR / "data.yaml")
    if not yaml_file.exists():
        raise FileNotFoundError(f"data.yaml not found at {yaml_file}. Run preprocessing first.")

    out_weights = output_weights_path or settings.DEFAULT_MODEL_PATH
    out_weights.parent.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("      StockSense Pro - YOLOv8 Model Training      ")
    print("==================================================")
    print(f"Data configuration: {yaml_file}")
    print(f"Epochs:             {epochs}")
    print(f"Image Size:         {imgsz}")
    print(f"Batch Size:         {batch_size}")
    print(f"Device:             {device}")
    print(f"Target Weights:     {out_weights}")
    print("==================================================")

    # Load pretrained backbone
    backbone = settings.BASE_DIR / settings.PRETRAINED_BACKBONE
    if not backbone.exists():
        model = YOLO("yolov8n.pt")
    else:
        model = YOLO(str(backbone))

    start_time = time.time()
    
    # Run training
    results = model.train(
        data=str(yaml_file),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        workers=0,  # 0 dataloader workers for CPU stability on Windows
        project=str(settings.BASE_DIR / "runs"),
        name=project_run_name,
        exist_ok=True,
        plots=True,
        verbose=True,
        save=True,
        val=True,
        optimizer="AdamW",
        lr0=0.003,
        lrf=0.01,
        weight_decay=0.0005,
        patience=10
    )

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time:.1f}s ({total_training_time/60:.2f} mins)")

    # Locate best.pt
    run_dir = settings.BASE_DIR / "runs" / project_run_name
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = run_dir / "weights" / "last.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, out_weights)
        print(f"Best model weights successfully saved to: {out_weights} ({os.path.getsize(out_weights)/(1024*1024):.2f} MB)")
    else:
        raise FileNotFoundError(f"Could not find trained weights at {best_pt}")

    # Copy plots to assets/eda and reports/
    plots_to_copy = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png"
    ]
    for p in plots_to_copy:
        src = run_dir / p
        if src.exists():
            shutil.copy2(src, settings.EDA_DIR / p)
            shutil.copy2(src, settings.REPORTS_DIR / p)

    # Save training metadata
    meta = {
        "model_architecture": "YOLOv8 Nano (yolov8n)",
        "pretrained_backbone": "yolov8n.pt (COCO pretrained)",
        "fine_tuned_classes": settings.CLASS_NAMES,
        "num_classes": settings.NUM_CLASSES,
        "epochs": epochs,
        "image_size": imgsz,
        "batch_size": batch_size,
        "training_time_seconds": round(total_training_time, 2),
        "device": device,
        "weights_path": str(out_weights.relative_to(settings.BASE_DIR)),
        "weights_size_bytes": os.path.getsize(out_weights),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(settings.ASSETS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta

if __name__ == "__main__":
    train_stocksense_yolo(epochs=6, imgsz=416, batch_size=16)
