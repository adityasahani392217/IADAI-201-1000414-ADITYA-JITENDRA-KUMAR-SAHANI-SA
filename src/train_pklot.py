"""
ParkVision AI - PKLot YOLOv8 Training Script
============================================
Trains a YOLOv8 Nano parking space occupancy detection model
on the PKLot dataset (60,188 annotated spaces: space-empty vs space-occupied)
using hardware-accelerated GPU training on NVIDIA GeForce RTX 3050.
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
YAML_PATH = BASE_DIR / "data" / "pklot" / "pklot.yaml"
BASE_WEIGHTS = BASE_DIR / "weights" / "yolov8n.pt"
OUTPUT_WEIGHTS = BASE_DIR / "weights" / "pklot_yolov8n.pt"
ASSETS_EDA = BASE_DIR / "assets" / "eda"


def train_pklot_model(epochs: int = 5, batch_size: int = 16, imgsz: int = 640):
    print("=" * 60)
    print("PARKVISION AI - PKLOT MODEL TRAINING PIPELINE")
    print("=" * 60)

    device = "0" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[INFO] Compute Device: {device} ({device_name})")
    print(f"[INFO] Base Architecture: YOLOv8n ({BASE_WEIGHTS})")
    print(f"[INFO] Dataset Config: {YAML_PATH}")
    print(f"[INFO] Hyperparameters: Epochs={epochs}, Batch={batch_size}, ImgSize={imgsz}")

    # Load base YOLO model
    model = YOLO(str(BASE_WEIGHTS))

    # Run training
    results = model.train(
        data=str(YAML_PATH),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        workers=2,
        project="runs/detect",
        name="pklot_train",
        exist_ok=True,
        verbose=True,
        plots=True
    )

    print("\n[INFO] Training complete! Saving best model weights...")
    best_weights_path = Path("runs/detect/pklot_train/weights/best.pt")
    if not best_weights_path.exists():
        best_weights_path = Path("runs/detect/pklot_train/weights/last.pt")

    if best_weights_path.exists():
        shutil.copy(best_weights_path, OUTPUT_WEIGHTS)
        print(f"[SUCCESS] Trained weights saved to: {OUTPUT_WEIGHTS}")

    # Copy generated training curves to assets/eda
    run_dir = Path("runs/detect/pklot_train")
    if run_dir.exists():
        for plot_name in ["results.png", "confusion_matrix.png", "F1_curve.png", "PR_curve.png"]:
            src_plot = run_dir / plot_name
            if src_plot.exists():
                dst_plot = ASSETS_EDA / f"pklot_{plot_name}"
                shutil.copy(src_plot, dst_plot)
                print(f"[INFO] Copied {plot_name} -> {dst_plot}")

    print("=" * 60)
    print("[SUCCESS] PKLot Training Pipeline Finished Successfully!")
    print("=" * 60)
    return results


if __name__ == "__main__":
    train_pklot_model(epochs=5, batch_size=16, imgsz=640)
