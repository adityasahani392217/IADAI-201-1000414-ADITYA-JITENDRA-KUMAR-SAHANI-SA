"""
ParkVision AI - Sample Media & Asset Generator
Generates realistic parking lot scene images and video clips with asphalt textures,
painted bay lines, and multi-colored vehicles for instant Demo & Upload testing.
"""

from pathlib import Path
import cv2
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings


def draw_realistic_car(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, body_color=(200, 50, 40)):
    """Draw a stylized overhead perspective vehicle with windshield, shadow, and roof."""
    # Drop shadow
    shadow_offset = 6
    cv2.rectangle(img, (x1 - shadow_offset, y1 - shadow_offset), (x2 + shadow_offset, y2 + shadow_offset), (30, 30, 30), -1)

    # Main car body
    cv2.rectangle(img, (x1, y1), (x2, y2), body_color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), 2)

    # Windshield / Roof
    w = x2 - x1
    h = y2 - y1
    roof_x1 = x1 + int(w * 0.15)
    roof_x2 = x2 - int(w * 0.15)
    roof_y1 = y1 + int(h * 0.25)
    roof_y2 = y2 - int(h * 0.25)

    cv2.rectangle(img, (roof_x1, roof_y1), (roof_x2, roof_y2), (40, 40, 40), -1)
    # Headlights & Taillights
    cv2.rectangle(img, (x1 + 3, y1 + 2), (x1 + 12, y1 + 6), (50, 240, 255), -1)
    cv2.rectangle(img, (x2 - 12, y1 + 2), (x2 - 3, y1 + 6), (50, 240, 255), -1)
    cv2.rectangle(img, (x1 + 3, y2 - 6), (x1 + 12, y2 - 2), (20, 20, 220), -1)
    cv2.rectangle(img, (x2 - 12, y2 - 6), (x2 - 3, y2 - 2), (20, 20, 220), -1)


def generate_parking_scene(slots_occupied_indices=(0, 2, 7, 9, 10), width=1280, height=720) -> np.ndarray:
    """Generate a realistic parking lot background with bays and parked cars."""
    img = np.ones((height, width, 3), dtype=np.uint8)
    # Asphalt dark slate background with subtle texture
    img[:] = (55, 55, 60)

    # Driveway central lane
    lane_y1 = int(height * 0.64)
    lane_y2 = int(height * 0.70)
    img[lane_y1:lane_y2, :] = (48, 48, 52)
    # Dashed white center line
    for x in range(0, width, 40):
        cv2.line(img, (x, int((lane_y1 + lane_y2) / 2)), (x + 20, int((lane_y1 + lane_y2) / 2)), (200, 200, 200), 2)

    # Draw Painted Parking Slot Lines for each configured slot
    for slot in settings.DEFAULT_OUTDOOR_PRESET:
        poly = slot["polygon"]
        pixel_pts = np.array([[int(p[0] * width), int(p[1] * height)] for p in poly], dtype=np.int32)
        # White parking stall dividers
        cv2.polylines(img, [pixel_pts], isClosed=True, color=(240, 240, 240), thickness=3, lineType=cv2.LINE_AA)
        
        # Stenciled slot label on ground
        cx = int(np.mean([p[0] * width for p in poly]))
        cy = int(np.mean([p[1] * height for p in poly]))
        cv2.putText(img, slot["id"], (cx - 15, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # Palette of vehicle colors
    car_colors = [
        (220, 40, 30),    # Red
        (230, 230, 235),  # Silver/White
        (30, 140, 210),   # Blue
        (30, 30, 30),     # Black
        (40, 180, 70),    # Green
        (20, 120, 200),   # Amber/Orange
        (160, 160, 165),  # Gray
        (220, 220, 50),   # Yellow
    ]

    # Place cars in the specified occupied slot indices
    for idx in slots_occupied_indices:
        if idx >= len(settings.DEFAULT_OUTDOOR_PRESET):
            continue
        slot = settings.DEFAULT_OUTDOOR_PRESET[idx]
        poly = slot["polygon"]
        
        sx1 = min(p[0] for p in poly) * width
        sx2 = max(p[0] for p in poly) * width
        sy1 = min(p[1] for p in poly) * height
        sy2 = max(p[1] for p in poly) * height

        # Vehicle slightly smaller than bay boundaries
        car_x1 = int(sx1 + (sx2 - sx1) * 0.12)
        car_x2 = int(sx2 - (sx2 - sx1) * 0.12)
        car_y1 = int(sy1 + (sy2 - sy1) * 0.08)
        car_y2 = int(sy2 - (sy2 - sy1) * 0.08)

        color = car_colors[idx % len(car_colors)]
        draw_realistic_car(img, car_x1, car_y1, car_x2, car_y2, body_color=color)

    return img


def generate_all_sample_assets():
    """Create sample images and a 60-frame demo video clip."""
    settings.SAMPLE_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    settings.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Standard Balanced Lot (5 / 12 occupied -> 41.7%)
    balanced_img = generate_parking_scene(slots_occupied_indices=(0, 2, 7, 9, 10))
    cv2.imwrite(str(settings.SAMPLE_MEDIA_DIR / "sample_parking_lot.jpg"), balanced_img)
    cv2.imwrite(str(settings.SAMPLES_DIR / "sample_parking_lot.jpg"), balanced_img)

    # 2. Sparse Parking Lot (2 / 12 occupied -> 16.7%)
    sparse_img = generate_parking_scene(slots_occupied_indices=(1, 8))
    cv2.imwrite(str(settings.SAMPLE_MEDIA_DIR / "sample_sparse_lot.jpg"), sparse_img)
    cv2.imwrite(str(settings.SAMPLES_DIR / "sample_sparse_lot.jpg"), sparse_img)

    # 3. High Capacity / Near Full (11 / 12 occupied -> 91.7%)
    full_img = generate_parking_scene(slots_occupied_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    cv2.imwrite(str(settings.SAMPLE_MEDIA_DIR / "sample_full_lot.jpg"), full_img)
    cv2.imwrite(str(settings.SAMPLES_DIR / "sample_full_lot.jpg"), full_img)

    # 4. Generate Short MP4 Video (60 frames at 15 FPS = 4 seconds)
    # Shows cars in lot and a vehicle smoothly parking into bay A02
    video_path = settings.SAMPLE_MEDIA_DIR / "demo_parking_feed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 15.0, (1280, 720))

    base_frame = generate_parking_scene(slots_occupied_indices=(0, 7, 9, 10))
    slot_a02 = settings.DEFAULT_OUTDOOR_PRESET[1]["polygon"]
    target_x1 = int(min(p[0] for p in slot_a02) * 1280 + 10)
    target_x2 = int(max(p[0] for p in slot_a02) * 1280 - 10)
    target_y1 = int(min(p[1] for p in slot_a02) * 720 + 5)
    target_y2 = int(max(p[1] for p in slot_a02) * 720 - 5)

    start_x = 1200
    start_y = int(720 * 0.66)

    for f_idx in range(60):
        frame = base_frame.copy()
        alpha = min(1.0, f_idx / 40.0)
        # Interpolate car position turning into bay
        cur_x1 = int((1.0 - alpha) * (start_x - 120) + alpha * target_x1)
        cur_x2 = int((1.0 - alpha) * start_x + alpha * target_x2)
        cur_y1 = int((1.0 - alpha) * (start_y - 80) + alpha * target_y1)
        cur_y2 = int((1.0 - alpha) * start_y + alpha * target_y2)

        draw_realistic_car(frame, cur_x1, cur_y1, cur_x2, cur_y2, body_color=(240, 240, 245))
        out.write(frame)

    out.release()
    print("Sample parking images and video generated successfully!")


if __name__ == "__main__":
    generate_all_sample_assets()
