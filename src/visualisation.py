"""
StockSense Pro - Visualisation & Analytics Presentation Engine
Renders high-contrast bounding boxes on retail shelf imagery and generates
interactive editorial charts for executive analytics.
"""

from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

def hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    """Convert hex color string like '#2563EB' to BGR tuple for OpenCV."""
    hex_str = hex_str.lstrip('#')
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return (b, g, r)

def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert hex color string like '#2563EB' to RGB tuple."""
    hex_str = hex_str.lstrip('#')
    return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))


def render_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    confidence_threshold: float = 0.35,
    selected_categories: Optional[List[str]] = None,
    show_labels: bool = True,
    show_conf: bool = True,
    box_thickness: int = 2
) -> np.ndarray:
    """
    Overlay sleek, high-contrast bounding boxes, label chips, and confidence badges on image.
    image: BGR numpy array
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    for det in detections:
        conf = det.get("confidence", 0.0)
        cname = det.get("class_name", "Unknown")

        if conf < confidence_threshold:
            continue
        if selected_categories and cname not in selected_categories:
            continue

        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue

        # bbox is [x1, y1, x2, y2] in pixels
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        color_hex = settings.CLASS_COLORS.get(cname, "#2563EB")
        bgr_color = hex_to_bgr(color_hex)

        # Draw crisp bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr_color, box_thickness)

        if show_labels:
            # Build label text
            label_text = f"{cname}"
            if show_conf:
                label_text += f" {int(conf * 100)}%"

            font_scale = max(0.38, min(0.6, w / 1200))
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Draw background pill with top boundary protection
            pill_y1 = max(0, y1 - th - 8)
            pill_y2 = y1
            pill_x2 = min(w - 1, x1 + tw + 8)
            
            # Badge background
            cv2.rectangle(annotated, (x1, pill_y1), (pill_x2, pill_y2), bgr_color, -1)
            # Badge text
            cv2.putText(
                annotated,
                label_text,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

    return annotated


def create_category_bar_chart(eval_result: Dict[str, Any]) -> go.Figure:
    """
    Generate an interactive Plotly horizontal bar chart showing
    detected product counts per category with thematic retail colors.
    """
    items = eval_result.get("items", [])
    categories = [it["category"] for it in items]
    counts = [it["detected_count"] for it in items]
    colors = [settings.CLASS_COLORS.get(c, "#2563EB") for c in categories]
    hover_texts = [f"<b>{c}</b><br>Detected Count: {cnt}<br>Status: {it['status']}" for c, cnt, it in zip(categories, counts, items)]

    fig = go.Figure(go.Bar(
        x=counts,
        y=categories,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='#0F172A', width=1.2)
        ),
        text=[f" {c}" if c > 0 else " 0" for c in counts],
        textposition='outside',
        hoverinfo='text',
        hovertext=hover_texts
    ))

    fig.update_layout(
        title=dict(
            text="<b>Visible Facing Volume by Product Category</b>",
            font=dict(size=14, color="#0F172A")
        ),
        xaxis=dict(
            title="Detected Instances (Units on Shelf)",
            showgrid=True,
            gridcolor="#E2E8F0",
            zeroline=True,
            zerolinecolor="#94A3B8"
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=12, color="#0F172A")
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=40, r=40, t=50, b=40),
        height=320
    )
    return fig


def create_status_donut_chart(eval_result: Dict[str, Any]) -> go.Figure:
    """
    Generate a sleek donut chart of stock statuses (Sufficient, Low, Out of Stock).
    """
    items = eval_result.get("items", [])
    status_counts = {
        settings.STATUS_SUFFICIENT: 0,
        settings.STATUS_LOW_STOCK: 0,
        settings.STATUS_OUT_OF_STOCK: 0
    }

    for it in items:
        status_counts[it["status"]] = status_counts.get(it["status"], 0) + 1

    labels = list(status_counts.keys())
    values = list(status_counts.values())
    colors = [settings.STATUS_COLORS[k] for k in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        textinfo='value+label',
        hoverinfo='label+value+percent'
    )])

    fig.update_layout(
        title=dict(
            text="<b>Category Health Composition</b>",
            font=dict(size=14, color="#0F172A")
        ),
        showlegend=False,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, t=50, b=20),
        height=320
    )
    return fig


def create_confidence_violin_plot(detections: List[Dict[str, Any]]) -> Optional[go.Figure]:
    """Plot detection confidence distributions per category."""
    if not detections:
        return None

    cats = [d["class_name"] for d in detections]
    confs = [d["confidence"] for d in detections]

    fig = px.box(
        x=cats,
        y=confs,
        color=cats,
        color_discrete_map=settings.CLASS_COLORS,
        points="all",
        labels={"x": "Product Category", "y": "Detection Confidence"},
        title="<b>Detection Confidence Distribution Across Visible Facings</b>"
    )

    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        showlegend=False,
        margin=dict(l=30, r=30, t=50, b=40),
        height=320,
        yaxis=dict(range=[0.0, 1.05], gridcolor="#E2E8F0")
    )
    return fig
