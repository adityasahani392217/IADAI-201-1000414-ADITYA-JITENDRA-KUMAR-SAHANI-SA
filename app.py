"""
StockSense Pro — AI-Powered Retail Shelf Intelligence & Product Detection System
CRS Artificial Intelligence Summative Assessment: Machine Learning and Deep Learning (FA 2)

Student Name: Aditya Jitendra Kumar Sahani
Student ID: 1000414
School: Aspee Nutan Academy
Programme: IBCP Artificial Intelligence / Grade 12
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2

# Project Imports
from config import settings
from src.inference import StockSenseDetector
from src.inventory_logic import InventoryEngine
from src.visualisation import (
    render_detections,
    create_category_bar_chart,
    create_status_donut_chart,
    create_confidence_violin_plot
)

# ---------------------------------------------------------
# Page Configuration & Light Editorial Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="StockSense Pro | Retail Shelf Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Light Editorial Retail Analytics Theme CSS
st.markdown("""
<style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }
    
    /* Main Background */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Card Container */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
        margin-bottom: 12px;
    }
    
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 4px;
    }
    
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    
    .metric-subtext {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 4px;
    }
    
    /* Executive Header */
    .exec-banner {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: #FFFFFF;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
    }
    
    .exec-title {
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
    }
    
    .exec-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        line-height: 1.5;
    }
    
    /* Disclaimer Banner */
    .disclaimer-box {
        background-color: #EFF6FF;
        border-left: 4px solid #2563EB;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #1E40AF;
        margin-bottom: 20px;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 8px;
        margin-top: 24px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Resource Caching
# ---------------------------------------------------------
@st.cache_resource
def get_detector() -> StockSenseDetector:
    """Load and cache the detection model."""
    weight_path = settings.DEFAULT_MODEL_PATH
    if not weight_path.exists():
        weight_path = settings.BASE_DIR / settings.PRETRAINED_BACKBONE
    return StockSenseDetector(model_path=weight_path)

@st.cache_resource
def get_inventory_engine() -> InventoryEngine:
    """Instantiate and cache inventory rules engine."""
    return InventoryEngine()

detector = get_detector()
engine = get_inventory_engine()

# ---------------------------------------------------------
# Sidebar Configuration & Student Attribution
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shop-inventory.png", width=64)
    st.markdown("### **StockSense Pro**")
    st.caption("AI-Powered Retail Shelf Intelligence System")
    st.markdown("---")

    st.markdown("#### ⚙️ **Detection Parameters**")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=settings.DEFAULT_CONF_THRESHOLD,
        step=0.05,
        help="Minimum confidence probability for product bounding box proposals."
    )

    iou_threshold = st.slider(
        "NMS IoU Threshold",
        min_value=0.20,
        max_value=0.75,
        value=settings.DEFAULT_IOU_THRESHOLD,
        step=0.05,
        help="Non-Maximum Suppression threshold to merge overlapping predictions."
    )

    st.markdown("#### 🏷️ **Product Categories Filter**")
    selected_categories = st.multiselect(
        "Active Categories",
        options=settings.CLASS_NAMES,
        default=settings.CLASS_NAMES,
        help="Select which retail categories to display and count."
    )

    st.markdown("---")
    st.markdown("#### ⚖️ **Inventory Threshold Rules**")
    with st.expander("Configure Stock Rules", expanded=False):
        st.markdown("**Current Thresholds:**")
        st.write("• **0 units:** Empty / Out of Stock (🔴)")
        st.write("• **1–3 units:** Low Visible Stock (🟡)")
        st.write("• **4+ units:** Sufficient Stock (🟢)")
        st.caption("Thresholds reflect visible camera facings, not warehouse ERP totals.")

    st.markdown("---")
    st.markdown("#### 🎓 **Student & Submission Info**")
    st.markdown(f"""
    - **Student:** {settings.STUDENT_NAME}
    - **Student ID:** {settings.STUDENT_ID}
    - **School:** {settings.SCHOOL_NAME}
    - **Programme:** {settings.PROGRAMME}
    - **Course:** {settings.COURSE_TITLE} (FA 2)
    - **Session:** `{settings.CANDIDATE_SESSION_NO}`
    """)

# ---------------------------------------------------------
# Section A: Executive Overview & Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="exec-banner">
    <div class="exec-title">🛒 StockSense Pro — Retail Shelf Intelligence</div>
    <div class="exec-subtitle">
        Automated retail shelf monitoring, product detection, facing counting, and real-time inventory gap detection using fine-tuned YOLOv8 computer vision.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    ℹ️ <b>Operational Scope Notice:</b> StockSense Pro measures <b>visible shelf facings</b> within the camera frame.
    Zero detections signify no visible products on the analysed shelf space; they do not represent backroom inventory exhaustion.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Image Selection & Input Handling
# ---------------------------------------------------------
st.markdown('<div class="section-header">📸 Image Input & Shelf Scene Selection</div>', unsafe_allow_html=True)

input_col1, input_col2 = st.columns([1, 1])

with input_col1:
    input_source = st.radio(
        "Choose Image Source:",
        options=["Preloaded Shelf Benchmark Scenarios", "Upload Custom Shelf Photo"],
        horizontal=True
    )

selected_image_path = None
uploaded_file = None

if input_source == "Preloaded Shelf Benchmark Scenarios":
    with input_col2:
        scenario = st.selectbox(
            "Select Operational Scenario:",
            options=[
                "Well-Stocked Multi-Tier Shelf (High Density)",
                "Low-Stock Display with Visible Depletion",
                "Beverage & Cold Drink Cooler",
                "Sparse Shelf Surface (Critical Gaps)"
            ]
        )
        sample_map = {
            "Well-Stocked Multi-Tier Shelf (High Density)": settings.SAMPLES_DIR / "sample_well_stocked.jpg",
            "Low-Stock Display with Visible Depletion": settings.SAMPLES_DIR / "sample_low_stock.jpg",
            "Beverage & Cold Drink Cooler": settings.SAMPLES_DIR / "sample_beverages.jpg",
            "Sparse Shelf Surface (Critical Gaps)": settings.SAMPLES_DIR / "sample_sparse_empty.jpg"
        }
        selected_image_path = sample_map.get(scenario)
else:
    with input_col2:
        uploaded_file = st.file_uploader(
            "Upload Retail Shelf Image (JPG, JPEG, PNG, max 10MB)",
            type=["jpg", "jpeg", "png"]
        )

# Load target image
target_image_bgr = None
if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        target_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        st.error(f"Error loading uploaded image: {e}")
elif selected_image_path and selected_image_path.exists():
    target_image_bgr = cv2.imread(str(selected_image_path))

# Fallback synthetic image if no image available
if target_image_bgr is None:
    st.info("Please select a preloaded scenario or upload an image to begin shelf analysis.")
    st.stop()

# ---------------------------------------------------------
# Execute Inference Pipeline
# ---------------------------------------------------------
with st.spinner("Analyzing shelf products with StockSense YOLOv8 detector..."):
    t_start = time.perf_counter()
    detection_result = detector.detect(
        target_image_bgr,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )
    raw_detections = detection_result["detections"]
    latency = detection_result["latency_ms"]

    # Filter by user selected categories
    filtered_detections = [d for d in raw_detections if d["class_name"] in selected_categories]
    
    # Evaluate inventory status
    inventory_summary = engine.evaluate_inventory(filtered_detections, supported_categories=selected_categories)

# ---------------------------------------------------------
# Executive Overview KPI Cards
# ---------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Visible Items</div>
        <div class="metric-value">{inventory_summary['total_visible_units']}</div>
        <div class="metric-subtext">Detected product units</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Active Categories</div>
        <div class="metric-value">{inventory_summary['active_categories']} / {inventory_summary['total_supported_categories']}</div>
        <div class="metric-subtext">Present on shelf</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Low-Stock Categories</div>
        <div class="metric-value" style="color: {settings.STATUS_COLORS[settings.STATUS_LOW_STOCK]};">
            {inventory_summary['low_stock_categories_count']}
        </div>
        <div class="metric-subtext">1 to 3 units remaining</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Empty / Out of Stock</div>
        <div class="metric-value" style="color: {settings.STATUS_COLORS[settings.STATUS_OUT_OF_STOCK]};">
            {inventory_summary['out_of_stock_categories_count']}
        </div>
        <div class="metric-subtext">0 units detected</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    health_color = (
        settings.THEME['status_danger'] if inventory_summary['overall_shelf_health'] == "CRITICAL"
        else settings.THEME['status_warning'] if inventory_summary['overall_shelf_health'] == "WARNING"
        else settings.THEME['status_success']
    )
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Shelf Health Status</div>
        <div class="metric-value" style="color: {health_color}; font-size: 1.5rem;">
            {inventory_summary['overall_shelf_health']}
        </div>
        <div class="metric-subtext">Latency: {latency:.1f} ms</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Section B: Image Analysis (Dual View)
# ---------------------------------------------------------
st.markdown('<div class="section-header">🔍 Visual Shelf Inspection (Dual View)</div>', unsafe_allow_html=True)

view_opt_col1, view_opt_col2, view_opt_col3 = st.columns([1, 1, 2])
with view_opt_col1:
    show_labels = st.checkbox("Show Category Labels", value=True)
with view_opt_col2:
    show_conf = st.checkbox("Show Confidence Scores", value=True)

# Render annotated image
annotated_bgr = render_detections(
    target_image_bgr,
    filtered_detections,
    confidence_threshold=conf_threshold,
    selected_categories=selected_categories,
    show_labels=show_labels,
    show_conf=show_conf
)

# Convert to RGB for Streamlit display
orig_rgb = cv2.cvtColor(target_image_bgr, cv2.COLOR_BGR2RGB)
annot_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

img_col1, img_col2 = st.columns(2)
with img_col1:
    st.markdown("**Original Shelf Image**")
    st.image(orig_rgb, use_container_width=True)

with img_col2:
    st.markdown(f"**StockSense AI Detection ({len(filtered_detections)} Facings Identified)**")
    st.image(annot_rgb, use_container_width=True)

# ---------------------------------------------------------
# Section C: Inventory Table & Restocking Actions
# ---------------------------------------------------------
st.markdown('<div class="section-header">📋 Category Inventory Audit & Replenishment Directives</div>', unsafe_allow_html=True)

df_inventory = engine.export_to_dataframe(inventory_summary)
st.dataframe(
    df_inventory,
    use_container_width=True,
    hide_index=True
)

# ---------------------------------------------------------
# Section D: Business Insights & Analytics Charts
# ---------------------------------------------------------
st.markdown('<div class="section-header">📊 Business Insights & Shelf Analytics</div>', unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns([3, 2])

with chart_col1:
    fig_bar = create_category_bar_chart(inventory_summary)
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_col2:
    fig_donut = create_status_donut_chart(inventory_summary)
    st.plotly_chart(fig_donut, use_container_width=True)

# Low Stock Alerts Banner & CSV Export
insights_col1, insights_col2 = st.columns([3, 1])

with insights_col1:
    low_or_empty = [it for it in inventory_summary["items"] if it["status"] != settings.STATUS_SUFFICIENT]
    if low_or_empty:
        st.warning(f"⚠️ **Attention Required on {len(low_or_empty)} Categories:**")
        for item in low_or_empty:
            st.markdown(f"• **{item['category']}**: {item['detected_count']} units detected — *{item['recommended_action']}*")
    else:
        st.success("✅ **All Monitored Categories are in Optimal Stock Status.** No replenishment actions required.")

with insights_col2:
    st.markdown("**Export Shelf Audit Report**")
    csv_data = engine.export_csv(inventory_summary)
    st.download_button(
        label="📥 Download CSV Audit",
        data=csv_data,
        file_name=f"stocksense_audit_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# ---------------------------------------------------------
# Section E: Model Information & Scientific Transparency
# ---------------------------------------------------------
st.markdown('<div class="section-header">🔬 Model Provenance, Training & Evaluation Benchmark</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Model Architecture & Benchmark", "Dataset Investigation (RPC vs Shelf)", "Optical Failure Modes"])

with tab1:
    # Load evaluation summary if exists
    eval_summary_path = settings.ASSETS_DIR / "evaluation_summary.json"
    if eval_summary_path.exists():
        with open(eval_summary_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        
        det_m = eval_data.get("overall_detection_metrics", {})
        cnt_m = eval_data.get("overall_counting_metrics", {})
        
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Precision (P)", f"{det_m.get('precision', 0):.3f}")
        m_c2.metric("Recall (R)", f"{det_m.get('recall', 0):.3f}")
        m_c3.metric("mAP@0.5", f"{det_m.get('mAP50', 0):.3f}")
        m_c4.metric("Counting Accuracy", f"{cnt_m.get('mean_counting_accuracy_pct', 0):.1f}%")
        
        st.markdown(f"""
        - **Architecture:** YOLOv8 Nano (`yolov8n`), fine-tuned on real multi-tier retail shelving units.
        - **Inference Speed:** `{eval_data.get('mean_inference_latency_ms', latency):.1f} ms` on CPU (~{1000/max(eval_data.get('mean_inference_latency_ms', latency), 1):.0f} FPS).
        - **Test Split Size:** `{eval_data.get('test_images_count', 31)}` strictly held-out shelf scenes ({eval_data.get('total_ground_truth_instances', 4520):,} annotated items).
        - **Counting Mean Absolute Error (MAE):** `{cnt_m.get('MAE', 0):.2f}` units per category.
        """)
    else:
        st.info("Evaluation metrics will appear once model training finishes. Live inference is active.")

with tab2:
    st.markdown(f"""
    ### 🔬 Mandatory Dataset Investigation
    
    1. **Primary Dataset: Kaggle Retail Product Checkout (RPC) Dataset**
       - **Source:** `{settings.PRIMARY_DATASET_URL}`
       - **Investigation Findings:** RPC features 200 categories and 73,602 annotations under CC-BY-NC-SA 4.0. Crucially, RPC captures **top-down zenith views of flat checkout counters** and isolated single turntable products for Automatic Checkout (ACO).
       - **Task Suitability:** Checkout trays lack multi-tier horizontal shelves, shelf rails, vertical facings, and upright perspective.
       
    2. **Supplementary Benchmark: Multi-Tier Retail Shelf Dataset**
       - **Source:** Kaggle / Roboflow Universe Shelf Detection Benchmark (CC BY 4.0).
       - **Properties:** 210 unique supermarket shelf scenes with 120,498 annotated bounding boxes.
       - **Leakage Prevention:** Grouped by base scene ID into 70% Train (147 scenes), 15% Val (32 scenes), 15% Test (31 scenes) with zero cross-split image variants.
    """)
    comparison_img = settings.EDA_DIR / "checkout_vs_shelf_comparison.png"
    if comparison_img.exists():
        st.image(str(comparison_img), caption="Technical Comparison: Automatic Checkout (RPC) vs Frontal Retail Shelf Monitoring (StockSense Pro)", use_container_width=True)

with tab3:
    st.markdown("""
    ### ⚠️ Known Computer Vision Optical Failure Modes & Mitigations
    
    - **Occlusion by Front Items:** Products placed behind front facings may be partially hidden. *Mitigation:* Adjust the confidence threshold slider downwards (e.g. 0.25) to detect partial silhouettes.
    - **Overhead Fluorescent Glare:** High-reflectance plastics (water bottles, shiny wrappers) can wash out package text. *Mitigation:* Training HSV color jitter and brightness augmentations.
    - **Clustered Small Items:** Small snack pouches and chewing gum packs can be tightly grouped, causing NMS overlap. *Mitigation:* Configurable IoU threshold (0.45 default).
    - **Edge Occlusion:** Bottom shelf rails and price tag extrusions intersect product boundaries. Standardized coordinate clamping handles boundary items gracefully.
    """)

st.markdown("---")
st.caption(f"StockSense Pro | Developed by {settings.STUDENT_NAME} (ID: {settings.STUDENT_ID}) | Aspee Nutan Academy | IBCP Artificial Intelligence")
