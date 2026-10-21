"""
StockSense Pro — AI-Powered Retail Shelf Intelligence & Product Detection System
Enterprise-Grade Computer Vision & Retail Inventory Intelligence Platform

Student Name: Aditya Jitendra Kumar Sahani
Student ID: 1000414
School: Aspee Nutan Academy
Programme: IBCP Artificial Intelligence / Grade 12
Course: Machine Learning and Deep Learning (CRS FA 2 / IADAI-201)
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import plotly.express as px
import plotly.graph_objects as go

# Internal System Imports
from config import settings
from src.inference import StockSenseDetector
from src.inventory_logic import InventoryEngine
from src.history_manager import HistoryManager
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

# Clean Light Editorial Styling CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }
    
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Top Banner */
    .app-banner {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
    }
    
    .app-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    
    .app-subtitle {
        font-size: 0.92rem;
        color: #64748B;
        line-height: 1.4;
    }
    
    /* Card Container */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        margin-bottom: 12px;
    }
    
    .metric-label {
        font-size: 0.80rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 4px;
    }
    
    .metric-value {
        font-size: 1.80rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.15;
    }
    
    .metric-subtext {
        font-size: 0.76rem;
        color: #64748B;
        margin-top: 4px;
    }
    
    /* Action Items */
    .action-box {
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    
    .action-critical {
        background-color: #FEF2F2;
        border-left: 4px solid #DC2626;
        color: #991B1B;
    }
    
    .action-warning {
        background-color: #FFFBEB;
        border-left: 4px solid #D97706;
        color: #92400E;
    }
    
    .action-review {
        background-color: #F0F9FF;
        border-left: 4px solid #0284C7;
        color: #075985;
    }
    
    .action-normal {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        color: #166534;
    }
    
    /* Section Headings */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 6px;
        margin-top: 18px;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Resource Caching
# ---------------------------------------------------------
@st.cache_resource
def get_detector() -> StockSenseDetector:
    """Load and cache fine-tuned detector."""
    weight_path = settings.DEFAULT_MODEL_PATH
    if not weight_path.exists():
        weight_path = settings.BASE_DIR / settings.PRETRAINED_BACKBONE
    return StockSenseDetector(model_path=weight_path)

@st.cache_resource
def get_inventory_engine() -> InventoryEngine:
    return InventoryEngine()

@st.cache_resource
def get_history_manager() -> HistoryManager:
    return HistoryManager()

detector = get_detector()
engine = get_inventory_engine()
history_mgr = get_history_manager()

# ---------------------------------------------------------
# Navigation & Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shop-inventory.png", width=54)
    st.markdown("### **StockSense Pro**")
    st.caption("Retail Shelf Intelligence & Product Detection")
    st.markdown("---")

    nav_mode = st.radio(
        "Application Mode",
        options=[
            "📸 Shelf Analysis",
            "📋 Inspection History",
            "📊 Operations Summary",
            "🔬 Model & Technical Info"
        ]
    )

    st.markdown("---")
    st.markdown("#### ⚙️ **Detection Tuning**")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=settings.DEFAULT_CONF_THRESHOLD,
        step=0.05,
        help="Lower confidence reveals occluded items; higher confidence eliminates false positives."
    )

    iou_threshold = st.slider(
        "NMS IoU Threshold",
        min_value=0.20,
        max_value=0.75,
        value=settings.DEFAULT_IOU_THRESHOLD,
        step=0.05,
        help="Non-Maximum Suppression threshold to merge overlapping boxes."
    )

    st.markdown("---")
    st.markdown("#### 🎓 **Academic Attribution**")
    st.markdown(f"""
    - **Student:** {settings.STUDENT_NAME}
    - **Student ID:** {settings.STUDENT_ID}
    - **School:** {settings.SCHOOL_NAME}
    - **Course:** {settings.COURSE_TITLE} (FA 2)
    - **Programme:** {settings.PROGRAMME}
    - **Session:** `{settings.CANDIDATE_SESSION_NO}`
    """)

# ---------------------------------------------------------
# Top Banner
# ---------------------------------------------------------
st.markdown("""
<div class="app-banner">
    <div class="app-title">🛒 StockSense Pro — Visual Retail Shelf Intelligence</div>
    <div class="app-subtitle">
        Frontline shelf auditing, visible facing counting, and real-time replenishment guidance powered by fine-tuned YOLOv8 computer vision.
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# VIEW 1: SHELF ANALYSIS (PRIMARY OPERATIONAL WORKFLOW)
# =========================================================
if nav_mode == "📸 Shelf Analysis":
    st.markdown("""
    <div style="background-color: #EFF6FF; border-left: 4px solid #2563EB; border-radius: 6px; padding: 10px 14px; font-size: 0.85rem; color: #1E40AF; margin-bottom: 16px;">
        ℹ️ <b>Store Associate Notice:</b> StockSense Pro audits <b>visible product facings</b> in the camera frame.
        Zero detections indicate no items visible on the analysed shelf space; check backroom storage before assuming complete store out-of-stock.
    </div>
    """, unsafe_allow_html=True)

    # Step 1: Input Selection
    st.markdown('<div class="section-title">Step 1 — Select or Upload Shelf Image</div>', unsafe_allow_html=True)

    input_c1, input_c2, input_c3 = st.columns([1.5, 2, 1.5])
    with input_c1:
        source_mode = st.radio(
            "Image Source",
            ["Preloaded Benchmark Scenarios", "Upload Photo from Device"],
            horizontal=True
        )

    target_image_bgr = None
    shelf_location_tag = "General Grocery Shelf"

    if source_mode == "Preloaded Benchmark Scenarios":
        with input_c2:
            scenario_choice = st.selectbox(
                "Operational Scenario",
                [
                    "Aisle 6: Well-Stocked Multi-Tier Shelf (High Density)",
                    "Aisle 2: Low-Stock Display (Visible Shortages)",
                    "Aisle 3: Cold Drink Cooler (Bottles & Cans)",
                    "Aisle 5: Sparse / Depleted Shelf Surface"
                ]
            )
            sample_map = {
                "Aisle 6: Well-Stocked Multi-Tier Shelf (High Density)": (settings.SAMPLES_DIR / "sample_well_stocked.jpg", "Aisle 6 - Packaged Snacks & Bakery"),
                "Aisle 2: Low-Stock Display (Visible Shortages)": (settings.SAMPLES_DIR / "sample_low_stock.jpg", "Aisle 2 - Breakfast & Cartons"),
                "Aisle 3: Cold Drink Cooler (Bottles & Cans)": (settings.SAMPLES_DIR / "sample_beverages.jpg", "Aisle 3 - Beverage Cooler"),
                "Aisle 5: Sparse / Depleted Shelf Surface": (settings.SAMPLES_DIR / "sample_sparse_empty.jpg", "Aisle 5 - Promotional Endcap")
            }
            img_file_path, default_tag = sample_map[scenario_choice]
            shelf_location_tag = default_tag
            if img_file_path.exists():
                target_image_bgr = cv2.imread(str(img_file_path))
    else:
        with input_c2:
            uploaded_file = st.file_uploader(
                "Upload Shelf Image (JPG, JPEG, PNG, max 10MB)",
                type=["jpg", "jpeg", "png"]
            )
            if uploaded_file is not None:
                try:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    target_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                except Exception as e:
                    st.error(f"Image decode error: {e}")

    with input_c3:
        shelf_location_tag = st.text_input("Shelf Location Tag", value=shelf_location_tag)

    if target_image_bgr is None:
        st.info("Select an operational scenario above or upload an image to start shelf analysis.")
        st.stop()

    # Step 2: Execute Analysis
    st.markdown('<div class="section-title">Step 2 — Shelf Analysis & Facing Audit</div>', unsafe_allow_html=True)

    with st.spinner("Analyzing visible shelf facings with fine-tuned YOLOv8 detector..."):
        t0 = time.perf_counter()
        det_result = detector.detect(target_image_bgr, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        latency = det_result["latency_ms"]
        raw_detections = det_result["detections"]
        inv_eval = engine.evaluate_inventory(raw_detections)

    # Step 3: Executive Metric Cards (Results First)
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Visible Facings</div>
            <div class="metric-value">{inv_eval['total_visible_units']}</div>
            <div class="metric-subtext">Total products identified</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active Categories</div>
            <div class="metric-value">{inv_eval['active_categories']} / {inv_eval['total_supported_categories']}</div>
            <div class="metric-subtext">Present on shelf surface</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Low-Stock Categories</div>
            <div class="metric-value" style="color: {settings.STATUS_COLORS[settings.STATUS_LOW_STOCK]};">{inv_eval['low_stock_categories_count']}</div>
            <div class="metric-subtext">1 to 3 units remaining</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Empty / Out of Stock</div>
            <div class="metric-value" style="color: {settings.STATUS_COLORS[settings.STATUS_OUT_OF_STOCK]};">{inv_eval['out_of_stock_categories_count']}</div>
            <div class="metric-subtext">0 units detected</div>
        </div>
        """, unsafe_allow_html=True)

    with m5:
        # Determine low confidence items requiring manual review
        low_conf_items = [d for d in raw_detections if d["confidence"] < 0.38]
        health_color = (
            "#DC2626" if inv_eval['overall_shelf_health'] == "CRITICAL"
            else "#D97706" if inv_eval['overall_shelf_health'] == "WARNING"
            else "#16A34A"
        )
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Shelf Condition</div>
            <div class="metric-value" style="color: {health_color}; font-size: 1.45rem;">{inv_eval['overall_shelf_health']}</div>
            <div class="metric-subtext">Review Queue: {len(low_conf_items)} items</div>
        </div>
        """, unsafe_allow_html=True)

    # Step 4: Dual Visual Inspection
    st.markdown('<div class="section-title">Step 3 — Visual Shelf Inspection (Dual View)</div>', unsafe_allow_html=True)
    
    v_opt1, v_opt2, v_opt3 = st.columns([1, 1, 2])
    with v_opt1:
        show_labels = st.checkbox("Show Category Chips", value=True)
    with v_opt2:
        show_conf = st.checkbox("Show Confidence Scores", value=True)
    with v_opt3:
        category_filter = st.multiselect("Filter Visible Categories", options=settings.CLASS_NAMES, default=settings.CLASS_NAMES)

    filtered_dets = [d for d in raw_detections if d["class_name"] in category_filter]
    annotated_bgr = render_detections(
        target_image_bgr,
        filtered_dets,
        confidence_threshold=conf_threshold,
        selected_categories=category_filter,
        show_labels=show_labels,
        show_conf=show_conf
    )

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown("**Original Shelf Image**")
        st.image(cv2.cvtColor(target_image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with img_col2:
        st.markdown(f"**StockSense AI Detection ({len(filtered_dets)} Items Annotated)**")
        st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Step 5: Product Breakdown & Action Priority Queue
    st.markdown('<div class="section-title">Step 4 — Product Breakdown & Next Operational Actions</div>', unsafe_allow_html=True)

    table_col, action_col = st.columns([3, 2])

    with table_col:
        st.markdown("**Category Inventory Audit**")
        df_inv = engine.export_to_dataframe(inv_eval)
        st.dataframe(df_inv, use_container_width=True, hide_index=True)

    with action_col:
        st.markdown("**Action Priority Queue (Store Staff Guidance)**")
        
        # 1. Critical Restock
        critical_items = [it for it in inv_eval["items"] if it["status"] == settings.STATUS_OUT_OF_STOCK]
        for it in critical_items:
            st.markdown(f"""
            <div class="action-box action-critical">
                <b>🔴 Urgent Restock — {it['category']}:</b> Zero visible units detected. Audit backroom storage and dispatch replenishment immediately.
            </div>
            """, unsafe_allow_html=True)

        # 2. Warning Restock
        warning_items = [it for it in inv_eval["items"] if it["status"] == settings.STATUS_LOW_STOCK]
        for it in warning_items:
            st.markdown(f"""
            <div class="action-box action-warning">
                <b>🟡 Replenish Soon — {it['category']}:</b> Only {it['detected_count']} units remaining. Queue restock before next peak footfall.
            </div>
            """, unsafe_allow_html=True)

        # 3. Model Uncertainty / Manual Review Queue
        if low_conf_items:
            st.markdown(f"""
            <div class="action-box action-review">
                <b>🔍 Model Uncertainty Alert ({len(low_conf_items)} Items):</b> Some detections exhibited confidence near threshold ({conf_threshold}). Verify shelf visually to confirm if items are partially hidden or occluded.
            </div>
            """, unsafe_allow_html=True)

        # 4. Healthy stock
        healthy_items = [it for it in inv_eval["items"] if it["status"] == settings.STATUS_SUFFICIENT]
        if healthy_items and not critical_items and not warning_items:
            st.markdown("""
            <div class="action-box action-normal">
                <b>🟢 Optimal Inventory:</b> All monitored product categories have sufficient visible depth. Maintain standard product rotation and alignment.
            </div>
            """, unsafe_allow_html=True)

    # Step 6: Save & Export Options
    st.markdown('<div class="section-title">Step 5 — Save Audit & Export Report</div>', unsafe_allow_html=True)

    btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)

    with btn_c1:
        if st.button("💾 Save Inspection to History", use_container_width=True, type="primary"):
            insp_id = history_mgr.record_inspection(
                shelf_tag=shelf_location_tag,
                inventory_result=inv_eval,
                confidence_threshold=conf_threshold,
                notes=f"Audit completed with {inv_eval['total_visible_units']} units identified."
            )
            st.success(f"Inspection recorded successfully as `{insp_id}`! View in Inspection History tab.")

    with btn_c2:
        csv_data = engine.export_csv(inv_eval)
        st.download_button(
            label="📥 Download Audit CSV",
            data=csv_data,
            file_name=f"stocksense_audit_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with btn_c3:
        # Encode annotated image to JPG bytes for direct download
        is_success, buffer = cv2.imencode(".jpg", annotated_bgr)
        if is_success:
            st.download_button(
                label="🖼️ Download Annotated Image",
                data=buffer.tobytes(),
                file_name=f"stocksense_annotated_{time.strftime('%Y%m%d_%H%M%S')}.jpg",
                mime="image/jpeg",
                use_container_width=True
            )

    with btn_c4:
        if st.button("🔄 Clear & Start New Audit", use_container_width=True):
            st.rerun()


# =========================================================
# VIEW 2: INSPECTION HISTORY (STORE MANAGER AUDIT LOG)
# =========================================================
elif nav_mode == "📋 Inspection History":
    st.markdown('<div class="section-title">📋 Store Shelf Inspection History & Audit Trail</div>', unsafe_allow_html=True)
    st.caption("Review completed shelf audits from the current and previous shifts. Track shelf condition over time.")

    all_inspections = history_mgr.get_all_inspections()
    if not all_inspections:
        st.info("No inspection records found. Complete and save a shelf audit in the Shelf Analysis tab.")
    else:
        df_hist = history_mgr.get_history_dataframe()

        h_kpi1, h_kpi2, h_kpi3 = st.columns(3)
        h_kpi1.metric("Total Completed Inspections", len(all_inspections))
        critical_count = sum(1 for i in all_inspections if i.get("overall_shelf_health") == "CRITICAL")
        h_kpi2.metric("Critical Health Alerts", critical_count)
        avg_units = int(np.mean([i.get("total_visible_units", 0) for i in all_inspections]))
        h_kpi3.metric("Average Facings per Shelf", f"{avg_units} units")

        st.markdown("**Historical Inspection Audit Log**")
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        exp_c1, exp_c2 = st.columns([1, 3])
        with exp_c1:
            st.download_button(
                label="📥 Export Complete Audit Log (CSV)",
                data=history_mgr.export_csv(),
                file_name=f"stocksense_full_audit_log_{time.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )


# =========================================================
# VIEW 3: OPERATIONS SUMMARY (STORE MANAGER & ANALYST)
# =========================================================
elif nav_mode == "📊 Operations Summary":
    st.markdown('<div class="section-title">📊 Store Operations & Recurring Shortage Analytics</div>', unsafe_allow_html=True)
    st.caption("Strategic retail analytics identifying recurring problem categories and replenishment bottlenecks.")

    shortages = history_mgr.get_recurring_shortages()
    
    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        # Shortage Frequency Bar Chart
        categories = list(shortages.keys())
        counts = list(shortages.values())
        fig_short = px.bar(
            x=counts,
            y=categories,
            orientation='h',
            labels={"x": "Times Flagged as Low or Out of Stock", "y": "Product Category"},
            title="<b>Recurring Shelf Depletion Frequency by Category</b>",
            color=counts,
            color_continuous_scale="Reds"
        )
        fig_short.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", height=340)
        st.plotly_chart(fig_short, use_container_width=True)

    with chart_c2:
        # Shelf Health Distribution
        all_inspections = history_mgr.get_all_inspections()
        health_counts = Counter([i.get("overall_shelf_health", "UNKNOWN") for i in all_inspections])
        fig_h = px.pie(
            names=list(health_counts.keys()),
            values=list(health_counts.values()),
            title="<b>Shift Shelf Health Condition Composition</b>",
            color=list(health_counts.keys()),
            color_discrete_map={"OPTIMAL": "#16A34A", "WARNING": "#D97706", "CRITICAL": "#DC2626"}
        )
        fig_h.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", height=340)
        st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("""
    ### 💡 Operational Restocking Directives for Store Managers
    1. **High-Frequency Depletion Items:** Categories with recurring shortages require safety stock adjustment in the replenishment ERP.
    2. **Peak Footfall Pre-Stocking:** Schedule shelf facing maintenance 45 minutes before morning (10:00 AM) and evening (6:00 PM) footfall peaks.
    3. **Floor Staff Routing:** Prioritize Aisle 2 (Breakfast) and Aisle 3 (Cold Beverages) for frequent visual inspection rounds.
    """)


# =========================================================
# VIEW 4: MODEL & TECHNICAL INFO (RETAIL OPERATIONS ANALYST)
# =========================================================
elif nav_mode == "🔬 Model & Technical Info":
    st.markdown('<div class="section-title">🔬 Model Architecture, Evaluation Benchmark & Scientific Provenance</div>', unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs([
        "Evaluation Benchmark (Test Set)",
        "8 Realistic Scenario Tests",
        "Dataset Research (RPC vs Shelf)",
        "Optical Failure Modes & Mitigations"
    ])

    with t1:
        # Load genuine evaluation results
        eval_path = settings.ASSETS_DIR / "evaluation_summary.json"
        if eval_path.exists():
            with open(eval_path) as f:
                ev = json.load(f)
            det = ev["overall_detection_metrics"]
            cnt = ev["overall_counting_metrics"]

            st.markdown("#### **Quantitative Benchmark Results (Held-Out Test Set)**")
            st.caption(f"Evaluated on {ev['test_images_count']} unseen supermarket shelf scenes containing {ev['total_ground_truth_instances']:,} annotated product units.")

            tc1, tc2, tc3, tc4 = st.columns(4)
            tc1.metric("Precision (P)", f"{det['precision']:.3f}")
            tc2.metric("Recall (R)", f"{det['recall']:.3f}")
            tc3.metric("mAP@0.50", f"{det['mAP50']:.3f}")
            tc4.metric("Mean Counting Accuracy", f"{cnt['mean_counting_accuracy_pct']:.1f}%")

            st.markdown("#### **Per-Class Detection & Counting Performance**")
            p_rows = []
            for c in settings.CLASS_NAMES:
                d = ev["per_class_detection"][c]
                k = ev["per_class_counting"][c]
                p_rows.append({
                    "Product Category": c,
                    "Precision": f"{d['precision']:.3f}",
                    "Recall": f"{d['recall']:.3f}",
                    "mAP@0.5": f"{d['mAP50']:.3f}",
                    "Counting MAE": f"{k['MAE']:.2f} units",
                    "Counting Accuracy (%)": f"{k['counting_accuracy_pct']:.1f}%"
                })
            st.dataframe(pd.DataFrame(p_rows), use_container_width=True, hide_index=True)

            pr_chart = settings.EDA_DIR / "test_pr_metrics_by_class.png"
            if pr_chart.exists():
                st.image(str(pr_chart), caption="Figure: Per-Class Precision, Recall, and mAP@0.5 on Held-Out Test Set", use_container_width=True)

    with t2:
        st.markdown("#### **8 Realistic Operational Scenario & Edge-Case Audits**")
        st.caption("Verification of model resilience across varied supermarket lighting, severe occlusion, and malformed inputs.")

        matrix_path = settings.EDA_DIR / "scenario_testing_matrix.json"
        if matrix_path.exists():
            with open(matrix_path) as f:
                scenarios = json.load(f)
            sc_rows = []
            for sc in scenarios:
                sc_rows.append({
                    "Scenario ID": sc["scenario_id"],
                    "Operational Scenario": sc["name"],
                    "Expected Behavior": sc["expected_behavior"],
                    "Detections": sc["observed_detections"],
                    "Latency": f"{sc['latency_ms']} ms",
                    "Verdict": sc["status"],
                    "Mitigation Strategy": sc["operational_mitigation"]
                })
            st.dataframe(pd.DataFrame(sc_rows), use_container_width=True, hide_index=True)

    with t3:
        st.markdown(f"""
        #### **Mandatory Dataset Research & Critique**
        
        1. **Primary Dataset: Kaggle Retail Product Checkout (RPC)**
           - **Source:** `{settings.PRIMARY_DATASET_URL}`
           - **Paper:** Wei et al., *RPC: A Large-Scale Retail Product Checkout Dataset*, arXiv:1901.07249 (2019).
           - **Investigation:** 200 categories, 73,602 annotations, CC-BY-NC-SA 4.0.
           - **Task Disparity:** RPC captures top-down automatic checkout tray scenes with horizontally piled items.
           
        2. **Supplementary Benchmark: Multi-Tier Retail Shelf Dataset**
           - **Source:** Kaggle / Roboflow Universe Shelf Detection Benchmark (CC BY 4.0).
           - **Scale:** 210 unique supermarket shelf scenes with 120,498 annotated bounding boxes.
           - **Leakage Prevention:** Grouped by parent scene hash into 70% Train, 15% Val, 15% Test with **0% cross-split leakage**.
        """)
        comp_img = settings.EDA_DIR / "checkout_vs_shelf_comparison.png"
        if comp_img.exists():
            st.image(str(comp_img), caption="Technical Disparity: Top-Down Checkout (RPC) vs Frontal Retail Shelf Monitoring", use_container_width=True)

    with t4:
        st.markdown("""
        #### **Optical Failure Modes & Employee Mitigations**
        
        - **Depth Occlusion:** Rear-row merchandise blocked by front-facing items. *Mitigation:* Confidence slider allows dialing sensitivity for partially visible rear packaging.
        - **Overhead Fluorescent Glare:** High-reflectance plastics (water bottles, transparent bags). *Mitigation:* Training included HSV saturation and brightness jitter.
        - **Dense Packaging Clutter:** Adjacent identical items suppressed by NMS. *Mitigation:* IoU threshold tuned to 0.45.
        - **Shelf Edge Obstructions:** Price rails and barcodes intersecting product bottoms. Handled via coordinate clamping.
        """)

st.markdown("---")
st.caption(f"StockSense Pro | Developed by {settings.STUDENT_NAME} (ID: {settings.STUDENT_ID}) | Aspee Nutan Academy | IBCP Artificial Intelligence")
