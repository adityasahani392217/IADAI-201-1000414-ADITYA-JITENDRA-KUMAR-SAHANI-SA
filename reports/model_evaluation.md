# StockSense Pro — Technical Model Evaluation Report
**Project:** StockSense Pro: AI-Powered Retail Shelf Intelligence & Product Detection System  
**Author:** Aditya Jitendra Kumar Sahani (Student ID: 1000414)  
**Institution:** Aspee Nutan Academy | IBCP Artificial Intelligence Grade 12  
**Evaluation Date:** 2026-10-21 09:48:31  

---

## 1. Executive Summary & Benchmark Results

The fine-tuned **StockSense Pro YOLOv8** model was evaluated on the strictly held-out test split comprising **31 unseen multi-tier retail shelf scenes** containing **4,520 annotated product units**.

### Global Detection & Counting Performance
| Metric | Value | Technical Meaning |
| :--- | :--- | :--- |
| **Precision (P)** | **0.5967** | Proportion of detected product bounding boxes that correctly matched a true item. |
| **Recall (R)** | **0.2016** | Proportion of actual shelf items correctly detected by the model. |
| **mAP @ 0.5** | **0.1566** | Mean Average Precision across all categories at standard IoU threshold 0.50. |
| **mAP @ 0.5:0.95** | **0.0717** | COCO primary metric measuring spatial localization across IoU range [0.50, 0.95]. |
| **Mean Absolute Error (MAE)** | **14.99 units** | Average deviation between visible ground-truth count and predicted count per category. |
| **Root Mean Squared Error (RMSE)** | **46.34 units** | Penalized quadratic deviation highlighting variance on dense shelf facings. |
| **Mean Counting Accuracy** | **79.8%** | Normalized counting accuracy: $\text{Acc} = \max\left(0, 1 - \frac{|Y_{true} - Y_{pred}|}{\max(Y_{true}, 1)}\right) \times 100\%$. |
| **Inference Latency** | **17.9 ms** | Average CPU inference latency per shelf image on Intel Core i7 (over 20 FPS). |

---

## 2. Per-Class Performance Breakdown

| Product Category | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Counting MAE | Counting Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bottle** | 0.844 | 0.551 | 0.524 | 0.290 | 2.03 | 97.7% |
| **Box** | 0.260 | 0.210 | 0.061 | 0.033 | 10.16 | 77.8% |
| **Bread** | 0.000 | 0.000 | 0.000 | 0.000 | 0.71 | 96.8% |
| **Can** | 1.000 | 0.005 | 0.005 | 0.003 | 12.19 | 74.3% |
| **Pack** | 0.769 | 0.041 | 0.041 | 0.019 | 7.45 | 90.7% |
| **Snack** | 0.704 | 0.126 | 0.114 | 0.048 | 21.06 | 58.3% |
| **Vegetable** | 0.600 | 0.479 | 0.351 | 0.109 | 51.32 | 63.0% |

---

## 3. Error Analysis and Optical Failure Modes

In accordance with rigorous computer vision methodology, model errors on the held-out test set were categorized into four primary typologies:

1. **Occlusion & Depth Masking:**
   - Products situated in rear shelf tiers are frequently partially occluded by front facings. While the human eye can infer the presence of a second row, bounding boxes on partially visible products exhibit lower confidence scores.
   - *Mitigation:* Configurable confidence threshold (0.25 - 0.45) in the Streamlit UI allows store managers to balance precision against occlusion recall.

2. **Specular Reflection & Glare:**
   - Plastic bottles and beverage packaging exhibit harsh reflections under overhead fluorescent supermarket strip lighting, occasionally breaking continuous edge gradients.
   - *Mitigation:* Data augmentation including brightness jitter and HSV shifts during training.

3. **High-Density Clutter & Small Scale:**
   - Dense multi-facing displays (such as small snack pouches or canned beverages) create tightly clustered bounding boxes where Non-Maximum Suppression (NMS) may suppress adjacent identical facings if the IoU threshold is set too conservatively.
   - *Mitigation:* IoU threshold tuned to 0.45 with adjustable user control.

4. **Shelf Edge Obstructions:**
   - Price tags, shelf rail dividers, and barcode channel strips can intersect lower bounding box boundaries. The data preprocessing pipeline applies geometric clipping and coordinate clamping to eliminate degenerate box proposals.

---

## 4. Operational Inventory Rules & Action Directives

StockSense Pro translates visible instance counts into real-time operational alerts:

- **0 Visible Units (🔴 Critical Alert):**
  *Status:* No Detections / Empty Shelf Facing  
  *Action Directive:* Immediate backroom inventory audit and shelf restock dispatch.
  
- **1 – 3 Visible Units (🟡 Warning Alert):**
  *Status:* Low Visible Stock  
  *Action Directive:* Queue restock order before peak footfall hours.

- **≥ 4 Visible Units (🟢 Normal Status):**
  *Status:* Sufficient Stock  
  *Action Directive:* Healthy inventory volume; maintain routine facing and rotation.

*Important Operational Caveat:* StockSense Pro measures **visible facings** in the captured camera view, not total ERP inventory. Zero detections do not prove that warehouse stock is exhausted.
