# ParkVision AI — Video Presentation Script & Demonstration Walkthrough

**CRS Artificial Intelligence Summative Assessment: Machine Learning & Deep Learning (60 Marks)**  
- **Student Name:** Aditya Jitendra Kumar Sahani  
- **Student ID:** 1000414  
- **Candidate Session Number:** `[Candidate Registration / Session Number: To be provided]`  
- **School:** Aspee Nutan Academy  
- **Programme:** IBCP Artificial Intelligence / Grade 12  
- **Target Repository:** `IADAI-201-1000414-ADITYA-JITENDRA-KUMAR-SAHANI-SA`  
- **Total Duration:** Exactly 5 Minutes (00:00 – 05:00)

---

## Scene 1: Introduction, Problem Scope & User Personas (00:00 – 00:45)

**Visual on Screen:**  
Start on the Streamlit dashboard on the **Overview & Architecture** page. Point out the top header badge (`CRS AI SUMMATIVE ASSESSMENT — 60 MARKS`), the clean editorial light theme, and the live status indicators.

**Spoken Narration:**  
> "Good day, examiners. My name is Aditya Jitendra Kumar Sahani, student ID 1000414 from Aspee Nutan Academy, Grade 12 IBCP Artificial Intelligence. Today, I am proud to present **ParkVision AI: A Real-Time Smart Parking Intelligence Platform**, built for my Machine Learning and Deep Learning Summative Assessment.
>
> Urban cruising for parking accounts for over 30% of downtown traffic congestion, while facility managers struggle with costly, breakable in-ground loop sensors. ParkVision AI transforms ubiquitous optical CCTV feeds into an intelligent, autonomous space management network.
>
> Rather than a generic vehicle counter, ParkVision AI explicitly models three real-world stakeholders:
> 1. **The Parking Facility Operator**, who requires real-time occupancy rates, automated capacity thresholds, and gate redirection logic;
> 2. **Field Security Staff**, who receive instant alerts for boundary-straddling vehicles or unauthorized parking in EV and accessible stalls;
> 3. **The Commercial Operations Analyst**, who requires audit manifests, dwell-time analytics, and CSV/PDF compliance exports."

---

## Scene 2: Decoupled Computer Vision Pipeline & Mathematical Formulation (00:45 – 01:45)

**Visual on Screen:**  
Scroll to the **Operational Workflow** and **Rubric Coverage** section. Show the latency breakdown and radar performance charts.

**Spoken Narration:**  
> "A frequent failure in smart parking software is treating the system as a monolithic black-box classifier. If a vehicle's roof projects across adjacent stalls due to oblique camera angles, naive bounding box IoU generates false positives.
>
> ParkVision AI solves this through a rigorous, two-stage decoupled pipeline:
> - **Stage 1: Deep Learning Vehicle Localization.** We deploy the lightweight YOLOv8 Nano architecture, accelerated locally on an NVIDIA GeForce RTX 3050 Laptop GPU with PyTorch CUDA 12.6. Warmup routines eliminate cold-start lag, achieving an ultra-fast inference latency of just 5.6 milliseconds—exceeding 47 frames per second.
> - **Stage 2: Deterministic Spatial Geometry & Temporal Smoothing.** Instead of bounding box IoU, we calculate **Intersection-over-Area (IoA)** relative to normalized four-point stall polygons. Furthermore, we compute the vehicle's bottom-center ground contact point—where tires meet pavement—to anchor the vehicle to its true physical bay.
> - Finally, greedy bipartite matching resolves multi-vehicle overlaps, and a 5-frame sliding window with hysteresis filters out pedestrians and lighting flicker."

---

## Scene 3: Live Monitor Interactive Demonstration (01:45 – 03:00)

**Visual on Screen:**  
Switch to the **Live Monitor** page. Demonstrate the control bar:
1. Select **Built-in Demo Loop** and drag the timeline scrubber from frame 10 to frame 50. Show the car moving down the lane, pulling into Bay P05, and watch the status badge transition from green Available to red Occupied.
2. Switch to **Static Test Image** and load `real_parking_1.jpg`. Show the semi-transparent colored polygons, the vehicle bounding boxes, and the live manifest table on the right.
3. Show the live KPI cards updating dynamically: 3 Available, 3 Occupied, 50% Occupancy Rate, 5.6 ms Latency.

**Spoken Narration:**  
> "Let us navigate to the **Live Monitor** page. Notice the five ingestion modes: Built-in Demo Loop, Static Test Image, Video Upload, Live Webcam, and RTSP Stream.
>
> In our live demo loop, we observe a vehicle entering from the lane. As it drives past, notice that Bay P05 starts in emerald green as **Available**. When the vehicle maneuvers into the bay, our IoA overlap climbs: passing 20%, triggering the amber **Uncertain** alert, and once fully parked with IoA exceeding 75%, our 5-frame hysteresis confirms the transition to crimson red **Occupied**.
>
> Simultaneously, the telemetry panel updates in real time. We see three available bays, three occupied bays, our instantaneous latency of 5.6 milliseconds, and our active bay manifest detailing each slot's classification, section, and vehicle class. When capacity exceeds 85%, automated critical alerts notify facility operations."

---

## Scene 4: Geometry Layout Editor & Business Analytics (03:00 – 04:00)

**Visual on Screen:**  
1. Navigate to the **Parking Layout Editor** page. Show the calibrated polygon overlays. Click the 'Presets' tab, switch to 'Underground Garage (B1)', and then back to 'Commercial Retail (CCTV-04)'. Show the JSON export download button.
2. Navigate to the **Analytics & Intelligence** page. Show the diurnal 24-hour occupancy curve, the bay dwell time table, and the historical event log.

**Spoken Narration:**  
> "Next, we explore the **Parking Layout Editor**. Because camera installations vary across facilities, ParkVision AI decouples geometry from code. Operators can calibrate new stall polygons directly using normalized coordinates from zero to one. These configurations can be saved, edited, or exported as portable JSON files that load seamlessly across different camera streams.
>
> Moving to **Analytics & Intelligence**, facility executives gain longitudinal operational insights. Here we see our diurnal occupancy profile showing peak morning and afternoon capacity spikes against warning thresholds. Below, our bay dwell-time analytics identify average parking durations across standard, EV, and accessible stalls, revealing which spaces experience the highest turnover."

---

## Scene 5: Verification, Automated Testing, PDF Report & Conclusion (04:00 – 05:00)

**Visual on Screen:**  
1. Navigate to **Audit Reports & Export**. Click the download buttons for CSV manifests and download the formal PDF report. Open or show the 4-page PDF document.
2. Navigate to **Model Evaluation & Benchmarks**. Point out the confusion matrix (100% precision, 100% recall), radar chart, and ablation study table.
3. Show terminal with all 12 Pytest tests passing.

**Spoken Narration:**  
> "Under **Audit Reports & Export**, operators can generate timestamped CSV manifests and download our formal, publication-grade 4-page technical PDF report. Built with ReportLab's Platypus engine and NumberedCanvas, this document contains complete mathematical formulations, candidate declarations, and evaluation tables matching all seven assessment criteria.
>
> On the **Model Evaluation** page, our empirical benchmark across multi-camera test sets proves the superiority of our approach: achieving 100% precision, 100% recall, and zero flicker across evaluated bays. Our ablation table highlights why combining IoA with ground contact and hysteresis outperforms standard bounding box IoU by over 33%.
>
> Our comprehensive Pytest suite confirms 100% pass rates across polygon geometry, bipartite matching, and temporal queues. All code, model weights, sample media, and documentation are committed cleanly to my official Summative Assessment repository.
>
> Thank you for your time and evaluation."
