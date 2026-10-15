# RoadSense AI — Comprehensive Engineering & Operational Guide
**Autonomous Multimodal Road Distress Detection, Spatial Memory & Municipal Asset Management Platform**

*Official Technical Reference, Architecture Specification & Deployment Manual*  
*Target Geography: Chhatrapati Sambhajinagar (Aurangabad), Maharashtra, India*

---

## Table of Contents
1. [Executive Overview & Problem Statement](#1-executive-overview--problem-statement)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Technology Stack & Dependency Inventory](#3-technology-stack--dependency-inventory)
4. [Sensor Waveform Science & Kinematics (Bump vs. Pothole)](#4-sensor-waveform-science--kinematics-bump-vs-pothole)
5. [The 30 Kinematic Feature Engineering Pipeline](#5-the-30-kinematic-feature-engineering-pipeline)
6. [Computer Vision & YOLO Object Detection Pipeline](#6-computer-vision--yolo-object-detection-pipeline)
7. [Geospatial Intelligence & Road Memory Engine](#7-geospatial-intelligence--road-memory-engine)
8. [Google Maps & Live Street Footage Integration](#8-google-maps--live-street-footage-integration)
9. [Municipal Road Authority Portal (CSMC / PWD Maharashtra)](#9-municipal-road-authority-portal-csmc--pwd-maharashtra)
10. [Maharashtra PWD DSR Material & Repair Cost Calculator](#10-maharashtra-pwd-dsr-material--repair-cost-calculator)
11. [Driver Proximity Radar & Heads-Up Display (HUD)](#11-driver-proximity-radar--heads-up-display-hud)
12. [Step-by-Step Installation & Execution Guide](#12-step-by-step-installation--execution-guide)
13. [API Endpoints Directory](#13-api-endpoints-directory)
14. [Testing & Quality Assurance](#14-testing--quality-assurance)

---

## 1. Executive Overview & Problem Statement

Road surface degradation (potholes, structural cracking, unmarked speed breakers, sub-base collapses) accounts for over **20,000 traffic accidents** and billions of rupees in vehicular damage, fuel inefficiencies, and logistical delays across urban corridors annually. Traditional road inspection methods rely on slow, expensive manual surveys or specialized laser profiler vans costing tens of lakhs of rupees.

**RoadSense AI** bridges this gap through a high-throughput, low-cost, multi-modal edge AI architecture:
1. **Smartphone IMU Telemetry (Edge Crowdsourcing):** High-frequency 3-axis Accelerometer ($a_x, a_y, a_z$) and 3-axis Gyroscope ($\omega_x, \omega_y, \omega_z$) recordings classified into road anomalies using a trained Random Forest model.
2. **Computer Vision (YOLO):** Real-time dashcam vision running on an RDD2022-trained YOLO model for multi-class visual pavement distress detection.
3. **Spatial Road Memory Engine:** Deduplicates, clusters, and tracks chronic recurring hazards using the Haversine metric.
4. **Municipal Public Works Portal (CSMC / PWD):** Triage system for Chhatrapati Sambhajinagar Municipal Corporation with automated Maharashtra PWD District Schedule of Rates (DSR) asphalt cost estimation in Indian Rupees (₹ INR) and work order generation.
5. **Driver Navigation HUD:** Early proximity warnings advising safe braking speeds before entering hazardous zones.

---

## 2. End-to-End System Architecture

RoadSense AI uses a decoupled **Client-Server-Model** architecture:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION STREAMS                          │
├───────────────────────────────────┬────────────────────────────────────┤
│  Smartphone IMU CSV (100 Hz)      │  Vehicle Dashcam Video / Camera    │
│  Acc (X,Y,Z), Gyro (X,Y,Z), GPS   │  Frames (640x640 RGB)              │
└─────────────────┬─────────────────┴──────────────────┬─────────────────┘
                  │                                    │
                  ▼                                    ▼
┌───────────────────────────────────┐┌───────────────────────────────────┐
│     FASTAPI BACKEND (Port 8000)   ││  ML INFERENCE ENGINES             │
│  - REST Endpoints (/api/...)      ││  1. SensorPredictor:              │
│  - 30-Kinematic Feature Extractor ││     - roadsense_model.joblib      │
│  - Haversine Spatial Memory Engine││     - 30 Kinematic Features       │
│  - SQLAlchemy ORM Data Layer      ││  2. YOLOPredictor:                │
│  - SQLite Database Storage        ││     - best.pt (RDD2022 weights)   │
└─────────────────┬─────────────────┘└─────────────────┬─────────────────┘
                  │                                    │
                  ▼                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 STREAMLIT COMMAND DASHBOARD (Port 8501)                │
├────────────────────────────────────────────────────────────────────────┤
│  1. 📊 Executive Command Center: Network Health PQI, Spatial Map      │
│  2. 📲 Sensor Waveform Analyzer: Multi-axis plots, 30 features, Map    │
│  3. 📈 Waveform Science: Bump vs Pothole Physics, Jerk, Pitch          │
│  4. 🏛️ Municipal Authority: CSMC Wards, Work Orders, PWD DSR Cost (₹)  │
│  5. 🚗 Driver Navigation: Radar HUD, Google Street View, Alert Sim     │
│  6. 🔌 API Console: REST Documentation, JSON payload tester            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack & Dependency Inventory

### Core Languages & Runtime
- **Python 3.11+**: Core runtime environment.
- **SQLite 3**: Embedded transactional relational storage.

### Backend Infrastructure
- **FastAPI 0.110+**: Asynchronous, high-performance REST API backend.
- **Uvicorn 0.28+**: Lightning-fast ASGI web server implementation.
- **SQLAlchemy 2.0+**: Enterprise Object Relational Mapper for database modeling.
- **Pydantic V2**: Robust request/response schema validation and serialisation.
- **python-multipart**: Multipart/form-data parsing for CSV and video uploads.

### Machine Learning & Data Science
- **Scikit-Learn 1.4+**: Inference pipeline for serialized Random Forest classifier.
- **Joblib 1.3+**: Binary model bundle serialization and feature registry loading.
- **Ultralytics YOLO**: Deep neural network for real-time pavement object detection.
- **OpenCV (cv2) 4.13+**: Dashcam image preprocessing, color conversion, overlay rendering.
- **Pillow (PIL) 11.1+**: Image manipulation, typography rendering, synthetic frame creation.
- **NumPy 1.26+ & Pandas 2.2+**: High-dimensional vector manipulation and time-series aggregation.
- **SciPy 1.12+**: Digital signal processing, derivative estimation, and peak detection (`find_peaks`).

### Frontend Visualization & Geospatial GIS
- **Streamlit 1.40+ / 1.64+**: Interactive web dashboard framework.
- **Plotly Express & Plotly Graph Objects 6.0+ / 7.1+**: Dynamic, synchronized interactive multi-trace charts (compatible with `scatter_map`).
- **Folium 0.16+ & streamlit-folium 0.20+**: Leaflet-powered GIS mapping engine with Google Maps layer integration.

---

## 4. Sensor Waveform Science & Kinematics (Bump vs. Pothole)

Distinguishing between an **elevation crest** (Speed Bump) and a **depression cavity** (Pothole) is accomplished through vertical acceleration, jerk derivative, and pitch rate dynamics:

```
      SPEED BUMP TRAVERSAL                           POTHOLE COLLISION
      (Elevation Crest Anomaly)                  (Depression Cavity Anomaly)
 
 Az (m/s²)                                  Az (m/s²)
    ▲                                          ▲
    │         ┌───┐ Ramp Compression           │      ▲ Primary Rim Strike (+28 m/s²)
 +17│        /     \ (+16.5 m/s²)           +28│      │
 +14│       /       \                       +21│      │
 +10├───---/---------\----- 1G Baseline     +14│      │       ┌──┐ Damped Ringing
    │     /           \                     +10├───┐  │  ┌───/    \─── 1G Baseline
  +6│    /             \                     +6│   │  │ /
    │   /               \ Crest Unload         │   └──┘/
  +2└───                 ─────────           +2└───▼──────────────────────► Time (s)
    0─────────────────────────────► Time (s)   0   Freefall Dip (~3.2 m/s²)
```

### 1. Speed Bump (Elevation Crest)
- **Initiation:** Vehicle tire strikes an upward incline. Suspension springs compress smoothly over $0.25$–$0.45\text{ s}$.
- **Vertical Accel ($a_z$):** Smooth positive rise above baseline gravity ($1G \approx 9.81\text{ m/s}^2$), peaking between $+14$ and $+18\text{ m/s}^2$. **Crucially, there is no prior freefall drop.**
- **Jerk ($da_z/dt$):** Moderate derivative gradient ($\Delta a_z / \Delta t \le 65\text{ m/s}^3$).
- **Pitch Dynamics ($\omega_x$):** Front axle ascends first ($\dot{\theta} > 0$), followed by rear axle ascent ($\dot{\theta} < 0$).
- **Frequency Profile:** Low-frequency dominant mechanical oscillation ($1$–$5\text{ Hz}$).

### 2. Pothole (Depression Cavity)
- **Phase A (Freefall Drop):** Tire rolls over the edge into empty air. Suspension rapidly expands downward; accelerometer experiences brief near-zero-g freefall where $a_z$ plunges down to $2$–$5\text{ m/s}^2$.
- **Phase B (Rim Strike Impact):** Wheel hits the opposing vertical asphalt lip. Suspension bottoms out violently against mechanical bump stops. Produces an extreme acceleration spike ($+22$ to $+35+\text{ m/s}^2$).
- **Jerk ($da_z/dt$):** Severe shock gradient ($\Delta a_z / \Delta t > 150$–$350+\text{ m/s}^3$).
- **Phase C (Harmonic Ringing):** Structural chassis reverberation produces $4$–$8$ high-frequency harmonic peaks (`peak_count` $\ge 5$).
- **Pitch Dynamics ($\omega_x$):** Sudden downward nose dive ($\dot{\theta} \ll 0$) followed by violent recoil snap ($\dot{\theta} \gg 0$).

---

## 5. The 30 Kinematic Feature Engineering Pipeline

Every $1.0$-second window (minimum 50 samples at $\ge 50\text{ Hz}$) extracts exactly **30 mathematical features**:

### A. Per-Channel Statistics (24 Features)
Computed independently across 6 channels:
1. `accelerometer_x` (Lateral cornering force)
2. `accelerometer_y` (Longitudinal braking / acceleration)
3. `accelerometer_z` (Vertical road normal shock)
4. `gyroscope_x` (Pitch rate: nose dive / climb)
5. `gyroscope_y` (Roll rate: side-to-side camber tilt)
6. `gyroscope_z` (Yaw rate: directional heading swing)

For each channel $C \in \{a_x, a_y, a_z, \omega_x, \omega_y, \omega_z\}$, four statistical moments are extracted:
1. **Standard Deviation (`std`):**
   $$\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2}$$
2. **Peak-to-Peak Range (`range`):**
   $$\text{Range} = \max(x) - \min(x)$$
3. **Maximum Absolute Amplitude (`max_abs`):**
   $$\text{MaxAbs} = \max(|x_i|)$$
4. **Root Mean Square Energy (`rms`):**
   $$\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}x_i^2}$$

### B. Composite Shock & Jerk Dynamics (6 Features)
1. **Total Acceleration Magnitude:**
   $$M(t) = \sqrt{a_x(t)^2 + a_y(t)^2 + a_z(t)^2}$$
   - `magnitude_std`: Standard deviation of spatial G-force.
   - `magnitude_max`: Absolute peak G-force magnitude.
   - `magnitude_rms`: Root mean square total energy.
2. **Kinematic Jerk Vector ($J = dM/dt$):**
   $$J(t) = \frac{M(t_{i+1}) - M(t_i)}{\Delta t}$$
   - `jerk_std`: Standard deviation of the jerk signal.
   - `jerk_max_abs`: Absolute peak transient jerk ($\text{m/s}^3$). Potholes register values $>150\text{ m/s}^3$.
3. **Harmonic Peak Count (`peak_count`):**
   $$\text{Peaks} = \text{find\_peaks}(M, \text{prominence}=0.75 \cdot \sigma_M)$$
   Measures chassis ringing after impact.

---

## 6. Computer Vision & YOLO Object Detection Pipeline

- **Architecture:** Ultralytics YOLOv8 Architecture trained on the international **Road Damage Dataset (RDD2022)**.
- **Classes Detected:**
  - `0`: Alligator Crack (Structural fatigue)
  - `1`: Transverse Crack (Perpendicular thermal/joint fissure)
  - `2`: Longitudinal Crack (Parallel wheel-path seam)
  - `3`: Other Corruption (Pavement weathering, delamination)
  - `4`: Pothole (Depression crater, rim strike hazard)
- **Input Resolution:** $640 \times 640\text{ px}$ letterboxed RGB frames.
- **Confidence Threshold:** Default $0.25$, adjustable up to $0.95$ with Non-Maximum Suppression (IoU $\ge 0.45$).
- **Telemetry Fusion:** Geotagged camera frames associate bounding box detections with contemporaneous IMU shock events.

---

## 7. Geospatial Intelligence & Road Memory Engine

When individual vehicles encounter road defects, sensor signals can vary based on vehicle speed, suspension stiffness, and tire pressure. The **Road Memory Engine** solves this through spatial aggregation:

### Haversine Spatial Clustering
Events within a radius $R = 30\text{ meters}$ are clustered together:
$$d = 2 R_{\text{earth}} \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)$$

Cluster centroid coordinates update dynamically with sample count $N$:
$$\text{Lat}_{\text{new}} = \frac{\text{Lat}_{\text{old}} \cdot (N - 1) + \text{Lat}_{\text{event}}}{N}, \quad \text{Lon}_{\text{new}} = \frac{\text{Lon}_{\text{old}} \cdot (N - 1) + \text{Lon}_{\text{event}}}{N}$$

### Chronic Hazard Priority Score
$$\text{Priority Score} = \min\left(100.0, \, \text{Recurrence} \times 4.5 + \text{Severity Weight} \times 25.0 + \text{Confidence} \times 30.0\right)$$
- $\ge 10$ detections promotes an anomaly to a **Chronic Hotspot Cluster**, triggering automatic municipal repair escalation.

### Composite Pavement Quality Index (PQI)
$$\text{PQI} = \max\left(0.0, \, 100.0 - (\text{P1 Hazards} \times 8.5) - (\text{P2 Hazards} \times 4.0) - (\text{Chronic Hotspots} \times 6.0)\right)$$

---

## 8. Google Maps & Live Street Footage Integration

The system includes deep geospatial integration focused on **Chhatrapati Sambhajinagar (Aurangabad), Maharashtra**:

### Tile Layer Integration in Folium
Users can toggle between four layer sets via Leaflet LayerControl:
1. **Google Maps (Streets):** `https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}`
2. **Google Satellite Imagery:** `https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}`
3. **Google Hybrid (Satellite + Roads):** `https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}`
4. **OpenStreetMap:** Default fallback tile source.

### Deep-Link Navigation
Every marker popup provides direct external links:
- **Google Street View:**
  `https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}`
- **Google Maps Pin:**
  `https://www.google.com/maps/search/?api=1&query={lat},{lon}`

### Synchronized Live Dashcam Footage Viewer
Selecting any detected pothole or bump renders:
1. A synthesized dashcam frame depicting road perspective, asphalt texture, lane markings, horizon, and weather conditions.
2. A YOLO bounding box overlay indicating exact distress geometry with confidence percentage.
3. Telemetry HUD showing vehicle speed, GPS coordinates, and assigned municipal ward.

---

## 9. Municipal Road Authority Portal (CSMC / PWD Maharashtra)

Engineered specifically for municipal operations in **Chhatrapati Sambhajinagar Municipal Corporation (CSMC)**:

### Municipal Ward & Division Hierarchy
- **Ward 01: Kranti Chowk & Nirala Bazar Zone** (Central Division — Er. Sachin Shinde)
- **Ward 02: Jalna Road Commercial Corridor (NH-753F)** (East Arterial Division — Er. Rajesh Kulkarni)
- **Ward 03: CIDCO Town Centre & Cannaught Ward** (New Aurangabad Division — Er. Priya Deshmukh)
- **Ward 04: Chikalthana MIDC & Airport Belt** (Industrial Division — Er. Vilas Jadhav)
- **Ward 05: Beed Bypass & Satara Parisar Zone** (South Arterial Division — Er. Ganesh Patil)

### 6-Stage Repair Workflow State Machine
1. `REPORTED`: Auto-flagged by sensor crowdsourcing or driver dashcam.
2. `UNDER_REVIEW`: Triage by junior highway engineer; field verification check.
3. `ASSIGNED`: Work order issued to field contractor or departmental maintenance squad.
4. `ASPHALT_LAYING`: Bituminous hot-mix paving actively in progress.
5. `COMPACTED_CURED`: Mini vibratory roller compaction and cooling verification.
6. `RESOLVED & QA PASSED`: Post-repair audit confirmed; incident closed in database.

---

## 10. Maharashtra PWD DSR Material & Repair Cost Calculator

Estimates repair costs and material quantities in Indian Rupees (₹ INR) based on standard **Maharashtra Public Works Department District Schedule of Rates (PWD DSR 2025–26)**:

| Material / Item Specification | Unit Metric | PWD DSR Standard Unit Rate |
|:---|:---:|:---:|
| Bituminous Concrete (BC / DBM Hot-Mix) | Metric Tonne (MT) | ₹6,800 / MT |
| Tack Coat (SS-1 Bitumen Emulsion) | Liter (L) | ₹95 / Liter |
| Granular Sub-Base (GSB Crushed Stone) | Cubic Meter ($m^3$) | ₹1,450 / $m^3$ |
| Skilled PWD Repair Crew (1 Supervisor + 4 Workers) | Shift Day | ₹5,200 / Day |
| 3-Ton Mini Vibratory Roller & Asphalt Cutter | Machine Day | ₹7,500 / Day |
| Departmental Contingency & Quality Testing | Percentage | 8.0% of Subtotal |

### Mathematical Formulas
1. **Total Void Volume ($m^3$):**
   $$V = N_{\text{potholes}} \times \text{Area}_{\text{avg}} \,(\text{m}^2) \times \text{Depth}_{\text{avg}} \,(\text{m})$$
2. **Compacted Asphalt Hot-Mix Tonnage ($T$):**
   $$T = V \times 2.4 \text{ tonnes/m}^3$$
3. **Tack Coat Emulsion ($L$):**
   $$L = N_{\text{potholes}} \times \text{Area}_{\text{avg}} \times 0.45 \text{ L/m}^2$$
4. **Total Sanction Budget (₹ INR):**
   $$\text{Budget} = \left(\text{Cost}_{\text{asphalt}} + \text{Cost}_{\text{tack}} + \text{Cost}_{\text{gsb}} + \text{Cost}_{\text{labor}} + \text{Cost}_{\text{machinery}}\right) \times 1.08$$

---

## 11. Driver Proximity Radar & Heads-Up Display (HUD)

- **Proximity Radar:** Computes radial distance to all recorded chronic road memory hotspots using Haversine equations.
- **Scanning Radius:** Configurable from $100\text{ m}$ to $1,500\text{ m}$ (Default $500\text{ m}$).
- **Safe Braking Speed Limit:**
  - High/Critical Pothole within $150\text{ m}$: Recommends slowing to $\le 25\text{ km/h}$.
  - Medium Speed Bump within $250\text{ m}$: Recommends slowing to $\le 30\text{ km/h}$.
  - Clear Corridor: Displays safe cruising indicator.
- **Corridor Quick-Jump Controls:** Instant tele-transport to Aurangabad landmarks:
  - *Kranti Chowk* (`19.87372, 75.32485`)
  - *Seven Hills Flyover* (`19.87698, 75.35380`)
  - *CIDCO Cannaught Place* (`19.87785, 75.36250`)
  - *Airport Road* (`19.88065, 75.38950`)
  - *Beed Bypass* (`19.84500, 75.34000`)

---

## 12. Step-by-Step Installation & Execution Guide

### Prerequisites
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS Sonoma+.
- **Python Version:** Python 3.10, 3.11, or 3.12 (Python 3.11 recommended).
- **RAM:** Minimum 4 GB (8 GB recommended for YOLO inference).
- **Disk Space:** ~2 GB free disk space.

---

### Step 1: Open Terminal / Shell
Open PowerShell (Windows) or Terminal (Linux/macOS) and navigate to the project directory:

```powershell
# Windows PowerShell
cd "d:\New folder (4)\RoadSense-Sensor-Dashboard"
```

```bash
# Linux / macOS
cd /path/to/RoadSense-Sensor-Dashboard
```

---

### Step 2: Create & Activate Virtual Environment

```powershell
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Required Dependencies
Install the required packages using `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*Required packages installed: `fastapi`, `uvicorn`, `streamlit`, `plotly`, `folium`, `streamlit-folium`, `scikit-learn`, `joblib`, `ultralytics`, `opencv-python`, `pillow`, `pandas`, `numpy`, `scipy`, `requests`, `sqlalchemy`, `pydantic`, `pytest`.*

---

### Step 4: Run the FastAPI Backend Server
The FastAPI backend serves the ML inference models, database, and REST APIs:

```powershell
# Windows PowerShell
$env:PYTHONPATH = "backend"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

```bash
# Linux / macOS
export PYTHONPATH=backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Backend API Base:** `http://localhost:8000`
- **Swagger Interactive API Docs:** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/api/health`

---

### Step 5: Run the Streamlit Dashboard (In a Second Terminal)
Open a new terminal window, activate the virtual environment, and launch Streamlit:

```powershell
# Windows PowerShell
cd "d:\New folder (4)\RoadSense-Sensor-Dashboard"
.\venv\Scripts\Activate.ps1
streamlit run app.py --server.port 8501
```

```bash
# Linux / macOS
cd /path/to/RoadSense-Sensor-Dashboard
source venv/bin/activate
streamlit run app.py --server.port 8501
```

- **Frontend Web UI:** `http://localhost:8501`

---

### Step 6: 1-Click Launch Options (Alternative)

#### Option A: Unified Python Launcher
Runs both FastAPI and Streamlit concurrently in a single command:
```powershell
python run_all.py
```

#### Option B: Windows Batch Script
Double-click `run_localhost.bat` or run:
```cmd
run_localhost.bat
```

#### Option C: Windows PowerShell Script
```powershell
powershell -ExecutionPolicy Bypass -File .\start_localhost.ps1
```

#### Option D: Docker Compose
```bash
docker-compose up --build
```

---

## 13. API Endpoints Directory

| Method | Endpoint | Description |
|:---:|:---|:---|
| `GET` | `/api/health` | Comprehensive system health, DB status, and loaded ML model registry |
| `POST`| `/api/sensors/uploads` | Ingests smartphone sensor CSV, extracts 30 features, runs RF model |
| `GET` | `/api/sensors/uploads` | Lists historical sensor upload logs |
| `GET` | `/api/sensors/uploads/{id}` | Retrieves window-level features and classification results |
| `GET` | `/api/sensors/uploads/{id}/map-events` | Returns geolocated map coordinates and route polylines |
| `POST`| `/api/vision/predict/image` | Executes YOLO inference on dashcam images |
| `POST`| `/api/dashcam/frame` | Low-latency base64 dashcam frame defect detection |
| `GET` | `/api/maps/incidents` | Returns all geotagged road defect incidents |
| `GET` | `/api/maps/hotspots` | Returns spatial road memory chronic hotspot clusters |
| `GET` | `/api/maps/nearby` | Proximity radar endpoint returning hazards within $R$ meters |
| `GET` | `/api/incidents` | Filterable incident queue (distress type, severity, status) |
| `POST`| `/api/incidents` | Submits a new road defect with automatic deduplication |
| `PATCH`| `/api/incidents/{id}` | Updates status, contractor assignment, and notes |
| `POST`| `/api/authority/incidents/{id}/assign` | Dispatches work order to PWD / municipal contractor |
| `POST`| `/api/authority/incidents/{id}/resolve`| Closes incident and records verification audit |
| `GET` | `/api/authority/summary` | Executive operations summary (P1 count, throughput) |
| `GET` | `/api/authority/work-orders/export` | Downloads official CSV work order dispatch sheet |
| `GET` | `/api/dashboard/summary` | Command center KPIs (PQI score, telemetry totals) |

---

## 14. Testing & Quality Assurance

The system includes a 25-test automated backend test suite covering API contracts, auth, database operations, GPS interpolation, and ML inference:

```powershell
# Run all tests from backend directory
cd backend
python -m pytest tests -v
```

### Verified Test Suite Results
```
tests/test_api_integration.py ....                                       [ 16%]
tests/test_auth.py ...                                                   [ 28%]
tests/test_database.py ..                                                [ 36%]
tests/test_e2e_api.py .                                                  [ 40%]
tests/test_gps_sensor.py ......                                          [ 64%]
tests/test_incidents_and_road_memory.py ...                              [ 76%]
tests/test_sensor.py ....                                                [ 92%]
tests/test_yolo.py ..                                                    [100%]

====================== 25 passed in 14.93s =======================
```

---

## 15. Default Credentials & Sample Assets

- **Admin Account:** `admin` / `admin123`
- **Field Driver Account:** `driver1` / `driver123`
- **Municipal Authority Account:** `authority1` / `authority123`
- **Sample Sensor Logs:**
  - `data/10.csv` (High-density road anomaly accelerometer and gyroscope recording)
  - `data/sample_sensor_drive.csv` (Field drive recording)
- **Model Files:**
  - `backend/app/ml/weights/roadsense_model.joblib` (Random Forest 30-feature model)
  - `backend/app/ml/weights/best.pt` (Trained YOLO weights)

---
*RoadSense AI © 2025–2026. Designed for Smart Cities & Municipal Infrastructure Resilience.*
