# RoadSense AI – Smartphone Sensor Road Damage & Memory System
## Member 4 Deliverable: Frontend and Analytics (Dashboard, Map, History, Driver & Authority Interfaces)

---

## 🎯 Member 4 Scope & Overview
In this final-year engineering project, **Member 4** is responsible for the user-facing spatial intelligence, driver safety alert systems, and municipal decision analytics layer:

```text
ANDROID SMARTPHONE (Acc + Gyro + GPS + Time)
        │
        ▼
 SENSOR CSV UPLOAD ──> ML EVENT CLASSIFIER ──> SQLITE DB
                                                   │
                                          ROAD MEMORY ENGINE
                                                   │
                  ┌────────────────────────────────┴────────────────────────────────┐
                  ▼                                                                 ▼
          DRIVER INTERFACE                                                 AUTHORITY INTERFACE
    • Real-time Hazard Map                                           • Road Health Analytics
    • Impact Event History                                           • Chronic Recurring Locations
    • Proximity Nearby Warnings                                      • Prioritized Inspection Work Orders
```

---

## 🚀 How to Run the Dashboard

### 1. Open Terminal in Project Directory
```powershell
cd C:\Users\Dipak\OneDrive\Desktop\RoadSense-Sensor-Dashboard
```

### 2. Install Required Packages
```powershell
pip install -r requirements.txt
```

### 3. Launch Streamlit Dashboard
```powershell
streamlit run app.py
```
Open your browser and navigate to: **`http://localhost:8501`**.

---

## 📁 Project Architecture & Components

```text
RoadSense-Sensor-Dashboard/
│
├── .streamlit/
│   └── config.toml             # Sleek dark command-center theme
├── app.py                      # Main entrypoint, sidebar navigation, and live telemetry
├── requirements.txt            # Project dependencies (Streamlit, Plotly, Pandas, NumPy, Scikit-learn)
├── roadsense_sensor.db         # SQLite database storing raw sensor shocks & aggregated road memory
├── README.md                   # Full documentation & viva defense guide
│
├── data/
│   └── sample_sensor_drive.csv # Realistic smartphone accelerometer + gyroscope + GPS log for 1-click viva demos
│
├── utils/
│   ├── __init__.py
│   ├── database.py             # SQLite schemas & data seeder
│   ├── sensor_processor.py     # Kinematic calculations (||a||, vertical shock, angular tilt, ML event classifier)
│   └── road_memory.py          # Spatial clustering engine (Haversine ~25m radius) & triage scoring
│
└── pages/
    ├── __init__.py
    ├── sensor_upload.py        # CSV upload & multi-axis waveform signal visualizer
    ├── driver_view.py          # Driver Interface: Hazard Map, Event History, Proximity Warnings
    ├── authority_view.py       # Authority Interface: Road Health Analytics, Chronic Hotspots, Work Orders
    └── about_system.py         # System architecture flow & Member 4 viva defense Q&A
```

---

## 💡 Key Features Implemented by Member 4

### 1. Driver Interface (`pages/driver_view.py`)
- **Proximity Heads-Up Warning System**: Calculates real-time distance from current driver GPS coordinates to registered hazards, showing flashing alerts when approaching within 350 meters.
- **Road Hazard Map**: Interactive Plotly OpenStreetMap showing all confirmed potholes (Red), speed bumps (Amber), and rough roads (Blue).
- **Driving Event History**: Chronological log of recent vehicle impacts and shocks.

### 2. Authority Interface (`pages/authority_view.py`)
- **Pavement Quality Index (PQI)**: Composite road network health score (0–100).
- **Chronic Recurring Locations**: Hotspots where multiple independent vehicle drives recorded repeated impacts.
- **Prioritized Triage Matrix**:
  - **Priority 1 (Urgent):** Score ≥ 80 &rarr; Immediate asphalt patching within 48h.
  - **Priority 2 (Scheduled):** Score 50–79 &rarr; Joint leveling & signage repainting within 14 days.
  - **Priority 3 (Routine):** Score < 50 &rarr; Continued monitoring.
- **Downloadable Municipal Work Order (CSV)**: Export ready for public works civil contractors.

### 3. Sensor Ingestion & Signal Analyzer (`pages/sensor_upload.py`)
- Multi-axis time-series waveform graph showing $a_z$, total acceleration magnitude $||a||$, and gravity reference ($9.81\,\text{m/s}^2$).
- **One-Click Pre-recorded Demo Button**: Instantly loads `data/sample_sensor_drive.csv` during viva presentations.

---

## 🎓 Member 4 Viva Defense Quick Reference

| Question | Member 4 Core Answer |
|:---|:---|
| **What is the Road Memory Engine?** | A spatial clustering engine that groups multiple sensor passes within a 25-meter radius. It filters out spurious noise (like dropped phones) and mathematically proves permanent road damage when recurrence counts are high. |
| **How do you classify a Pothole vs a Bump?** | Potholes feature a sharp downward drop ($a_z < 6.5\,\text{m/s}^2$) followed by violent upward rebound. Speed bumps produce a coordinated upward heave ($a_z > 13.0\,\text{m/s}^2$) with significant vehicle pitch rotation ($\omega_{\text{pitch}}$). |
| **How are priority scores calculated?** | $\text{Priority} = (0.45 \times \text{Severity}) + (0.45 \times \text{Recurrence Density}) + 10$. Combines impact violence with repeat failure frequency. |
