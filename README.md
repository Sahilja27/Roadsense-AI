# RoadSense AI – Integrated Road Damage Detection & Spatial Memory System

> **Production Full-Stack Architecture:** FastAPI Backend, SQLAlchemy/Alembic ORM, PostgreSQL / SQLite Spatial Persistence, Ultralytics YOLO Vision Engine (RDD2022 Dataset), Smartphone Sensor Kinematics ML Model (Random Forest), Real-Time Road Memory Engine, and Interactive Command Center.

---

## 🏗️ System Architecture

```text
┌───────────────────────────────────────────────┐     ┌────────────────────────────────────────────────┐
│             DRIVER & CAMERA INTERFACE         │     │           SMARTPHONE SENSOR INGESTION          │
│   • Browser WebCam / Dashcam Stream           │     │   • Acc (x,y,z) + Gyro (x,y,z) CSV Upload      │
│   • Video File Upload (.mp4, .avi)            │     │   • 30 Kinematic Features / 1-Second Window    │
│   • Live GPS Geolocation Tagging              │     │   • Data Quality & Null Column Validation      │
└───────────────────────┬───────────────────────┘     └───────────────────────┬────────────────────────┘
                        │                                                     │
                        ▼                                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   FASTAPI MODULAR BACKEND (:8000)                                   │
│                                                                                                      │
│  ┌─────────────────────────┐   ┌──────────────────────────┐   ┌───────────────────────────────────┐  │
│  │   YOLO Vision Service   │   │  Sensor Predictor (ML)   │   │        Road Memory Engine         │  │
│  │   • best.pt (RDD2022)   │   │  • roadsense_model.joblib│   │   • Spatial Haversine Clustering  │  │
│  │   • 5 Defect Classes    │   │  • Bump vs Pothole       │   │   • 0-100 Priority Triage Score   │  │
│  │   • Video Job Worker    │   │  • Exact 30 Feature Math │   │   • Chronic Hotspot Management    │  │
│  └────────────┬────────────┘   └─────────────┬────────────┘   └─────────────────┬─────────────────┘  │
│               │                              │                                  │                    │
│               └──────────────────────────────┼──────────────────────────────────┘                    │
│                                              ▼                                                       │
│                               ┌─────────────────────────────┐                                        │
│                               │ SQLAlchemy ORM & Repository │                                        │
│                               │  • PostgreSQL / SQLite DB   │                                        │
│                               │  • Alembic Migrations       │                                        │
│                               └──────────────┬──────────────┘                                        │
└──────────────────────────────────────────────┼───────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                STREAMLIT COMMAND-CENTER DASHBOARD (:8501)                            │
│                                                                                                      │
│   📊 Executive Command Center: Real-time PQI Index, Critical P1 Alerts, Interactive Geospatial Map   │
│   🚗 Driver Interface: Imminent Hazard Proximity HUD, Live Camera / Dashcam YOLO Processing         │
│   🏛️ Authority Interface: Municipal Triage, Work Order Assignments, Resolution Notes, CSV Export    │
│   📲 Sensor CSV Waveform Ingestion: 30-feature signal extraction, GPS maps & CSV export             │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart Guide (Windows PowerShell)

### Prerequisites
- Python 3.10+ (Verified on Python 3.11)
- Git & PowerShell

### 1. Install Dependencies
In the root directory, install the required libraries:
```powershell
pip install -r requirements.txt
```

### 2. Configure Environment
Copy the sample environment file:
```powershell
Copy-Item "backend\.env.example" -Destination "backend\.env"
```

### 3. Initialize Database & Run Migrations
Generate tables and seed administrative accounts:
```powershell
cd backend
python -m app.db.init_db
alembic upgrade head
cd ..
```

### 4. Launch the FastAPI Backend
Start the backend server on **port 8000**:
```powershell
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger API Documentation: **`http://localhost:8000/docs`**
- Alternative ReDoc Documentation: **`http://localhost:8000/redoc`**
- Health Check Endpoint: **`http://localhost:8000/api/health`**

### 5. Launch the Streamlit Dashboard
In a separate PowerShell window, start the frontend on **port 8501**:
```powershell
python -m streamlit run app.py --server.port 8501
```
Navigate your browser to: **`http://localhost:8501`**.

---

## 🧪 Automated Testing

Execute the test suite with pytest:
```powershell
cd backend
pytest -v
```
**Test Coverage Includes:**
- Database connectivity, model relationships, and seed fixtures.
- User registration, password hashing (PBKDF2), and JWT authentication.
- Sensor CSV validation, required column error handling, and exact 30-feature extraction.
- Sensor ML inference using the trained `roadsense_model.joblib`.
- Computer vision YOLO inference using trained `best.pt` with RDD2022 classes.
- Video generation, frame sampling, and background job state updates.
- Incident creation, geographic deduplication (<30m), and multimodal sensor/vision correlation.
- Road Memory Engine spatial clustering and driver proximity alert queries.
- Complete API integration tests for health, dashboard summaries, and authority endpoints.

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health, DB status, and ML model readiness |
| `POST` | `/api/auth/register` | Register user account |
| `POST` | `/api/auth/login` | Obtain JWT Bearer access token |
| `GET` | `/api/dashboard/summary` | Real-time network health, PQI, open defects, chronic hotspots |
| `GET` | `/api/dashboard/activity` | Unified activity timeline |
| `POST` | `/api/sensors/uploads` | Ingest sensor CSV & run 30-feature ML model |
| `GET` | `/api/sensors/uploads` | List sensor upload history |
| `GET` | `/api/sensors/uploads/{id}` | Detailed prediction windows and window breakdown |
| `GET` | `/api/sensors/uploads/{id}/results` | Download model predictions as CSV |
| `POST` | `/api/vision/predict/image` | Run YOLO on snapshot, draw bounding boxes |
| `POST` | `/api/vision/jobs` | Submit video for asynchronous YOLO processing |
| `GET` | `/api/vision/jobs/{id}` | Poll video job progress & frame detection count |
| `GET` | `/api/vision/jobs/{id}/detections` | Get list of all defects detected across frames |
| `POST` | `/api/dashcam/sessions` | Create live vehicle monitoring session |
| `POST` | `/api/dashcam/frame` | Process live base64 browser camera frame |
| `GET` | `/api/incidents` | List incidents with search, status, and severity filters |
| `POST` | `/api/incidents` | Submit incident with spatial deduplication & multimodal verification |
| `PATCH` | `/api/incidents/{id}` | Update incident status, severity, or notes |
| `GET` | `/api/maps/incidents` | Return geotagged incidents for GIS mapping |
| `GET` | `/api/maps/hotspots` | Return Road Memory clusters and priority scores |
| `GET` | `/api/maps/nearby` | Driver HUD proximity query for obstacles ahead |
| `GET` | `/api/authority/summary` | Municipal triage breakdown (Urgent P1, Assigned, Resolved) |
| `POST` | `/api/authority/incidents/{id}/assign` | Dispatch work order to field contractor |
| `POST` | `/api/authority/incidents/{id}/resolve` | Mark repair resolved with completion notes |
| `GET` | `/api/authority/work-orders/export` | Download filtered work orders as CSV |

---

## 🐳 Docker Deployment

To run in production with PostgreSQL 16:
```bash
docker-compose up --build -d
```
This spins up:
1. `db`: PostgreSQL database on port `5432` with automatic health checks.
2. `backend`: FastAPI backend on port `8000`.
3. `dashboard`: Streamlit command center on port `8501`.

---

## 🔧 Troubleshooting

- **Missing Model Weights:**
  Ensure `roadsense_model.joblib` and `best.pt` exist in `backend/app/ml/weights/` (or paths configured in `.env`). The API will report degraded status on `/api/health` if weights cannot be loaded.
- **Camera Access in Browser:**
  When testing the Live Camera Stream on the Driver page, grant camera permissions in your web browser. If no physical camera is attached, use the "Upload Road Video / Photo" tab.
- **GPU vs. CPU Execution:**
  Ultralytics YOLO automatically detects CUDA-capable GPUs. If no NVIDIA GPU or CUDA runtime is present, CPU inference will be used transparently with no code changes needed.
- **Port Conflicts:**
  If port 8000 or 8501 is occupied, change the `--port` argument when launching or set `PORT` in your environment.
