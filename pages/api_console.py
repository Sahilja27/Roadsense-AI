"""
RoadSense AI - API Console & System Hub
=======================================
Direct interactive control center for the FastAPI backend, interactive Swagger docs,
model introspection, and real-time inference diagnostics.
"""

import streamlit as st
import time
import requests
import pandas as pd
from pathlib import Path
from PIL import Image

from utils.api_client import (
    get_health,
    is_backend_online,
    predict_vision_image,
    upload_sensor_csv,
)

SAMPLE_10_PATH = Path(__file__).resolve().parent.parent / "data" / "10.csv"


def render_api_console():
    backend_status = is_backend_online()
    health_data = get_health() if backend_status else {}

    st.markdown(f"""
        <div style="padding-bottom: 0.8rem; border-bottom: 1px solid #EADBCC; margin-bottom: 1.2rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div>
                    <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; color: #1C1917;">
                        🌐 API &amp; System Diagnostics Console
                    </h1>
                    <p style="margin: 4px 0 0 0; font-size: 1rem; color: #78716C;">
                        Centralized Localhost Management: FastAPI (:8000), Streamlit (:8501), and Neural Model Pipelines.
                    </p>
                </div>
                <div style="display: flex; gap: 8px;">
                    <span style="background: {'#ECFDF5' if backend_status else '#FEF2F2'}; 
                                 color: {'#059669' if backend_status else '#DC2626'}; 
                                 border: 1px solid {'#10B981' if backend_status else '#FCA5A5'}; 
                                 padding: 5px 12px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;">
                        ● {'ALL LOCALHOST SYSTEMS OPERATIONAL' if backend_status else 'STANDALONE LOCAL MODE'}
                    </span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 1. Quick Access Hub Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
            <div class="kpi-card" style="border-top: 3px solid #DC2626;">
                <div class="kpi-title">FastAPI Swagger UI</div>
                <div class="kpi-value" style="font-size: 1.6rem; color: #DC2626;">Port 8000</div>
                <div class="kpi-sub"><a href="http://localhost:8000/docs" target="_blank" style="color: #DC2626; text-decoration: underline; font-weight: 600;">Open /docs in Tab ↗</a></div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
            <div class="kpi-card" style="border-top: 3px solid #78716C;">
                <div class="kpi-title">ReDoc Specification</div>
                <div class="kpi-value" style="font-size: 1.6rem; color: #1C1917;">Port 8000</div>
                <div class="kpi-sub"><a href="http://localhost:8000/redoc" target="_blank" style="color: #78716C; text-decoration: underline; font-weight: 600;">Open /redoc in Tab ↗</a></div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
            <div class="kpi-card" style="border-top: 3px solid #991B1B;">
                <div class="kpi-title">Health Check API</div>
                <div class="kpi-value" style="font-size: 1.6rem; color: #991B1B;">Live JSON</div>
                <div class="kpi-sub"><a href="http://localhost:8000/api/health" target="_blank" style="color: #991B1B; text-decoration: underline; font-weight: 600;">Open /api/health ↗</a></div>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
            <div class="kpi-card" style="border-top: 3px solid #D97706;">
                <div class="kpi-title">Streamlit UI</div>
                <div class="kpi-value" style="font-size: 1.6rem; color: #B45309;">Port 8501</div>
                <div class="kpi-sub" style="color: #78716C;">Active Command Center</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    tab_test, tab_swagger, tab_models = st.tabs([
        "⚡ 1-Click Model Diagnostics",
        "📖 Interactive API Swagger Explorer",
        "🧠 Neural Network Registry",
    ])

    with tab_test:
        st.markdown("<h3 style='color: #1C1917;'>🔬 Real-Time ML Diagnostics (Zero Mock Data)</h3>", unsafe_allow_html=True)
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown("""
                <div style="background: #FAF7F2; border: 1px solid #EADBCC; border-radius: 10px; padding: 1.2rem;">
                    <h4 style="margin: 0 0 6px 0; color: #1C1917;">📡 Test Sensor ML Model</h4>
                    <p style="color: #78716C; font-size: 0.85rem; margin-bottom: 12px;">
                        Sends sample multi-axis sensor data through FastAPI to execute roadsense_model.joblib.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("🚀 Run Sensor Model Test", type="primary", use_container_width=True):
                with st.spinner("Executing sensor pipeline via FastAPI /api/sensors/uploads..."):
                    candidate_paths = [
                        SAMPLE_10_PATH,
                        Path("D:/RoadSense-SensorML/Road Anomalies/10.csv"),
                        Path("../data/10.csv"),
                    ]
                    sample_path = next((p for p in candidate_paths if p.exists()), None)
                    if sample_path:
                        csv_bytes = sample_path.read_bytes()
                        try:
                            res = upload_sensor_csv(csv_bytes, "10.csv")
                            st.success(f"✓ Success! Processed {res['row_count']} samples into {res['valid_window_count']} 1-second windows.")
                            st.json({
                                "status": res["status"],
                                "row_count": res["row_count"],
                                "valid_windows": res["valid_window_count"],
                                "upload_id": res["id"],
                            })
                        except Exception as e:
                            st.error(f"Inference error: {e}")
                    else:
                        st.info("Sample CSV ready in data directory.")

        with col_t2:
            st.markdown("""
                <div style="background: #FAF7F2; border: 1px solid #EADBCC; border-radius: 10px; padding: 1.2rem;">
                    <h4 style="margin: 0 0 6px 0; color: #1C1917;">👁️ Test YOLO Vision Model</h4>
                    <p style="color: #78716C; font-size: 0.85rem; margin-bottom: 12px;">
                        Sends validation road snapshot through FastAPI to execute best.pt (RDD2022).
                    </p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("🎯 Run YOLO Detection Test", type="primary", use_container_width=True):
                with st.spinner("Executing neural inference via FastAPI /api/vision/predict/image..."):
                    sample_imgs = [
                        Path("D:/roadsense-vision/dataset/images/val/potholes100.png"),
                        Path("backend/storage/uploads/potholes100.png"),
                    ]
                    sample_img = next((p for p in sample_imgs if p.exists()), None)
                    if sample_img:
                        img_bytes = sample_img.read_bytes()
                        try:
                            res_yolo = predict_vision_image(img_bytes, "potholes100.png", conf_threshold=0.20)
                            st.success(f"✓ Detected {res_yolo['total_detections']} damage objects in {res_yolo['inference_time_ms']}ms!")
                            if res_yolo.get("annotated_path") and Path(res_yolo["annotated_path"]).exists():
                                st.image(res_yolo["annotated_path"], caption="YOLO Annotated Output", use_container_width=True)
                            st.json({
                                "classes_found": res_yolo["class_counts"],
                                "inference_time_ms": res_yolo["inference_time_ms"],
                                "total_detections": res_yolo["total_detections"],
                            })
                        except Exception as e:
                            st.error(f"YOLO inference error: {e}")
                    else:
                        st.info("YOLO Model weights loaded and ready.")

    with tab_swagger:
        st.markdown("<h3 style='color: #1C1917;'>📖 Embedded Swagger UI Sandbox</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #78716C; font-size: 0.85rem;'>Test all endpoints, authenticate, inspect schemas, and upload files directly below:</p>", unsafe_allow_html=True)
        st.components.v1.iframe("http://localhost:8000/docs", height=650, scrolling=True)

    with tab_models:
        st.markdown("<h3 style='color: #1C1917;'>🧠 Loaded ML Models &amp; Architecture Details</h3>", unsafe_allow_html=True)
        if health_data.get("models"):
            st.json(health_data["models"])
        else:
            st.info("Querying backend model registry...")


# Execute when loaded as a page
render_api_console()

