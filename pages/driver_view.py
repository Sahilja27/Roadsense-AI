"""
RoadSense AI - Driver Navigation & Dashcam Vision Interface
===========================================================
Primary real-time driver monitoring station:
1. Real-Time Heads-Up Display (HUD): Proximity warnings for imminent road hazards.
2. Dashcam Video & Camera Ingestion: Live browser camera capture and video file upload with YOLO defect detection.
3. Multimodal Damage Review: 1-click official incident submission with GPS association.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import time
from pathlib import Path
from PIL import Image

from utils.api_client import (
    get_nearby_hazards,
    predict_vision_image,
    create_incident,
    is_backend_online,
)
from utils.ui_theme import apply_theme
from utils.street_view_dataset import (
    get_google_maps_url,
    get_google_street_view_url,
    find_nearest_aurangabad_landmark,
)


def render_driver_view():
    apply_theme()
    backend_status = is_backend_online()

    # Header Section
    st.markdown(f"""
        <div style="padding-bottom: 0.8rem; border-bottom: 1px solid #EADBCC; margin-bottom: 1.2rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div>
                    <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; color: #1C1917;">
                        🚗 Driver Navigation &amp; Real-Time Dashcam Vision
                    </h1>
                    <p style="margin: 4px 0 0 0; font-size: 1rem; color: #78716C;">
                        AI Dashcam Object Detection, Road Memory Proximity Alerts, and Field Incident Dispatch.
                    </p>
                </div>
                <div style="display: flex; gap: 8px;">
                    <span style="background: {'#ECFDF5' if backend_status else '#FEF2F2'}; 
                                 color: {'#059669' if backend_status else '#DC2626'}; 
                                 border: 1px solid {'#10B981' if backend_status else '#FCA5A5'}; 
                                 padding: 5px 12px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;">
                        ● {'DASHCAM ENGINE ACTIVE' if backend_status else 'STANDALONE MODE'}
                    </span>
                    <span style="background: #FAF7F2; color: #DC2626; border: 1px solid #EADBCC; padding: 5px 12px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;">
                        CHHATRAPATI SAMBHAJINAGAR
                    </span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 1. Driver Location Coordinates & GPS Configuration
    with st.expander("📍 Driver GPS Telemetry & Aurangabad Corridor Quick-Jump", expanded=True):
        st.markdown("<strong style='color: #1C1917;'>⚡ Quick-Jump to Landmark along Jalna Road Corridor:</strong>", unsafe_allow_html=True)
        q1, q2, q3, q4, q5 = st.columns(5)
        
        if "driver_lat" not in st.session_state:
            st.session_state["driver_lat"] = 19.87698
            st.session_state["driver_lon"] = 75.35380

        with q1:
            if st.button("📍 Kranti Chowk", use_container_width=True):
                st.session_state["driver_lat"] = 19.87372
                st.session_state["driver_lon"] = 75.32485
                st.rerun()
        with q2:
            if st.button("📍 Seven Hills", use_container_width=True):
                st.session_state["driver_lat"] = 19.87698
                st.session_state["driver_lon"] = 75.35380
                st.rerun()
        with q3:
            if st.button("📍 CIDCO Cannaught", use_container_width=True):
                st.session_state["driver_lat"] = 19.87785
                st.session_state["driver_lon"] = 75.36250
                st.rerun()
        with q4:
            if st.button("📍 Airport Road", use_container_width=True):
                st.session_state["driver_lat"] = 19.88065
                st.session_state["driver_lon"] = 75.38950
                st.rerun()
        with q5:
            if st.button("📍 Beed Bypass", use_container_width=True):
                st.session_state["driver_lat"] = 19.84500
                st.session_state["driver_lon"] = 75.34000
                st.rerun()

        c_gps1, c_gps2, c_gps3 = st.columns([1.2, 1.2, 1.6])
        with c_gps1:
            driver_lat = st.number_input(
                "Vehicle Latitude",
                value=float(st.session_state.get("driver_lat", 19.87698)),
                format="%.6f",
                help="Current GPS latitude in Chhatrapati Sambhajinagar",
            )
        with c_gps2:
            driver_lon = st.number_input(
                "Vehicle Longitude",
                value=float(st.session_state.get("driver_lon", 75.35380)),
                format="%.6f",
                help="Current GPS longitude in Chhatrapati Sambhajinagar",
            )
        with c_gps3:
            scan_radius = st.slider("HUD Radar Scanning Radius (m)", min_value=100, max_value=1500, value=500, step=50)

    # 2. Heads-Up Display (HUD) Proximity Alert
    nearby = get_nearby_hazards(driver_lat, driver_lon, radius=float(scan_radius))

    if nearby:
        closest = nearby[0]
        dist = closest["distance_meters"]
        hazard = closest["primary_hazard"]
        sev = closest["avg_severity"]
        name = closest["cluster_name"]
        safe_spd = f"{closest.get('safe_speed_kmh', 30)} km/h"

        alert_bg = "#FEF2F2" if sev in ["High", "Critical"] else "#FFFBEB"
        alert_border = "#DC2626" if sev in ["High", "Critical"] else "#F59E0B"
        alert_title = "#DC2626" if sev in ["High", "Critical"] else "#B45309"

        gstreet_url = get_google_street_view_url(driver_lat, driver_lon)
        gmaps_url = get_google_maps_url(driver_lat, driver_lon)

        st.markdown(f"""
            <div style="background: {alert_bg}; border: 2px solid {alert_border}; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem; box-shadow: 0 2px 10px rgba(220, 38, 38, 0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                    <div style="display: flex; align-items: center; gap: 14px;">
                        <span style="font-size: 2.2rem;">⚠️</span>
                        <div>
                            <div style="font-size: 0.82rem; font-weight: 800; color: {alert_title}; text-transform: uppercase; letter-spacing: 0.08em;">
                                CRITICAL PROXIMITY HAZARD WARNING • CHHATRAPATI SAMBHAJINAGAR
                            </div>
                            <div style="font-size: 1.45rem; font-weight: 800; color: #1C1917; margin-top: 2px;">
                                {hazard.upper()} IMMINENT IN {int(dist)} METERS
                            </div>
                            <div style="color: #57534E; font-size: 0.85rem; margin-top: 3px;">
                                Location: <strong>{name}</strong> • Recurrence: <strong>{closest['recurrence_count']} detections</strong>
                            </div>
                            <div style="margin-top: 6px; display: flex; gap: 8px;">
                                <a href="{gstreet_url}" target="_blank" style="background: #FFFFFF; color: #DC2626; border: 1px solid #DC2626; padding: 2px 8px; border-radius: 4px; font-size: 0.76rem; font-weight: 700; text-decoration: none;">
                                    🌐 View in Google Street View
                                </a>
                                <a href="{gmaps_url}" target="_blank" style="background: #DC2626; color: #FFFFFF; padding: 2px 8px; border-radius: 4px; font-size: 0.76rem; font-weight: 700; text-decoration: none;">
                                    📍 Open on Google Maps
                                </a>
                            </div>
                        </div>
                    </div>
                    <div style="text-align: right; background: #FFFFFF; padding: 9px 18px; border-radius: 8px; border: 1px solid #EADBCC; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                        <div style="font-size: 0.74rem; color: #78716C; text-transform: uppercase; font-weight: 700;">Safe Speed Limit</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #059669;">{safe_spd}</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="background: #ECFDF5; border: 1px solid #10B981; border-radius: 10px; padding: 0.9rem 1.2rem; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 1.5rem;">🛡️</span>
                <div>
                    <span style="font-weight: 700; color: #059669; font-size: 0.95rem;">Clear Corridor Ahead:</span>
                    <span style="color: #57534E; font-size: 0.88rem; margin-left: 6px;">No high-priority road distress clusters recorded within {scan_radius}m of vehicle.</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # 3. Real-Time Dashcam Detection Station
    st.markdown("<h3 style='color: #1C1917; margin-bottom: 0.4rem;'>📹 Intelligent Road Dashcam Inspection</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #78716C; font-size: 0.88rem; margin-bottom: 1rem;'>Live road damage vision processing powered by trained YOLO (RDD2022 dataset: Potholes, Alligator Cracks, Transverse/Longitudinal Cracks).</p>", unsafe_allow_html=True)

    tab_photo, tab_camera = st.tabs(["📁 Upload Road Video / Photo", "📸 Live Camera Stream"])

    with tab_photo:
        c_in1, c_in2 = st.columns([1.5, 1])
        with c_in1:
            uploaded_file = st.file_uploader(
                "Upload Dashcam Video or Image Snapshot",
                type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
                key="dashcam_upload",
            )
        with c_in2:
            conf_thresh = st.slider("YOLO Confidence Threshold", 0.15, 0.90, 0.25, 0.05, key="conf_slider_photo")
            has_gps = st.checkbox("Tag with Current GPS Coordinates", value=True)

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            is_video = uploaded_file.name.lower().endswith((".mp4", ".avi", ".mov"))

            if is_video:
                st.info("🎥 Video uploaded! Submitting to FastAPI YOLO video processing service...")
                try:
                    import requests
                    files = {"file": (uploaded_file.name, file_bytes, "video/mp4")}
                    data = {
                        "conf_threshold": str(conf_thresh),
                        "sampling_fps": "2.0",
                    }
                    if has_gps:
                        data["latitude"] = str(driver_lat)
                        data["longitude"] = str(driver_lon)

                    job_res = requests.post("http://localhost:8000/api/vision/jobs", files=files, data=data, timeout=30.0).json()
                    job_id = job_res.get("id")
                    st.success(f"✓ Video processing job #{job_id} queued! Tracking progress...")

                    prog_bar = st.progress(0)
                    for _ in range(25):
                        time.sleep(1.0)
                        status_res = requests.get(f"http://localhost:8000/api/vision/jobs/{job_id}", timeout=5.0).json()
                        prog = int(status_res.get("progress_percent", 0))
                        prog_bar.progress(prog)
                        if status_res.get("status") in ["completed", "failed"]:
                            break

                    st.success(f"✓ Video processing complete! Found {status_res.get('total_detections', 0)} road defects.")
                except Exception as e:
                    st.error(f"Video job submission error: {e}")
            else:
                # Single Image Inspection via Backend API
                with st.spinner("Analyzing image through YOLO neural network..."):
                    try:
                        res = predict_vision_image(
                            image_bytes=file_bytes,
                            filename=uploaded_file.name,
                            conf_threshold=conf_thresh,
                            latitude=float(driver_lat) if has_gps else None,
                            longitude=float(driver_lon) if has_gps else None,
                        )

                        col_img1, col_img2 = st.columns([1.2, 1])
                        with col_img1:
                            if res.get("annotated_path") and Path(res["annotated_path"]).exists():
                                st.image(res["annotated_path"], caption=f"YOLO Damage Detections ({res['total_detections']} found)", use_container_width=True)
                            else:
                                st.image(file_bytes, caption="Uploaded Road Frame", use_container_width=True)

                        with col_img2:
                            st.markdown(f"""
                                <div class="kpi-card">
                                    <div class="kpi-title">Inference Speed</div>
                                    <div class="kpi-value" style="color: #DC2626; font-size: 1.8rem;">{res['inference_time_ms']} ms</div>
                                    <div class="kpi-sub">Total Damage Objects: <strong>{res['total_detections']}</strong></div>
                                </div>
                            """, unsafe_allow_html=True)

                            st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
                            st.markdown("<strong style='color: #1C1917;'>Identified Defects:</strong>", unsafe_allow_html=True)

                            if res["detections"]:
                                for idx, d in enumerate(res["detections"], 1):
                                    cls = d["class_name"]
                                    conf = d["confidence"]
                                    st.markdown(f"- **{cls.title()}** • Confidence: `{conf:.2f}`")

                                # 1-Click Incident Dispatch
                                st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
                                if st.button("🚨 Submit as Official Road Incident", type="primary", use_container_width=True):
                                    primary_d = res["detections"][0]
                                    inc_data = {
                                        "title": f"Dashcam Detected {primary_d['class_name'].title()}",
                                        "distress_type": primary_d["class_name"].title(),
                                        "severity": "High" if "pothole" in primary_d["class_name"].lower() else "Medium",
                                        "confidence": primary_d["confidence"],
                                        "latitude": float(driver_lat) if has_gps else None,
                                        "longitude": float(driver_lon) if has_gps else None,
                                        "notes": f"Detected via Dashcam YOLO inference. Total objects: {res['total_detections']}",
                                        "evidence_type": "vision_detection",
                                        "evidence_metadata": primary_d,
                                    }
                                    created_inc = create_incident(inc_data)
                                    st.success(f"✓ Incident #{created_inc.get('id')} submitted to Municipal Authority triage queue!")
                            else:
                                st.info("Pavement is in normal condition. No defects detected above confidence threshold.")

                    except Exception as e:
                        st.error(f"Inference error: {e}")

    with tab_camera:
        st.markdown("<p style='color: #78716C; font-size: 0.85rem;'>Use your connected vehicle webcam or smartphone browser camera:</p>", unsafe_allow_html=True)
        cam_image = st.camera_input("Capture Dashcam Frame")

        if cam_image is not None:
            c_bytes = cam_image.read()
            with st.spinner("Executing YOLO detection on live camera snapshot..."):
                try:
                    res_cam = predict_vision_image(
                        image_bytes=c_bytes,
                        filename="camera_frame.jpg",
                        conf_threshold=0.25,
                        latitude=float(driver_lat),
                        longitude=float(driver_lon),
                    )

                    if res_cam.get("annotated_path") and Path(res_cam["annotated_path"]).exists():
                        st.image(res_cam["annotated_path"], caption=f"Live Camera Detection: {res_cam['total_detections']} defects", use_container_width=True)
                    else:
                        st.image(c_bytes, caption="Live Camera Frame", use_container_width=True)

                    if res_cam["detections"]:
                        st.warning(f"⚠️ {res_cam['total_detections']} road anomalies detected in live camera feed!")
                        if st.button("Submit Camera Incident to Authority", type="primary"):
                            first_det = res_cam["detections"][0]
                            create_incident({
                                "title": f"Live Camera Detected {first_det['class_name'].title()}",
                                "distress_type": first_det["class_name"].title(),
                                "severity": "High" if "pothole" in first_det["class_name"].lower() else "Medium",
                                "confidence": first_det["confidence"],
                                "latitude": float(driver_lat),
                                "longitude": float(driver_lon),
                                "notes": "Captured via live vehicle browser camera feed.",
                            })
                            st.success("✓ Live camera defect logged to Road Memory database.")
                    else:
                        st.success("✓ Road surface is clear.")
                except Exception as e:
                    st.error(f"Live camera inference error: {e}")


# Execute when loaded as a page
render_driver_view()

