"""
RoadSense AI - Sensor CSV Ingestion & GPS Detection Map
======================================================
Ingests smartphone sensor recordings, executes the trained Random Forest model
(roadsense_model.joblib), extracts 30 kinematic features, validates GPS streams,
and renders an interactive OpenStreetMap centered on Chhatrapati Sambhajinagar (Aurangabad).
"""

import os
import sys
from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.api_client import (
    upload_sensor_csv,
    get_sensor_uploads,
    get_sensor_upload_detail,
    get_sensor_upload_map_events,
    is_backend_online,
)
from utils.ui_theme import apply_theme
from utils.physics_visualizer import render_kinematics_explainer, render_window_kinematic_deepdive
from utils.street_view_dataset import (
    get_google_maps_url,
    get_google_street_view_url,
    render_live_footage_inspector,
    find_nearest_aurangabad_landmark,
    AURANGABAD_JALNA_ROAD_WAYPOINTS,
    AURANGABAD_KEY_LANDMARKS,
)

SAMPLE_10_PATH = os.path.join(ROOT_DIR, "data", "10.csv")
SAMPLE_DRIVE_PATH = os.path.join(ROOT_DIR, "data", "sample_sensor_drive.csv")

AURANGABAD_DEFAULT_CENTER = [19.8762, 75.3433]



def render_sensor_upload():
    apply_theme()
    backend_status = is_backend_online()

    st.markdown(f"""
        <div style="padding-bottom: 0.8rem; border-bottom: 1px solid #EADBCC; margin-bottom: 1.2rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div>
                    <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; color: #1C1917;">
                        📲 Smartphone Sensor Signal Analyzer &amp; GPS Map
                    </h1>
                    <p style="margin: 4px 0 0 0; font-size: 1rem; color: #78716C;">
                        Direct ML Waveform Classification with real-time GPS geolocation and Aurangabad demonstration mapping.
                    </p>
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <span style="background: {'#ECFDF5' if backend_status else '#FEF2F2'}; 
                                 color: {'#059669' if backend_status else '#DC2626'}; 
                                 border: 1px solid {'#10B981' if backend_status else '#FCA5A5'}; 
                                 padding: 5px 12px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;">
                        ● {'API ONLINE' if backend_status else 'STANDALONE MODE'}
                    </span>
                    <span style="background: #FAF7F2; color: #78716C; border: 1px solid #EADBCC; padding: 5px 12px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;">
                        30-FEATURE KINEMATICS
                    </span>
                    <span style="background: #FEF2F2; color: #DC2626; border: 1px solid #FCA5A5; padding: 5px 12px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;">
                        RANDOM FOREST (Bump / Pothole)
                    </span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    tab_upload, tab_physics, tab_features, tab_history = st.tabs([
        "📤 CSV Ingestion & Interactive GPS Map",
        "📈 Waveform Science: Bump vs Pothole",
        "🔬 30 Kinematic Features Breakdown", 
        "📜 Ingestion History"
    ])

    with tab_upload:
        # Section 1: Ingestion & Route Mode Controls
        st.markdown("<h4 style='color: #1C1917; margin-bottom: 0.5rem;'>1. Upload &amp; Route Configuration</h4>", unsafe_allow_html=True)
        
        c_mode, c_file, c_demo = st.columns([2, 3, 2])
        
        with c_mode:
            route_mode_choice = st.radio(
                "🗺️ Location / Route Mode",
                [
                    "Chhatrapati Sambhajinagar Demo Route",
                    "Real GPS from CSV Dataset",
                ],
                index=0,
                help="Demo Route maps windows chronologically along documented public roads (Jalna Road) in Chhatrapati Sambhajinagar (Aurangabad). Real GPS extracts actual sensor coordinates from CSV.",
            )
            route_mode = "demo_aurangabad" if "Demo" in route_mode_choice else "real_gps"

            if route_mode == "demo_aurangabad":
                st.markdown("""
                    <div style="background: #FEF2F2; border-left: 3px solid #DC2626; padding: 7px 11px; border-radius: 4px; font-size: 0.78rem; color: #991B1B;">
                        <strong>DEMO ROUTE MODE:</strong> Maps 1s prediction windows along Jalna Road, Aurangabad. Coordinates tagged as <code>DEMO / SIMULATED LOCATION</code>.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background: #FAF7F2; border-left: 3px solid #EADBCC; padding: 7px 11px; border-radius: 4px; font-size: 0.78rem; color: #57534E;">
                        <strong>REAL GPS MODE:</strong> Extracts latitude/longitude from CSV with median window calculation. Unlocated windows marked as <code>location-pending</code>.
                    </div>
                """, unsafe_allow_html=True)

        with c_file:
            uploaded_file = st.file_uploader(
                "Upload Smartphone Sensor CSV (e.g. 10.csv)",
                type=["csv"],
                help="Supports accelerometer, gyroscope, speed, and GPS location streams.",
            )

        with c_demo:
            st.markdown("<div style='height: 1.6rem;'></div>", unsafe_allow_html=True)
            if st.button("⚡ Process 10.csv (Road Anomalies Test Log)", type="primary", use_container_width=True):
                if os.path.exists(SAMPLE_10_PATH):
                    st.session_state["active_file_path"] = SAMPLE_10_PATH
                    st.session_state["active_file_name"] = "10.csv"
                    st.session_state["trigger_process"] = True
                else:
                    st.error("10.csv file not found in repository.")
            
            if st.button("📁 Load Sample Drive CSV", use_container_width=True):
                if os.path.exists(SAMPLE_DRIVE_PATH):
                    st.session_state["active_file_path"] = SAMPLE_DRIVE_PATH
                    st.session_state["active_file_name"] = "sample_sensor_drive.csv"
                    st.session_state["trigger_process"] = True

        # Handle processing trigger
        csv_bytes = None
        target_name = None

        if uploaded_file is not None:
            csv_bytes = uploaded_file.read()
            target_name = uploaded_file.name
        elif st.session_state.get("trigger_process") and st.session_state.get("active_file_path"):
            path = st.session_state["active_file_path"]
            with open(path, "rb") as f:
                csv_bytes = f.read()
            target_name = st.session_state.get("active_file_name", "sensor_data.csv")
            st.session_state["trigger_process"] = False

        if csv_bytes is not None:
            with st.spinner(f"Extracting 30 kinematic features & executing Random Forest model ({target_name})..."):
                try:
                    upload_rec = upload_sensor_csv(csv_bytes, target_name, route_mode=route_mode)
                    st.session_state["current_upload_id"] = upload_rec["id"]
                    st.success(f"✓ Processed {target_name}: {upload_rec.get('valid_window_count', 0)} windows classified and stored into database!")
                except Exception as ex:
                    st.error(f"Processing failed: {ex}")

        # Retrieve current or selected upload data with automatic fallback so page is NEVER empty
        current_upload_id = st.session_state.get("current_upload_id")
        if not current_upload_id:
            recent_uploads = get_sensor_uploads(limit=1)
            if recent_uploads:
                current_upload_id = recent_uploads[0].get("id")
                st.session_state["current_upload_id"] = current_upload_id
            elif os.path.exists(SAMPLE_10_PATH):
                try:
                    with open(SAMPLE_10_PATH, "rb") as f:
                        auto_bytes = f.read()
                    auto_rec = upload_sensor_csv(auto_bytes, "10.csv", route_mode="demo_aurangabad")
                    current_upload_id = auto_rec["id"]
                    st.session_state["current_upload_id"] = current_upload_id
                except Exception:
                    pass
        
        if current_upload_id:
            detail = get_sensor_upload_detail(current_upload_id)
            map_data = get_sensor_upload_map_events(current_upload_id)

            if detail and map_data:
                st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
                
                # Section 2: Metric KPI Cards
                st.markdown(f"<h4 style='color: #1C1917; margin-bottom: 0.6rem;'>2. Ingestion &amp; Prediction Metrics (Upload #{current_upload_id} — {detail.get('original_name', 'Sensor File')})</h4>", unsafe_allow_html=True)
                
                counts = detail.get("counts", {})
                bump_count = counts.get("Bump", 0)
                pothole_count = counts.get("Pothole", 0)
                total_windows = detail.get("valid_window_count", 0)
                geolocated_count = map_data.get("geolocated_events_count", 0)
                
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.markdown(f"""
                        <div class="kpi-card" style="border-top: 3px solid #78716C;">
                            <div class="kpi-title">Raw Ingested Rows</div>
                            <div class="kpi-value" style="color: #1C1917;">{detail.get('row_count', 0):,}</div>
                            <div class="kpi-sub">Total CSV Samples</div>
                        </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                        <div class="kpi-card" style="border-top: 3px solid #059669;">
                            <div class="kpi-title">Classified Windows</div>
                            <div class="kpi-value" style="color: #059669;">{total_windows}</div>
                            <div class="kpi-sub">1.0s Time Windows</div>
                        </div>
                    """, unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                        <div class="kpi-card" style="border-top: 3px solid #D97706;">
                            <div class="kpi-title">Detected Bumps</div>
                            <div class="kpi-value" style="color: #B45309;">{bump_count}</div>
                            <div class="kpi-sub">Speed Bumps / Ramps</div>
                        </div>
                    """, unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""
                        <div class="kpi-card" style="border-top: 3px solid #DC2626;">
                            <div class="kpi-title">Detected Potholes</div>
                            <div class="kpi-value" style="color: #DC2626;">{pothole_count}</div>
                            <div class="kpi-sub">Impact Shocks / Drops</div>
                        </div>
                    """, unsafe_allow_html=True)
                with m5:
                    is_demo = detail.get("route_mode") == "demo_aurangabad"
                    badge_border = "#DC2626" if is_demo else "#059669"
                    badge_color = "#DC2626" if is_demo else "#059669"
                    badge_label = "DEMO ROUTE" if is_demo else "REAL GPS"
                    st.markdown(f"""
                        <div class="kpi-card" style="border-top: 3px solid {badge_border};">
                            <div class="kpi-title">Geolocated Events</div>
                            <div class="kpi-value" style="color: {badge_color};">{geolocated_count}</div>
                            <div class="kpi-sub"><span style="background: {'#FEF2F2' if is_demo else '#ECFDF5'}; color: {badge_color}; border: 1px solid {'#FCA5A5' if is_demo else '#10B981'}; padding: 2px 7px; border-radius: 4px; font-weight:700; font-size: 0.72rem;">{badge_label}</span></div>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

                # Section 3: Interactive GPS Map
                st.markdown("""
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; margin-bottom: 0.6rem;">
                        <h4 style="color: #1C1917; margin: 0;">3. Interactive GPS Anomaly Map — Chhatrapati Sambhajinagar</h4>
                        <div style="display: flex; gap: 14px; font-size: 0.85rem; font-weight: 600;">
                            <span style="color: #B45309;">● Bump ({})</span>
                            <span style="color: #DC2626;">● Pothole ({})</span>
                            <span style="color: #991B1B;">━ Demo Corridor (Jalna Rd)</span>
                        </div>
                    </div>
                """.format(bump_count, pothole_count), unsafe_allow_html=True)

                # Map Controls & Filters
                all_events = map_data.get("events", [])
                
                fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 3])
                with fc1:
                    class_filter = st.selectbox(
                        "Filter by Class",
                        ["All Events", "Bump Only", "Pothole Only"],
                        index=0,
                    )
                with fc2:
                    min_conf = st.slider("Min Confidence (%)", min_value=50, max_value=95, value=50, step=5)
                with fc3:
                    provenance_filter = st.selectbox(
                        "Location Source",
                        ["All Sources", "Demo Route Only", "Real GPS Only"],
                        index=0,
                    )
                with fc4:
                    event_options = ["None"] + [
                        f"Event #{e['id']} | Win {e['window_index']} | {e['predicted_class']} ({e['confidence_percent']}%) at {e['start_seconds']}s"
                        for e in all_events
                    ]
                    selected_event_str = st.selectbox("🎯 Highlight Event on Map", event_options, index=0)

                filtered_events = []
                has_demo_prov = any(e.get("coordinate_provenance") == "demo_route" for e in all_events)
                has_real_prov = any(e.get("coordinate_provenance") in ["actual_gps", "interpolated_gps"] for e in all_events)

                for e in all_events:
                    if class_filter == "Bump Only" and e["predicted_class"] != "Bump":
                        continue
                    if class_filter == "Pothole Only" and e["predicted_class"] != "Pothole":
                        continue
                    if e["confidence_percent"] < min_conf:
                        continue
                    if provenance_filter == "Demo Route Only":
                        if has_demo_prov and e.get("coordinate_provenance") != "demo_route":
                            continue
                    elif provenance_filter == "Real GPS Only":
                        if has_real_prov and e.get("coordinate_provenance") not in ["actual_gps", "interpolated_gps"]:
                            continue
                    filtered_events.append(e)

                highlighted_event_id = None
                if selected_event_str != "None":
                    try:
                        highlighted_event_id = int(selected_event_str.split("|")[0].replace("Event #", "").strip())
                    except Exception:
                        highlighted_event_id = None

                # Build Folium Map strictly centered on Chhatrapati Sambhajinagar
                raw_center = map_data.get("center", {})
                c_lat = raw_center.get("lat", AURANGABAD_DEFAULT_CENTER[0])
                c_lon = raw_center.get("lon", AURANGABAD_DEFAULT_CENTER[1])
                if not (19.70 <= c_lat <= 20.05 and 75.10 <= c_lon <= 75.60):
                    c_lat, c_lon = AURANGABAD_DEFAULT_CENTER[0], AURANGABAD_DEFAULT_CENTER[1]

                center_coords = [c_lat, c_lon]
                zoom_level = 13
                if highlighted_event_id is not None:
                    h_event = next((e for e in all_events if e["id"] == highlighted_event_id), None)
                    if h_event:
                        center_coords = [h_event["latitude"], h_event["longitude"]]
                        zoom_level = 16

                m = folium.Map(
                    location=center_coords,
                    zoom_start=zoom_level,
                    tiles=None,
                    control_scale=True,
                )

                # Base Layer 1: Google Maps Streets (DEFAULT active for clear road names)
                folium.TileLayer(
                    tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
                    attr="Google Maps (Streets)",
                    name="Google Maps (Streets)",
                    max_zoom=20,
                    control=True,
                    show=True,
                ).add_to(m)

                # Base Layer 2: Google Hybrid (Satellite + Roads)
                folium.TileLayer(
                    tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
                    attr="Google Satellite with Road Labels",
                    name="Google Hybrid",
                    max_zoom=20,
                    control=True,
                    show=False,
                ).add_to(m)

                # Base Layer 3: Google Satellite Imagery
                folium.TileLayer(
                    tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                    attr="Google Satellite Imagery",
                    name="Google Satellite",
                    max_zoom=20,
                    control=True,
                    show=False,
                ).add_to(m)

                # Base Layer 4: OpenStreetMap
                folium.TileLayer(
                    "OpenStreetMap",
                    name="OpenStreetMap",
                    control=True,
                    show=False,
                ).add_to(m)

                # Jalna Road (NH-753F) Arterial Corridor Polyline in Chhatrapati Sambhajinagar
                corridor_pts = map_data.get("demo_polyline") or [[wp[0], wp[1]] for wp in AURANGABAD_JALNA_ROAD_WAYPOINTS]
                if corridor_pts:
                    folium.PolyLine(
                        corridor_pts,
                        color="#B91C1C",
                        weight=6,
                        opacity=0.85,
                        tooltip="📍 Jalna Road (NH-753F) Arterial Corridor — Chhatrapati Sambhajinagar",
                    ).add_to(m)

                # Reference Landmark Markers along Jalna Road
                for lm in AURANGABAD_KEY_LANDMARKS:
                    folium.CircleMarker(
                        location=[lm["lat"], lm["lon"]],
                        radius=5,
                        color="#1E3A8A",
                        weight=2,
                        fill=True,
                        fill_color="#60A5FA",
                        fill_opacity=0.9,
                        tooltip=lm["label"],
                        popup=folium.Popup(f"<div style='font-family:sans-serif;'><b>{lm['label']}</b><br><span style='color:#6B7280; font-size:11px;'>Jalna Road Municipal Checkpoint</span></div>", max_width=240),
                    ).add_to(m)

                active_selected_event = None

                if filtered_events:
                    for e in filtered_events:
                        is_highlighted = (e["id"] == highlighted_event_id)
                        if is_highlighted:
                            active_selected_event = e
                        is_bump = (e["predicted_class"] == "Bump")
                        
                        fill_color = "#F59E0B" if is_bump else "#DC2626"
                        radius = 14 if is_highlighted else 9
                        weight = 3 if is_highlighted else 2
                        border_color = "#FFFFFF" if is_highlighted else ("#92400E" if is_bump else "#991B1B")

                        prov = e.get("coordinate_provenance", "none")
                        if prov == "demo_route":
                            prov_badge = "<span style='background:#FEF2F2; color:#DC2626; border:1px solid #FCA5A5; padding:2px 6px; border-radius:4px; font-weight:700;'>DEMO / JALNA RD ROUTE</span>"
                        elif prov == "actual_gps":
                            prov_badge = "<span style='background:#ECFDF5; color:#059669; border:1px solid #10B981; padding:2px 6px; border-radius:4px; font-weight:700;'>REAL SENSOR GPS</span>"
                        elif prov == "interpolated_gps":
                            prov_badge = "<span style='background:#FFFBEB; color:#B45309; border:1px solid #F59E0B; padding:2px 6px; border-radius:4px; font-weight:700;'>INTERPOLATED GPS (&lt;5s GAP)</span>"
                        else:
                            prov_badge = "<span style='background:#F5F5F4; color:#78716C; border:1px solid #E7E5E4; padding:2px 6px; border-radius:4px;'>LOCATION PENDING</span>"

                        landmark_match = find_nearest_aurangabad_landmark(e["latitude"], e["longitude"])
                        landmark_name = landmark_match.get("nearest_corridor_landmark") or landmark_match.get("ward_name", "Jalna Road Corridor")

                        e_gmaps = get_google_maps_url(e["latitude"], e["longitude"])
                        e_street = get_google_street_view_url(e["latitude"], e["longitude"])

                        popup_html = f"""
                            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; width: 255px; font-size: 13px; line-height: 1.4; color: #1C1917;">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px; border-bottom: 1px solid #EADBCC; padding-bottom: 4px;">
                                    <strong style="color: {'#B45309' if is_bump else '#DC2626'}; font-size: 15px;">
                                        {'⚠️ ' if is_bump else '🚨 '}{e['predicted_class'].upper()}
                                    </strong>
                                    <span style="font-size: 11px; color: #78716C;">Event #{e['id']}</span>
                                </div>
                                <div style="margin-bottom: 4px;"><strong>Confidence:</strong> {e['confidence_percent']}%</div>
                                <div style="margin-bottom: 4px;"><strong>Location:</strong> {landmark_name}</div>
                                <div style="margin-bottom: 4px;"><strong>Time:</strong> {e['start_seconds']}s – {e['end_seconds']}s (Win {e['window_index']})</div>
                                <div style="margin-bottom: 4px;"><strong>Coordinates:</strong> {e['latitude']:.5f}, {e['longitude']:.5f}</div>
                                <div style="margin-bottom: 6px;"><strong>Provenance:</strong><br>{prov_badge}</div>
                                <div style="margin-top: 8px; display: flex; gap: 6px;">
                                    <a href="{e_street}" target="_blank" style="flex: 1; text-align: center; background: #FAF7F2; color: #DC2626; border: 1px solid #DC2626; padding: 4px 6px; border-radius: 4px; font-size: 11px; font-weight: 700; text-decoration: none;">
                                        🌐 Street View
                                    </a>
                                    <a href="{e_gmaps}" target="_blank" style="flex: 1; text-align: center; background: #DC2626; color: #FFFFFF; padding: 4px 6px; border-radius: 4px; font-size: 11px; font-weight: 700; text-decoration: none;">
                                        📍 Google Maps
                                    </a>
                                </div>
                            </div>
                        """

                        tooltip_text = f"#{e['id']} {e['predicted_class']} ({e['confidence_percent']}%) at {e['start_seconds']}s — {landmark_name}"

                        folium.CircleMarker(
                            location=[e["latitude"], e["longitude"]],
                            radius=radius,
                            color=border_color,
                            weight=weight,
                            fill=True,
                            fill_color=fill_color,
                            fill_opacity=0.95 if is_highlighted else 0.85,
                            popup=folium.Popup(popup_html, max_width=300),
                            tooltip=tooltip_text,
                        ).add_to(m)

                    # Auto-fit bounds to the anomalies along the corridor
                    if highlighted_event_id is None:
                        pts_lat = [e["latitude"] for e in filtered_events if e.get("latitude")]
                        pts_lon = [e["longitude"] for e in filtered_events if e.get("longitude")]
                        if pts_lat and pts_lon:
                            m.fit_bounds([[min(pts_lat) - 0.002, min(pts_lon) - 0.002], [max(pts_lat) + 0.002, max(pts_lon) + 0.002]])
                else:
                    # If filtered_events is empty, fit bounds to the Jalna Road corridor
                    if highlighted_event_id is None and corridor_pts:
                        c_lats = [pt[0] for pt in corridor_pts]
                        c_lons = [pt[1] for pt in corridor_pts]
                        m.fit_bounds([[min(c_lats) - 0.003, min(c_lons) - 0.003], [max(c_lats) + 0.003, max(c_lons) + 0.003]])

                # Add Layer Control to toggle Google Maps layers
                folium.LayerControl(position="topright", collapsed=False).add_to(m)

                st_map = st_folium(
                    m,
                    use_container_width=True,
                    height=520,
                    returned_objects=["last_object_clicked"],
                )

                if not filtered_events:
                    st.info("ℹ️ No anomalies match the active filters. Showing Chhatrapati Sambhajinagar Jalna Road corridor. Try adjusting 'Filter by Class', 'Min Confidence', or 'Location Source'.")

                # Fallback to first filtered event (or any event in dataset) if none explicitly highlighted
                if not active_selected_event:
                    if filtered_events:
                        active_selected_event = filtered_events[0]
                    elif all_events:
                        active_selected_event = all_events[0]

                # Section 3B: Integrated Live Dashcam & Google Street View Anomaly Inspector
                if active_selected_event:
                    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
                    st.markdown("""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                            <h4 style="color: #1C1917; margin: 0;">
                                📹 Live Dashcam Footage &amp; Google Street View Pothole Inspector
                            </h4>
                            <span style="background: #FAF7F2; color: #78716C; border: 1px solid #EADBCC; padding: 3px 9px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">
                                Synchronized to Selected Anomaly (Chhatrapati Sambhajinagar)
                            </span>
                        </div>
                    """, unsafe_allow_html=True)

                    lm_info = find_nearest_aurangabad_landmark(active_selected_event.get("latitude", AURANGABAD_DEFAULT_CENTER[0]), active_selected_event.get("longitude", AURANGABAD_DEFAULT_CENTER[1]))
                    e_loc_name = f"Jalna Road (NH-753F) near {lm_info.get('ward_name', 'Window #' + str(active_selected_event.get('window_index', 0)))}"
                    render_live_footage_inspector(
                        distress_type=active_selected_event.get("predicted_class", "Pothole"),
                        confidence=active_selected_event.get("confidence_percent", 90.0) / 100.0,
                        location_name=e_loc_name,
                        latitude=float(active_selected_event.get("latitude", AURANGABAD_DEFAULT_CENTER[0])),
                        longitude=float(active_selected_event.get("longitude", AURANGABAD_DEFAULT_CENTER[1])),
                        speed_kmh=float(active_selected_event.get("speed") or 40.0),
                    )

                st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

                # Section 4: Raw Sensor Waveforms (Accelerometer & Gyroscope)
                st.markdown("<h4 style='color: #1C1917; margin-bottom: 0.6rem;'>4. Multi-Axis Kinematic Signal Waveforms</h4>", unsafe_allow_html=True)
                
                # Check for raw CSV data in local paths
                raw_df = None
                if os.path.exists(SAMPLE_10_PATH):
                    try:
                        raw_df = pd.read_csv(SAMPLE_10_PATH)
                    except Exception:
                        raw_df = None

                if raw_df is not None and not raw_df.empty:
                    # Accelerometer Waveforms
                    fig_acc = go.Figure()
                    t = raw_df["seconds_elapsed"] if "seconds_elapsed" in raw_df.columns else np.arange(len(raw_df)) / 50.0
                    
                    if "accelerometer_x" in raw_df.columns:
                        fig_acc.add_trace(go.Scatter(x=t, y=raw_df["accelerometer_x"], name="Acc X (Lateral)", line=dict(color="#2563EB", width=1.2)))
                    if "accelerometer_y" in raw_df.columns:
                        fig_acc.add_trace(go.Scatter(x=t, y=raw_df["accelerometer_y"], name="Acc Y (Forward)", line=dict(color="#D97706", width=1.2)))
                    if "accelerometer_z" in raw_df.columns:
                        fig_acc.add_trace(go.Scatter(x=t, y=raw_df["accelerometer_z"], name="Acc Z (Vertical Shock)", line=dict(color="#DC2626", width=1.5)))

                    fig_acc.update_layout(
                        title="3-Axis Accelerometer Kinematic Waves (Shock & Impact Transients)",
                        xaxis_title="Time Elapsed (seconds)",
                        yaxis_title="Acceleration (m/s²)",
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FAF7F2",
                        font=dict(color="#1C1917"),
                        height=280,
                        margin=dict(l=40, r=40, t=40, b=40),
                        xaxis=dict(gridcolor="#EADBCC"),
                        yaxis=dict(gridcolor="#EADBCC"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig_acc, use_container_width=True)

                    # Gyroscope Waveforms
                    fig_gyro = go.Figure()
                    if "gyroscope_x" in raw_df.columns:
                        fig_gyro.add_trace(go.Scatter(x=t, y=raw_df["gyroscope_x"], name="Gyro X (Pitch Rate)", line=dict(color="#059669", width=1.2)))
                    if "gyroscope_y" in raw_df.columns:
                        fig_gyro.add_trace(go.Scatter(x=t, y=raw_df["gyroscope_y"], name="Gyro Y (Roll Rate)", line=dict(color="#7C3AED", width=1.2)))
                    if "gyroscope_z" in raw_df.columns:
                        fig_gyro.add_trace(go.Scatter(x=t, y=raw_df["gyroscope_z"], name="Gyro Z (Yaw Rate)", line=dict(color="#991B1B", width=1.2)))

                    fig_gyro.update_layout(
                        title="3-Axis Gyroscope Waves (Angular Pitch & Roll Dynamics)",
                        xaxis_title="Time Elapsed (seconds)",
                        yaxis_title="Angular Velocity (rad/s)",
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FAF7F2",
                        font=dict(color="#1C1917"),
                        height=260,
                        margin=dict(l=40, r=40, t=40, b=40),
                        xaxis=dict(gridcolor="#EADBCC"),
                        yaxis=dict(gridcolor="#EADBCC"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig_gyro, use_container_width=True)

                st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

                # Section 5: Predicted Road Distress Anomaly Timeline
                st.markdown("<h4 style='color: #1C1917; margin-bottom: 0.6rem;'>5. Predicted Road Distress Anomaly Timeline</h4>", unsafe_allow_html=True)

                df_win = pd.DataFrame(detail.get("windows", []))
                if not df_win.empty:
                    fig_timeline = go.Figure()
                    for pred_cls, color, symbol in [("Bump", "#D97706", "diamond"), ("Pothole", "#DC2626", "cross")]:
                        subset = df_win[df_win["predicted_class"] == pred_cls]
                        if not subset.empty:
                            fig_timeline.add_trace(go.Scatter(
                                x=subset["start_seconds"],
                                y=subset["confidence"] * 100.0,
                                mode="markers",
                                name=f"{pred_cls} ({len(subset)})",
                                marker=dict(size=11, color=color, symbol=symbol, line=dict(width=1, color="#FFFFFF")),
                                hovertemplate="Time: %{x}s<br>Confidence: %{y:.1f}%<extra></extra>",
                            ))

                    fig_timeline.update_layout(
                        title="Road Distress Classification Events Along Recording Duration",
                        xaxis_title="Elapsed Time (seconds)",
                        yaxis_title="Model Confidence (%)",
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FAF7F2",
                        font=dict(color="#1C1917"),
                        height=280,
                        margin=dict(l=40, r=40, t=40, b=40),
                        xaxis=dict(gridcolor="#EADBCC"),
                        yaxis=dict(gridcolor="#EADBCC"),
                        legend=dict(
                            orientation="h", 
                            yanchor="bottom", 
                            y=1.02, 
                            xanchor="right", 
                            x=1,
                            bgcolor="rgba(250, 247, 242, 0.95)",
                            bordercolor="#EADBCC",
                            borderwidth=1,
                        ),
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)

                st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

                # Section 6: Geolocated Prediction Windows Table & CSV Download
                st.markdown("<h4 style='color: #1C1917; margin-bottom: 0.6rem;'>6. Geolocated Prediction Windows Table</h4>", unsafe_allow_html=True)

                if not df_win.empty:
                    table_cols = [
                        "window_index",
                        "start_seconds",
                        "end_seconds",
                        "predicted_class",
                        "confidence_percent",
                        "latitude",
                        "longitude",
                        "coordinate_provenance",
                        "is_location_pending",
                    ]
                    available_cols = [c for c in table_cols if c in df_win.columns]
                    display_df = df_win[available_cols].rename(columns={
                        "window_index": "Window #",
                        "start_seconds": "Start (s)",
                        "end_seconds": "End (s)",
                        "predicted_class": "Event Class",
                        "confidence_percent": "Confidence (%)",
                        "latitude": "Latitude",
                        "longitude": "Longitude",
                        "coordinate_provenance": "Provenance Source",
                        "is_location_pending": "Location Pending",
                    })

                    st.dataframe(display_df, use_container_width=True)

                    csv_export_data = display_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Results CSV (Predictions + GPS Provenance)",
                        data=csv_export_data,
                        file_name=f"roadsense_predictions_gps_{current_upload_id}.csv",
                        mime="text/csv",
                    )
            else:
                st.warning(f"Could not load details for Upload #{current_upload_id}.")
        else:
            st.info("👆 Please upload a smartphone sensor CSV or click 'Process 10.csv' above to execute model inference and view the interactive GPS map.")

    with tab_physics:
        render_kinematics_explainer()
        
        # Deep Dive on specific recorded windows from the upload
        if current_upload_id:
            detail_phys = get_sensor_upload_detail(current_upload_id)
            if detail_phys and detail_phys.get("windows"):
                st.markdown("<h4 style='color: #1C1917; margin-top: 1.5rem; margin-bottom: 0.6rem;'>🔍 Real Dataset Window Waveform &amp; Physics Deep-Dive</h4>", unsafe_allow_html=True)
                p_windows = detail_phys["windows"]
                phys_options = [
                    f"Window #{w['window_index']} ({w['start_seconds']}s - {w['end_seconds']}s) | {w['predicted_class']} ({w['confidence_percent']}%)"
                    for w in p_windows
                ]
                selected_phys_str = st.selectbox("Choose Detected Window to Inspect", phys_options, key="phys_window_selector")
                sel_w_idx = int(selected_phys_str.split("#")[1].split(" ")[0])
                chosen_window = next((w for w in p_windows if w["window_index"] == sel_w_idx), p_windows[0])
                
                # Fetch raw dataframe if available
                raw_df_phys = None
                if os.path.exists(SAMPLE_10_PATH):
                    try:
                        raw_df_phys = pd.read_csv(SAMPLE_10_PATH)
                    except Exception:
                        pass
                render_window_kinematic_deepdive(chosen_window, raw_df=raw_df_phys)

    with tab_features:
        st.markdown("<h4 style='color: #1C1917; margin-bottom: 0.6rem;'>🔬 30-Kinematic Feature Vector Deep Dive (Random Forest Model Inputs)</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color: #78716C; font-size: 0.9rem;'>Every 1.0-second sensor window extracts exactly 30 mathematical features capturing lateral, forward, and vertical shocks and angular pitch/roll dynamics.</p>", unsafe_allow_html=True)

        if current_upload_id:
            detail_feat = get_sensor_upload_detail(current_upload_id)
            if detail_feat and detail_feat.get("windows"):
                windows = detail_feat["windows"]
                win_selector_options = [
                    f"Window #{w['window_index']} ({w['start_seconds']}s - {w['end_seconds']}s) | Class: {w['predicted_class']} ({w['confidence_percent']}%)"
                    for w in windows
                ]
                selected_win_str = st.selectbox("Select Window to Inspect 30 Features", win_selector_options, index=0)
                selected_win_idx = int(selected_win_str.split("#")[1].split(" ")[0])
                target_win = next((w for w in windows if w["window_index"] == selected_win_idx), windows[0])
                feats = target_win.get("features", {})

                st.markdown(f"""
                    <div style="background: #FAF7F2; border: 1px solid #EADBCC; border-radius: 8px; padding: 12px; margin-bottom: 1rem;">
                        <strong>Inspecting Window #{selected_win_idx}:</strong> Time: <code>{target_win['start_seconds']}s - {target_win['end_seconds']}s</code> • 
                        Predicted Class: <strong style="color: {'#B45309' if target_win['predicted_class'] == 'Bump' else '#DC2626'};">{target_win['predicted_class'].upper()}</strong> • 
                        Confidence: <code>{target_win['confidence_percent']}%</code> • 
                        Coordinates: <code>{target_win['latitude']:.5f}, {target_win['longitude']:.5f}</code>
                    </div>
                """, unsafe_allow_html=True)

                col_f1, col_f2, col_f3 = st.columns(3)
                
                with col_f1:
                    st.markdown("<h5 style='color: #1C1917; border-bottom: 2px solid #2563EB; padding-bottom: 4px;'>📊 Accelerometer Kinematics (X, Y, Z)</h5>", unsafe_allow_html=True)
                    acc_keys = [
                        ("Acc X Std Dev", "accelerometer_x_std"),
                        ("Acc X Peak-to-Peak", "accelerometer_x_range"),
                        ("Acc X Max Absolute", "accelerometer_x_max_abs"),
                        ("Acc X Root Mean Sq", "accelerometer_x_rms"),
                        ("Acc Y Std Dev", "accelerometer_y_std"),
                        ("Acc Y Peak-to-Peak", "accelerometer_y_range"),
                        ("Acc Y Max Absolute", "accelerometer_y_max_abs"),
                        ("Acc Y Root Mean Sq", "accelerometer_y_rms"),
                        ("Acc Z (Shock) Std Dev", "accelerometer_z_std"),
                        ("Acc Z (Shock) Range", "accelerometer_z_range"),
                        ("Acc Z (Shock) Max Abs", "accelerometer_z_max_abs"),
                        ("Acc Z (Shock) RMS", "accelerometer_z_rms"),
                    ]
                    acc_rows = []
                    for label, key in acc_keys:
                        val = feats.get(key, 0.0)
                        acc_rows.append({"Feature Name": label, "Value": f"{val:.5f}"})
                    st.dataframe(pd.DataFrame(acc_rows), use_container_width=True, hide_index=True)

                with col_f2:
                    st.markdown("<h5 style='color: #1C1917; border-bottom: 2px solid #059669; padding-bottom: 4px;'>🔄 Gyroscope Dynamics (Pitch, Roll, Yaw)</h5>", unsafe_allow_html=True)
                    gyro_keys = [
                        ("Gyro X (Pitch) Std Dev", "gyroscope_x_std"),
                        ("Gyro X (Pitch) Range", "gyroscope_x_range"),
                        ("Gyro X (Pitch) Max Abs", "gyroscope_x_max_abs"),
                        ("Gyro X (Pitch) RMS", "gyroscope_x_rms"),
                        ("Gyro Y (Roll) Std Dev", "gyroscope_y_std"),
                        ("Gyro Y (Roll) Range", "gyroscope_y_range"),
                        ("Gyro Y (Roll) Max Abs", "gyroscope_y_max_abs"),
                        ("Gyro Y (Roll) RMS", "gyroscope_y_rms"),
                        ("Gyro Z (Yaw) Std Dev", "gyroscope_z_std"),
                        ("Gyro Z (Yaw) Range", "gyroscope_z_range"),
                        ("Gyro Z (Yaw) Max Abs", "gyroscope_z_max_abs"),
                        ("Gyro Z (Yaw) RMS", "gyroscope_z_rms"),
                    ]
                    gyro_rows = []
                    for label, key in gyro_keys:
                        val = feats.get(key, 0.0)
                        gyro_rows.append({"Feature Name": label, "Value": f"{val:.5f}"})
                    st.dataframe(pd.DataFrame(gyro_rows), use_container_width=True, hide_index=True)

                with col_f3:
                    st.markdown("<h5 style='color: #1C1917; border-bottom: 2px solid #DC2626; padding-bottom: 4px;'>💥 Magnitude, Jerk &amp; Peak Transients</h5>", unsafe_allow_html=True)
                    dyn_keys = [
                        ("Total Magnitude Std Dev", "magnitude_std"),
                        ("Total Magnitude Max", "magnitude_max"),
                        ("Total Magnitude RMS", "magnitude_rms"),
                        ("Kinematic Jerk Std Dev", "jerk_std"),
                        ("Kinematic Jerk Max Abs", "jerk_max_abs"),
                        ("Shock Peak Count", "peak_count"),
                    ]
                    dyn_rows = []
                    for label, key in dyn_keys:
                        val = feats.get(key, 0.0)
                        dyn_rows.append({"Feature Name": label, "Value": f"{val:.5f}"})
                    st.dataframe(pd.DataFrame(dyn_rows), use_container_width=True, hide_index=True)

                # Feature comparison across all windows
                st.markdown("<h5 style='color: #1C1917; margin-top: 1rem;'>📈 Feature Distribution Across Windows (Shock Severity Comparison)</h5>", unsafe_allow_html=True)
                comparison_records = []
                for w in windows:
                    wf = w.get("features", {})
                    comparison_records.append({
                        "Win #": w["window_index"],
                        "Class": w["predicted_class"],
                        "Conf (%)": w["confidence_percent"],
                        "Acc_Z_RMS": wf.get("accelerometer_z_rms", 0),
                        "Acc_Z_Range": wf.get("accelerometer_z_range", 0),
                        "Jerk_Max": wf.get("jerk_max_abs", 0),
                        "Mag_Max": wf.get("magnitude_max", 0),
                        "Peaks": wf.get("peak_count", 0),
                    })
                st.dataframe(pd.DataFrame(comparison_records), use_container_width=True, hide_index=True)

    with tab_history:
        st.markdown("<h4 style='color: #1C1917; margin-bottom: 0.6rem;'>📋 Historical Sensor Uploads</h4>", unsafe_allow_html=True)
        uploads = get_sensor_uploads(limit=30)

        if uploads:
            records = []
            for u in uploads:
                records.append({
                    "Upload ID": u.get("id"),
                    "File Name": u.get("original_name"),
                    "Raw Rows": u.get("row_count"),
                    "Classified Windows": u.get("valid_window_count"),
                    "Route Mode": u.get("route_mode", "real_gps"),
                    "GPS Enabled": "Yes" if u.get("has_gps") else "No",
                    "Status": u.get("status"),
                    "Timestamp": str(u.get("created_at"))[:19].replace("T", " "),
                })
            
            df_hist = pd.DataFrame(records)
            st.dataframe(df_hist, use_container_width=True)

            st.markdown("<strong style='color: #1C1917;'>Select an upload to view on the interactive map:</strong>", unsafe_allow_html=True)
            hist_options = [u["id"] for u in uploads]
            chosen_upload = st.selectbox("Choose Historical Upload ID", hist_options, index=0)
            if st.button("🗺️ Load Selected Upload to Map"):
                st.session_state["current_upload_id"] = chosen_upload
                st.success(f"✓ Loaded Upload #{chosen_upload}. Switching to Map tab...")
                st.rerun()
        else:
            st.info("No sensor recordings uploaded yet.")


# Execute when loaded as a page
render_sensor_upload()
