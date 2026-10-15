"""
RoadSense AI - Shared UI & Styling Theme Helper
==============================================
Provides unified styling for the Light Command Center theme:
Pure White background, Warm Beige containers, and Crimson Red accents.
"""

import streamlit as st
from utils.api_client import get_dashboard_summary, is_backend_online

THEME_CSS = """
<style>
    /* Global Base */
    .stApp {
        background-color: #FFFFFF;
        color: #1C1917;
    }
    
    /* Modern Warm Beige KPI Cards */
    .kpi-card {
        background-color: #FAF7F2;
        border: 1px solid #E8DFD5;
        border-radius: 12px;
        padding: 1.3rem 1.2rem;
        box-shadow: 0 2px 10px rgba(185, 28, 28, 0.04), 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        border-color: #DC2626;
        box-shadow: 0 6px 20px rgba(220, 38, 38, 0.10);
        transform: translateY(-2px);
    }
    .kpi-title {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #78716C;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 2.3rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.3rem;
        color: #1C1917;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #78716C;
        font-weight: 500;
    }

    /* Buttons */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #E5DCD2;
        background-color: #FAF7F2;
        color: #1C1917;
        transition: all 0.15s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        border-color: #DC2626;
        color: #DC2626;
        background-color: #FEF2F2;
    }
    div.stButton > button[kind="primary"] {
        background-color: #DC2626 !important;
        color: #FFFFFF !important;
        border: 1px solid #B91C1C !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #B91C1C !important;
        color: #FFFFFF !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F8F5F0;
        border-right: 1px solid #EADBCC;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #EADBCC;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        color: #57534E;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #DC2626 !important;
        border-bottom-color: #DC2626 !important;
    }
</style>
"""


def apply_theme():
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def render_sidebar_telemetry():
    with st.sidebar:
        st.markdown("""
            <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #EADBCC; text-align: center;">
                <div style="font-size: 2.2rem; margin-bottom: 4px;">📡</div>
                <h2 style="margin: 0; color: #1C1917; font-size: 1.45rem; font-weight: 800; letter-spacing: -0.3px;">
                    RoadSense <span style="color: #DC2626;">AI</span>
                </h2>
                <p style="margin: 3px 0 0 0; color: #78716C; font-size: 0.8rem; font-weight: 500;">
                    Sensor Road Anomaly &amp; Memory System
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

        backend_up = is_backend_online()
        summary = get_dashboard_summary()
        hotspot_count = summary.get("chronic_hotspots", 0)
        urgent_count = summary.get("critical_incidents", 0)

        st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #EADBCC; border-radius: 10px; padding: 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                <div style="color: #78716C; font-size: 0.74rem; text-transform: uppercase; font-weight: 700; display: flex; align-items: center; gap: 6px;">
                    <span>🧠</span> Road Memory Engine
                </div>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 8px;">
                    <span style="color: #1C1917; font-size: 0.88rem; font-weight: 500;">Active Clusters:</span>
                    <span style="color: #DC2626; font-weight: 800; font-size: 1.15rem;">{hotspot_count}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 3px;">
                    <span style="color: #1C1917; font-size: 0.88rem; font-weight: 500;">Urgent Triage (P1):</span>
                    <span style="color: #B91C1C; font-weight: 800; font-size: 1.15rem;">{urgent_count}</span>
                </div>
                <div style="border-top: 1px solid #F0EAE1; margin-top: 8px; padding-top: 8px; font-size: 0.72rem; color: #78716C;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>FastAPI Backend:</span>
                        <span style="color: {'#059669' if backend_up else '#DC2626'}; font-weight: 700;">
                            {'Online :8000' if backend_up else 'Offline'}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 2px;">
                        <span>Sensor ML:</span>
                        <span style="color: #059669; font-weight: 700;">Random Forest</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 2px;">
                        <span>Vision Model:</span>
                        <span style="color: #059669; font-weight: 700;">YOLO (RDD2022)</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.caption("RoadSense AI • Road Damage Monitoring System v2.0")
