"""
RoadSense AI – Smartphone Sensor Road Monitoring System
=======================================================
Executive Dashboard, Driver Interface, Municipal Authority Interface,
Sensor CSV Waveform Analyzer, and Road Memory Spatial Engine.
"""

import os
import sys
import streamlit as st

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.database import init_db
from utils.ui_theme import apply_theme, render_sidebar_telemetry

# Page Configuration
st.set_page_config(
    page_title="RoadSense AI – Sensor Road Damage & Memory System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize SQLite database on boot
init_db()

# Apply White, Warm Beige & Crimson Red Theme
apply_theme()

# Modern Multi-Page Navigation
dashboard_page = st.Page("pages/dashboard_overview.py", title="Executive Dashboard", icon="📊", default=True)
driver_page = st.Page("pages/driver_view.py", title="Driver Interface", icon="🚗")
authority_page = st.Page("pages/authority_view.py", title="Authority Interface", icon="🏛️")
sensor_page = st.Page("pages/sensor_upload.py", title="Sensor CSV Ingestion", icon="📲")
api_page = st.Page("pages/api_console.py", title="API & System Console", icon="🌐")

pg = st.navigation({
    "RoadSense Command Center": [
        dashboard_page,
        driver_page,
        authority_page,
        sensor_page,
        api_page,
    ]
})

# Render Sidebar Telemetry
render_sidebar_telemetry()

# Run the active page
pg.run()
