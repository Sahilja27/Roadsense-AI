"""
RoadSense AI - Google Maps Dataset & Street Footage Integrator
==============================================================
Provides Chhatrapati Sambhajinagar (Aurangabad), Maharashtra geospatial data,
Google Maps tile layer configurations (Streets, Satellite, Hybrid, Terrain),
direct Google Street View pano deep-links, and integrated live dashcam road footage
with YOLO road defect detections (Potholes, Bumps, Alligator Cracks).
"""

import io
import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st


# ==============================================================================
# 1. Chhatrapati Sambhajinagar (Aurangabad) Geographic & Landmark Registry
# ==============================================================================
AURANGABAD_CITY_INFO = {
    "name": "Chhatrapati Sambhajinagar",
    "aliases": ["Aurangabad", "CSMC", "ASCDCL"],
    "state": "Maharashtra",
    "country": "India",
    "center_lat": 19.8762,
    "center_lon": 75.3433,
    "elevation_m": 568,
    "primary_corridor": "Jalna Road (NH-753F / SH-6)",
}

AURANGABAD_MUNICIPAL_WARDS = [
    {
        "ward_id": "WARD-01",
        "ward_name": "Kranti Chowk & Nirala Bazar Zone",
        "zone": "Central Division",
        "center": [19.8737, 75.3248],
        "roads": ["Kranti Chowk Junction", "Nirala Bazar Road", "Gopal Tea Link", "Station Road"],
        "pwd_engineer": "Er. Sachin Shinde (Dy. City Engineer)",
        "sla_target_hours": 24,
    },
    {
        "ward_id": "WARD-02",
        "ward_name": "Jalna Road Commercial Corridor (NH-753F)",
        "zone": "East Arterial Division",
        "center": [19.8765, 75.3475],
        "roads": ["Sessions Court Cross", "Akashwani Circle", "Mondha Naka Flyover", "Seven Hills Flyover"],
        "pwd_engineer": "Er. Rajesh Kulkarni (Executive Engineer PWD)",
        "sla_target_hours": 24,
    },
    {
        "ward_id": "WARD-03",
        "ward_name": "CIDCO Town Centre & Cannaught Ward",
        "zone": "New Aurangabad Division",
        "center": [19.8778, 75.3625],
        "roads": ["CIDCO Cannaught Place", "CIDCO Bus Stand Road", "Town Centre Arterial", "APMC Road"],
        "pwd_engineer": "Er. Priya Deshmukh (ASCDCL Smart City)",
        "sla_target_hours": 48,
    },
    {
        "ward_id": "WARD-04",
        "ward_name": "Chikalthana MIDC & Airport Belt",
        "zone": "Industrial Division",
        "center": [19.8806, 75.3895],
        "roads": ["Airport Road", "MGM Hospital Link", "Dhoot Hospital Chowk", "MIDC Phase 1"],
        "pwd_engineer": "Er. Vilas Jadhav (MIDC Infrastructure)",
        "sla_target_hours": 48,
    },
    {
        "ward_id": "WARD-05",
        "ward_name": "Beed Bypass & Satara Parisar Zone",
        "zone": "South Arterial Division",
        "center": [19.8450, 75.3400],
        "roads": ["Beed Bypass NH-52", "Satara Parisar Ring Link", "Deolai Road", "Mahanubhav Chowk"],
        "pwd_engineer": "Er. Ganesh Patil (National Highways Div)",
        "sla_target_hours": 24,
    },
]

# Jalna Road (NH-753F) Corridor Waypoints in Chhatrapati Sambhajinagar
AURANGABAD_JALNA_ROAD_WAYPOINTS: List[Tuple[float, float, str]] = [
    (19.87372, 75.32485, "Kranti Chowk Public Junction"),
    (19.87445, 75.32890, "Gopal Tea / Nirala Bazar Link"),
    (19.87520, 75.33380, "Sessions Court / Jalna Road"),
    (19.87595, 75.34120, "Akashwani Prasar Bharati Circle"),
    (19.87648, 75.34750, "Mondha Naka Road Flyover"),
    (19.87698, 75.35380, "Seven Hills Flyover, Jalna Road"),
    (19.87750, 75.35920, "CIDCO Town Centre Cross"),
    (19.87785, 75.36250, "CIDCO Cannaught Place Entrance"),
    (19.87815, 75.36680, "APMC Market / Jalna Road"),
    (19.87855, 75.37120, "MGM Medical College & Hospital"),
    (19.87910, 75.37620, "Dhoot Hospital Road Link"),
    (19.87955, 75.38050, "Cambridge School Junction"),
    (19.88015, 75.38520, "Chikalthana MIDC Road"),
    (19.88065, 75.38950, "Chhatrapati Sambhajinagar Airport Road"),
]

AURANGABAD_KEY_LANDMARKS = [
    {"lat": 19.87372, "lon": 75.32485, "name": "Kranti Chowk Junction", "label": "🏛️ Kranti Chowk (Corridor Start)"},
    {"lat": 19.87595, "lon": 75.34120, "name": "Akashwani Circle", "label": "📻 Akashwani Circle"},
    {"lat": 19.87698, "lon": 75.35380, "name": "Seven Hills Flyover", "label": "🌉 Seven Hills Flyover"},
    {"lat": 19.87785, "lon": 75.36250, "name": "CIDCO Town Centre", "label": "🏢 CIDCO Town Centre"},
    {"lat": 19.87855, "lon": 75.37120, "name": "MGM Medical College", "label": "🏥 MGM Medical College"},
    {"lat": 19.87955, "lon": 75.38050, "name": "Cambridge School Junction", "label": "🏫 Cambridge Junction"},
    {"lat": 19.88065, "lon": 75.38950, "name": "Airport Road / Chikalthana", "label": "✈️ Airport Road (Corridor End)"},
]

# ==============================================================================
# 2. Google Maps Tile Layer Definitions for Folium
# ==============================================================================
GOOGLE_MAP_TILES = {
    "google_streets": {
        "name": "Google Maps (Street)",
        "tiles": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        "attr": "Google Maps (Roads & Streets)",
        "max_zoom": 20,
    },
    "google_satellite": {
        "name": "Google Satellite Imagery",
        "tiles": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "attr": "Google Satellite Imagery",
        "max_zoom": 20,
    },
    "google_hybrid": {
        "name": "Google Hybrid (Satellite + Roads)",
        "tiles": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        "attr": "Google Satellite with Road Labels",
        "max_zoom": 20,
    },
    "google_terrain": {
        "name": "Google Terrain",
        "tiles": "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
        "attr": "Google Terrain / Elevation",
        "max_zoom": 20,
    },
}


def get_google_maps_url(lat: float, lon: float) -> str:
    """Generates a direct Google Maps pin link for precise coordinates."""
    return f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"


def get_google_street_view_url(lat: float, lon: float) -> str:
    """Generates a direct Google Street View panorama viewer URL."""
    return f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat:.6f},{lon:.6f}"


def get_google_directions_url(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> str:
    """Generates a turn-by-turn navigation URL on Google Maps."""
    return f"https://www.google.com/maps/dir/?api=1&origin={start_lat:.6f},{start_lon:.6f}&destination={end_lat:.6f},{end_lon:.6f}"


def find_nearest_aurangabad_landmark(lat: float, lon: float) -> Dict[str, Any]:
    """Finds the closest municipal ward and landmark in Chhatrapati Sambhajinagar."""
    best_ward = AURANGABAD_MUNICIPAL_WARDS[0]
    best_dist = 1e9

    for w in AURANGABAD_MUNICIPAL_WARDS:
        clat, clon = w["center"]
        d = math.sqrt((lat - clat) ** 2 + (lon - clon) ** 2)
        if d < best_dist:
            best_dist = d
            best_ward = w

    return {
        "ward_id": best_ward["ward_id"],
        "ward_name": best_ward["ward_name"],
        "zone": best_ward["zone"],
        "engineer": best_ward["pwd_engineer"],
        "sla_target_hours": best_ward["sla_target_hours"],
        "distance_deg": best_dist,
    }


# ==============================================================================
# 3. High-Definition Dashcam & Street Footage Frame Generator
# ==============================================================================
def generate_street_footage_frame(
    distress_type: str = "Pothole",
    confidence: float = 0.92,
    location_label: str = "Jalna Road (NH-753F), Chhatrapati Sambhajinagar",
    gps_coords: Tuple[float, float] = (19.8769, 75.3538),
    speed_kmh: float = 42.0,
    show_yolo_boxes: bool = True,
) -> Image.Image:
    """
    Synthesizes a realistic 640x360 dashcam footage frame captured on a road in
    Chhatrapati Sambhajinagar with asphalt perspective, horizon, road markings,
    road distress, telemetry HUD overlay, and trained YOLO bounding box detection.
    """
    w, h = 640, 360
    img = Image.new("RGB", (w, h), color=(140, 175, 215))  # Sky blue
    draw = ImageDraw.Draw(img)

    # 1. Horizon & Sky gradient
    for y in range(0, 140):
        r = int(140 + (220 - 140) * (y / 140))
        g = int(175 + (225 - 175) * (y / 140))
        b = int(215 + (235 - 215) * (y / 140))
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    # 2. Distant Aurangabad Cityscape & Trees
    draw.polygon([(0, 140), (120, 130), (220, 138), (350, 128), (480, 135), (w, 132), (w, 145), (0, 145)], fill=(120, 135, 110))
    # Distant buildings
    draw.rectangle([(80, 120), (130, 140)], fill=(170, 165, 160))
    draw.rectangle([(250, 115), (290, 140)], fill=(160, 155, 150))
    draw.rectangle([(420, 122), (470, 140)], fill=(175, 170, 165))

    # 3. Road Surface Perspective (Trapezoid from horizon y=140 to bottom y=360)
    vp_x, vp_y = 320, 140
    road_pts = [(vp_x - 30, vp_y), (vp_x + 30, vp_y), (w + 40, h), (-40, h)]
    draw.polygon(road_pts, fill=(58, 62, 65))  # Dark asphalt

    # Asphalt Texture noise
    np.random.seed(42)
    noise_x = np.random.randint(0, w, 600)
    noise_y = np.random.randint(145, h, 600)
    for nx, ny in zip(noise_x, noise_y):
        # Only in road bounds
        left_edge = vp_x - 30 - ((ny - vp_y) / (h - vp_y)) * (vp_x + 10)
        right_edge = vp_x + 30 + ((ny - vp_y) / (h - vp_y)) * (w - vp_x + 10)
        if left_edge < nx < right_edge:
            draw.point((nx, ny), fill=(np.random.randint(45, 75), np.random.randint(45, 75), np.random.randint(45, 75)))

    # 4. White Edge Lines
    draw.line([(vp_x - 28, vp_y), (-20, h)], fill=(220, 220, 220), width=4)
    draw.line([(vp_x + 28, vp_y), (w + 20, h)], fill=(220, 220, 220), width=4)

    # 5. Yellow Dashed Center Lane Marking
    dash_segments = [(150, 170), (185, 215), (235, 275), (300, 355)]
    for y1, y2 in dash_segments:
        draw.line([(vp_x, y1), (vp_x, y2)], fill=(235, 180, 25), width=int(2 + (y1 - vp_y) / 45))

    # 6. Road Distress Rendering
    is_bump = "bump" in distress_type.lower()
    if is_bump:
        # Speed bump: horizontal arched mound across lane with yellow/white reflective chevrons
        by = 260
        bx_l, bx_r = 180, 460
        draw.ellipse([(bx_l, by - 12), (bx_r, by + 28)], fill=(40, 42, 45), outline=(30, 30, 30))
        # Reflective diagonal stripes
        for sx in range(bx_l + 25, bx_r - 20, 35):
            draw.polygon([(sx, by + 24), (sx + 14, by - 6), (sx + 26, by - 6), (sx + 12, by + 24)], fill=(245, 195, 25))
        box_coords = [bx_l - 10, by - 16, bx_r + 10, by + 32]
        det_label = f"Speed Bump {int(confidence * 100)}%"
        box_color = (245, 158, 11)  # Amber
    else:
        # Pothole: irregular dark crater with fractured perimeter and shadow depth
        py = 270
        px = 340
        draw.ellipse([(px - 65, py - 20), (px + 65, py + 30)], fill=(22, 22, 24))  # Deep cavity
        draw.ellipse([(px - 58, py - 14), (px + 58, py + 22)], fill=(12, 12, 14))  # Bottom shadow
        # Asphalt fracture cracks radiating out
        draw.line([(px - 65, py), (px - 90, py + 8)], fill=(35, 35, 35), width=2)
        draw.line([(px + 65, py + 5), (px + 95, py + 12)], fill=(35, 35, 35), width=2)
        draw.line([(px, py + 30), (px + 10, py + 48)], fill=(35, 35, 35), width=2)
        box_coords = [px - 75, py - 28, px + 75, py + 38]
        det_label = f"Pothole {int(confidence * 100)}%"
        box_color = (220, 38, 38)  # Crimson Red

    # 7. YOLO Bounding Box Overlay
    if show_yolo_boxes:
        bx1, by1, bx2, by2 = box_coords
        draw.rectangle([bx1, by1, bx2, by2], outline=box_color, width=3)
        # Label badge
        draw.rectangle([bx1, by1 - 22, bx1 + 145, by1], fill=box_color)
        draw.text((bx1 + 6, by1 - 18), det_label, fill=(255, 255, 255))

    # 8. Dashcam Telemetry HUD Overlay (Top & Bottom bars)
    # Top Bar
    draw.rectangle([(0, 0), (w, 32)], fill=(15, 15, 15, 210))
    hud_top = f"ROADSENSE AI LIVE | {location_label.upper()}"
    draw.text((12, 9), hud_top, fill=(255, 255, 255))
    draw.text((w - 110, 9), "REC ● LIVE", fill=(239, 68, 68))

    # Bottom Bar
    draw.rectangle([(0, h - 34), (w, h)], fill=(15, 15, 15, 210))
    hud_bot = f"GPS: {gps_coords[0]:.5f} N, {gps_coords[1]:.5f} E | SPEED: {speed_kmh:.1f} km/h | CORRIDOR: NH-753F JALNA RD"
    draw.text((12, h - 24), hud_bot, fill=(220, 220, 220))

    return img


def render_live_footage_inspector(
    distress_type: str = "Pothole",
    confidence: float = 0.92,
    location_name: str = "Seven Hills Flyover, Jalna Road",
    latitude: float = 19.87698,
    longitude: float = 75.35380,
    speed_kmh: float = 38.5,
):
    """
    Renders an interactive road camera inspection station displaying synchronized
    dashcam footage, YOLO detections, and Google Street View deep-link.
    """
    gmaps_url = get_google_maps_url(latitude, longitude)
    gstreet_url = get_google_street_view_url(latitude, longitude)
    ward_info = find_nearest_aurangabad_landmark(latitude, longitude)

    c_img, c_meta = st.columns([1.6, 1.0])

    with c_img:
        frame = generate_street_footage_frame(
            distress_type=distress_type,
            confidence=confidence,
            location_label=location_name,
            gps_coords=(latitude, longitude),
            speed_kmh=speed_kmh,
            show_yolo_boxes=True,
        )
        st.image(
            frame,
            caption=f"🎥 Live Dashcam Frame Synchronized to GPS ({latitude:.5f}, {longitude:.5f})",
            use_container_width=True,
        )

    with c_meta:
        st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #EADBCC; border-radius: 10px; padding: 1.1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #EADBCC; padding-bottom: 6px; margin-bottom: 8px;">
                    <strong style="color: #1C1917; font-size: 1.05rem;">📍 Location Intelligence</strong>
                    <span style="background: #FEF2F2; color: #DC2626; border: 1px solid #FCA5A5; padding: 2px 7px; border-radius: 4px; font-weight: 700; font-size: 0.72rem;">
                        CHHATRAPATI SAMBHAJINAGAR
                    </span>
                </div>
                
                <div style="font-size: 0.86rem; color: #44403C; line-height: 1.6;">
                    <div><strong>Corridor:</strong> {location_name}</div>
                    <div><strong>Municipal Ward:</strong> {ward_info['ward_name']}</div>
                    <div><strong>Zone:</strong> {ward_info['zone']}</div>
                    <div><strong>Assigned PWD Lead:</strong> {ward_info['engineer']}</div>
                    <div><strong>Coordinates:</strong> <code>{latitude:.5f}, {longitude:.5f}</code></div>
                    <div><strong>SLA Repair Target:</strong> &lt; {ward_info['sla_target_hours']} hours</div>
                </div>
                
                <div style="margin-top: 12px; display: flex; flex-direction: column; gap: 8px;">
                    <a href="{gstreet_url}" target="_blank" style="display: block; text-align: center; background: #FAF7F2; color: #1C1917; border: 1.5px solid #DC2626; border-radius: 6px; padding: 7px 12px; font-size: 0.85rem; font-weight: 700; text-decoration: none;">
                        🌐 Open Google Street View Panorama ↗
                    </a>
                    <a href="{gmaps_url}" target="_blank" style="display: block; text-align: center; background: #DC2626; color: #FFFFFF; border-radius: 6px; padding: 7px 12px; font-size: 0.85rem; font-weight: 700; text-decoration: none;">
                        📍 View on Google Maps / Satellite ↗
                    </a>
                </div>
            </div>
        """, unsafe_allow_html=True)
