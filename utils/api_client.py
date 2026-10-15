"""
RoadSense AI - Frontend API Client
==================================
Communicates with the FastAPI backend REST API (http://localhost:8000/api)
with fallback mechanisms to ensure complete UI stability.
"""

import os
import requests
from typing import Dict, Any, List, Optional
import pandas as pd

API_BASE_URL = os.getenv("ROADSENSE_API_URL", "http://localhost:8000/api")
TIMEOUT = 30.0


def is_backend_online() -> bool:
    """Checks if the FastAPI backend server is responsive."""
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=2.5)
        return res.status_code == 200
    except Exception:
        return False


def get_health() -> Dict[str, Any]:
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=5.0)
        return res.json()
    except Exception as e:
        return {"status": "offline", "error": str(e)}


def get_dashboard_summary() -> Dict[str, Any]:
    try:
        res = requests.get(f"{API_BASE_URL}/dashboard/summary", timeout=10.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    # Fallback to local default structure
    return {
        "pqi_score": 100.0,
        "total_sensor_uploads": 0,
        "total_sensor_windows": 0,
        "total_vision_jobs": 0,
        "total_vision_detections": 0,
        "open_incidents": 0,
        "critical_incidents": 0,
        "chronic_hotspots": 0,
        "resolved_incidents": 0,
        "verified_incidents": 0,
        "hazard_distribution": {},
        "severity_distribution": {},
        "recent_activity": [],
    }


def upload_sensor_csv(
    file_bytes: bytes,
    filename: str,
    route_mode: str = "demo_aurangabad",
    session_id: Optional[int] = None,
) -> Dict[str, Any]:
    files = {"file": (filename, file_bytes, "text/csv")}
    params = {"route_mode": route_mode}
    if session_id:
        params["session_id"] = session_id
    res = requests.post(f"{API_BASE_URL}/sensors/uploads", files=files, params=params, timeout=TIMEOUT)
    if res.status_code != 200:
        error_msg = res.json().get("detail", res.text)
        raise ValueError(error_msg)
    return res.json()


def get_sensor_uploads(limit: int = 50) -> List[Dict[str, Any]]:
    try:
        res = requests.get(f"{API_BASE_URL}/sensors/uploads?limit={limit}", timeout=10.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []


def get_sensor_upload_detail(upload_id: int) -> Optional[Dict[str, Any]]:
    try:
        res = requests.get(f"{API_BASE_URL}/sensors/uploads/{upload_id}", timeout=10.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None


def get_sensor_upload_map_events(upload_id: int) -> Optional[Dict[str, Any]]:
    try:
        res = requests.get(f"{API_BASE_URL}/sensors/uploads/{upload_id}/map-events", timeout=10.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None


def predict_vision_image(
    image_bytes: bytes,
    filename: str,
    conf_threshold: float = 0.25,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> Dict[str, Any]:
    files = {"file": (filename, image_bytes, "image/jpeg")}
    data = {"conf_threshold": str(conf_threshold)}
    if latitude is not None and longitude is not None:
        data["latitude"] = str(latitude)
        data["longitude"] = str(longitude)

    res = requests.post(f"{API_BASE_URL}/vision/predict/image", files=files, data=data, timeout=TIMEOUT)
    if res.status_code != 200:
        raise ValueError(res.json().get("detail", res.text))
    return res.json()


def process_dashcam_frame(
    frame_b64: str,
    conf_threshold: float = 0.25,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> Dict[str, Any]:
    payload = {
        "frame_base64": frame_b64,
        "conf_threshold": conf_threshold,
        "latitude": latitude,
        "longitude": longitude,
    }
    res = requests.post(f"{API_BASE_URL}/dashcam/frame", json=payload, timeout=TIMEOUT)
    if res.status_code != 200:
        raise ValueError(res.json().get("detail", res.text))
    return res.json()


def get_map_incidents() -> List[Dict[str, Any]]:
    try:
        res = requests.get(f"{API_BASE_URL}/maps/incidents", timeout=10.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []


def get_map_hotspots() -> List[Dict[str, Any]]:
    try:
        res = requests.get(f"{API_BASE_URL}/maps/hotspots", timeout=10.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []


def get_nearby_hazards(lat: float, lon: float, radius: float = 350.0) -> List[Dict[str, Any]]:
    try:
        res = requests.get(
            f"{API_BASE_URL}/maps/nearby?latitude={lat}&longitude={lon}&radius_meters={radius}",
            timeout=5.0,
        )
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []


def list_incidents(
    distress_type: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    params = {}
    if distress_type:
        params["distress_type"] = distress_type
    if severity:
        params["severity"] = severity
    if status:
        params["status"] = status

    try:
        res = requests.get(f"{API_BASE_URL}/incidents", params=params, timeout=10.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []


def create_incident(data: Dict[str, Any]) -> Dict[str, Any]:
    res = requests.post(f"{API_BASE_URL}/incidents", json=data, timeout=10.0)
    if res.status_code != 200:
        raise ValueError(res.json().get("detail", res.text))
    return res.json()


def update_incident(incident_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
    res = requests.patch(f"{API_BASE_URL}/incidents/{incident_id}", json=data, timeout=10.0)
    if res.status_code != 200:
        raise ValueError(res.json().get("detail", res.text))
    return res.json()


def assign_incident(incident_id: int, assigned_to: str, notes: Optional[str] = None) -> Dict[str, Any]:
    payload = {"assigned_to": assigned_to, "notes": notes}
    res = requests.post(f"{API_BASE_URL}/authority/incidents/{incident_id}/assign", json=payload, timeout=10.0)
    if res.status_code != 200:
        raise ValueError(res.json().get("detail", res.text))
    return res.json()


def resolve_incident(incident_id: int, resolution_notes: str) -> Dict[str, Any]:
    payload = {"resolution_notes": resolution_notes}
    res = requests.post(f"{API_BASE_URL}/authority/incidents/{incident_id}/resolve", json=payload, timeout=10.0)
    if res.status_code != 200:
        raise ValueError(res.json().get("detail", res.text))
    return res.json()
