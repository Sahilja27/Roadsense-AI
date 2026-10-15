import pytest
import os
from pathlib import Path
from starlette.testclient import TestClient
from app.main import app

def test_sensor_upload_and_map_events_e2e():
    """
    End-to-End integration test for sensor CSV upload, 
    Random Forest inference, spatial map events, and CSV export.
    """
    candidate_paths = [
        Path("data/10.csv"),
        Path("../data/10.csv"),
        Path("D:/New folder (4)/RoadSense-Sensor-Dashboard/data/10.csv"),
    ]
    csv_path = None
    for p in candidate_paths:
        if p.exists():
            csv_path = p
            break
            
    if not csv_path:
        pytest.skip("10.csv not found")

    client = TestClient(app)
    with open(csv_path, "rb") as f:
        csv_bytes = f.read()

    # 1. Test Demo Route Upload
    res_demo = client.post(
        "/api/sensors/uploads?route_mode=demo_aurangabad",
        files={"file": ("10.csv", csv_bytes, "text/csv")},
    )
    assert res_demo.status_code == 200, f"Upload failed: {res_demo.text}"
    demo_data = res_demo.json()
    demo_id = demo_data["id"]
    assert demo_data["valid_window_count"] == 27
    assert demo_data["route_mode"] == "demo_aurangabad"

    # 2. Test Map Events endpoint
    map_res = client.get(f"/api/sensors/uploads/{demo_id}/map-events")
    assert map_res.status_code == 200, f"Map events failed: {map_res.text}"
    map_json = map_res.json()
    assert len(map_json["events"]) == 27
    assert map_json["counts"]["Bump"] == 21
    assert map_json["counts"]["Pothole"] == 6
    assert len(map_json["demo_polyline"]) > 0

    # 3. Test Real GPS Upload
    res_real = client.post(
        "/api/sensors/uploads?route_mode=real_gps",
        files={"file": ("10.csv", csv_bytes, "text/csv")},
    )
    assert res_real.status_code == 200, f"Real GPS upload failed: {res_real.text}"
    real_data = res_real.json()
    real_id = real_data["id"]
    assert real_data["valid_window_count"] == 27

    map_res_real = client.get(f"/api/sensors/uploads/{real_id}/map-events")
    assert map_res_real.status_code == 200

    # 4. Test Results CSV endpoint
    csv_res = client.get(f"/api/sensors/uploads/{demo_id}/results")
    assert csv_res.status_code == 200
    lines = csv_res.text.strip().split("\n")
    assert len(lines) == 28  # header + 27 windows
