import pytest
import os
import io
import pandas as pd
from pathlib import Path
from app.ml.sensor_predictor import SensorPredictor, is_valid_coordinate
from app.core.demo_routes import AURANGABAD_CENTER, AURANGABAD_JALNA_ROAD_WAYPOINTS, generate_route_coordinates

TEST_10_CSV_PATH = Path("D:/New folder (3)/Combined CSV with GIS and Weather Data/Road Anomalies/10.csv")
if not TEST_10_CSV_PATH.exists():
    TEST_10_CSV_PATH = Path("../data/10.csv")


def test_coordinate_validator():
    assert is_valid_coordinate(19.8762, 75.3433) is True
    assert is_valid_coordinate(24.3737, 88.5618) is True
    assert is_valid_coordinate(0.0, 0.0) is False  # Null island rejected
    assert is_valid_coordinate(95.0, 75.0) is False  # Lat out of range
    assert is_valid_coordinate(19.0, 195.0) is False  # Lon out of range
    assert is_valid_coordinate(None, 75.0) is False
    assert is_valid_coordinate(float("nan"), 75.0) is False


def test_demo_route_generation():
    points = generate_route_coordinates(27)
    assert len(points) == 27
    # Verify deterministic ordering and Aurangabad bounding box
    for p in points:
        assert 19.85 <= p["latitude"] <= 19.90
        assert 75.30 <= p["longitude"] <= 75.40
        assert p["source"] == "demo_route"
        assert "Aurangabad" in p["provenance_label"]

    # Verify chronological progression along Jalna Road (longitude increases going east)
    assert points[0]["longitude"] < points[-1]["longitude"]


def test_real_sensor_model_on_10_csv_predictions():
    if not TEST_10_CSV_PATH.exists():
        pytest.skip("10.csv not found in test environment.")

    sp = SensorPredictor.get_instance()
    df = pd.read_csv(TEST_10_CSV_PATH)
    
    # 1. Test Demo Mode
    res_demo = sp.predict_dataframe(df, route_mode="demo_aurangabad")
    assert res_demo["total_windows"] == 27
    assert res_demo["counts"]["Bump"] == 21
    assert res_demo["counts"]["Pothole"] == 6
    assert len(res_demo["windows"]) == 27
    for w in res_demo["windows"]:
        assert w["coordinate_provenance"] == "demo_route"
        assert w["is_location_pending"] is False
        assert w["latitude"] is not None
        assert w["longitude"] is not None
        assert len(w["features"]) == 30

    # 2. Test Real GPS Mode
    res_real = sp.predict_dataframe(df, route_mode="real_gps")
    assert res_real["total_windows"] == 27
    assert res_real["counts"]["Bump"] == 21
    assert res_real["counts"]["Pothole"] == 6
    assert res_real["metadata"]["has_gps"] is True
    for w in res_real["windows"]:
        assert w["coordinate_provenance"] == "actual_gps"
        assert w["is_location_pending"] is False
        # Coordinates in 10.csv are ~24.37, ~88.56
        assert 24.30 <= w["latitude"] <= 24.45
        assert 88.50 <= w["longitude"] <= 88.60


def test_missing_gps_marks_location_pending():
    if not TEST_10_CSV_PATH.exists():
        pytest.skip("10.csv not found.")

    sp = SensorPredictor.get_instance()
    df = pd.read_csv(TEST_10_CSV_PATH)
    
    # Drop all location columns
    loc_cols = [c for c in df.columns if "location" in c.lower() or "lat" in c.lower() or "lon" in c.lower()]
    df_no_gps = df.drop(columns=loc_cols)
    
    res = sp.predict_dataframe(df_no_gps, route_mode="real_gps")
    assert res["total_windows"] == 27
    assert res["metadata"]["has_gps"] is False
    for w in res["windows"]:
        assert w["coordinate_provenance"] == "none"
        assert w["is_location_pending"] is True
        assert w["latitude"] is None
        assert w["longitude"] is None


def test_api_upload_and_map_events(client):
    if not TEST_10_CSV_PATH.exists():
        pytest.skip("10.csv not found.")

    with open(TEST_10_CSV_PATH, "rb") as f:
        csv_bytes = f.read()

    # 1. Post upload with demo_aurangabad
    res = client.post(
        "/api/sensors/uploads?route_mode=demo_aurangabad",
        files={"file": ("test_upload_10.csv", csv_bytes, "text/csv")},
    )
    assert res.status_code == 200
    upload = res.json()
    upload_id = upload["id"]
    assert upload["valid_window_count"] == 27
    assert upload["route_mode"] == "demo_aurangabad"

    # 2. Get map events
    map_res = client.get(f"/api/sensors/uploads/{upload_id}/map-events")
    assert map_res.status_code == 200
    map_data = map_res.json()
    assert map_data["total_windows"] == 27
    assert map_data["geolocated_events_count"] == 27
    assert map_data["counts"]["Bump"] == 21
    assert map_data["counts"]["Pothole"] == 6
    assert map_data["center"]["lat"] == AURANGABAD_CENTER["lat"]
    assert len(map_data["demo_polyline"]) > 0

    # 3. Get results CSV
    csv_res = client.get(f"/api/sensors/uploads/{upload_id}/results")
    assert csv_res.status_code == 200
    assert "text/csv" in csv_res.headers["content-type"]
    assert "coordinate_provenance" in csv_res.text
    assert "demo_route" in csv_res.text


def test_api_maps_incidents_upload_filter(client):
    res = client.get("/api/maps/incidents")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    res_filt = client.get("/api/maps/incidents?upload_id=999999")
    assert res_filt.status_code == 200
    assert isinstance(res_filt.json(), list)
