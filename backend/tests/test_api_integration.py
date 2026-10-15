def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "models" in data
    assert data["models"]["sensor_model"]["loaded"] is True
    assert data["models"]["yolo_model"]["loaded"] is True


def test_dashboard_summary_endpoint(client):
    res = client.get("/api/dashboard/summary")
    assert res.status_code == 200
    data = res.json()
    assert "pqi_score" in data
    assert 0 <= data["pqi_score"] <= 100
    assert "total_sensor_uploads" in data
    assert "open_incidents" in data
    assert "chronic_hotspots" in data


def test_authority_summary_endpoint(client):
    res = client.get("/api/authority/summary")
    assert res.status_code == 200
    data = res.json()
    assert "p1_urgent_action_required" in data


def test_maps_hotspots_endpoint(client):
    res = client.get("/api/maps/hotspots")
    assert res.status_code == 200
    assert isinstance(res.json(), list)
