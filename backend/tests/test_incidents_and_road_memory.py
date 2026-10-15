from datetime import datetime, timezone
from app.services.incident_service import IncidentService
from app.services.road_memory_service import RoadMemoryService, haversine_distance_m
from app.models.incident import RoadIncident
from app.models.road_memory import HotspotCluster


def test_haversine_distance():
    # Distance between two close coordinates in Delhi (~111 meters per 0.001 deg lat)
    dist = haversine_distance_m(28.6139, 77.2090, 28.6149, 77.2090)
    assert 105.0 < dist < 118.0


def test_incident_deduplication(db_session):
    # Create first incident
    inc1 = IncidentService.create_incident(
        db=db_session,
        title="Ring Road Pothole",
        distress_type="Pothole",
        severity="High",
        latitude=28.6150,
        longitude=77.2100,
        notes="First detection report",
    )
    assert inc1.id is not None

    # Submit second incident within 15 meters
    inc2 = IncidentService.create_incident(
        db=db_session,
        title="Ring Road Pothole Repeated",
        distress_type="Pothole",
        severity="High",
        latitude=28.6151,  # ~11 meters away
        longitude=77.2100,
        notes="Second detection report",
    )
    # Deduplication matches existing incident
    assert inc2.id == inc1.id


def test_road_memory_clustering_and_nearby(db_session):
    # Create a cluster
    cluster = HotspotCluster(
        cluster_name="Sector 5 Flyover Hotspot",
        center_lat=28.6200,
        center_lon=77.2200,
        radius_meters=25.0,
        primary_hazard="Pothole",
        avg_severity="High",
        recurrence_count=12,
        priority_score=92.5,
        status="Action Required",
        last_detected_at=datetime.now(timezone.utc),
    )
    db_session.add(cluster)
    db_session.commit()


    # Query vehicle 100 meters away
    nearby = RoadMemoryService.find_nearby_hazards(
        db=db_session,
        current_lat=28.6208,
        current_lon=77.2200,
        radius_meters=350.0,
    )
    assert len(nearby) > 0
    assert nearby[0]["cluster_name"] == "Sector 5 Flyover Hotspot"
    assert nearby[0]["distance_meters"] < 150.0
    assert nearby[0]["safe_speed_kmh"] == 25
