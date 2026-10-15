import pytest
from app.models.user import User, UserRole
from app.models.incident import RoadIncident
from app.models.road_memory import HotspotCluster


def test_database_connectivity_and_seed(db_session):
    """Verifies database connectivity and pre-seeded administrative accounts."""
    admin = db_session.query(User).filter(User.username == "admin").first()
    assert admin is not None
    assert admin.role == UserRole.ADMIN.value
    assert admin.email == "admin@roadsense.ai"

    driver = db_session.query(User).filter(User.username == "driver1").first()
    assert driver is not None
    assert driver.role == UserRole.DRIVER.value


def test_incident_model_persistence(db_session):
    """Verifies that road incident records persist and commit correctly."""
    incident = RoadIncident(
        title="Test Severe Pothole Sector 4",
        distress_type="Pothole",
        severity="High",
        status="reported",
        verification_status="unverified",
        confidence=0.92,
        latitude=28.6139,
        longitude=77.2090,
        is_location_pending=False,
    )
    db_session.add(incident)
    db_session.commit()
    db_session.refresh(incident)

    assert incident.id is not None
    assert incident.title == "Test Severe Pothole Sector 4"
    assert incident.severity == "High"
