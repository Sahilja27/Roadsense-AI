from sqlalchemy import Column, String, Integer, Float, ForeignKey, Text, Boolean, JSON
from sqlalchemy.orm import relationship
from app.db.base import Base


class RoadSegment(Base):
    __tablename__ = "road_segments"

    segment_name = Column(String(100), nullable=False)
    road_name = Column(String(100), nullable=True, index=True)
    city = Column(String(100), default="Delhi NCR", index=True)
    district = Column(String(100), nullable=True)
    start_lat = Column(Float, nullable=False)
    start_lon = Column(Float, nullable=False)
    end_lat = Column(Float, nullable=False)
    end_lon = Column(Float, nullable=False)
    surface_type = Column(String(50), default="Asphalt")
    length_meters = Column(Float, nullable=True)

    incidents = relationship("RoadIncident", back_populates="road_segment")


class RoadIncident(Base):
    __tablename__ = "road_incidents"

    session_id = Column(Integer, ForeignKey("monitoring_sessions.id"), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    road_segment_id = Column(Integer, ForeignKey("road_segments.id"), nullable=True, index=True)
    hotspot_id = Column(Integer, ForeignKey("hotspot_clusters.id"), nullable=True, index=True)

    title = Column(String(150), nullable=False)
    distress_type = Column(String(50), nullable=False, index=True)  # Pothole, Bump, Alligator Crack, etc.
    severity = Column(String(20), default="Medium", index=True)     # Low, Medium, High, Critical
    status = Column(String(30), default="reported", index=True)     # reported, under_review, assigned, scheduled, resolved
    verification_status = Column(
        String(30),
        default="unverified",
        index=True,
    )  # unverified, sensor_confirmed, vision_confirmed, multi_modal_confirmed
    confidence = Column(Float, default=0.80)
    latitude = Column(Float, nullable=True, index=True)
    longitude = Column(Float, nullable=True, index=True)
    is_location_pending = Column(Boolean, default=False, nullable=False)
    address_text = Column(String(255), nullable=True)
    assigned_to = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)

    reported_by_user = relationship("User", back_populates="reported_incidents")
    session = relationship("MonitoringSession", back_populates="incidents")
    road_segment = relationship("RoadSegment", back_populates="incidents")
    hotspot = relationship("HotspotCluster", back_populates="incidents")
    evidence = relationship("IncidentEvidence", back_populates="incident", cascade="all, delete-orphan")
    actions = relationship("AuthorityAction", back_populates="incident", cascade="all, delete-orphan")


class IncidentEvidence(Base):
    __tablename__ = "incident_evidence"

    incident_id = Column(Integer, ForeignKey("road_incidents.id"), nullable=False, index=True)
    evidence_type = Column(String(30), nullable=False)  # sensor_window, vision_detection, image_crop, video_clip
    file_path = Column(String(500), nullable=True)
    metadata_json = Column(JSON, nullable=True)

    incident = relationship("RoadIncident", back_populates="evidence")
