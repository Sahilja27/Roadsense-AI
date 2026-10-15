from sqlalchemy import Column, String, Integer, Float, DateTime, Text
from sqlalchemy.orm import relationship
from app.db.base import Base


class HotspotCluster(Base):
    __tablename__ = "hotspot_clusters"

    cluster_name = Column(String(150), nullable=False)
    center_lat = Column(Float, nullable=False, index=True)
    center_lon = Column(Float, nullable=False, index=True)
    radius_meters = Column(Float, default=25.0)
    primary_hazard = Column(String(50), nullable=False, index=True)  # Pothole, Bump, etc.
    avg_severity = Column(String(20), default="Medium")
    recurrence_count = Column(Integer, default=1, index=True)
    priority_score = Column(Float, default=50.0, index=True)  # 0.0 to 100.0
    status = Column(
        String(30),
        default="new",
        index=True,
    )  # new, recurring, under_review, assigned, scheduled, resolved
    last_detected_at = Column(DateTime, nullable=False)
    notes = Column(Text, nullable=True)

    incidents = relationship("RoadIncident", back_populates="hotspot")
    authority_actions = relationship("AuthorityAction", back_populates="hotspot")
