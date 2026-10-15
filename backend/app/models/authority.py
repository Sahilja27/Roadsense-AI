from sqlalchemy import Column, String, Integer, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.db.base import Base


class AuthorityAction(Base):
    __tablename__ = "authority_actions"

    incident_id = Column(Integer, ForeignKey("road_incidents.id"), nullable=True, index=True)
    hotspot_id = Column(Integer, ForeignKey("hotspot_clusters.id"), nullable=True, index=True)
    performed_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action_type = Column(
        String(50),
        nullable=False,
    )  # status_change, assignment, work_order, resolution, priority_override
    previous_status = Column(String(50), nullable=True)
    new_status = Column(String(50), nullable=True)
    assigned_to = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)

    incident = relationship("RoadIncident", back_populates="actions")
    hotspot = relationship("HotspotCluster", back_populates="authority_actions")
