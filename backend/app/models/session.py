from sqlalchemy import Column, String, Integer, Float, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from app.db.base import Base


class MonitoringSession(Base):
    __tablename__ = "monitoring_sessions"

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=True, index=True)
    session_token = Column(String(100), unique=True, index=True, nullable=False)
    status = Column(String(20), default="active", nullable=False, index=True)  # active, paused, completed, cancelled
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    initial_lat = Column(Float, nullable=True)
    initial_lon = Column(Float, nullable=True)
    final_lat = Column(Float, nullable=True)
    final_lon = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)

    user = relationship("User", back_populates="monitoring_sessions")
    vehicle = relationship("Vehicle", back_populates="monitoring_sessions")
    sensor_uploads = relationship("SensorUpload", back_populates="session")
    sensor_windows = relationship("SensorWindow", back_populates="session")
    vision_jobs = relationship("VisionJob", back_populates="session")
    incidents = relationship("RoadIncident", back_populates="session")
