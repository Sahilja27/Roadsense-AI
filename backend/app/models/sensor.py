from sqlalchemy import Column, String, Integer, Float, ForeignKey, Text, Boolean, JSON
from sqlalchemy.orm import relationship
from app.db.base import Base


class SensorUpload(Base):
    __tablename__ = "sensor_uploads"

    session_id = Column(Integer, ForeignKey("monitoring_sessions.id"), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    filename = Column(String(255), nullable=False)
    original_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    file_hash = Column(String(64), nullable=True, index=True)
    row_count = Column(Integer, default=0)
    valid_window_count = Column(Integer, default=0)
    sample_rate_hz = Column(Float, nullable=True)
    status = Column(String(20), default="uploaded", index=True)  # uploaded, processing, completed, failed
    error_message = Column(Text, nullable=True)
    
    # GPS and route attributes
    route_mode = Column(String(50), default="real_gps", nullable=False)  # 'real_gps', 'demo_aurangabad'
    has_gps = Column(Boolean, default=False, nullable=False)
    geolocated_window_count = Column(Integer, default=0, nullable=False)

    user = relationship("User", back_populates="sensor_uploads")
    session = relationship("MonitoringSession", back_populates="sensor_uploads")
    windows = relationship("SensorWindow", back_populates="upload", cascade="all, delete-orphan")


class SensorWindow(Base):
    __tablename__ = "sensor_windows"

    upload_id = Column(Integer, ForeignKey("sensor_uploads.id"), nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("monitoring_sessions.id"), nullable=True, index=True)
    window_index = Column(Integer, nullable=False)
    start_seconds = Column(Float, nullable=False)
    end_seconds = Column(Float, nullable=False)
    sample_count = Column(Integer, nullable=False)
    predicted_class = Column(String(50), nullable=False, index=True)  # 'Bump', 'Pothole'
    confidence = Column(Float, nullable=False)
    features_json = Column(JSON, nullable=True)
    latitude = Column(Float, nullable=True, index=True)
    longitude = Column(Float, nullable=True, index=True)
    speed = Column(Float, nullable=True)
    is_location_pending = Column(Boolean, default=False, nullable=False)

    # GPS Provenance & interpolation tracking
    coordinate_provenance = Column(String(50), default="none", nullable=False)  # 'actual_gps', 'interpolated_gps', 'demo_route', 'none'
    is_interpolated = Column(Boolean, default=False, nullable=False)
    gps_timestamp = Column(String(50), nullable=True)

    upload = relationship("SensorUpload", back_populates="windows")
    session = relationship("MonitoringSession", back_populates="sensor_windows")
