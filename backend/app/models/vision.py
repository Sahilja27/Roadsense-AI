from sqlalchemy import Column, String, Integer, Float, ForeignKey, Text, Boolean, JSON, DateTime
from sqlalchemy.orm import relationship
from app.db.base import Base


class VisionJob(Base):
    __tablename__ = "vision_jobs"

    session_id = Column(Integer, ForeignKey("monitoring_sessions.id"), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    job_type = Column(String(20), default="image", index=True)  # image, video, dashcam_stream
    source_file_path = Column(String(500), nullable=False)
    annotated_output_path = Column(String(500), nullable=True)
    file_size_bytes = Column(Integer, nullable=False)
    mime_type = Column(String(50), nullable=True)
    status = Column(String(20), default="queued", index=True)  # queued, processing, completed, failed, cancelled
    progress_percent = Column(Float, default=0.0)
    total_frames = Column(Integer, default=1)
    processed_frames = Column(Integer, default=0)
    total_detections = Column(Integer, default=0)
    conf_threshold = Column(Float, default=0.25)
    sampling_rate_fps = Column(Float, default=2.0)
    error_message = Column(Text, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="vision_jobs")
    session = relationship("MonitoringSession", back_populates="vision_jobs")
    detections = relationship("VisionDetection", back_populates="job", cascade="all, delete-orphan")


class VisionDetection(Base):
    __tablename__ = "vision_detections"

    job_id = Column(Integer, ForeignKey("vision_jobs.id"), nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("monitoring_sessions.id"), nullable=True, index=True)
    frame_number = Column(Integer, default=0)
    timestamp_seconds = Column(Float, default=0.0)
    class_id = Column(Integer, nullable=False)
    class_name = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    bbox_xyxy = Column(JSON, nullable=False)        # [xmin, ymin, xmax, ymax]
    bbox_normalized = Column(JSON, nullable=False)  # [x_center, y_center, w, h]
    latitude = Column(Float, nullable=True, index=True)
    longitude = Column(Float, nullable=True, index=True)
    is_location_pending = Column(Boolean, default=False, nullable=False)

    job = relationship("VisionJob", back_populates="detections")
