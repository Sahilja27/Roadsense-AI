from sqlalchemy import Column, String, Boolean, Enum
from sqlalchemy.orm import relationship
import enum
from app.db.base import Base


class UserRole(str, enum.Enum):
    DRIVER = "driver"
    AUTHORITY = "authority"
    ADMIN = "admin"


class User(Base):
    __tablename__ = "users"

    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(120), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    role = Column(String(20), default=UserRole.DRIVER.value, nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False)

    vehicles = relationship("Vehicle", back_populates="user", cascade="all, delete-orphan")
    monitoring_sessions = relationship("MonitoringSession", back_populates="user")
    sensor_uploads = relationship("SensorUpload", back_populates="user")
    vision_jobs = relationship("VisionJob", back_populates="user")
    reported_incidents = relationship("RoadIncident", back_populates="reported_by_user")
