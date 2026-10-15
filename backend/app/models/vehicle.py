from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base import Base


class Vehicle(Base):
    __tablename__ = "vehicles"

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    registration_number = Column(String(50), unique=True, index=True, nullable=False)
    make = Column(String(50), nullable=True)
    model = Column(String(50), nullable=True)
    year = Column(Integer, nullable=True)
    device_id = Column(String(100), nullable=True)

    user = relationship("User", back_populates="vehicles")
    monitoring_sessions = relationship("MonitoringSession", back_populates="vehicle")
