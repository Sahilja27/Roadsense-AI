from sqlalchemy import Column, String, Integer, ForeignKey, JSON
from app.db.base import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    action = Column(String(100), nullable=False, index=True)
    entity_type = Column(String(50), nullable=False, index=True)
    entity_id = Column(Integer, nullable=True)
    ip_address = Column(String(45), nullable=True)
    details_json = Column(JSON, nullable=True)
