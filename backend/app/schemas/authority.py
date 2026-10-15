from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class AuthorityActionCreate(BaseModel):
    action_type: str  # assignment, status_change, work_order, resolution
    new_status: Optional[str] = None
    assigned_to: Optional[str] = None
    notes: Optional[str] = None


class AuthorityActionOut(BaseModel):
    id: int
    incident_id: Optional[int] = None
    hotspot_id: Optional[int] = None
    performed_by_user_id: Optional[int] = None
    action_type: str
    previous_status: Optional[str] = None
    new_status: Optional[str] = None
    assigned_to: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
