from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class IncidentEvidenceOut(BaseModel):
    id: int
    evidence_type: str
    file_path: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class IncidentCreate(BaseModel):
    title: str
    distress_type: str
    severity: str = "Medium"
    confidence: Optional[float] = 0.85
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address_text: Optional[str] = None
    notes: Optional[str] = None
    evidence_type: Optional[str] = None
    evidence_metadata: Optional[Dict[str, Any]] = None
    session_id: Optional[int] = None


class IncidentUpdate(BaseModel):
    status: Optional[str] = None
    severity: Optional[str] = None
    assigned_to: Optional[str] = None
    notes: Optional[str] = None
    verification_status: Optional[str] = None


class IncidentOut(BaseModel):
    id: int
    title: str
    distress_type: str
    severity: str
    status: str
    verification_status: str
    confidence: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    is_location_pending: bool
    address_text: Optional[str] = None
    assigned_to: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    evidence: List[IncidentEvidenceOut] = []

    class Config:
        from_attributes = True
