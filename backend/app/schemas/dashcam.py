from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.schemas.vision import DetectionItem


class DashcamSessionCreate(BaseModel):
    vehicle_id: Optional[int] = None
    notes: Optional[str] = None
    initial_lat: Optional[float] = None
    initial_lon: Optional[float] = None


class DashcamSessionOut(BaseModel):
    id: int
    session_token: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    initial_lat: Optional[float] = None
    initial_lon: Optional[float] = None
    notes: Optional[str] = None

    class Config:
        from_attributes = True


class DashcamFrameIn(BaseModel):
    session_id: Optional[int] = None
    frame_base64: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    conf_threshold: Optional[float] = 0.25


class DashcamFrameResult(BaseModel):
    status: str
    detections: List[DetectionItem]
    total_detections: int
    inference_time_ms: float
    timestamp: str
