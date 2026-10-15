from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class SensorWindowOut(BaseModel):
    id: Optional[int] = None
    window_index: int
    start_seconds: float
    end_seconds: float
    sample_count: int
    predicted_class: str
    confidence: float
    confidence_percent: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    speed: Optional[float] = None
    coordinate_provenance: Optional[str] = "none"
    is_interpolated: Optional[bool] = False
    is_location_pending: bool = False
    features: Optional[Dict[str, float]] = None

    class Config:
        from_attributes = True


class SensorUploadOut(BaseModel):
    id: int
    filename: str
    original_name: str
    row_count: int
    valid_window_count: int
    status: str
    route_mode: Optional[str] = "real_gps"
    has_gps: Optional[bool] = False
    geolocated_window_count: Optional[int] = 0
    created_at: datetime
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class SensorUploadDetail(SensorUploadOut):
    counts: Dict[str, int]
    windows: List[SensorWindowOut]


class SensorMapEventItem(BaseModel):
    id: Optional[int] = None
    upload_id: int
    window_index: int
    start_seconds: float
    end_seconds: float
    predicted_class: str
    confidence: float
    confidence_percent: float
    latitude: float
    longitude: float
    speed: Optional[float] = None
    coordinate_provenance: str
    is_interpolated: bool
    is_location_pending: bool


class SensorUploadMapEventsOut(BaseModel):
    upload_id: int
    filename: str
    route_mode: str
    has_gps: bool
    total_windows: int
    geolocated_events_count: int
    counts: Dict[str, int]
    center: Dict[str, float]
    demo_polyline: List[List[float]] = []
    events: List[SensorMapEventItem]
