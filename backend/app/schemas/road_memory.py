from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class HotspotClusterOut(BaseModel):
    id: int
    cluster_name: str
    center_lat: float
    center_lon: float
    radius_meters: float
    primary_hazard: str
    avg_severity: str
    recurrence_count: int
    priority_score: float
    status: str
    last_detected_at: datetime
    notes: Optional[str] = None

    class Config:
        from_attributes = True


class HotspotClusterUpdate(BaseModel):
    status: Optional[str] = None
    priority_score: Optional[float] = None
    notes: Optional[str] = None


class NearbyHazardOut(BaseModel):
    id: int
    cluster_name: str
    latitude: float
    longitude: float
    primary_hazard: str
    avg_severity: str
    recurrence_count: int
    priority_score: float
    status: str
    distance_meters: float
    safe_speed_kmh: int
