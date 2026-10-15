from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime


class ActivityItem(BaseModel):
    id: int
    activity_type: str  # sensor_upload, vision_job, incident_created, authority_action
    title: str
    description: str
    timestamp: datetime
    status: str
    severity: Optional[str] = None


class DashboardSummary(BaseModel):
    pqi_score: float
    total_sensor_uploads: int
    total_sensor_windows: int
    total_vision_jobs: int
    total_vision_detections: int
    open_incidents: int
    critical_incidents: int
    chronic_hotspots: int
    resolved_incidents: int
    verified_incidents: int
    hazard_distribution: Dict[str, int]
    severity_distribution: Dict[str, int]
    recent_activity: List[ActivityItem]
