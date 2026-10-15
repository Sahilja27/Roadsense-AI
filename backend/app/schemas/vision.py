from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]
    bbox_normalized: List[float]
    frame_number: Optional[int] = 0
    timestamp_seconds: Optional[float] = 0.0
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    is_location_pending: bool = False


class VisionPredictionResponse(BaseModel):
    status: str
    inference_time_ms: float
    total_detections: int
    max_confidence: float
    class_counts: Dict[str, int]
    detections: List[DetectionItem]
    annotated_url: Optional[str] = None


class VisionJobOut(BaseModel):
    id: int
    job_type: str
    status: str
    progress_percent: float
    total_frames: int
    processed_frames: int
    total_detections: int
    conf_threshold: float
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class VisionJobDetail(VisionJobOut):
    detections: List[DetectionItem]
    annotated_output_url: Optional[str] = None
