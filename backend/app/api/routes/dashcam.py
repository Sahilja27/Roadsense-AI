import base64
import uuid
import time
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.session import MonitoringSession
from app.schemas.dashcam import (
    DashcamSessionCreate,
    DashcamSessionOut,
    DashcamFrameIn,
    DashcamFrameResult,
)
from app.ml.yolo_predictor import YOLOPredictor
from app.api.deps import get_current_user_optional
from app.models.user import User

router = APIRouter()


@router.post("/sessions", response_model=DashcamSessionOut)
def start_session(
    session_in: DashcamSessionCreate,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """Starts a live vehicle road monitoring session."""
    user_id = current_user.id if current_user else 2  # default driver1 if unauthenticated

    session = MonitoringSession(
        user_id=user_id,
        vehicle_id=session_in.vehicle_id,
        session_token=f"sess_{uuid.uuid4().hex[:12]}",
        status="active",
        start_time=datetime.now(timezone.utc),
        initial_lat=session_in.initial_lat,
        initial_lon=session_in.initial_lon,
        notes=session_in.notes,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/sessions/{session_id}", response_model=DashcamSessionOut)
def get_session(session_id: int, db: Session = Depends(get_db)):
    """Retrieves monitoring session status."""
    session = db.query(MonitoringSession).filter(MonitoringSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


@router.post("/sessions/{session_id}/stop", response_model=DashcamSessionOut)
def stop_session(session_id: int, db: Session = Depends(get_db)):
    """Ends and releases resources for an active monitoring session."""
    session = db.query(MonitoringSession).filter(MonitoringSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    session.status = "completed"
    session.end_time = datetime.now(timezone.utc)
    db.commit()
    db.refresh(session)
    return session


@router.post("/frame", response_model=DashcamFrameResult)
def process_live_frame(frame_in: DashcamFrameIn):
    """
    Processes a live dashcam/webcam snapshot from the driver's browser.
    Decodes base64 JPEG/PNG and runs YOLO damage detection.
    """
    raw_b64 = frame_in.frame_base64
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(raw_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 frame encoding.")

    predictor = YOLOPredictor.get_instance()
    try:
        pred_res = predictor.predict_image(
            image_input=img_bytes,
            conf_threshold=frame_in.conf_threshold or 0.25,
            annotate=False,
        )

        is_loc_pending = frame_in.latitude is None or frame_in.longitude is None
        for d in pred_res["detections"]:
            d["latitude"] = frame_in.latitude
            d["longitude"] = frame_in.longitude
            d["is_location_pending"] = is_loc_pending

        return {
            "status": "success",
            "detections": pred_res["detections"],
            "total_detections": pred_res["total_detections"],
            "inference_time_ms": pred_res["inference_time_ms"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
