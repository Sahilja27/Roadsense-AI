from typing import Optional, List
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.core.config import settings
from app.models.vision import VisionJob, VisionDetection
from app.schemas.vision import VisionPredictionResponse, VisionJobOut, VisionJobDetail
from app.services.vision_service import VisionService
from app.api.deps import get_current_user_optional
from app.models.user import User

router = APIRouter()


@router.post("/predict/image", response_model=VisionPredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    session_id: Optional[int] = Form(None),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """
    Runs trained Ultralytics YOLO model on an uploaded road snapshot.
    Returns detected damages (Pothole, Cracks) and saves annotated image.
    """
    valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    if not any(file.filename.lower().endswith(ext) for ext in valid_exts):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image format. Allowed: {valid_exts}",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty image uploaded.")

    user_id = current_user.id if current_user else None

    try:
        res = VisionService.process_image(
            db=db,
            image_bytes=content,
            filename=file.filename,
            conf_threshold=conf_threshold,
            user_id=user_id,
            session_id=session_id,
            latitude=latitude,
            longitude=longitude,
        )
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vision inference error: {str(e)}",
        )


@router.post("/jobs", response_model=VisionJobOut)
async def create_video_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    sampling_fps: float = Form(2.0),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    session_id: Optional[int] = Form(None),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """
    Uploads a road video for asynchronous YOLO damage detection.
    Processes frames in a background worker, updating progress and saving annotated video.
    """
    valid_exts = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    if not any(file.filename.lower().endswith(ext) for ext in valid_exts):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format. Allowed: {valid_exts}",
        )

    content = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Video exceeds maximum allowed size ({settings.MAX_UPLOAD_SIZE_MB}MB).",
        )

    user_id = current_user.id if current_user else None
    safe_name = f"video_{int(datetime.now().timestamp())}_{file.filename}"
    src_path = settings.UPLOAD_DIR / safe_name
    src_path.write_bytes(content)

    job = VisionJob(
        session_id=session_id,
        user_id=user_id,
        job_type="video",
        source_file_path=str(src_path),
        file_size_bytes=len(content),
        mime_type=file.content_type,
        status="queued",
        progress_percent=0.0,
        conf_threshold=conf_threshold,
        sampling_rate_fps=sampling_fps,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Launch background worker
    background_tasks.add_task(
        VisionService.process_video_background,
        job_id=job.id,
        conf_threshold=conf_threshold,
        sampling_fps=sampling_fps,
        latitude=latitude,
        longitude=longitude,
    )

    return job


@router.get("/jobs/{job_id}", response_model=VisionJobOut)
def get_job_status(job_id: int, db: Session = Depends(get_db)):
    """Polls progress status for an asynchronous video inference job."""
    job = db.query(VisionJob).filter(VisionJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vision job #{job_id} not found.",
        )
    return job


@router.get("/jobs/{job_id}/detections", response_model=VisionJobDetail)
def get_job_detections(job_id: int, db: Session = Depends(get_db)):
    """Retrieves full detection details and annotated video URL for a completed job."""
    job = db.query(VisionJob).filter(VisionJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job #{job_id} not found.")

    dets = (
        db.query(VisionDetection)
        .filter(VisionDetection.job_id == job_id)
        .order_by(VisionDetection.frame_number.asc())
        .all()
    )

    ann_url = None
    if job.annotated_output_path and Path(job.annotated_output_path).exists():
        filename = Path(job.annotated_output_path).name
        ann_url = f"/api/vision/files/detections/{filename}"

    return {
        "id": job.id,
        "job_type": job.job_type,
        "status": job.status,
        "progress_percent": job.progress_percent,
        "total_frames": job.total_frames,
        "processed_frames": job.processed_frames,
        "total_detections": job.total_detections,
        "conf_threshold": job.conf_threshold,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "error_message": job.error_message,
        "annotated_output_url": ann_url,
        "detections": [
            {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bbox_xyxy": d.bbox_xyxy,
                "bbox_normalized": d.bbox_normalized,
                "frame_number": d.frame_number,
                "timestamp_seconds": d.timestamp_seconds,
                "latitude": d.latitude,
                "longitude": d.longitude,
                "is_location_pending": d.is_location_pending,
            }
            for d in dets
        ],
    }


@router.get("/files/detections/{filename}")
def serve_detection_file(filename: str):
    """Serves annotated images or processed video files with path traversal protection."""
    file_path = (settings.DETECTION_DIR / filename).resolve()
    # Prevent path traversal
    if not str(file_path).startswith(str(settings.DETECTION_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Access denied.")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(path=str(file_path))
