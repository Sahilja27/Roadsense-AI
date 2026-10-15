from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Query, HTTPException, status, Response
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.sensor import SensorUpload
from app.schemas.sensor import SensorUploadOut, SensorUploadDetail, SensorUploadMapEventsOut
from app.services.sensor_service import SensorService
from app.api.deps import get_current_user_optional
from app.models.user import User

router = APIRouter()


@router.post("/uploads", response_model=SensorUploadOut)
async def upload_sensor_csv(
    file: UploadFile = File(...),
    route_mode: str = Query("demo_aurangabad", description="Route mode: 'demo_aurangabad' or 'real_gps'"),
    session_id: Optional[int] = None,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """
    Ingests smartphone sensor CSV, validates required accelerometer/gyroscope signals,
    executes real ML inference pipeline (roadsense_model.joblib), and persists windows with GPS provenance to DB.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a CSV (.csv) file.",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    user_id = current_user.id if current_user else None

    try:
        upload_record = SensorService.process_and_save_upload(
            db=db,
            file_bytes=content,
            filename=file.filename,
            route_mode=route_mode,
            user_id=user_id,
            session_id=session_id,
        )
        return upload_record
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@router.get("/uploads", response_model=List[SensorUploadOut])
def list_uploads(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """Lists past sensor upload records."""
    uploads = (
        db.query(SensorUpload)
        .order_by(SensorUpload.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return uploads


@router.get("/uploads/{upload_id}", response_model=SensorUploadDetail)
def get_upload_detail(
    upload_id: int,
    db: Session = Depends(get_db),
):
    """Retrieves full prediction details, window features, and GPS provenance for an upload."""
    detail = SensorService.get_upload_detail(db, upload_id)
    if not detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sensor upload #{upload_id} not found.",
        )
    return detail


@router.get("/uploads/{upload_id}/map-events", response_model=SensorUploadMapEventsOut)
def get_upload_map_events(
    upload_id: int,
    db: Session = Depends(get_db),
):
    """Retrieves interactive map events, center coordinates, and route polyline for an upload."""
    map_events = SensorService.get_upload_map_events(db, upload_id)
    if not map_events:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sensor upload #{upload_id} not found.",
        )
    return map_events


@router.get("/uploads/{upload_id}/results")
def export_results_csv(
    upload_id: int,
    db: Session = Depends(get_db),
):
    """Downloads model predictions and GPS coordinates for the upload as CSV."""
    csv_content = SensorService.export_results_csv(db, upload_id)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=sensor_predictions_{upload_id}.csv"
        },
    )
