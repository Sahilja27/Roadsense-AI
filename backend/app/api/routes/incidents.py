from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.incident import RoadIncident
from app.schemas.incident import IncidentCreate, IncidentUpdate, IncidentOut
from app.services.incident_service import IncidentService
from app.api.deps import get_current_user_optional
from app.models.user import User

router = APIRouter()


@router.get("", response_model=List[IncidentOut])
def list_incidents(
    distress_type: Optional[str] = None,
    severity: Optional[str] = None,
    status_filter: Optional[str] = Query(None, alias="status"),
    verification_status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """Retrieves road incidents with filtering and pagination."""
    query = db.query(RoadIncident)

    if distress_type:
        query = query.filter(RoadIncident.distress_type == distress_type)
    if severity:
        query = query.filter(RoadIncident.severity == severity)
    if status_filter:
        query = query.filter(RoadIncident.status == status_filter)
    if verification_status:
        query = query.filter(RoadIncident.verification_status == verification_status)

    incidents = query.order_by(RoadIncident.created_at.desc()).offset(skip).limit(limit).all()
    return incidents


@router.get("/{incident_id}", response_model=IncidentOut)
def get_incident(incident_id: int, db: Session = Depends(get_db)):
    """Retrieves full incident details with evidence and audit history."""
    inc = db.query(RoadIncident).filter(RoadIncident.id == incident_id).first()
    if not inc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident #{incident_id} not found.",
        )
    return inc


@router.post("", response_model=IncidentOut)
def create_incident(
    inc_in: IncidentCreate,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """
    Submits a new road defect incident.
    Performs automatic deduplication (30m threshold) and multimodal verification.
    """
    user_id = current_user.id if current_user else None

    incident = IncidentService.create_incident(
        db=db,
        title=inc_in.title,
        distress_type=inc_in.distress_type,
        severity=inc_in.severity,
        confidence=inc_in.confidence or 0.85,
        latitude=inc_in.latitude,
        longitude=inc_in.longitude,
        user_id=user_id,
        session_id=inc_in.session_id,
        notes=inc_in.notes,
        address_text=inc_in.address_text,
        evidence_type=inc_in.evidence_type,
        evidence_metadata=inc_in.evidence_metadata,
    )
    return incident


@router.patch("/{incident_id}", response_model=IncidentOut)
def update_incident(
    incident_id: int,
    inc_update: IncidentUpdate,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """Updates status, assignment, severity, or notes for an incident."""
    user_id = current_user.id if current_user else None

    inc = db.query(RoadIncident).filter(RoadIncident.id == incident_id).first()
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found.")

    if inc_update.status:
        inc = IncidentService.update_incident_status(
            db=db,
            incident_id=incident_id,
            new_status=inc_update.status,
            user_id=user_id,
            assigned_to=inc_update.assigned_to,
            notes=inc_update.notes,
        )

    if inc_update.severity:
        inc.severity = inc_update.severity
    if inc_update.verification_status:
        inc.verification_status = inc_update.verification_status

    db.commit()
    db.refresh(inc)
    return inc
