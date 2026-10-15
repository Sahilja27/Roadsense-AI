from typing import List, Optional
import io
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status, Response, Body
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.incident import RoadIncident
from app.models.authority import AuthorityAction
from app.schemas.authority import AuthorityActionOut, AuthorityActionCreate
from app.schemas.incident import IncidentOut
from app.services.incident_service import IncidentService
from app.api.deps import get_current_user_optional, require_role
from app.models.user import User

router = APIRouter()


@router.get("/summary")
def get_authority_summary(db: Session = Depends(get_db)):
    """Summary of active work orders, pending triage, and repair throughput."""
    p1_urgent = (
        db.query(RoadIncident)
        .filter(RoadIncident.severity.in_(["High", "Critical"]), RoadIncident.status != "resolved")
        .count()
    )
    scheduled = db.query(RoadIncident).filter(RoadIncident.status == "scheduled").count()
    assigned = db.query(RoadIncident).filter(RoadIncident.status == "assigned").count()
    under_review = db.query(RoadIncident).filter(RoadIncident.status == "under_review").count()
    resolved = db.query(RoadIncident).filter(RoadIncident.status == "resolved").count()

    return {
        "p1_urgent_action_required": p1_urgent,
        "under_review": under_review,
        "assigned_to_crews": assigned,
        "scheduled_patching": scheduled,
        "total_resolved": resolved,
    }


@router.post("/incidents/{incident_id}/assign", response_model=IncidentOut)
def assign_incident(
    incident_id: int,
    assigned_to: str = Body(..., embed=True),
    notes: Optional[str] = Body(None, embed=True),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """Assigns road damage repair work order to field contractor or engineering team."""
    user_id = current_user.id if current_user else None
    inc = IncidentService.update_incident_status(
        db=db,
        incident_id=incident_id,
        new_status="assigned",
        user_id=user_id,
        assigned_to=assigned_to,
        notes=notes or f"Assigned to {assigned_to}",
    )
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found.")
    return inc


@router.post("/incidents/{incident_id}/resolve", response_model=IncidentOut)
def resolve_incident(
    incident_id: int,
    resolution_notes: str = Body(..., embed=True),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """Marks a road defect repaired and updates the incident lifecycle."""
    user_id = current_user.id if current_user else None
    inc = IncidentService.update_incident_status(
        db=db,
        incident_id=incident_id,
        new_status="resolved",
        user_id=user_id,
        notes=f"Resolved: {resolution_notes}",
    )
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found.")
    return inc


@router.get("/work-orders/export")
def export_work_orders(
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Exports filtered road maintenance incidents into a CSV work order."""
    query = db.query(RoadIncident)
    if status_filter:
        query = query.filter(RoadIncident.status == status_filter)

    incidents = query.order_by(RoadIncident.created_at.desc()).all()
    rows = [
        {
            "WorkOrder_ID": f"WO-{inc.id:04d}",
            "Title": inc.title,
            "Distress_Type": inc.distress_type,
            "Severity": inc.severity,
            "Status": inc.status,
            "Verification": inc.verification_status,
            "Confidence": round(inc.confidence, 2),
            "Latitude": inc.latitude if inc.latitude is not None else "Pending",
            "Longitude": inc.longitude if inc.longitude is not None else "Pending",
            "Assigned_To": inc.assigned_to or "Unassigned",
            "Reported_At": inc.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "Notes": inc.notes or "",
        }
        for inc in incidents
    ]
    df = pd.DataFrame(rows)
    csv_bytes = df.to_csv(index=False)

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=roadsense_work_orders.csv"},
    )
