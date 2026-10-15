from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.analytics import DashboardSummary
from app.services.analytics_service import AnalyticsService

router = APIRouter()


@router.get("/summary", response_model=DashboardSummary)
def get_dashboard_summary(db: Session = Depends(get_db)):
    """
    Computes real-time executive dashboard KPIs strictly from database records.
    (PQI, open hazards, critical defects, chronic hotspots, hazard breakdowns).
    """
    return AnalyticsService.get_dashboard_summary(db)


@router.get("/activity")
def get_activity_feed(db: Session = Depends(get_db)):
    """Timeline of recent sensor uploads, vision detections, and authority maintenance actions."""
    summary = AnalyticsService.get_dashboard_summary(db)
    return {"activity": summary["recent_activity"]}
