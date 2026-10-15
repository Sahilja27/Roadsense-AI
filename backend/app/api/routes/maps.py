from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.incident import RoadIncident
from app.models.road_memory import HotspotCluster
from app.models.sensor import SensorUpload
from app.schemas.incident import IncidentOut
from app.schemas.road_memory import HotspotClusterOut, NearbyHazardOut
from app.services.road_memory_service import RoadMemoryService

router = APIRouter()


@router.get("/incidents", response_model=List[IncidentOut])
def get_map_incidents(
    upload_id: Optional[int] = Query(None, description="Filter incidents by associated sensor upload ID"),
    db: Session = Depends(get_db),
):
    """Retrieves all confirmed geotagged road incidents for GIS map rendering."""
    query = db.query(RoadIncident).filter(RoadIncident.latitude.isnot(None), RoadIncident.longitude.isnot(None))
    
    if upload_id is not None:
        upload = db.query(SensorUpload).filter(SensorUpload.id == upload_id).first()
        if upload and upload.session_id:
            query = query.filter(RoadIncident.session_id == upload.session_id)
            
    incidents = query.all()
    return incidents


@router.get("/hotspots", response_model=List[HotspotClusterOut])
def get_map_hotspots(
    db: Session = Depends(get_db),
):
    """Retrieves all consolidated Road Memory hotspots and priority scores."""
    hotspots = db.query(HotspotCluster).order_by(HotspotCluster.priority_score.desc()).all()
    return hotspots


@router.get("/nearby", response_model=List[NearbyHazardOut])
def get_nearby_hazards(
    latitude: float = Query(..., description="Vehicle current latitude"),
    longitude: float = Query(..., description="Vehicle current longitude"),
    radius_meters: float = Query(350.0, description="Scanning radius in meters"),
    db: Session = Depends(get_db),
):
    """
    Finds road memory hazards within a given radius of the driver's current coordinates.
    Dispatched to the Driver Interface for active proximity warning HUD alerts.
    """
    nearby = RoadMemoryService.find_nearby_hazards(
        db=db,
        current_lat=latitude,
        current_lon=longitude,
        radius_meters=radius_meters,
    )
    return nearby
