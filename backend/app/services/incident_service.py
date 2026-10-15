import math
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.incident import RoadIncident, IncidentEvidence
from app.models.authority import AuthorityAction
from app.models.audit import AuditLog
from app.models.sensor import SensorWindow
from app.models.vision import VisionDetection
from app.services.road_memory_service import haversine_distance_m, RoadMemoryService


class IncidentService:
    @staticmethod
    def create_incident(
        db: Session,
        title: str,
        distress_type: str,
        severity: str = "Medium",
        confidence: float = 0.85,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        user_id: Optional[int] = None,
        session_id: Optional[int] = None,
        notes: Optional[str] = None,
        address_text: Optional[str] = None,
        evidence_type: Optional[str] = None,
        evidence_file: Optional[str] = None,
        evidence_metadata: Optional[Dict[str, Any]] = None,
    ) -> RoadIncident:
        """
        Creates an incident with deduplication and multimodal correlation checks.
        """
        is_loc_pending = latitude is None or longitude is None

        # 1. Deduplication check: Check if an active incident exists within 30m
        if not is_loc_pending:
            recent_incidents = (
                db.query(RoadIncident)
                .filter(
                    RoadIncident.status != "resolved",
                    RoadIncident.latitude.isnot(None),
                    RoadIncident.longitude.isnot(None),
                )
                .all()
            )
            for inc in recent_incidents:
                dist = haversine_distance_m(latitude, longitude, inc.latitude, inc.longitude)
                if dist <= settings.CORRELATION_MAX_DISTANCE_M:
                    # Match found! Attach evidence to existing incident instead of duplicating
                    if evidence_type:
                        ev = IncidentEvidence(
                            incident_id=inc.id,
                            evidence_type=evidence_type,
                            file_path=evidence_file,
                            metadata_json=evidence_metadata,
                        )
                        db.add(ev)
                    inc.confidence = max(inc.confidence, confidence)
                    db.commit()
                    db.refresh(inc)
                    return inc

        # 2. Multi-modal correlation check
        verification_status = "unverified"
        if not is_loc_pending:
            # Check if there is a corresponding sensor shock within 30m
            matching_sensor = (
                db.query(SensorWindow)
                .filter(SensorWindow.latitude.isnot(None), SensorWindow.longitude.isnot(None))
                .all()
            )
            has_sensor_match = any(
                haversine_distance_m(latitude, longitude, sw.latitude, sw.longitude) <= settings.CORRELATION_MAX_DISTANCE_M
                for sw in matching_sensor
            )

            # Check if there is a corresponding vision detection within 30m
            matching_vision = (
                db.query(VisionDetection)
                .filter(VisionDetection.latitude.isnot(None), VisionDetection.longitude.isnot(None))
                .all()
            )
            has_vision_match = any(
                haversine_distance_m(latitude, longitude, vd.latitude, vd.longitude) <= settings.CORRELATION_MAX_DISTANCE_M
                for vd in matching_vision
            )

            if has_sensor_match and has_vision_match:
                verification_status = "multi_modal_confirmed"
            elif has_sensor_match:
                verification_status = "sensor_confirmed"
            elif has_vision_match:
                verification_status = "vision_confirmed"

        incident = RoadIncident(
            session_id=session_id,
            user_id=user_id,
            title=title,
            distress_type=distress_type,
            severity=severity,
            status="reported",
            verification_status=verification_status,
            confidence=confidence,
            latitude=latitude,
            longitude=longitude,
            is_location_pending=is_loc_pending,
            address_text=address_text,
            notes=notes,
        )
        db.add(incident)
        db.flush()

        # Attach evidence if provided
        if evidence_type:
            ev = IncidentEvidence(
                incident_id=incident.id,
                evidence_type=evidence_type,
                file_path=evidence_file,
                metadata_json=evidence_metadata,
            )
            db.add(ev)

        # Log audit entry
        audit = AuditLog(
            user_id=user_id,
            action="incident_created",
            entity_type="road_incident",
            entity_id=incident.id,
            details_json={
                "title": title,
                "distress_type": distress_type,
                "verification_status": verification_status,
            },
        )
        db.add(audit)

        db.commit()
        db.refresh(incident)

        # Trigger spatial memory sync
        if not is_loc_pending:
            RoadMemoryService.sync_hotspots_from_events(db)

        return incident

    @staticmethod
    def update_incident_status(
        db: Session,
        incident_id: int,
        new_status: str,
        user_id: Optional[int] = None,
        assigned_to: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Optional[RoadIncident]:
        """Updates incident lifecycle status, assignment, and logs audit action."""
        incident = db.query(RoadIncident).filter(RoadIncident.id == incident_id).first()
        if not incident:
            return None

        prev_status = incident.status
        incident.status = new_status
        if assigned_to:
            incident.assigned_to = assigned_to
        if notes:
            incident.notes = (incident.notes or "") + f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {notes}"

        # Record authority action
        action = AuthorityAction(
            incident_id=incident.id,
            performed_by_user_id=user_id,
            action_type="status_change",
            previous_status=prev_status,
            new_status=new_status,
            assigned_to=assigned_to,
            notes=notes,
        )
        db.add(action)

        db.commit()
        db.refresh(incident)
        return incident
