from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.sensor import SensorUpload, SensorWindow
from app.models.vision import VisionJob, VisionDetection
from app.models.incident import RoadIncident
from app.models.road_memory import HotspotCluster
from app.models.authority import AuthorityAction


class AnalyticsService:
    @staticmethod
    def get_dashboard_summary(db: Session) -> Dict[str, Any]:
        """Calculates all executive KPIs strictly from persisted database records."""
        total_uploads = db.query(SensorUpload).count()
        total_windows = db.query(SensorWindow).count()
        total_vjobs = db.query(VisionJob).count()
        total_vdets = db.query(VisionDetection).count()

        open_incidents = (
            db.query(RoadIncident).filter(RoadIncident.status != "resolved").count()
        )
        critical_incidents = (
            db.query(RoadIncident)
            .filter(
                RoadIncident.status != "resolved",
                RoadIncident.severity.in_(["High", "Critical"]),
            )
            .count()
        )
        resolved_incidents = (
            db.query(RoadIncident).filter(RoadIncident.status == "resolved").count()
        )
        verified_incidents = (
            db.query(RoadIncident)
            .filter(RoadIncident.verification_status != "unverified")
            .count()
        )

        chronic_hotspots = (
            db.query(HotspotCluster)
            .filter(HotspotCluster.recurrence_count >= 10)
            .count()
        )

        # Dynamic Pavement Quality Index (PQI) derived from real road health:
        # 100 base score minus penalties for critical defects and active incidents
        raw_pqi = 100.0 - (critical_incidents * 7.5) - (open_incidents * 1.8)
        pqi_score = round(max(10.0, min(100.0, raw_pqi)), 1)

        # Hazard distribution
        hazard_query = (
            db.query(RoadIncident.distress_type, func.count(RoadIncident.id))
            .group_by(RoadIncident.distress_type)
            .all()
        )
        hazard_dist = {h[0]: h[1] for h in hazard_query} if hazard_query else {}

        # Severity distribution
        sev_query = (
            db.query(RoadIncident.severity, func.count(RoadIncident.id))
            .group_by(RoadIncident.severity)
            .all()
        )
        sev_dist = {s[0]: s[1] for s in sev_query} if sev_query else {}

        # Recent activity stream
        activity_items = []

        recent_incidents = (
            db.query(RoadIncident).order_by(RoadIncident.created_at.desc()).limit(5).all()
        )
        for inc in recent_incidents:
            activity_items.append({
                "id": inc.id,
                "activity_type": "incident_created",
                "title": f"Incident Reported: {inc.title}",
                "description": f"{inc.distress_type} ({inc.severity}) - {inc.verification_status}",
                "timestamp": inc.created_at,
                "status": inc.status,
                "severity": inc.severity,
            })

        recent_uploads = (
            db.query(SensorUpload).order_by(SensorUpload.created_at.desc()).limit(3).all()
        )
        for u in recent_uploads:
            activity_items.append({
                "id": u.id,
                "activity_type": "sensor_upload",
                "title": f"Sensor Log Ingested: {u.original_name}",
                "description": f"{u.valid_window_count} windows processed across accelerometer/gyroscope",
                "timestamp": u.created_at,
                "status": u.status,
                "severity": "Info",
            })

        recent_vision = (
            db.query(VisionJob).order_by(VisionJob.created_at.desc()).limit(3).all()
        )
        for vj in recent_vision:
            activity_items.append({
                "id": vj.id,
                "activity_type": "vision_job",
                "title": f"YOLO Vision Job: #{vj.id} ({vj.job_type})",
                "description": f"{vj.total_detections} damage objects identified",
                "timestamp": vj.created_at,
                "status": vj.status,
                "severity": "Info",
            })

        # Sort all activity by timestamp descending
        activity_items.sort(key=lambda x: x["timestamp"], reverse=True)

        return {
            "pqi_score": pqi_score,
            "total_sensor_uploads": total_uploads,
            "total_sensor_windows": total_windows,
            "total_vision_jobs": total_vjobs,
            "total_vision_detections": total_vdets,
            "open_incidents": open_incidents,
            "critical_incidents": critical_incidents,
            "chronic_hotspots": chronic_hotspots,
            "resolved_incidents": resolved_incidents,
            "verified_incidents": verified_incidents,
            "hazard_distribution": hazard_dist,
            "severity_distribution": sev_dist,
            "recent_activity": activity_items[:8],
        }
