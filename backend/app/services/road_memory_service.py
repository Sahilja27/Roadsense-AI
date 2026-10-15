import math
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.road_memory import HotspotCluster
from app.models.sensor import SensorWindow
from app.models.vision import VisionDetection
from app.models.incident import RoadIncident


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two GPS coordinates in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0)**2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


class RoadMemoryService:
    @staticmethod
    def sync_hotspots_from_events(db: Session, cluster_radius_m: Optional[float] = None) -> int:
        """
        Reclusters all geotagged sensor anomalies and visual detections into
        consolidated Road Memory hotspots.
        """
        if cluster_radius_m is None:
            cluster_radius_m = settings.HOTSPOT_CLUSTER_RADIUS_M

        # Gather geotagged sensor windows
        sensor_windows = (
            db.query(SensorWindow)
            .filter(SensorWindow.latitude.isnot(None), SensorWindow.longitude.isnot(None))
            .all()
        )

        # Gather geotagged vision detections
        vision_dets = (
            db.query(VisionDetection)
            .filter(VisionDetection.latitude.isnot(None), VisionDetection.longitude.isnot(None))
            .all()
        )

        all_events = []
        for sw in sensor_windows:
            all_events.append({
                "type": "sensor",
                "hazard": sw.predicted_class,
                "severity": "High" if sw.predicted_class == "Pothole" else "Medium",
                "latitude": sw.latitude,
                "longitude": sw.longitude,
                "timestamp": sw.created_at,
            })

        for vd in vision_dets:
            all_events.append({
                "type": "vision",
                "hazard": vd.class_name.title(),
                "severity": "High" if "pothole" in vd.class_name.lower() else "Medium",
                "latitude": vd.latitude,
                "longitude": vd.longitude,
                "timestamp": vd.created_at,
            })

        if not all_events:
            return 0

        # Cluster events spatially using Haversine
        clusters: List[Dict[str, Any]] = []
        for ev in all_events:
            e_lat, e_lon = ev["latitude"], ev["longitude"]
            matched_cluster = None

            for c in clusters:
                dist = haversine_distance_m(e_lat, e_lon, c["center_lat"], c["center_lon"])
                if dist <= cluster_radius_m:
                    matched_cluster = c
                    break

            if matched_cluster:
                matched_cluster["events"].append(ev)
                n = len(matched_cluster["events"])
                matched_cluster["center_lat"] = ((matched_cluster["center_lat"] * (n - 1)) + e_lat) / n
                matched_cluster["center_lon"] = ((matched_cluster["center_lon"] * (n - 1)) + e_lon) / n
            else:
                clusters.append({
                    "center_lat": e_lat,
                    "center_lon": e_lon,
                    "events": [ev],
                })

        # Persist to hotspot_clusters
        # Clear or reconcile clusters
        for idx, c in enumerate(clusters, 1):
            evs = c["events"]
            recurrence = len(evs)

            hazards = [e["hazard"] for e in evs]
            primary_hazard = max(set(hazards), key=hazards.count)

            sev_scores = {"High": 100, "Medium": 60, "Low": 30}
            avg_sev_num = sum(sev_scores.get(e["severity"], 50) for e in evs) / recurrence
            avg_sev = "High" if avg_sev_num >= 75 else ("Medium" if avg_sev_num >= 45 else "Low")

            # Priority 0 - 100 formula
            priority_score = min(
                99.0,
                round((0.45 * avg_sev_num) + (0.45 * min(100.0, recurrence * 8.0)) + 10.0, 1)
            )

            if recurrence >= 10:
                status = "recurring"
            elif priority_score >= 80.0:
                status = "under_review"
            else:
                status = "new"

            last_detected = max(e["timestamp"] for e in evs)
            cluster_name = f"Zone #{idx} ({primary_hazard} Hotspot)"

            # Check if cluster already exists near this coordinate
            existing = (
                db.query(HotspotCluster)
                .filter(
                    HotspotCluster.center_lat.between(c["center_lat"] - 0.0003, c["center_lat"] + 0.0003),
                    HotspotCluster.center_lon.between(c["center_lon"] - 0.0003, c["center_lon"] + 0.0003),
                )
                .first()
            )

            if existing:
                existing.cluster_name = cluster_name
                existing.center_lat = c["center_lat"]
                existing.center_lon = c["center_lon"]
                existing.primary_hazard = primary_hazard
                existing.avg_severity = avg_sev
                existing.recurrence_count = recurrence
                existing.priority_score = priority_score
                existing.last_detected_at = last_detected
            else:
                cluster_obj = HotspotCluster(
                    cluster_name=cluster_name,
                    center_lat=c["center_lat"],
                    center_lon=c["center_lon"],
                    radius_meters=cluster_radius_m,
                    primary_hazard=primary_hazard,
                    avg_severity=avg_sev,
                    recurrence_count=recurrence,
                    priority_score=priority_score,
                    status=status,
                    last_detected_at=last_detected,
                )
                db.add(cluster_obj)

        db.commit()
        return len(clusters)

    @staticmethod
    def find_nearby_hazards(
        db: Session,
        current_lat: float,
        current_lon: float,
        radius_meters: float = 350.0,
    ) -> List[Dict[str, Any]]:
        """Queries active road memory hotspots within radius for Driver HUD proximity alerts."""
        clusters = db.query(HotspotCluster).all()
        nearby = []

        for c in clusters:
            dist = haversine_distance_m(current_lat, current_lon, c.center_lat, c.center_lon)
            if dist <= radius_meters:
                safe_spd = 25 if c.avg_severity == "High" else 35
                nearby.append({
                    "id": c.id,
                    "cluster_name": c.cluster_name,
                    "latitude": c.center_lat,
                    "longitude": c.center_lon,
                    "primary_hazard": c.primary_hazard,
                    "avg_severity": c.avg_severity,
                    "recurrence_count": c.recurrence_count,
                    "priority_score": c.priority_score,
                    "status": c.status,
                    "distance_meters": round(dist, 1),
                    "safe_speed_kmh": safe_spd,
                })

        nearby.sort(key=lambda x: x["distance_meters"])
        return nearby
