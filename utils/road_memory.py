"""
RoadSense Sensor Analytics - Road Memory Engine
================================================
Aggregates and clusters individual sensor shocks into persistent Road Memory.
1. Geospatial Clustering: Groups events within ~25m radius using Haversine metric.
2. Anomaly Recurrence Scoring: Distinguishes chronic physical road damage from isolated shocks.
3. Municipal Priority Triage: Computes 0-100 inspection urgency index.
4. Proximity Alert Engine: Queries nearby hazardous clusters for driver warning system.
"""

import math
from typing import List, Dict, Any, Tuple
import pandas as pd
from datetime import datetime
from utils.database import get_connection, get_all_events, DB_PATH


def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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


def sync_road_memory_from_raw_events(cluster_radius_m: float = 25.0, db_path: str = DB_PATH) -> int:
    """
    Rebuilds the road_memory cluster registry by aggregating all raw_sensor_events.
    Returns the count of consolidated clusters.
    """
    events_df = get_all_events(db_path)
    if events_df.empty:
        return 0

    clusters: List[Dict[str, Any]] = []

    # Simple spatial clustering within cluster_radius_m
    for _, row in events_df.iterrows():
        r_lat, r_lon = row["latitude"], row["longitude"]
        matched_cluster = None

        for c in clusters:
            dist = haversine_distance_meters(r_lat, r_lon, c["center_lat"], c["center_lon"])
            if dist <= cluster_radius_m:
                matched_cluster = c
                break

        if matched_cluster:
            matched_cluster["events"].append(row)
            # Update weighted centroid
            n = len(matched_cluster["events"])
            matched_cluster["center_lat"] = ((matched_cluster["center_lat"] * (n - 1)) + r_lat) / n
            matched_cluster["center_lon"] = ((matched_cluster["center_lon"] * (n - 1)) + r_lon) / n
        else:
            clusters.append({
                "center_lat": r_lat,
                "center_lon": r_lon,
                "events": [row]
            })

    # Prepare consolidated road_memory records
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM road_memory")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='road_memory'")

    for idx, c in enumerate(clusters, 1):
        evs = c["events"]
        recurrence = len(evs)
        
        # Primary hazard by frequency
        hazard_types = [e["event_type"] for e in evs]
        primary_hazard = max(set(hazard_types), key=hazard_types.count)

        # Average severity score
        sev_scores = {"High": 100, "Medium": 60, "Low": 30}
        avg_sev_num = sum(sev_scores.get(e["severity"], 50) for e in evs) / recurrence
        avg_severity = "High" if avg_sev_num >= 75 else ("Medium" if avg_sev_num >= 45 else "Low")

        # Priority calculation (0 - 100)
        # Combines severity intensity + recurrence density
        priority_score = min(98.5, round((0.45 * avg_sev_num) + (0.45 * min(100, recurrence * 8.0)) + 10.0, 1))

        # Action status
        if priority_score >= 78.0:
            status = "Action Required"
        elif priority_score >= 50.0:
            status = "Scheduled"
        else:
            status = "Monitored"

        # Most recent timestamp
        last_detected = max(str(e["timestamp"]) for e in evs)
        cluster_name = f"Zone #{idx} ({primary_hazard} Hotspot)"

        cursor.execute("""
            INSERT INTO road_memory (
                cluster_name, latitude, longitude, primary_hazard,
                avg_severity, recurrence_count, last_detected, priority_score, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cluster_name, round(c["center_lat"], 6), round(c["center_lon"], 6),
            primary_hazard, avg_severity, recurrence, last_detected, priority_score, status
        ))

    conn.commit()
    conn.close()
    return len(clusters)


def find_nearby_hazards(
    current_lat: float,
    current_lon: float,
    radius_meters: float = 350.0,
    db_path: str = DB_PATH
) -> List[Dict[str, Any]]:
    """
    Finds road memory hazards within a given radius of the driver's current coordinates.
    Used by the Driver Interface for real-time proximity warning dispatch.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM road_memory ORDER BY priority_score DESC")
    rows = cursor.fetchall()
    conn.close()

    nearby = []
    for r in rows:
        dist = haversine_distance_meters(current_lat, current_lon, r["latitude"], r["longitude"])
        if dist <= radius_meters:
            hazard_data = dict(r)
            hazard_data["distance_meters"] = round(dist, 1)
            nearby.append(hazard_data)

    nearby.sort(key=lambda x: x["distance_meters"])
    return nearby
