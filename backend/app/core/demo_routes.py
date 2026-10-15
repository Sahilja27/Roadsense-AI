"""
RoadSense AI - Documented Demonstration Routes
==============================================
Predefined, verified public road coordinates for presentation & demonstration mode.
Specifically provides public road coordinates along Jalna Road (NH-753F / SH-6)
in Chhatrapati Sambhajinagar (Aurangabad), Maharashtra, India.
"""

from typing import List, Tuple, Dict, Any
import numpy as np


# Jalna Road (NH-753F) Public Road Corridor in Chhatrapati Sambhajinagar (Aurangabad)
# Starts at Kranti Chowk, runs east past Akashwani, Seven Hills, CIDCO, to Chikalthana.
AURANGABAD_JALNA_ROAD_WAYPOINTS: List[Tuple[float, float, str]] = [
    (19.87372, 75.32485, "Kranti Chowk Public Junction"),
    (19.87445, 75.32890, "Gopal Tea / Nirala Bazar Link"),
    (19.87520, 75.33380, "Sessions Court / Jalna Road"),
    (19.87595, 75.34120, "Akashwani Prasar Bharati Circle"),
    (19.87648, 75.34750, "Mondha Naka Road Flyover"),
    (19.87698, 75.35380, "Seven Hills Flyover, Jalna Road"),
    (19.87750, 75.35920, "CIDCO Town Centre Cross"),
    (19.87785, 75.36250, "CIDCO Cannaught Place Entrance"),
    (19.87815, 75.36680, "APMC Market / Jalna Road"),
    (19.87855, 75.37120, "MGM Medical College & Hospital"),
    (19.87910, 75.37620, "Dhoot Hospital Road Link"),
    (19.87955, 75.38050, "Cambridge School Junction"),
    (19.88015, 75.38520, "Chikalthana MIDC Road"),
    (19.88065, 75.38950, "Chhatrapati Sambhajinagar Airport Road"),
]

AURANGABAD_CENTER = {"lat": 19.8762, "lon": 75.3433, "city": "Chhatrapati Sambhajinagar (Aurangabad)", "state": "Maharashtra"}


def generate_route_coordinates(num_points: int) -> List[Dict[str, Any]]:
    """
    Deterministically maps `num_points` chronologically spaced along the documented
    public road corridor in Chhatrapati Sambhajinagar (Aurangabad).
    
    Returns a list of dicts with latitude, longitude, and landmark info.
    """
    if num_points <= 0:
        return []
    
    waypoints = np.array([[wp[0], wp[1]] for wp in AURANGABAD_JALNA_ROAD_WAYPOINTS])
    total_segments = len(waypoints) - 1
    
    # Calculate cumulative distance along waypoints
    diffs = np.diff(waypoints, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_dist = np.insert(np.cumsum(seg_lengths), 0, 0.0)
    total_dist = cum_dist[-1]
    
    # Distribute num_points evenly along the path
    t_vals = np.linspace(0.0, total_dist, num_points)
    
    points = []
    for i, t in enumerate(t_vals):
        # Find which segment t falls into
        idx = np.searchsorted(cum_dist, t, side="right") - 1
        idx = min(max(idx, 0), total_segments - 1)
        
        seg_start_dist = cum_dist[idx]
        seg_len = seg_lengths[idx]
        ratio = (t - seg_start_dist) / seg_len if seg_len > 0 else 0.0
        
        lat = float(waypoints[idx, 0] + ratio * (waypoints[idx + 1, 0] - waypoints[idx, 0]))
        lon = float(waypoints[idx, 1] + ratio * (waypoints[idx + 1, 1] - waypoints[idx, 1]))
        
        # Round to 6 decimal places (typical GPS precision ~ 10cm)
        lat = round(lat, 6)
        lon = round(lon, 6)
        
        nearest_landmark = AURANGABAD_JALNA_ROAD_WAYPOINTS[idx][2]
        
        points.append({
            "index": i,
            "latitude": lat,
            "longitude": lon,
            "landmark": nearest_landmark,
            "source": "demo_route",
            "provenance_label": "DEMO / SIMULATED LOCATION (Aurangabad Public Road)",
        })
        
    return points


def get_demo_route_polyline() -> List[List[float]]:
    """Returns the ordered [lat, lon] coordinates of the entire demo route for map polyline rendering."""
    return [[wp[0], wp[1]] for wp in AURANGABAD_JALNA_ROAD_WAYPOINTS]
