"""
RoadSense Sensor Analytics - Preprocessing & Feature Extraction Engine
=======================================================================
Processes multi-axis accelerometer and gyroscope time-series streams:
1. Signal Preprocessing: Gravity compensation, magnitude, vertical acceleration shock.
2. Feature Extraction: Windowed peak-to-peak amplitude, rolling variance, angular tilt.
3. Event Classifier: ML/Heuristic classification for Potholes, Speed Bumps, and Rough Roads.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd

GRAVITY = 9.80665


def validate_and_load_sensor_csv(filepath_or_buffer) -> pd.DataFrame:
    """Loads and standardizes smartphone sensor log CSV."""
    df = pd.read_csv(filepath_or_buffer)
    
    # Standardize column naming
    col_mapping = {}
    for col in df.columns:
        c = col.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
        if c in ["acc_x", "ax", "accelerometer_x"] or c.endswith("_x") and ("acc" in c or "accelerometer" in c):
            col_mapping[col] = "acc_x"
        elif c in ["acc_y", "ay", "accelerometer_y"] or c.endswith("_y") and ("acc" in c or "accelerometer" in c):
            col_mapping[col] = "acc_y"
        elif c in ["acc_z", "az", "accelerometer_z"] or c.endswith("_z") and ("acc" in c or "accelerometer" in c):
            col_mapping[col] = "acc_z"
        elif c in ["gyro_x", "gx", "gyroscope_x"] or c.endswith("_x") and ("gyro" in c or "gyroscope" in c):
            col_mapping[col] = "gyro_x"
        elif c in ["gyro_y", "gy", "gyroscope_y"] or c.endswith("_y") and ("gyro" in c or "gyroscope" in c):
            col_mapping[col] = "gyro_y"
        elif c in ["gyro_z", "gz", "gyroscope_z"] or c.endswith("_z") and ("gyro" in c or "gyroscope" in c):
            col_mapping[col] = "gyro_z"
        elif "lat" in c:
            col_mapping[col] = "latitude"
        elif "lon" in c or "lng" in c:
            col_mapping[col] = "longitude"
        elif "spd" in c or "speed" in c:
            col_mapping[col] = "speed"
        elif "time" in c or "date" in c:
            col_mapping[col] = "timestamp"

    df = df.rename(columns=col_mapping)

    # Defaults for optional metadata if missing
    if "acc_x" not in df.columns or "acc_z" not in df.columns:
        raise ValueError("CSV must contain accelerometer streams (acc_x, acc_y, acc_z).")

    if "gyro_x" not in df.columns:
        df["gyro_x"] = 0.0
    if "gyro_y" not in df.columns:
        df["gyro_y"] = 0.0
    if "gyro_z" not in df.columns:
        df["gyro_z"] = 0.0

    if "latitude" not in df.columns or "longitude" not in df.columns:
        # Default mock urban trajectory if GPS was unrecorded
        df["latitude"] = 28.6139 + np.linspace(0, 0.015, len(df))
        df["longitude"] = 77.2090 + np.linspace(0, 0.018, len(df))

    if "speed" not in df.columns:
        df["speed"] = 40.0

    if "timestamp" not in df.columns:
        df["timestamp"] = [f"T+{i*0.1:.1f}s" for i in range(len(df))]

    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates kinematic and statistical features over the raw sensor streams."""
    proc = df.copy()

    # 1. Total Acceleration Magnitude: ||a|| = sqrt(ax^2 + ay^2 + az^2)
    proc["acc_mag"] = np.sqrt(proc["acc_x"]**2 + proc["acc_y"]**2 + proc["acc_z"]**2)

    # 2. Vertical Z-Axis Deviation from normal gravity: |az - g|
    proc["z_shock"] = np.abs(proc["acc_z"] - GRAVITY)

    # 3. Gyroscope Total Angular Pitch/Roll Velocity
    proc["gyro_mag"] = np.sqrt(proc["gyro_x"]**2 + proc["gyro_y"]**2)

    # 4. Trailing Windowed Statistics (Window = 5 samples ~0.5s at 10Hz)
    proc["rolling_z_std"] = proc["acc_z"].rolling(window=5, min_periods=1).std().fillna(0)
    proc["rolling_z_ptp"] = proc["acc_z"].rolling(window=5, min_periods=1).apply(lambda w: np.ptp(w) if len(w) > 0 else 0).fillna(0)

    return proc


def detect_road_events(df: pd.DataFrame, source_filename: str = "upload.csv") -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Evaluates sensor windows and classifies road distress anomalies:
    - Pothole: Sharp vertical drop + violent rebound spike
    - Speed Bump: Coordinated vertical lift with pitch rotation
    - Rough Road: Sustained high-frequency vibration
    """
    proc = extract_features(df)
    events: List[Dict[str, Any]] = []
    
    n_samples = len(proc)
    skip_until = 0

    for i in range(2, n_samples - 2):
        if i < skip_until:
            continue

        curr_az = proc.loc[i, "acc_z"]
        prev_az = proc.loc[i - 1, "acc_z"]
        next_az = proc.loc[i + 1, "acc_z"]
        curr_shock = proc.loc[i, "z_shock"]
        z_ptp = proc.loc[i, "rolling_z_ptp"]
        gyro_m = proc.loc[i, "gyro_mag"]
        z_std = proc.loc[i, "rolling_z_std"]
        lat = proc.loc[i, "latitude"]
        lon = proc.loc[i, "longitude"]
        spd = proc.loc[i, "speed"]
        t_stamp = str(proc.loc[i, "timestamp"])

        # Detection Rule 1: POTHOLE (sharp downward dip followed immediately by upward rebound, or violent vertical swing)
        is_dip_rebound = (curr_az < 6.8 and next_az > 12.5) or (prev_az < 6.8 and curr_az > 13.0) or (curr_shock > 5.5 and (curr_az < 7.0 or curr_az > 14.5))
        if is_dip_rebound:
            severity = "High" if curr_shock > 7.0 or z_ptp > 10.0 else "Medium"
            conf = min(0.98, 0.74 + (curr_shock / 30.0))
            events.append({
                "timestamp": t_stamp,
                "event_type": "Pothole",
                "confidence": round(conf, 3),
                "severity": severity,
                "z_acc_spike": round(curr_shock, 2),
                "gyro_pitch": round(gyro_m, 3),
                "latitude": float(lat),
                "longitude": float(lon),
                "speed": float(spd),
                "source_file": source_filename,
            })
            skip_until = i + 6  # Skip past the impact rebound

        # Detection Rule 2: SPEED BUMP (upward heave without severe drop + noticeable pitch rate)
        elif curr_az > 12.5 and gyro_m > 0.30 and curr_az > prev_az and curr_shock < 9.0:
            severity = "Medium" if spd >= 35.0 else "Low"
            conf = min(0.94, 0.76 + (gyro_m / 4.0))
            events.append({
                "timestamp": t_stamp,
                "event_type": "Speed Bump",
                "confidence": round(conf, 3),
                "severity": severity,
                "z_acc_spike": round(curr_shock, 2),
                "gyro_pitch": round(gyro_m, 3),
                "latitude": float(lat),
                "longitude": float(lon),
                "speed": float(spd),
                "source_file": source_filename,
            })
            skip_until = i + 6

        # Detection Rule 3: ROUGH ROAD / CORRUGATION (continuous sustained vibration, but no massive single shock)
        elif z_std > 2.0 and z_ptp > 4.5 and curr_shock < 5.0:
            events.append({
                "timestamp": t_stamp,
                "event_type": "Rough Road",
                "confidence": round(0.84, 3),
                "severity": "Medium",
                "z_acc_spike": round(curr_shock, 2),
                "gyro_pitch": round(gyro_m, 3),
                "latitude": float(lat),
                "longitude": float(lon),
                "speed": float(spd),
                "source_file": source_filename,
            })
            skip_until = i + 8

    return proc, events
