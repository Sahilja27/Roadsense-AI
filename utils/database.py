"""
RoadSense Sensor Analytics - Database Management Module
========================================================
Manages SQLite persistence for:
1. raw_sensor_events: Individual road anomaly events detected from smartphone sensor CSVs.
2. road_memory: Aggregated spatial memory clusters representing confirmed recurring distress.
"""

import os
import sqlite3
from datetime import datetime, timedelta
import random
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "roadsense_sensor.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_sensor_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT NOT NULL,
            z_acc_spike REAL NOT NULL,
            gyro_pitch REAL NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            speed REAL NOT NULL,
            source_file TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS road_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            primary_hazard TEXT NOT NULL,
            avg_severity TEXT NOT NULL,
            recurrence_count INTEGER NOT NULL,
            last_detected TEXT NOT NULL,
            priority_score REAL NOT NULL,
            status TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

    seed_demo_data_if_empty(db_path)


def insert_sensor_event(
    timestamp: str,
    event_type: str,
    confidence: float,
    severity: str,
    z_acc_spike: float,
    gyro_pitch: float,
    latitude: float,
    longitude: float,
    speed: float,
    source_file: str,
    db_path: str = DB_PATH,
) -> int:
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO raw_sensor_events (
            timestamp, event_type, confidence, severity, z_acc_spike,
            gyro_pitch, latitude, longitude, speed, source_file
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp, event_type, round(confidence, 3), severity,
        round(z_acc_spike, 2), round(gyro_pitch, 3),
        round(latitude, 6), round(longitude, 6), round(speed, 1), source_file
    ))
    event_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return event_id


def get_all_events(db_path: str = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    conn = get_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM raw_sensor_events ORDER BY id DESC", conn)
    conn.close()
    return df


def get_road_memory_clusters(db_path: str = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    conn = get_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM road_memory ORDER BY priority_score DESC", conn)
    conn.close()
    return df


def update_road_memory_status(cluster_id: int, new_status: str, db_path: str = DB_PATH) -> None:
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE road_memory SET status = ? WHERE id = ?", (new_status, cluster_id))
    conn.commit()
    conn.close()


def clear_all_events(db_path: str = DB_PATH) -> None:
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM raw_sensor_events")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='raw_sensor_events'")
    cursor.execute("DELETE FROM road_memory")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='road_memory'")
    conn.commit()
    conn.close()


def seed_demo_data_if_empty(db_path: str = DB_PATH) -> None:
    """Pre-seeds realistic municipal sensor logs and confirmed road memory clusters."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM road_memory")
    mem_count = cursor.fetchone()[0]

    if mem_count == 0:
        base_lat, base_lon = 28.6139, 77.2090
        now = datetime.now()

        # Seed pre-aggregated Road Memory Clusters (Chronic Road Distress)
        clusters = [
            ("Corridor Sector A-1 (Main Arterial)", base_lat + 0.0036, base_lon + 0.0045, "Pothole", "High", 14, (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"), 94.2, "Action Required"),
            ("Ring Road Flyover Entry", base_lat + 0.0084, base_lon + 0.0102, "Pothole", "High", 9, (now - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M"), 88.5, "Action Required"),
            ("School Zone Crossing West", base_lat + 0.0168, base_lon + 0.0210, "Speed Bump", "Medium", 22, (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"), 62.0, "Scheduled"),
            ("Market Access Link #4", base_lat - 0.0050, base_lon - 0.0062, "Rough Road", "Medium", 8, (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M"), 58.4, "Monitored"),
            ("Industrial Bypass km 12", base_lat + 0.0125, base_lon + 0.0155, "Pothole", "High", 11, (now - timedelta(hours=8)).strftime("%Y-%m-%d %H:%M"), 91.0, "Action Required"),
            ("Hospital Approach Road", base_lat - 0.0092, base_lon + 0.0078, "Speed Bump", "Low", 18, (now - timedelta(hours=12)).strftime("%Y-%m-%d %H:%M"), 42.5, "Monitored"),
        ]

        cursor.executemany("""
            INSERT INTO road_memory (
                cluster_name, latitude, longitude, primary_hazard,
                avg_severity, recurrence_count, last_detected, priority_score, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, clusters)

        # Seed realistic individual sensor events associated with these locations
        raw_events = []
        for c in clusters:
            name, c_lat, c_lon, hazard, sev, count, last_det, score, stat = c
            for i in range(min(count, 4)):
                t_stamp = (now - timedelta(hours=i*3 + 1)).strftime("%Y-%m-%d %H:%M:%S")
                conf = round(random.uniform(0.78, 0.96), 2)
                z_spike = round(random.uniform(5.5, 11.2) if sev == "High" else random.uniform(3.0, 5.4), 2)
                g_pitch = round(random.uniform(0.4, 1.2), 2)
                lat_jitter = c_lat + random.uniform(-0.00008, 0.00008)
                lon_jitter = c_lon + random.uniform(-0.00008, 0.00008)
                spd = round(random.uniform(32, 54), 1)
                raw_events.append((t_stamp, hazard, conf, sev, z_spike, g_pitch, lat_jitter, lon_jitter, spd, "fleet_logger_demo.csv"))

        cursor.executemany("""
            INSERT INTO raw_sensor_events (
                timestamp, event_type, confidence, severity, z_acc_spike,
                gyro_pitch, latitude, longitude, speed, source_file
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, raw_events)

        conn.commit()

    conn.close()
