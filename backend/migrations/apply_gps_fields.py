import sqlite3
import os

db_path = "roadsense_production.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("PRAGMA table_info(sensor_uploads)")
    cols = [r[1] for r in cur.fetchall()]
    print("sensor_uploads existing cols:", cols)
    if "route_mode" not in cols:
        cur.execute("ALTER TABLE sensor_uploads ADD COLUMN route_mode VARCHAR(50) DEFAULT 'real_gps'")
    if "has_gps" not in cols:
        cur.execute("ALTER TABLE sensor_uploads ADD COLUMN has_gps BOOLEAN DEFAULT 0")
    if "geolocated_window_count" not in cols:
        cur.execute("ALTER TABLE sensor_uploads ADD COLUMN geolocated_window_count INTEGER DEFAULT 0")
        
    cur.execute("PRAGMA table_info(sensor_windows)")
    w_cols = [r[1] for r in cur.fetchall()]
    print("sensor_windows existing cols:", w_cols)
    if "coordinate_provenance" not in w_cols:
        cur.execute("ALTER TABLE sensor_windows ADD COLUMN coordinate_provenance VARCHAR(50) DEFAULT 'none'")
    if "is_interpolated" not in w_cols:
        cur.execute("ALTER TABLE sensor_windows ADD COLUMN is_interpolated BOOLEAN DEFAULT 0")
    if "gps_timestamp" not in w_cols:
        cur.execute("ALTER TABLE sensor_windows ADD COLUMN gps_timestamp VARCHAR(50)")
        
    conn.commit()
    conn.close()
    print("Database schema successfully upgraded with GPS provenance columns!")
else:
    print("roadsense_production.db does not exist yet.")
