import os
import hashlib
import io
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.demo_routes import AURANGABAD_CENTER, get_demo_route_polyline
from app.models.sensor import SensorUpload, SensorWindow
from app.ml.sensor_predictor import SensorPredictor
from app.services.road_memory_service import RoadMemoryService


class SensorService:
    @staticmethod
    def process_and_save_upload(
        db: Session,
        file_bytes: bytes,
        filename: str,
        route_mode: str = "demo_aurangabad",
        user_id: Optional[int] = None,
        session_id: Optional[int] = None,
    ) -> SensorUpload:
        """
        Validates, persists raw CSV to storage, runs the trained ML pipeline,
        and saves windows & predictions with GPS provenance to the database.
        """
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        file_size = len(file_bytes)

        # Save to storage
        safe_name = f"{int(datetime.now().timestamp())}_{filename}"
        dest_path = settings.UPLOAD_DIR / safe_name
        dest_path.write_bytes(file_bytes)

        # Read into dataframe
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as e:
            raise ValueError(f"Failed to parse CSV file: {e}")

        # Run SensorPredictor with specified route_mode ('demo_aurangabad' or 'real_gps')
        predictor = SensorPredictor.get_instance()
        pred_result = predictor.predict_dataframe(df, route_mode=route_mode)

        meta = pred_result.get("metadata", {})
        has_gps = bool(meta.get("has_gps", False))
        geolocated_count = int(meta.get("geolocated_windows", 0))

        # Create SensorUpload DB record
        upload_record = SensorUpload(
            session_id=session_id,
            user_id=user_id,
            filename=safe_name,
            original_name=filename,
            file_path=str(dest_path),
            file_size_bytes=file_size,
            file_hash=file_hash,
            row_count=meta.get("total_rows", len(df)),
            valid_window_count=pred_result["total_windows"],
            sample_rate_hz=50.0,
            status="completed",
            route_mode=route_mode,
            has_gps=has_gps,
            geolocated_window_count=geolocated_count,
        )
        db.add(upload_record)
        db.flush()

        # Insert SensorWindow records
        window_entities = []
        for w in pred_result["windows"]:
            window_entity = SensorWindow(
                upload_id=upload_record.id,
                session_id=session_id,
                window_index=w["window_index"],
                start_seconds=w["start_seconds"],
                end_seconds=w["end_seconds"],
                sample_count=w["sample_count"],
                predicted_class=w["predicted_class"],
                confidence=w["confidence"],
                features_json=w["features"],
                latitude=w["latitude"],
                longitude=w["longitude"],
                speed=w["speed"],
                coordinate_provenance=w.get("coordinate_provenance", "none"),
                is_interpolated=w.get("is_interpolated", False),
                is_location_pending=w.get("is_location_pending", False),
            )
            window_entities.append(window_entity)

        db.bulk_save_objects(window_entities)
        db.commit()
        db.refresh(upload_record)

        # If actual GPS coordinates are available and real_gps mode was chosen, sync Road Memory Engine
        if has_gps and route_mode == "real_gps":
            RoadMemoryService.sync_hotspots_from_events(db)

        return upload_record

    @staticmethod
    def get_upload_detail(db: Session, upload_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves upload record along with its predicted windows, counts, and GPS info."""
        upload = db.query(SensorUpload).filter(SensorUpload.id == upload_id).first()
        if not upload:
            return None

        windows = (
            db.query(SensorWindow)
            .filter(SensorWindow.upload_id == upload_id)
            .order_by(SensorWindow.window_index.asc())
            .all()
        )

        counts = {"Bump": 0, "Pothole": 0}
        for w in windows:
            counts[w.predicted_class] = counts.get(w.predicted_class, 0) + 1

        return {
            "id": upload.id,
            "filename": upload.filename,
            "original_name": upload.original_name,
            "row_count": upload.row_count,
            "valid_window_count": upload.valid_window_count,
            "status": upload.status,
            "route_mode": getattr(upload, "route_mode", "real_gps"),
            "has_gps": getattr(upload, "has_gps", False),
            "geolocated_window_count": getattr(upload, "geolocated_window_count", 0),
            "created_at": upload.created_at,
            "error_message": upload.error_message,
            "counts": counts,
            "windows": [
                {
                    "id": w.id,
                    "window_index": w.window_index,
                    "start_seconds": w.start_seconds,
                    "end_seconds": w.end_seconds,
                    "sample_count": w.sample_count,
                    "predicted_class": w.predicted_class,
                    "confidence": w.confidence,
                    "confidence_percent": round(w.confidence * 100.0, 1),
                    "latitude": w.latitude,
                    "longitude": w.longitude,
                    "speed": w.speed,
                    "coordinate_provenance": getattr(w, "coordinate_provenance", "none"),
                    "is_interpolated": getattr(w, "is_interpolated", False),
                    "is_location_pending": w.is_location_pending,
                    "features": w.features_json,
                }
                for w in windows
            ],
        }

    @staticmethod
    def get_upload_map_events(db: Session, upload_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves geolocated map events for an upload, including Aurangabad center
        and route polyline if Demo Route Mode was used.
        """
        upload = db.query(SensorUpload).filter(SensorUpload.id == upload_id).first()
        if not upload:
            return None

        windows = (
            db.query(SensorWindow)
            .filter(SensorWindow.upload_id == upload_id)
            .order_by(SensorWindow.window_index.asc())
            .all()
        )

        events = []
        lats = []
        lons = []
        class_counts = {"Bump": 0, "Pothole": 0}

        for w in windows:
            class_counts[w.predicted_class] = class_counts.get(w.predicted_class, 0) + 1
            if w.latitude is not None and w.longitude is not None:
                lats.append(w.latitude)
                lons.append(w.longitude)
                events.append({
                    "id": w.id,
                    "upload_id": upload.id,
                    "window_index": w.window_index,
                    "start_seconds": w.start_seconds,
                    "end_seconds": w.end_seconds,
                    "predicted_class": w.predicted_class,
                    "confidence": w.confidence,
                    "confidence_percent": round(w.confidence * 100.0, 1),
                    "latitude": w.latitude,
                    "longitude": w.longitude,
                    "speed": w.speed,
                    "coordinate_provenance": getattr(w, "coordinate_provenance", "none"),
                    "is_interpolated": getattr(w, "is_interpolated", False),
                    "is_location_pending": w.is_location_pending,
                })

        route_mode = getattr(upload, "route_mode", "real_gps")

        # Map center determination: verify within Chhatrapati Sambhajinagar bounds
        is_aurangabad_coords = False
        if lats and lons:
            avg_lat = sum(lats) / len(lats)
            avg_lon = sum(lons) / len(lons)
            if 19.70 <= avg_lat <= 20.05 and 75.10 <= avg_lon <= 75.60:
                is_aurangabad_coords = True

        if route_mode == "demo_aurangabad" or not is_aurangabad_coords:
            center_lat = AURANGABAD_CENTER["lat"]
            center_lon = AURANGABAD_CENTER["lon"]
        else:
            center_lat = float(sum(lats) / len(lats))
            center_lon = float(sum(lons) / len(lons))

        # Always return the Jalna Road corridor polyline for demonstration & municipal context
        demo_polyline = get_demo_route_polyline()

        return {
            "upload_id": upload.id,
            "filename": upload.original_name,
            "route_mode": route_mode,
            "has_gps": getattr(upload, "has_gps", False),
            "total_windows": len(windows),
            "geolocated_events_count": len(events),
            "counts": class_counts,
            "center": {"lat": center_lat, "lon": center_lon},
            "demo_polyline": demo_polyline,
            "events": events,
        }

    @staticmethod
    def export_results_csv(db: Session, upload_id: int) -> str:
        """Exports predicted windows for an upload as standard CSV string with GPS and provenance fields."""
        windows = (
            db.query(SensorWindow)
            .filter(SensorWindow.upload_id == upload_id)
            .order_by(SensorWindow.window_index.asc())
            .all()
        )
        data = [
            {
                "window_index": w.window_index,
                "start_seconds": w.start_seconds,
                "end_seconds": w.end_seconds,
                "sample_count": w.sample_count,
                "predicted_class": w.predicted_class,
                "confidence": w.confidence,
                "confidence_percent": round(w.confidence * 100.0, 1),
                "latitude": w.latitude,
                "longitude": w.longitude,
                "speed": w.speed,
                "coordinate_provenance": getattr(w, "coordinate_provenance", "none"),
                "is_interpolated": getattr(w, "is_interpolated", False),
                "is_location_pending": w.is_location_pending,
            }
            for w in windows
        ]
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
