import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import cv2

from app.core.config import settings
from app.models.vision import VisionJob, VisionDetection
from app.ml.yolo_predictor import YOLOPredictor
from app.services.road_memory_service import RoadMemoryService
from app.db.session import SessionLocal


class VisionService:
    @staticmethod
    def process_image(
        db: Session,
        image_bytes: bytes,
        filename: str,
        conf_threshold: float = 0.25,
        user_id: Optional[int] = None,
        session_id: Optional[int] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Runs YOLO on uploaded image, stores detections and annotated image output."""
        file_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
        safe_src = f"src_{int(datetime.now().timestamp())}_{filename}"
        safe_ann = f"ann_{int(datetime.now().timestamp())}_{filename}"

        src_path = settings.UPLOAD_DIR / safe_src
        ann_path = settings.DETECTION_DIR / safe_ann
        src_path.write_bytes(image_bytes)

        predictor = YOLOPredictor.get_instance()
        pred_res = predictor.predict_image(
            image_input=image_bytes,
            conf_threshold=conf_threshold,
            annotate=True,
            save_annotated_path=ann_path,
        )

        job = VisionJob(
            session_id=session_id,
            user_id=user_id,
            job_type="image",
            source_file_path=str(src_path),
            annotated_output_path=str(ann_path),
            file_size_bytes=len(image_bytes),
            mime_type="image/jpeg",
            status="completed",
            progress_percent=100.0,
            total_frames=1,
            processed_frames=1,
            total_detections=pred_res["total_detections"],
            conf_threshold=conf_threshold,
            completed_at=datetime.now(timezone.utc),
        )
        db.add(job)
        db.flush()

        detection_records = []
        is_loc_pending = latitude is None or longitude is None

        for d in pred_res["detections"]:
            det_record = VisionDetection(
                job_id=job.id,
                session_id=session_id,
                frame_number=0,
                timestamp_seconds=0.0,
                class_id=d["class_id"],
                class_name=d["class_name"],
                confidence=d["confidence"],
                bbox_xyxy=d["bbox_xyxy"],
                bbox_normalized=d["bbox_normalized"],
                latitude=latitude,
                longitude=longitude,
                is_location_pending=is_loc_pending,
            )
            detection_records.append(det_record)
            d["latitude"] = latitude
            d["longitude"] = longitude
            d["is_location_pending"] = is_loc_pending

        db.bulk_save_objects(detection_records)
        db.commit()

        if not is_loc_pending:
            RoadMemoryService.sync_hotspots_from_events(db)

        return {
            "job_id": job.id,
            "status": "success",
            "inference_time_ms": pred_res["inference_time_ms"],
            "total_detections": pred_res["total_detections"],
            "max_confidence": pred_res["max_confidence"],
            "class_counts": pred_res["class_counts"],
            "detections": pred_res["detections"],
            "annotated_path": str(ann_path),
            "annotated_url": f"/api/vision/files/detections/{safe_ann}",
        }

    @staticmethod
    def process_video_background(
        job_id: int,
        conf_threshold: float = 0.25,
        sampling_fps: float = 2.0,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ):
        """Background worker executing video inference generator."""
        db = SessionLocal()
        try:
            job = db.query(VisionJob).filter(VisionJob.id == job_id).first()
            if not job:
                return

            job.status = "processing"
            db.commit()

            src_path = Path(job.source_file_path)
            safe_ann = f"ann_{job_id}_{src_path.stem}.mp4"
            ann_path = settings.DETECTION_DIR / safe_ann

            predictor = YOLOPredictor.get_instance()
            generator = predictor.process_video_generator(
                video_path=src_path,
                conf_threshold=conf_threshold,
                sampling_fps=sampling_fps,
                output_annotated_path=ann_path,
            )

            is_loc_pending = latitude is None or longitude is None
            all_dets = []

            for update in generator:
                job.progress_percent = update["progress_percent"]
                job.processed_frames = update["frame_number"]
                job.total_detections = update["total_detections_so_far"]
                db.commit()

                for d in update["frame_detections"]:
                    det_record = VisionDetection(
                        job_id=job.id,
                        session_id=job.session_id,
                        frame_number=d["frame_number"],
                        timestamp_seconds=d["timestamp_seconds"],
                        class_id=d["class_id"],
                        class_name=d["class_name"],
                        confidence=d["confidence"],
                        bbox_xyxy=d["bbox_xyxy"],
                        bbox_normalized=d["bbox_normalized"],
                        latitude=latitude,
                        longitude=longitude,
                        is_location_pending=is_loc_pending,
                    )
                    all_dets.append(det_record)

            if all_dets:
                db.bulk_save_objects(all_dets)

            job.status = "completed"
            job.progress_percent = 100.0
            job.annotated_output_path = str(ann_path)
            job.completed_at = datetime.now(timezone.utc)
            db.commit()

            if not is_loc_pending and all_dets:
                RoadMemoryService.sync_hotspots_from_events(db)

        except Exception as e:
            db.rollback()
            job = db.query(VisionJob).filter(VisionJob.id == job_id).first()
            if job:
                job.status = "failed"
                job.error_message = str(e)
                db.commit()
            print(f"[VisionService] Video job {job_id} error: {e}")
        finally:
            db.close()
