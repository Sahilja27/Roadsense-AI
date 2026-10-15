from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.session import get_db
from app.core.config import settings
from app.ml.sensor_predictor import SensorPredictor
from app.ml.yolo_predictor import YOLOPredictor

router = APIRouter()


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Comprehensive health check for DB connection and ML model states."""
    # Check DB
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        db_ok = False

    # Check Sensor Predictor
    sensor_pred = SensorPredictor.get_instance()
    # Check YOLO Predictor
    yolo_pred = YOLOPredictor.get_instance()

    return {
        "status": "healthy" if db_ok else "degraded",
        "app_name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "database": {
            "status": "connected" if db_ok else "disconnected",
            "dialect": settings.DATABASE_URL.split("://")[0],
        },
        "models": {
            "sensor_model": {
                "loaded": sensor_pred.is_loaded,
                "classes": sensor_pred.classes,
                "path": sensor_pred.model_path,
            },
            "yolo_model": {
                "loaded": yolo_pred.is_loaded,
                "classes": list(yolo_pred.names.values()),
                "path": yolo_pred.weights_path,
            },
        },
    }
