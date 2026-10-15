import os
from pathlib import Path
from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    PROJECT_NAME: str = "RoadSense AI"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Security
    SECRET_KEY: str = "roadsense_ai_production_super_secret_key_2026_change_in_prod"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # Database: PostgreSQL primary, SQLite configurable fallback for local dev
    DATABASE_URL: str = "sqlite:///./roadsense_production.db"

    # File Storage
    STORAGE_DIR: Path = BASE_DIR / "storage"
    UPLOAD_DIR: Path = BASE_DIR / "storage" / "uploads"
    DETECTION_DIR: Path = BASE_DIR / "storage" / "detections"
    MAX_UPLOAD_SIZE_MB: int = 150
    RETENTION_DAYS: int = 30

    # ML Model Weights (checks local weights directory first, then sibling repo paths)
    SENSOR_MODEL_PATH: str = str(
        BASE_DIR / "app" / "ml" / "weights" / "roadsense_model.joblib"
    )
    YOLO_MODEL_PATH: str = str(
        BASE_DIR / "app" / "ml" / "weights" / "best.pt"
    )

    # Road Memory Engine & Sensor/Vision Correlation
    CORRELATION_MAX_DISTANCE_M: float = 30.0  # meters
    CORRELATION_MAX_TIME_DIFF_S: float = 4.0   # seconds
    HOTSPOT_CLUSTER_RADIUS_M: float = 25.0    # meters

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
    ]

    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "allow"


settings = Settings()

# Ensure storage directories exist
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.DETECTION_DIR.mkdir(parents=True, exist_ok=True)
