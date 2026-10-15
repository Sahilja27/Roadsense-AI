import os
import sys
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Ensure backend root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.config import settings
from app.db.init_db import init_database
from app.api.routes import (
    health,
    auth,
    sensors,
    vision,
    dashcam,
    incidents,
    maps,
    authority,
    analytics,
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production Full-Stack REST API for RoadSense AI: Smartphone Sensor Anomaly Detection, Road Memory Engine, and YOLO Vision Processing.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/openapi.json",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static storage for uploaded and annotated detection files
app.mount(
    "/storage",
    StaticFiles(directory=str(settings.STORAGE_DIR)),
    name="storage",
)


@app.on_event("startup")
def on_startup():
    """Initializes database schema and ensures pre-trained ML models are ready."""
    print(f"[RoadSense AI] Starting server in {settings.ENVIRONMENT} mode...")
    init_database(seed_initial_data=True)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Safe global error handler to prevent internal trace exposure."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "An internal server error occurred.",
            "detail": str(exc) if settings.DEBUG else None,
        },
    )


# Register API Routers
app.include_router(health.router, prefix="/api", tags=["System Health"])
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(sensors.router, prefix="/api/sensors", tags=["Sensor Waveforms & Ingestion"])
app.include_router(vision.router, prefix="/api/vision", tags=["Computer Vision & YOLO"])
app.include_router(dashcam.router, prefix="/api/dashcam", tags=["Driver Dashcam & Live Stream"])
app.include_router(incidents.router, prefix="/api/incidents", tags=["Road Incidents"])
app.include_router(maps.router, prefix="/api/maps", tags=["Spatial Maps & Road Memory"])
app.include_router(authority.router, prefix="/api/authority", tags=["Municipal Authority"])
app.include_router(analytics.router, prefix="/api/dashboard", tags=["Executive Analytics"])


@app.get("/")
def root():
    return {
        "app": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/api/health",
        "status": "operational",
    }
