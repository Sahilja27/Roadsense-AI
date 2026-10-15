@echo off
title RoadSense AI - Full-Stack Localhost Server
cls
echo ==================================================================
echo           RoadSense AI - Full-Stack Localhost Launcher
echo ==================================================================
echo Starting FastAPI Backend (Port 8000) and Dashboard (Port 8501)...
echo.

cd /d "%~dp0"
python run_all.py

pause
