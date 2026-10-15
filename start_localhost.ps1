# RoadSense AI - PowerShell Localhost Launcher
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "       RoadSense AI - Full-Stack Localhost Server Launcher        " -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "Starting FastAPI Backend (:8000) and Streamlit Dashboard (:8501)..." -ForegroundColor Yellow

Set-Location -Path $PSScriptRoot
python run_all.py
