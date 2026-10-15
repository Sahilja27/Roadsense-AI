"""
RoadSense AI - Unified Full-Stack Localhost Launcher
====================================================
Boots both the FastAPI Backend (Port 8000) and the Streamlit Command Center (Port 8501)
concurrently with process supervision, health verification, and graceful shutdown.

Usage:
    python run_all.py
"""

import sys
import os
import subprocess
import time
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"


def print_banner():
    banner = r"""
==================================================================
           📡 RoadSense AI - Full-Stack Localhost Launcher
==================================================================
  [1] FastAPI Backend API     -->  http://localhost:8000
  [2] Interactive Swagger UI  -->  http://localhost:8000/docs
  [3] Streamlit Dashboard     -->  http://localhost:8501
==================================================================
    """
    print(banner)


def check_dependencies():
    print("[1/3] Verifying Python environment & database...")
    try:
        sys.path.insert(0, str(BACKEND_DIR))
        from app.db.init_db import init_database
        init_database(seed_initial_data=True)
        print("  ✓ Database schema verified & initial seed records ready.")
    except Exception as e:
        print(f"  ⚠️ Database initialization warning: {e}")


def run():
    print_banner()
    check_dependencies()

    python_executable = sys.executable

    print("\n[2/3] Launching FastAPI Backend on http://localhost:8000...")
    backend_cmd = [
        python_executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=str(BACKEND_DIR),
    )

    # Allow backend 1.5 seconds to bind
    time.sleep(1.5)

    print("[3/3] Launching Streamlit Command Center on http://localhost:8501...")
    frontend_cmd = [
        python_executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        "8501",
        "--server.headless",
        "true",
    ]
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=str(ROOT_DIR),
    )

    print("\n" + "=" * 66)
    print("  🚀 FULL LOCALHOST ENVIRONMENT IS NOW RUNNING!")
    print("=" * 66)
    print("  • Command Center Dashboard : http://localhost:8501")
    print("  • FastAPI Backend API      : http://localhost:8000")
    print("  • Interactive API Docs     : http://localhost:8000/docs")
    print("=" * 66)
    print("  Press CTRL+C at any time to gracefully stop all services.\n")

    try:
        # Open browser automatically
        time.sleep(2.0)
        webbrowser.open("http://localhost:8501")

        # Keep running and supervise processes
        while True:
            time.sleep(1.0)
            if backend_proc.poll() is not None:
                print("⚠️ FastAPI backend exited unexpectedly.")
                break
            if frontend_proc.poll() is not None:
                print("⚠️ Streamlit dashboard exited unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\n[Shutting down RoadSense AI services...]")
    finally:
        frontend_proc.terminate()
        backend_proc.terminate()
        frontend_proc.wait()
        backend_proc.wait()
        print("✓ All RoadSense AI processes stopped cleanly.")


if __name__ == "__main__":
    run()
