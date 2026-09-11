"""Aster Desktop Application Launcher.
Starts Aster RAG server, verifies port binding, opens default browser, and handles graceful shutdown.
"""
from __future__ import annotations

import os
import sys
import time
import threading
import webbrowser
from pathlib import Path

# Setup multiprocessing support for PyInstaller on Windows
import multiprocessing
multiprocessing.freeze_support()

import uvicorn
import httpx
from app import app
from config import APP_DIR, settings

PORT = int(os.getenv("PORT", "8000"))
HOST = "127.0.0.1"
URL = f"http://{HOST}:{PORT}"


def open_browser_when_ready():
    """Poll the health check endpoint and automatically open the user's browser."""
    print(f"[*] Initializing Aster Grounded AI Agent on {URL} ...")
    max_wait = 25
    start = time.time()
    while time.time() - start < max_wait:
        try:
            res = httpx.get(f"{URL}/", timeout=1.0)
            if res.status_code == 200:
                print(f"[✓] Aster is live! Opening {URL} in your browser...")
                webbrowser.open(URL)
                return
        except Exception:
            pass
        time.sleep(0.5)
    webbrowser.open(URL)


def main():
    print("=" * 65)
    print("       ASTER — Grounded Document Intelligence Agent")
    print(f"       Portable Desktop Edition | Running from {APP_DIR}")
    print("=" * 65)
    print(f"[*] Data & conversations stored in: {settings.data_dir}")
    print(f"[*] Press Ctrl+C in this window anytime to exit.")
    print("-" * 65)

    threading.Thread(target=open_browser_when_ready, daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
