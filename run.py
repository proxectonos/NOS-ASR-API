#!/usr/bin/env python3
"""Cross-platform dev launcher for the ASR API.

Linux/macOS -> gunicorn (with --reload). Windows -> waitress (gunicorn has no
Windows support). Configure via env vars; sensible defaults below.

    python run.py

Env overrides: PORT, USE_CUDA, MODELS_ROOT, ASR_API_CONFIG,
               WEB_THREADS, WEB_WORKERS, WEB_TIMEOUT
"""
import os
import sys
import subprocess

os.environ.setdefault("USE_CUDA", "0")
os.environ.setdefault("MODELS_ROOT", "models")
os.environ.setdefault("ASR_API_CONFIG", "config.json")

PORT = os.environ.get("PORT", "5052")
THREADS = os.environ.get("WEB_THREADS", "4")
WORKERS = os.environ.get("WEB_WORKERS", "1")
TIMEOUT = os.environ.get("WEB_TIMEOUT", "300")

if os.name == "nt":
    # gunicorn does not run on Windows; use waitress.
    cmd = [
        "waitress-serve",
        f"--listen=*:{PORT}",
        f"--threads={THREADS}",
        f"--channel-timeout={TIMEOUT}",
        "server:app",
    ]
else:
    cmd = [
        "gunicorn", "server:app",
        "-b", f":{PORT}",
        "--workers", WORKERS,
        "--threads", THREADS,
        "--timeout", TIMEOUT,
        "--reload",
    ]

print("launch:", " ".join(cmd))
sys.exit(subprocess.call(cmd))
