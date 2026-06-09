"""Plugin configuration — host creds, paths, thresholds."""

from __future__ import annotations

import os
from pathlib import Path

# ── MBP connection ──────────────────────────────────────────────────────────
MBP_HOST = os.environ.get("PORTRAIT_MBP_HOST", "192.168.10.121")
MBP_USER = os.environ.get("PORTRAIT_MBP_USER", "curtisfreeman")
MBP_PASS = os.environ.get("PORTRAIT_MBP_PASS", "11aaxx2wR")
SSH_TIMEOUT = int(os.environ.get("PORTRAIT_SSH_TIMEOUT", "10"))

# ── ComfyUI on MBP ─────────────────────────────────────────────────────────
COMFYUI_PORT = int(os.environ.get("PORTRAIT_COMFYUI_PORT", "8188"))
COMFYUI_DIR = os.environ.get("PORTRAIT_COMFYUI_DIR", "/Users/curtisfreeman/ComfyUI")
COMFYUI_STARTUP_TIMEOUT = int(os.environ.get("PORTRAIT_COMFYUI_STARTUP_TIMEOUT", "60"))
COMFYUI_POLL_INTERVAL = int(os.environ.get("PORTRAIT_COMFYUI_POLL_INTERVAL", "3"))
COMFYUI_GENERATION_TIMEOUT = int(os.environ.get("PORTRAIT_COMFYUI_GENERATION_TIMEOUT", "300"))

# ── Reference photo library on iMac ────────────────────────────────────────
REF_PHOTOS_DIR = Path(os.environ.get(
    "PORTRAIT_REF_DIR",
    os.path.expanduser("~/Pictures/ref-photos")
))

# ── Face identification ────────────────────────────────────────────────────
FACE_MATCH_THRESHOLD = float(os.environ.get("PORTRAIT_FACE_MATCH_THRESHOLD", "0.35"))
EMBEDDINGS_DB = REF_PHOTOS_DIR / ".embeddings.json"

# ── InsightFace model directory (on iMac) ──────────────────────────────────
INSIGHTFACE_DIR = Path(os.environ.get(
    "PORTRAIT_INSIGHTFACE_DIR",
    os.path.expanduser("~/.hermes/plugins/portrait-gen/models")
))

# ── ComfyUI temp staging on MBP ────────────────────────────────────────────
MBP_STAGING_DIR = "/tmp/portrait-gen"
