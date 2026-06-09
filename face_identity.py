"""Face identity — detect, embed, identify, auto-file reference photos.

Delegates InsightFace analysis to a subprocess (face_analyzer.py) so
onnxruntime/insightface never pollute the agent's Python process.
All heavy lifting happens in a clean venv Python every time.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from .config import (
    REF_PHOTOS_DIR,
    EMBEDDINGS_DB,
    INSIGHTFACE_DIR,
    FACE_MATCH_THRESHOLD,
)
from .models import FaceIdentity, FaceMatch, EmbeddingsDB

# Path to the standalone face analyzer script
_ANALYZER_SCRIPT = Path(__file__).parent / "scripts" / "face_analyzer.py"
_VENV_PYTHON = Path.home() / ".hermes/hermes-agent/venv/bin/python3"


def _run_analyzer(*args: str, timeout: int = 120) -> dict:
    """Run face_analyzer.py as a subprocess and return parsed JSON output."""
    cmd = [str(_VENV_PYTHON), str(_ANALYZER_SCRIPT)] + list(args)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        # Try to parse error from stdout (script writes JSON errors there)
        try:
            err = json.loads(result.stdout)
            raise RuntimeError(err.get("error", result.stderr.strip()))
        except (json.JSONDecodeError, ValueError):
            raise RuntimeError(
                f"face_analyzer failed (exit {result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"face_analyzer returned non-JSON: {result.stdout[:500]}")


def analyze_faces(image_path: str) -> list[dict]:
    """Detect all faces in an image and return their embeddings + bboxes.

    Returns list of dicts: {embedding, bbox, det_score, age, gender}
    """
    data = _run_analyzer("analyze", image_path)
    return data.get("faces", [])


def identify_face(image_path: str, db: Optional[EmbeddingsDB] = None) -> list[FaceMatch]:
    """Identify faces in an image against known identities.

    Returns list of FaceMatch (one per detected face).
    Uses subprocess for face detection + matching against the embeddings DB.
    """
    # Use subprocess for the full identify flow (detection + matching)
    data = _run_analyzer(
        "identify", image_path,
        "--db", str(EMBEDDINGS_DB),
        "--threshold", str(FACE_MATCH_THRESHOLD),
    )

    matches = []
    for m in data.get("matches", []):
        matches.append(FaceMatch(
            person=m.get("person"),
            confidence=m.get("confidence", 0.0),
            is_new=m.get("is_new", True),
            bbox=m.get("bbox", []),
        ))
    return matches


def register_identity(
    name: str,
    image_path: str,
    db: Optional[EmbeddingsDB] = None,
) -> FaceIdentity:
    """Register a new identity from a photo. Uses first detected face embedding.

    Auto-files the photo into the ref library.
    """
    # Register via subprocess (handles detection + DB write)
    data = _run_analyzer(
        "register", name, image_path,
        "--db", str(EMBEDDINGS_DB),
    )

    if "error" in data:
        raise ValueError(data["error"])

    # Auto-file the ref photo
    filed_path = file_ref_photo(name, image_path)

    identity = FaceIdentity(
        name=name.lower().strip(),
        embedding=[],  # stored in DB by subprocess
        ref_count=1,
    )
    identity.best_ref = str(filed_path)
    return identity


def file_ref_photo(
    person: str,
    image_path: str,
    quality: float = 0.0,
) -> Path:
    """Copy a photo into the ref library under person's folder.

    Returns the destination path.
    """
    person_dir = REF_PHOTOS_DIR / person.lower().strip()
    person_dir.mkdir(parents=True, exist_ok=True)
    src = Path(image_path)
    # Generate unique filename
    ts = int(time.time() * 1000)
    dest = person_dir / f"ref_{ts}{src.suffix}"
    shutil.copy2(str(src), str(dest))
    return dest


def get_ref_photos(person: str) -> list[Path]:
    """List all ref photos for a person, sorted by modification time (newest first)."""
    person_dir = REF_PHOTOS_DIR / person.lower().strip()
    if not person_dir.exists():
        return []
    photos = [
        p for p in person_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        and not p.name.startswith(".")
    ]
    photos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return photos


def get_ref_stats() -> dict:
    """Get stats about the ref photo library."""
    db = EmbeddingsDB(EMBEDDINGS_DB)
    stats = {}
    for name, identity in db.all().items():
        ref_dir = REF_PHOTOS_DIR / name
        ref_count = len(list(ref_dir.glob("*.[jJpPpPwW]*"))) if ref_dir.exists() else 0
        stats[name] = {
            "ref_count": ref_count,
            "best_ref": identity.best_ref,
            "known_since": identity.created_at,
        }
    return stats
