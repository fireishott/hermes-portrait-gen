"""Face identity — detect, embed, identify, auto-file reference photos."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .config import (
    REF_PHOTOS_DIR,
    EMBEDDINGS_DB,
    INSIGHTFACE_DIR,
    FACE_MATCH_THRESHOLD,
)
from .models import FaceIdentity, FaceMatch, EmbeddingsDB

# Lazy-loaded InsightFace app
_app = None


def _get_app():
    """Lazy-load InsightFace. Requires insightface + onnxruntime."""
    global _app
    if _app is None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface not installed. Run: "
                "~/.hermes/hermes-agent/venv/bin/pip3 install insightface onnxruntime"
            )
        models_dir = str(INSIGHTFACE_DIR / "models")
        provider = "CPUExecutionProvider"
        try:
            import onnxruntime
            avail = onnxruntime.get_available_providers()
            if "CoreMLExecutionProvider" in avail:
                provider = "CoreMLExecutionProvider"
            elif "CUDAExecutionProvider" in avail:
                provider = "CUDAExecutionProvider"
        except ImportError:
            pass
        _app = FaceAnalysis(
            name="antelopev2",
            root=models_dir,
            providers=[provider],
        )
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def analyze_faces(image_path: str) -> list[dict]:
    """Detect all faces in an image and return their embeddings + bboxes.

    Returns list of dicts: {embedding, bbox, det_score, age, gender}
    """
    app = _get_app()
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    faces = app.get(img)
    results = []
    for face in faces:
        results.append({
            "embedding": face.embedding.tolist(),
            "bbox": face.bbox.tolist(),
            "det_score": float(face.det_score),
            "age": int(face.age) if hasattr(face, "age") else None,
            "gender": face.gender if hasattr(face, "gender") else None,
        })
    return results


def identify_face(image_path: str, db: Optional[EmbeddingsDB] = None) -> list[FaceMatch]:
    """Identify faces in an image against known identities.

    Returns list of FaceMatch (one per detected face).
    """
    if db is None:
        db = EmbeddingsDB(EMBEDDINGS_DB)
    faces = analyze_faces(image_path)
    matches = []
    known = db.all()
    for face_data in faces:
        emb = np.array(face_data["embedding"])
        best_match = None
        best_score = 0.0
        for name, identity in known.items():
            known_emb = np.array(identity.embedding)
            score = _cosine_similarity(emb, known_emb)
            if score > best_score:
                best_score = score
                best_match = name
        if best_match and best_score >= FACE_MATCH_THRESHOLD:
            matches.append(FaceMatch(
                person=best_match,
                confidence=best_score,
                is_new=False,
                bbox=face_data["bbox"],
            ))
        else:
            matches.append(FaceMatch(
                person=None,
                confidence=best_score,
                is_new=True,
                bbox=face_data["bbox"],
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
    if db is None:
        db = EmbeddingsDB(EMBEDDINGS_DB)
    faces = analyze_faces(image_path)
    if not faces:
        raise ValueError(f"No faces detected in {image_path}")
    # Use highest-confidence face
    best = max(faces, key=lambda f: f["det_score"])
    identity = FaceIdentity(
        name=name.lower().strip(),
        embedding=best["embedding"],
        ref_count=1,
    )
    db.put(identity)
    # Auto-file the ref photo
    filed_path = file_ref_photo(name, image_path)
    identity.best_ref = str(filed_path)
    db.put(identity)
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
