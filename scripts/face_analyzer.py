#!/usr/bin/env python3
"""face_analyzer.py — Standalone face analysis helper.

Can be used independently of the plugin for testing/debugging.

Usage:
    python face_analyzer.py <image_path>
    python face_analyzer.py <image_path> --compare <other_image>
    python face_analyzer.py <image_path> --register <name>

Requires: insightface, onnxruntime, opencv-python, numpy
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis
except ImportError:
    print(
        "Missing dependencies. Install with:\n"
        "  pip install insightface onnxruntime opencv-python numpy",
        file=sys.stderr,
    )
    sys.exit(1)


def get_analyzer(model_dir: str = "~/.hermes/plugins/portrait-gen/models"):
    """Initialize InsightFace with antelopev2."""
    root = str(Path(model_dir).expanduser())
    app = FaceAnalysis(
        name="antelopev2",
        root=root,
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def analyze(image_path: str, app=None) -> list[dict]:
    """Detect and analyze faces in an image."""
    if app is None:
        app = get_analyzer()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    faces = app.get(img)
    results = []
    for i, face in enumerate(faces):
        results.append({
            "face_index": i,
            "bbox": face.bbox.tolist(),
            "det_score": float(face.det_score),
            "age": int(face.age) if hasattr(face, "age") else None,
            "gender": face.gender if hasattr(face, "gender") else None,
            "embedding_norm": float(np.linalg.norm(face.embedding)),
            "embedding_dim": len(face.embedding),
        })
    return results


def compare(image_a: str, image_b: str, app=None) -> dict:
    """Compare faces in two images."""
    if app is None:
        app = get_analyzer()
    import cv2

    img_a = cv2.imread(image_a)
    img_b = cv2.imread(image_b)
    if img_a is None:
        raise FileNotFoundError(f"Cannot read: {image_a}")
    if img_b is None:
        raise FileNotFoundError(f"Cannot read: {image_b}")

    faces_a = app.get(img_a)
    faces_b = app.get(img_b)

    if not faces_a:
        return {"error": f"No faces in {image_a}"}
    if not faces_b:
        return {"error": f"No faces in {image_b}"}

    emb_a = faces_a[0].embedding
    emb_b = faces_b[0].embedding
    sim = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8))

    return {
        "image_a": image_a,
        "image_b": image_b,
        "faces_a": len(faces_a),
        "faces_b": len(faces_b),
        "cosine_similarity": round(sim, 4),
        "match": sim >= 0.35,
        "threshold": 0.35,
    }


def main():
    parser = argparse.ArgumentParser(description="Face analysis helper")
    parser.add_argument("image", help="Image path to analyze")
    parser.add_argument("--compare", help="Compare with another image")
    parser.add_argument("--register", help="Register as identity (name)")
    parser.add_argument("--model-dir", default="~/.hermes/plugins/portrait-gen/models")
    args = parser.parse_args()

    app = get_analyzer(args.model_dir)

    if args.compare:
        result = compare(args.image, args.compare, app)
    else:
        result = analyze(args.image, app)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
