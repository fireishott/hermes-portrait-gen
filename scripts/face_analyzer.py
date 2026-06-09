#!/usr/bin/env python3
"""face_analyzer.py — Standalone face analysis helper.

Runs in a subprocess so onnxruntime/insightface never pollute the agent's
process space.  Returns JSON to stdout so the plugin can consume it cleanly.

Usage:
    python face_analyzer.py analyze <image_path>
    python face_analyzer.py compare <image_a> <image_b>
    python face_analyzer.py identify <image_path> --db <embeddings.json> [--threshold 0.35]
    python face_analyzer.py register <name> <image_path> --db <embeddings.json>

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
        json.dumps({"error": "Missing dependencies. Install insightface onnxruntime opencv-python numpy"}),
        file=sys.stderr,
    )
    sys.exit(1)


def get_analyzer(model_dir: str = "~/.hermes/plugins/portrait-gen"):
    """Initialize InsightFace with antelopev2."""
    root = str(Path(model_dir).expanduser())
    # Check available providers
    provider = "CPUExecutionProvider"
    try:
        import onnxruntime
        avail = onnxruntime.get_available_providers()
        if "CoreMLExecutionProvider" in avail:
            provider = "CoreMLExecutionProvider"
        elif "CUDAExecutionProvider" in avail:
            provider = "CUDAExecutionProvider"
    except Exception:
        pass
    app = FaceAnalysis(
        name="antelopev2",
        root=root,
        providers=[provider],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def cmd_analyze(image_path: str, model_dir: str):
    """Detect faces and return embeddings."""
    app = get_analyzer(model_dir)
    img = cv2.imread(image_path)
    if img is None:
        print(json.dumps({"error": f"Cannot read image: {image_path}"}))
        sys.exit(1)
    faces = app.get(img)
    results = []
    for face in faces:
        results.append({
            "embedding": face.embedding.tolist(),
            "bbox": face.bbox.tolist(),
            "det_score": float(face.det_score),
            "age": int(face.age) if hasattr(face, "age") and face.age is not None else None,
            "gender": str(face.gender) if hasattr(face, "gender") and face.gender is not None else None,
        })
    print(json.dumps({"faces": results}))


def cmd_compare(image_a: str, image_b: str, model_dir: str):
    """Compare faces in two images."""
    app = get_analyzer(model_dir)
    img_a = cv2.imread(image_a)
    img_b = cv2.imread(image_b)
    if img_a is None:
        print(json.dumps({"error": f"Cannot read: {image_a}"}))
        sys.exit(1)
    if img_b is None:
        print(json.dumps({"error": f"Cannot read: {image_b}"}))
        sys.exit(1)
    faces_a = app.get(img_a)
    faces_b = app.get(img_b)
    if not faces_a:
        print(json.dumps({"error": f"No faces in {image_a}"}))
        sys.exit(1)
    if not faces_b:
        print(json.dumps({"error": f"No faces in {image_b}"}))
        sys.exit(1)
    emb_a = faces_a[0].embedding
    emb_b = faces_b[0].embedding
    sim = cosine_similarity(emb_a, emb_b)
    print(json.dumps({
        "cosine_similarity": round(sim, 4),
        "match": sim >= 0.35,
        "faces_a": len(faces_a),
        "faces_b": len(faces_b),
    }))


def cmd_identify(image_path: str, db_path: str, threshold: float, model_dir: str):
    """Identify faces against known identities in the embeddings DB."""
    app = get_analyzer(model_dir)
    img = cv2.imread(image_path)
    if img is None:
        print(json.dumps({"error": f"Cannot read image: {image_path}"}))
        sys.exit(1)
    faces = app.get(img)
    if not faces:
        print(json.dumps({"faces": [], "matches": []}))
        return

    # Load embeddings DB
    db_file = Path(db_path)
    if db_file.exists():
        db = json.loads(db_file.read_text())
    else:
        db = {}

    matches = []
    for face in faces:
        emb = face.embedding
        best_match = None
        best_score = 0.0
        for name, identity in db.items():
            known_emb = np.array(identity["embedding"])
            score = cosine_similarity(emb, known_emb)
            if score > best_score:
                best_score = score
                best_match = name
        match_info = {
            "bbox": face.bbox.tolist(),
            "det_score": float(face.det_score),
            "embedding": emb.tolist(),
        }
        if best_match and best_score >= threshold:
            match_info["person"] = best_match
            match_info["confidence"] = round(best_score, 4)
            match_info["is_new"] = False
        else:
            match_info["person"] = None
            match_info["confidence"] = round(best_score, 4)
            match_info["is_new"] = True
        matches.append(match_info)

    print(json.dumps({"matches": matches}))


def cmd_register(name: str, image_path: str, db_path: str, model_dir: str):
    """Register a new identity and file the reference photo."""
    app = get_analyzer(model_dir)
    img = cv2.imread(image_path)
    if img is None:
        print(json.dumps({"error": f"Cannot read image: {image_path}"}))
        sys.exit(1)
    faces = app.get(img)
    if not faces:
        print(json.dumps({"error": "No faces detected"}))
        sys.exit(1)

    # Use highest-confidence face
    best = max(faces, key=lambda f: f.det_score)

    # Load DB
    db_file = Path(db_path)
    if db_file.exists():
        db = json.loads(db_file.read_text())
    else:
        db = {}

    db[name.lower().strip()] = {
        "embedding": best.embedding.tolist(),
        "ref_count": 1,
        "created_at": str(int(__import__("time").time())),
    }
    db_file.write_text(json.dumps(db, indent=2))

    print(json.dumps({
        "person": name.lower().strip(),
        "confidence": float(best.det_score),
        "registered": True,
    }))


def main():
    parser = argparse.ArgumentParser(description="Face analysis helper")
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze faces in an image")
    p_analyze.add_argument("image", help="Image path")
    p_analyze.add_argument("--model-dir", default="~/.hermes/plugins/portrait-gen")

    # compare
    p_compare = sub.add_parser("compare", help="Compare faces in two images")
    p_compare.add_argument("image_a", help="First image")
    p_compare.add_argument("image_b", help="Second image")
    p_compare.add_argument("--model-dir", default="~/.hermes/plugins/portrait-gen")

    # identify
    p_identify = sub.add_parser("identify", help="Identify faces against known DB")
    p_identify.add_argument("image", help="Image path")
    p_identify.add_argument("--db", required=True, help="Embeddings JSON path")
    p_identify.add_argument("--threshold", type=float, default=0.35)
    p_identify.add_argument("--model-dir", default="~/.hermes/plugins/portrait-gen")

    # register
    p_register = sub.add_parser("register", help="Register a new identity")
    p_register.add_argument("name", help="Person's name")
    p_register.add_argument("image", help="Image path")
    p_register.add_argument("--db", required=True, help="Embeddings JSON path")
    p_register.add_argument("--model-dir", default="~/.hermes/plugins/portrait-gen")

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze(args.image, args.model_dir)
    elif args.command == "compare":
        cmd_compare(args.image_a, args.image_b, args.model_dir)
    elif args.command == "identify":
        cmd_identify(args.image, args.db, args.threshold, args.model_dir)
    elif args.command == "register":
        cmd_register(args.name, args.image, args.db, args.model_dir)


if __name__ == "__main__":
    main()
