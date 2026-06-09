"""portrait-gen — Local portrait generation with face preservation.

Hermes plugin that identifies faces, auto-files reference photos,
and orchestrates ComfyUI + PuLID + Flux on a remote MBP for generation.

Zero base64. Everything is file-based over SSH.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("portrait-gen")


def _tool_schema(
    name: str,
    description: str,
    properties: dict,
    required: list[str] | None = None,
) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    }


# ── Tool handlers ──────────────────────────────────────────────────────────


def _handle_identify_face(args: dict, **kwargs) -> str:
    """Identify faces in an image and auto-file as reference photos."""
    from .face_identity import identify_face, register_identity, file_ref_photo, get_ref_stats
    from .models import EmbeddingsDB
    from .config import EMBEDDINGS_DB

    image_path = args.get("image_path", "")
    if not image_path:
        return json.dumps({"error": "image_path is required"})

    p = Path(image_path).expanduser()
    if not p.exists():
        return json.dumps({"error": f"File not found: {image_path}"})

    db = EmbeddingsDB(EMBEDDINGS_DB)
    matches = identify_face(str(p), db)

    results = []
    for match in matches:
        if match.is_new:
            results.append({
                "status": "unknown_face",
                "message": "New face detected. Provide a name to register this identity.",
                "confidence": round(match.confidence, 3),
                "bbox": match.bbox,
            })
        else:
            # Auto-file as ref photo for known person
            filed = file_ref_photo(match.person, str(p), quality=match.confidence)
            results.append({
                "status": "identified",
                "person": match.person,
                "confidence": round(match.confidence, 3),
                "filed_to": str(filed),
            })

    stats = get_ref_stats()
    return json.dumps({
        "faces": results,
        "ref_library": stats,
    }, indent=2)


def _handle_register_identity(args: dict, **kwargs) -> str:
    """Register a new identity from a photo (called after identify_face returns unknown)."""
    from .face_identity import register_identity

    name = args.get("name", "").strip().lower()
    image_path = args.get("image_path", "")

    if not name or not image_path:
        return json.dumps({"error": "name and image_path are required"})

    p = Path(image_path).expanduser()
    if not p.exists():
        return json.dumps({"error": f"File not found: {image_path}"})

    try:
        identity = register_identity(name, str(p))
        return json.dumps({
            "status": "registered",
            "person": identity.name,
            "ref_count": identity.ref_count,
            "best_ref": identity.best_ref,
            "message": f"Identity '{name}' registered. Future photos with this face will be auto-filed.",
        }, indent=2)
    except ValueError as e:
        return json.dumps({"error": str(e)})


def _handle_generate(args: dict, **kwargs) -> str:
    """Generate a portrait using PuLID + Flux on the MBP."""
    from .orchestrator import generate_portrait
    from .models import PortraitRequest
    from .face_identity import get_ref_photos
    from .config import EMBEDDINGS_DB
    from .models import EmbeddingsDB

    person = args.get("person", "").strip().lower()
    prompt = args.get("prompt", "")

    if not person or not prompt:
        return json.dumps({"error": "person and prompt are required"})

    # Get ref photos for this person
    ref_photos = get_ref_photos(person)
    if not ref_photos:
        return json.dumps({
            "error": f"No reference photos found for '{person}'. "
                     "Use portrait_identify_face to register them first.",
        })

    request = PortraitRequest(
        person=person,
        prompt=prompt,
        negative_prompt=args.get("negative_prompt", ""),
        width=args.get("width", 1024),
        height=args.get("height", 1024),
        steps=args.get("steps", 20),
        cfg=args.get("cfg", 3.5),
        seed=args.get("seed", -1),
        pulid_weight=args.get("pulid_weight", 0.8),
    )

    result = generate_portrait(request, ref_photos)

    if result.success:
        return json.dumps({
            "status": "success",
            "output_path": result.output_path,
            "person": result.person,
            "prompt": result.prompt,
            "seed": result.seed,
            "generation_time_s": result.generation_time_s,
            "ref_count": len(ref_photos),
            "message": f"Portrait generated in {result.generation_time_s}s. "
                       f"Used {len(ref_photos)} reference photo(s).",
        }, indent=2)
    else:
        return json.dumps({
            "status": "error",
            "error": result.error,
        })


def _handle_ref_status(args: dict, **kwargs) -> str:
    """Show ref library stats."""
    from .face_identity import get_ref_stats
    from .orchestrator import check_mbp_reachable, check_comfyui_running

    stats = get_ref_stats()
    mbp_up = check_mbp_reachable()
    comfy_up = check_comfyui_running() if mbp_up else False

    return json.dumps({
        "ref_library": stats,
        "mbp_reachable": mbp_up,
        "comfyui_running": comfy_up,
        "total_identities": len(stats),
    }, indent=2)


# ── Plugin registration ────────────────────────────────────────────────────


def register(ctx) -> None:
    """Called by Hermes plugin loader."""

    ctx.register_tool(
        name="portrait_identify_face",
        toolset="portrait",
        schema=_tool_schema(
            "portrait_identify_face",
            "Detect and identify faces in a photo. If the face is known, auto-files it as a "
            "reference photo. If unknown, returns 'unknown_face' — call again with a name to "
            "register the identity. Supports JPG, PNG, WebP.",
            properties={
                "image_path": {
                    "type": "string",
                    "description": "Path to the image file (local or URL)",
                },
            },
            required=["image_path"],
        ),
        handler=_handle_identify_face,
    )

    ctx.register_tool(
        name="portrait_register_identity",
        toolset="portrait",
        schema=_tool_schema(
            "portrait_register_identity",
            "Register a new face identity from a photo. Use after portrait_identify_face "
            "returns an unknown face. The photo is filed into the ref library automatically.",
            properties={
                "name": {
                    "type": "string",
                    "description": "Person's name (lowercase, e.g. 'curtis', 'kristie')",
                },
                "image_path": {
                    "type": "string",
                    "description": "Path to the image file containing this person's face",
                },
            },
            required=["name", "image_path"],
        ),
        handler=_handle_register_identity,
    )

    ctx.register_tool(
        name="portrait_generate",
        toolset="portrait",
        schema=_tool_schema(
            "portrait_generate",
            "Generate a portrait of a known person using PuLID + Flux on the MBP. "
            "Requires at least one reference photo (register via portrait_identify_face first). "
            "Spins up ComfyUI, generates, and returns the output path.",
            properties={
                "person": {
                    "type": "string",
                    "description": "Person's name (must be registered)",
                },
                "prompt": {
                    "type": "string",
                    "description": "Scene/setting prompt (e.g. 'professional headshot, studio lighting')",
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "What to avoid (default: empty)",
                },
                "width": {
                    "type": "integer",
                    "description": "Image width (default: 1024)",
                },
                "height": {
                    "type": "integer",
                    "description": "Image height (default: 1024)",
                },
                "steps": {
                    "type": "integer",
                    "description": "Sampling steps (default: 20)",
                },
                "cfg": {
                    "type": "number",
                    "description": "CFG guidance scale (default: 3.5)",
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed (-1 for random)",
                },
                "pulid_weight": {
                    "type": "number",
                    "description": "PuLID face preservation weight (0.0-5.0, default: 0.8)",
                },
            },
            required=["person", "prompt"],
        ),
        handler=_handle_generate,
    )

    ctx.register_tool(
        name="portrait_ref_status",
        toolset="portrait",
        schema=_tool_schema(
            "portrait_ref_status",
            "Show reference photo library stats: registered identities, ref counts, "
            "and MBP/ComfyUI availability.",
            properties={},
        ),
        handler=_handle_ref_status,
    )

    log.info("portrait-gen plugin registered (v0.0.1)")
