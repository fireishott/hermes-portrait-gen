---
name: portrait-gen
description: "Use when generating portraits with face preservation, identifying faces in photos, or managing the reference photo library. Orchestrates ComfyUI + PuLID + Flux on MBP for local generation."
version: 0.0.1
author: ignyte
license: private
metadata:
  hermes:
    tags: [portraits, face-preservation, comfyui, pulid, image-generation, homelab]
    related_skills: [homeassistant, homelab, portrait-generation]
---

# Portrait Generation (Local)

Local face-preserving portrait generation via ComfyUI + PuLID + Flux on the MBP. Zero cloud APIs, zero base64 — all file-based over SSH.

## When to Use

- Generating portraits of real people with face preservation
- Identifying or registering faces in photos
- Managing the reference photo library
- Checking MBP/ComfyUI availability

Don't use for: general image generation (use `image_generate`), abstract art, or images that don't need face preservation.

## Quick Start

### 1. Identify and register a face

```python
# Send a photo — plugin detects and identifies
portrait_identify_face(image_path="~/Pictures/curtis_headshot.png")

# If unknown, register the identity
portrait_register_identity(name="curtis", image_path="~/Pictures/curtis_headshot.png")
```

### 2. Generate a portrait

```python
portrait_generate(
    person="curtis",
    prompt="professional headshot, studio lighting, dark background"
)
```

### 3. Check status

```python
portrait_ref_status()
```

## Pipeline Flow

```
1. Photo arrives → face detection + identification
2. Known face → auto-file to ~/Pictures/ref-photos/{person}/
3. Unknown face → ask for name → register → file
4. Portrait request → pull refs → SSH to MBP → ComfyUI + PuLID + Flux
5. Output → pull back to iMac → deliver via MEDIA:
6. ComfyUI shut down → MBP freed
```

## Mnemosyne Integration

The plugin stores identity facts in Mnemosyne:
- `(person, has_ref, photo_path)` — each filed reference
- `(person, best_ref, path)` — highest quality reference
- `(portrait, depicts, person)` — generated portraits

Use `mnemosyne_recall` to look up identity info across sessions.

## Configuration

Defaults work out of the box for the FIH homelab. Override via env vars if needed:

| Env Var | Default | Description |
|---------|---------|-------------|
| `PORTRAIT_MBP_HOST` | `192.168.10.121` | MBP IP |
| `PORTRAIT_MBP_USER` | `curtisfreeman` | SSH user |
| `PORTRAIT_MBP_PASS` | (from config) | SSH password |
| `PORTRAIT_COMFYUI_PORT` | `8188` | ComfyUI port |
| `PORTRAIT_FACE_MATCH_THRESHOLD` | `0.35` | Face match threshold |

## Common Pitfalls

1. **MBP not reachable** — Ensure MBP is on and SSH is enabled. Check with `portrait_ref_status()`.
2. **InsightFace not installed** — Run `~/.hermes/hermes-agent/venv/bin/pip3 install insightface onnxruntime opencv-python-headless`.
3. **antelopev2 models missing** — InsightFace auto-downloads on first use. If blocked, manually place in `~/.hermes/plugins/portrait-gen/models/models/antelopev2/`.
4. **ComfyUI startup timeout** — Default 60s. Increase with `PORTRAIT_COMFYUI_STARTUP_TIMEOUT` if MBP is slow.
5. **No ref photos for person** — Must register at least one face photo before generating.
6. **sshpass not installed** — Required for SSH. `brew install sshpass`.
7. **Low face match accuracy** — Add more reference photos. More refs = better identity preservation.
8. **PuLID weight too low/high** — Default 0.8. Increase for stronger face lock (may reduce prompt adherence). Range: 0.0-5.0.

## Verification Checklist

- [ ] MBP reachable via SSH (`portrait_ref_status`)
- [ ] InsightFace deps installed in Hermes venv
- [ ] ComfyUI + PuLID models on MBP
- [ ] At least one face registered
- [ ] Test generation completes successfully

## Future Roadmap

- **v0.1.0** — Quality scoring, automatic ref ranking
- **v0.2.0** — LoRA training pipeline for stronger identity lock
- **v0.3.0** — Multiple workflow support (styles, sizes)
- **v0.4.0** — ComfyUI workflow customization via prompt
