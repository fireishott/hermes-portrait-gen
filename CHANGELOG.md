# Changelog

## [0.0.3] — 2026-06-09

### Fixed — Critical: onnxruntime module cache corruption
- **Subprocess delegation for all face analysis**: Replaced in-process InsightFace/onnxruntime imports with a standalone `scripts/face_analyzer.py` subprocess. The gateway process had poisoned `sys.modules` from a deleted Python 3.12 `hermes_env` venv — any in-process `import onnxruntime` would fail with `ModuleNotFoundError: No module named 'onnxruntime.capi._pybind_state'` regardless of what was actually installed. The subprocess spawns a clean Python 3.11 interpreter every time, completely bypassing the stale module cache.
- **Removed `_get_app()` entirely**: The v0.0.2 cache-flush retry approach (`sys.modules.pop` + re-import) was unreliable because the poisoned entries kept coming back. Subprocess isolation is the correct fix.
- **All four tools now use subprocess**: `analyze_faces`, `identify_face`, `register_identity` all delegate to `face_analyzer.py` via `_run_analyzer()`. No InsightFace or onnxruntime code runs in the gateway process.

### Architecture
- `face_identity.py` is now a thin wrapper: path config, subprocess calls, file operations (filing ref photos, reading the embeddings DB).
- `scripts/face_analyzer.py` is the standalone entry point: accepts `analyze`, `identify`, `register`, `compare` subcommands, outputs JSON to stdout.
- `_VENV_PYTHON` points to the Hermes venv (`~/.hermes/hermes-agent/venv/bin/python3`) so the subprocess gets the correct onnxruntime + insightface installation.

## [0.0.2] — 2026-06-09

### Fixed
- **Model path triple-nesting**: `FaceAnalysis(root=...)` appends `/models` internally; code was also appending `/models`, causing `models/models/models/antelopev2`. Changed `models_dir` from `INSIGHTFACE_DIR / "models"` to `INSIGHTFACE_DIR.parent`.
- **onnxruntime import (attempted fix)**: Added `sys.modules` cache-flush retry in `_get_app()` — this was superseded by the subprocess approach in v0.0.3.

### Dependencies
- Pinned `onnxruntime==1.21.0` (cp311 wheel). The 1.23.2 universal wheel was built against Python 3.12 ABI and fails to load the C extension under Python 3.11.

## [0.0.1] — 2026-06-09

Initial pre-release. Plugin scaffold with four tools:
- `portrait_identify_face` — detect and identify faces via InsightFace embeddings
- `portrait_register_identity` — register new face identity with auto-filing
- `portrait_generate` — generate portraits via ComfyUI + PuLID + Flux on MBP
- `portrait_ref_status` — show reference photo library stats
