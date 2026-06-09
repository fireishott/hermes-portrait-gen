# Changelog

## [0.0.2] — 2026-06-09

### Fixed
- **Model path triple-nesting**: `FaceAnalysis(root=...)` appends `/models` internally; code was also appending `/models`, causing `models/models/models/antelopev2`. Changed `models_dir` from `INSIGHTFACE_DIR / "models"` to `INSIGHTFACE_DIR.parent`.
- **onnxruntime import failure**: Stale `sys.modules` entries from a Python 3.12 onnxruntime build (removed `hermes_env` venv) caused `ModuleNotFoundError: No module named 'onnxruntime.capi._pybind_state'`. Added cache-flush retry in `_get_app()` that purges stale onnxruntime modules and re-imports.
- **Improved error diagnostics**: `_get_app()` now logs `sys.path`, `sys.modules` state, and onnxruntime source path on import failure instead of a generic "not installed" message.

### Dependencies
- Pinned `onnxruntime==1.21.0` (cp311 wheel). The 1.23.2 universal wheel was built against Python 3.12 ABI and fails to load the C extension under Python 3.11.

## [0.0.1] — 2026-06-09

Initial pre-release. Plugin scaffold with four tools:
- `portrait_identify_face` — detect and identify faces via InsightFace embeddings
- `portrait_register_identity` — register new face identity with auto-filing
- `portrait_generate` — generate portraits via ComfyUI + PuLID + Flux on MBP
- `portrait_ref_status` — show reference photo library stats
