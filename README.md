# hermes-portrait-gen

Local portrait generation with face preservation — a [Hermes Agent](https://hermes-agent.nousresearch.com) plugin.

## What It Does

Generates portraits of real people by preserving their face from reference photos, running entirely on your local network.

- **Face identification** — InsightFace detects and identifies faces, auto-files reference photos by person
- **Zero-shot face preservation** — PuLID + Flux injects identity from reference photos without training
- **Fully local** — no cloud APIs, no base64, all file transfer over SSH
- **Smart ref library** — grows automatically as you send photos, Mnemosyne tracks identity facts

## Architecture

```
iMac (Hermes + Plugin)              MBP (ComfyUI + GPU)
  │                                       │
  │ ~/Pictures/ref-photos/{person}/       │
  │    (source of truth)                  │
  │                                       │
  │ 1. scp ref(s) ──────────────────────►│ /tmp/portrait-gen/
  │ 2. ssh: start ComfyUI ──────────────►│ python main.py --port 8188
  │ 3. http: POST workflow ──────────────►│ :8188/api/prompt
  │ 4. poll until done                    │ PuLID + Flux generates
  │ 5. scp output ◄─────────────────────│ /ComfyUI/output/
  │ 6. ssh: stop ComfyUI ───────────────►│ freed
  │ 7. deliver portrait                   │
```

## Requirements

### iMac (Hermes host)
- macOS with SSH access to MBP
- `sshpass` installed (`brew install sshpass`)
- Python 3.11+ (Hermes venv)

### MBP (compute node)
- Apple Silicon Mac with 16GB+ RAM
- ComfyUI installed with venv
- Custom nodes: `ComfyUI-PuLID-Flux-Enhanced`, `ComfyUI-GGUF`
- Models downloaded:
  - `flux1-dev-Q6_K.gguf` (in `unet/`)
  - `clip_l.safetensors`, `t5xxl_fp8_e4m3fn.safetensors` (in `clip/`)
  - `ae.safetensors` (in `vae/`)
  - `pulid_flux_v0.9.0.safetensors` (in `pulid/`)
  - `antelopev2` InsightFace models (in `insightface/models/`)

## Installation

```bash
# 1. Clone into Hermes plugins dir
cd ~/.hermes/plugins
git clone git@github.com:fireishott/hermes-portrait-gen.git portrait-gen

# 2. Install Python deps in Hermes venv
~/.hermes/hermes-agent/venv/bin/pip3 install -r ~/.hermes/plugins/portrait-gen/requirements.txt

# 3. Enable in config.yaml
# Add 'portrait-gen' under plugins.enabled

# 4. Restart gateway (or /reset in chat)
```

## Tools

| Tool | Description |
|------|-------------|
| `portrait_identify_face` | Detect/identify faces, auto-file ref photos |
| `portrait_register_identity` | Register a new person from a photo |
| `portrait_generate` | Generate a portrait via PuLID + Flux on MBP |
| `portrait_ref_status` | Show ref library and MBP/ComfyUI status |

## Configuration

All settings have sane defaults. Override via environment variables:

| Env Var | Default | Description |
|---------|---------|-------------|
| `PORTRAIT_MBP_HOST` | `192.168.10.121` | MBP IP address |
| `PORTRAIT_MBP_USER` | `curtisfreeman` | MBP SSH user |
| `PORTRAIT_MBP_PASS` | (from config) | MBP SSH password |
| `PORTRAIT_COMFYUI_PORT` | `8188` | ComfyUI port |
| `PORTRAIT_FACE_MATCH_THRESHOLD` | `0.35` | Cosine similarity threshold for face match |
| `PORTRAIT_REF_DIR` | `~/Pictures/ref-photos` | Reference photo library path |

## Version History

- **v0.0.1** — Initial pre-release. Face identification, auto-filing, basic ComfyUI orchestration with PuLID + Flux.

## License

Private — for personal use.
