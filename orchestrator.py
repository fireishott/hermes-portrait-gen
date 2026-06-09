"""Orchestrator — SSH push refs → spin ComfyUI → generate → pull back."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from .config import (
    MBP_HOST,
    MBP_USER,
    MBP_PASS,
    SSH_TIMEOUT,
    COMFYUI_PORT,
    COMFYUI_DIR,
    COMFYUI_STARTUP_TIMEOUT,
    COMFYUI_POLL_INTERVAL,
    COMFYUI_GENERATION_TIMEOUT,
    MBP_STAGING_DIR,
)
from .models import PortraitRequest, PortraitResult
from .face_identity import get_ref_photos

log = logging.getLogger("portrait-gen.orchestrator")

# ── Default ComfyUI workflow template ──────────────────────────────────────
_WORKFLOW_PATH = Path(__file__).parent / "templates" / "default_workflow.json"


def _ssh_cmd(cmd: str, timeout: int = SSH_TIMEOUT) -> subprocess.CompletedProcess:
    """Run a command on the MBP via SSH."""
    full_cmd = [
        "sshpass", "-p", MBP_PASS,
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=5",
        f"{MBP_USER}@{MBP_HOST}",
        cmd,
    ]
    return subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _scp_to_mbp(local_path: str, remote_path: str) -> bool:
    """Copy a local file to the MBP."""
    full_cmd = [
        "sshpass", "-p", MBP_PASS,
        "scp",
        "-o", "StrictHostKeyChecking=no",
        local_path,
        f"{MBP_USER}@{MBP_HOST}:{remote_path}",
    ]
    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=30)
    return result.returncode == 0


def _scp_from_mbp(remote_path: str, local_path: str) -> bool:
    """Copy a file from the MBP to local."""
    full_cmd = [
        "sshpass", "-p", MBP_PASS,
        "scp",
        "-o", "StrictHostKeyChecking=no",
        f"{MBP_USER}@{MBP_HOST}:{remote_path}",
        local_path,
    ]
    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=60)
    return result.returncode == 0


def check_mbp_reachable() -> bool:
    """Check if MBP is reachable via SSH."""
    try:
        result = _ssh_cmd("echo ok", timeout=5)
        return result.returncode == 0 and "ok" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_comfyui_running() -> bool:
    """Check if ComfyUI is already running on the MBP."""
    try:
        url = f"http://{MBP_HOST}:{COMFYUI_PORT}/system_stats"
        req = urllib.request.Request(url, method="GET")
        req.add_header("User-Agent", "portrait-gen/0.0.1")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_comfyui() -> bool:
    """Start ComfyUI on the MBP. Returns True when ready.

    If already running, returns True immediately.
    """
    if check_comfyui_running():
        log.info("ComfyUI already running on MBP")
        return True

    log.info("Starting ComfyUI on MBP...")
    # Start in background on MBP
    start_cmd = (
        f"cd {COMFYUI_DIR} && "
        f"source venv/bin/activate && "
        f"nohup python main.py --port {COMFYUI_PORT} --listen 0.0.0.0 "
        f"> /tmp/comfyui.log 2>&1 &"
    )
    result = _ssh_cmd(start_cmd, timeout=15)
    if result.returncode != 0:
        log.error(f"Failed to start ComfyUI: {result.stderr}")
        return False

    # Wait for readiness
    start_time = time.time()
    while time.time() - start_time < COMFYUI_STARTUP_TIMEOUT:
        if check_comfyui_running():
            elapsed = time.time() - start_time
            log.info(f"ComfyUI ready in {elapsed:.1f}s")
            return True
        time.sleep(COMFYUI_POLL_INTERVAL)

    log.error(f"ComfyUI not ready after {COMFYUI_STARTUP_TIMEOUT}s")
    return False


def stop_comfyui():
    """Stop ComfyUI on the MBP."""
    log.info("Stopping ComfyUI on MBP...")
    _ssh_cmd(
        "pkill -f 'python main.py.*--port' 2>/dev/null; true",
        timeout=5,
    )


def push_refs_to_mbp(person: str, ref_paths: list[Path]) -> list[str]:
    """Push reference photos to MBP staging dir. Returns remote paths."""
    # Ensure staging dir exists
    _ssh_cmd(f"mkdir -p {MBP_STAGING_DIR}", timeout=5)

    remote_paths = []
    for ref in ref_paths:
        remote_name = f"{person}_{ref.name}"
        remote_path = f"{MBP_STAGING_DIR}/{remote_name}"
        if _scp_to_mbp(str(ref), remote_path):
            remote_paths.append(remote_path)
            log.info(f"Pushed {ref.name} → {remote_path}")
        else:
            log.warning(f"Failed to push {ref.name}")
    return remote_paths


def pull_output_from_mbp(remote_path: str, local_dir: Path) -> Optional[Path]:
    """Pull a generated image from MBP to local directory."""
    local_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(remote_path).name
    local_path = local_dir / filename
    if _scp_from_mbp(remote_path, str(local_path)):
        return local_path
    return None


def load_workflow_template() -> dict:
    """Load the default ComfyUI workflow template."""
    if not _WORKFLOW_PATH.exists():
        raise FileNotFoundError(
            f"Workflow template not found: {_WORKFLOW_PATH}"
        )
    return json.loads(_WORKFLOW_PATH.read_text())


def build_workflow(
    request: PortraitRequest,
    ref_remote_path: str,
    seed: Optional[int] = None,
) -> dict:
    """Build a ComfyUI workflow from template + request parameters.

    Injects the reference image path and prompt into the workflow.
    """
    workflow = load_workflow_template()

    if seed is None or seed == -1:
        import random
        seed = random.randint(0, 2**32 - 1)

    # Inject reference image path
    if "4" in workflow:
        workflow["4"]["inputs"]["image"] = ref_remote_path

    # Inject prompt
    if "6" in workflow:
        workflow["6"]["inputs"]["text"] = request.prompt

    # Inject generation params
    if "10" in workflow:
        workflow["10"]["inputs"]["seed"] = seed
        workflow["10"]["inputs"]["steps"] = request.steps
        workflow["10"]["inputs"]["cfg"] = request.cfg
        workflow["10"]["inputs"]["width"] = request.width
        workflow["10"]["inputs"]["height"] = request.height

    # Inject PuLID weight
    if "8" in workflow:
        workflow["8"]["inputs"]["weight"] = request.pulid_weight

    return workflow


def queue_prompt(workflow: dict) -> Optional[str]:
    """Submit a workflow to ComfyUI and return the prompt_id."""
    payload = json.dumps({"prompt": workflow}).encode()
    url = f"http://{MBP_HOST}:{COMFYUI_PORT}/api/prompt"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "portrait-gen/0.0.1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            return body.get("prompt_id")
    except Exception as e:
        log.error(f"Failed to queue prompt: {e}")
        return None


def poll_for_completion(prompt_id: str) -> Optional[dict]:
    """Poll ComfyUI /history until the prompt completes. Returns output info."""
    url = f"http://{MBP_HOST}:{COMFYUI_PORT}/history/{prompt_id}"
    start_time = time.time()
    while time.time() - start_time < COMFYUI_GENERATION_TIMEOUT:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "portrait-gen/0.0.1"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                history = json.loads(resp.read())
                if prompt_id in history:
                    return history[prompt_id]
        except Exception:
            pass
        time.sleep(COMFYUI_POLL_INTERVAL)
    return None


def find_output_image(history_entry: dict) -> Optional[str]:
    """Extract the output image path from a ComfyUI history entry."""
    outputs = history_entry.get("outputs", {})
    for node_id, node_output in outputs.items():
        images = node_output.get("images", [])
        for img in images:
            return img.get("filename")
    return None


def generate_portrait(
    request: PortraitRequest,
    ref_paths: list[Path],
) -> PortraitResult:
    """Full portrait generation pipeline.

    1. Push refs to MBP
    2. Start ComfyUI
    3. Queue workflow
    4. Poll for completion
    5. Pull output
    6. Stop ComfyUI
    """
    start_time = time.time()

    # Check MBP reachability
    if not check_mbp_reachable():
        return PortraitResult(
            success=False,
            error="MBP not reachable at {MBP_HOST}",
        )

    # Push refs to MBP
    remote_refs = push_refs_to_mbp(request.person, ref_paths)
    if not remote_refs:
        return PortraitResult(
            success=False,
            error="Failed to push any reference photos to MBP",
        )

    # Start ComfyUI
    if not start_comfyui():
        return PortraitResult(
            success=False,
            error="Failed to start ComfyUI on MBP",
        )

    try:
        # Build and queue workflow
        seed = request.seed if request.seed != -1 else None
        workflow = build_workflow(request, remote_refs[0], seed=seed)
        prompt_id = queue_prompt(workflow)
        if not prompt_id:
            return PortraitResult(
                success=False,
                error="Failed to queue workflow to ComfyUI",
            )

        log.info(f"Generation queued: {prompt_id}")

        # Poll for completion
        history = poll_for_completion(prompt_id)
        if not history:
            return PortraitResult(
                success=False,
                error=f"Generation timed out after {COMFYUI_GENERATION_TIMEOUT}s",
            )

        # Find output image
        output_filename = find_output_image(history)
        if not output_filename:
            return PortraitResult(
                success=False,
                error="No output image found in ComfyUI history",
            )

        # Pull output to iMac
        remote_output = f"{COMFYUI_DIR}/output/{output_filename}"
        local_dir = Path(tempfile.mkdtemp(prefix="portrait_"))
        local_path = pull_output_from_mbp(remote_output, local_dir)

        if not local_path:
            return PortraitResult(
                success=False,
                error=f"Failed to pull output from MBP: {output_filename}",
            )

        elapsed = time.time() - start_time
        return PortraitResult(
            success=True,
            output_path=str(local_path),
            person=request.person,
            prompt=request.prompt,
            seed=seed,
            generation_time_s=round(elapsed, 1),
        )

    finally:
        # Cleanup staging and stop ComfyUI
        _ssh_cmd(f"rm -rf {MBP_STAGING_DIR}", timeout=5)
        stop_comfyui()
