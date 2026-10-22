"""Environment doctor: command resolution, threads, disk, CUDA compute + gradients."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flycircuit.config import load_hardware, project_root
from flycircuit.resources.probe import probe
from flycircuit.tracking.manifest import env_versions, git_state

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _linux_search_path() -> str:
    """Interactive-shell PATH approximation: ~/.local/bin and the newest nvm Node first."""
    home = Path.home()
    extra = [str(home / ".local" / "bin")]
    nodes = sorted((home / ".nvm" / "versions" / "node").glob("v*/bin"))
    if nodes:
        extra.append(str(nodes[-1]))
    return os.pathsep.join([*extra, os.environ.get("PATH", "")])


def _cmd(name: str) -> dict[str, Any]:
    bare = shutil.which(name)
    p = shutil.which(name, path=_linux_search_path())
    return {
        "path": p,
        "windows_shadow": bool(p and p.startswith("/mnt/")),
        "found": p is not None,
        "on_bare_process_path": bare == p,
    }


def cuda_selftest() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"ok": False, "reason": "torch.cuda.is_available() is False"}
        x = torch.randn(512, 512, device="cuda", requires_grad=True)
        x.square().mean().backward()
        return {
            "ok": bool(x.grad is not None and x.grad.abs().sum().item() > 0),
            "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "what": "forward + backward on 512x512",
        }
    except Exception as exc:
        return {"ok": False, "reason": repr(exc)}


def run_doctor(write: bool = True) -> dict[str, Any]:
    root = project_root()
    hw = load_hardware(root=root)
    import torch

    rev, dirty = git_state(root)
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "project_root": str(root),
        "python_executable": sys.executable,
        "python_in_project_venv": str(root / ".venv") in sys.executable,
        "kernel": platform.release(),
        "commands": {c: _cmd(c) for c in ("python3", "uv", "node", "npm", "git", "gh", "rg")},
        "thread_env": {v: os.environ.get(v) for v in THREAD_VARS},
        "torch_threads": torch.get_num_threads(),
        "profile": hw.model_dump(),
        "resources": probe(root),
        "cuda_selftest": cuda_selftest(),
        "git": {"revision": rev, "dirty": dirty},
        "versions": env_versions(),
        "notes": [
            "WSL filesystem free space is thin-provisioned; use host_disk for physical free space.",
            "Windows Claude authentication is not inspected from Linux (by design).",
        ],
    }
    problems = []
    if not info["python_in_project_venv"]:
        problems.append("python is not the project .venv interpreter")
    for c, d in info["commands"].items():
        if d["windows_shadow"]:
            problems.append(f"{c} resolves to a Windows binary: {d['path']}")
    if not info["cuda_selftest"]["ok"]:
        problems.append("CUDA self-test failed")
    info["problems"] = problems
    if write:
        out = root / "results" / "environment.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
        info["written_to"] = str(Path(out))
    return info
