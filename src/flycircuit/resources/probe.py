"""Measure current host/WSL/CUDA resources. Always refreshed, never cached."""

from __future__ import annotations

import resource
import shutil
from pathlib import Path
from typing import Any

import psutil

GIB = 1024**3


def peak_ram_mib() -> float:
    """Peak RSS of this process plus reaped children (Linux ru_maxrss is KiB)."""
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return (own + kids) / 1024.0


def peak_cuda_mib() -> float | None:
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return torch.cuda.max_memory_allocated() / 1024**2
    except Exception:
        pass
    return None


def cuda_info() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        free, total = torch.cuda.mem_get_info()
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "free_gib": free / GIB,
            "total_gib": total / GIB,
            "torch_cuda": torch.version.cuda,
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "error": repr(exc)}


def probe(project_root: Path) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    wsl_disk = shutil.disk_usage(project_root)
    host = None
    for cand in ("/mnt/c",):
        if Path(cand).exists():
            d = shutil.disk_usage(cand)
            host = {"mount": cand, "free_gib": d.free / GIB, "total_gib": d.total / GIB}
    return {
        "wsl_ram_total_gib": vm.total / GIB,
        "wsl_ram_available_gib": vm.available / GIB,
        "wsl_swap_total_gib": sw.total / GIB,
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        # WSL virtual disk capacity is thin-provisioned: NOT physical free space.
        "wsl_fs_free_gib_UNRELIABLE": wsl_disk.free / GIB,
        "host_disk": host,
        "cuda": cuda_info(),
    }
