"""Live (never cached) Windows + WSL + GPU resource snapshot.

Windows memory is read by calling ``powershell.exe`` from WSL (read-only CIM query). If that
fails the Windows fields are ``None`` and callers must say so; nothing is guessed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from typing import Any

import psutil

GIB = 1024**3
_PS = (
    "$o=Get-CimInstance Win32_OperatingSystem;"
    "$v=(Get-Process vmmemWSL,vmmem -ErrorAction SilentlyContinue|Measure-Object WorkingSet64 -Sum).Sum;"
    "[pscustomobject]@{free_kib=$o.FreePhysicalMemory;total_kib=$o.TotalVisibleMemorySize;"
    "commit_free_kib=$o.FreeVirtualMemory;commit_total_kib=$o.TotalVirtualMemorySize;vmmem_b=$v}"
    "|ConvertTo-Json -Compress"
)


def windows_memory(timeout: float = 15.0) -> dict[str, Any] | None:
    exe = shutil.which("powershell.exe") or "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    try:
        out = subprocess.run(
            [exe, "-NoProfile", "-NonInteractive", "-Command", _PS],
            capture_output=True, text=True, timeout=timeout, check=True,
        ).stdout
        d = json.loads(out.strip().splitlines()[-1])
    except Exception:
        return None
    return {
        "free_gib": d["free_kib"] * 1024 / GIB,
        "total_gib": d["total_kib"] * 1024 / GIB,
        "commit_free_gib": d["commit_free_kib"] * 1024 / GIB,
        "commit_total_gib": d["commit_total_kib"] * 1024 / GIB,
        # host-side working set of the WSL VM (what the VM currently costs Windows); may be None
        "vmmem_gib": (d["vmmem_b"] / GIB) if d.get("vmmem_b") else None,
    }


def windows_growth_gib(need_gib: float, win: dict[str, Any] | None, wsl: dict[str, Any]) -> float:
    """HEURISTIC: how much of ``need_gib`` must be newly taken from Windows.

    If the VM's host footprint (vmmemWSL working set) is known, growth is whatever pushes
    (Linux used + need) beyond it; Linux page cache is assumed evictable. Otherwise fall back to
    need minus the VM's reclaimable cache. Not a measurement of the future; see docs/RESOURCES.md.
    """
    if win and win.get("vmmem_gib"):
        used = wsl["total_gib"] - wsl["available_gib"]
        return max(0.0, used + need_gib - win["vmmem_gib"])
    return max(0.0, need_gib - wsl["reclaimable_gib"])


def wsl_memory() -> dict[str, Any]:
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    return {
        "total_gib": vm.total / GIB,
        "available_gib": vm.available / GIB,
        "memfree_gib": vm.free / GIB,
        # page cache etc. that Linux can hand back without asking Windows for more memory
        "reclaimable_gib": max(0.0, (vm.available - vm.free) / GIB),
        "swap_used_gib": sw.used / GIB,
        "swap_total_gib": sw.total / GIB,
    }


def gpu_state() -> dict[str, Any]:
    """nvidia-smi view (includes other processes' use, unlike torch.cuda.memory_allocated)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "error": "nvidia-smi not found"}
    q = "memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu"
    try:
        out = subprocess.run([exe, f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15, check=True).stdout
        used, free, total, util, temp = (float(x) for x in out.strip().splitlines()[0].split(","))
        apps = subprocess.run([exe, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    return {
        "available": True, "used_gib": used / 1024, "free_gib": free / 1024,
        "total_gib": total / 1024, "util_pct": util, "temp_c": temp,
        "compute_apps": [ln.strip() for ln in apps.splitlines() if ln.strip()],
    }


def cpu_state() -> dict[str, Any]:
    return {"logical": psutil.cpu_count(True), "physical": psutil.cpu_count(False),
            "load_1min": psutil.getloadavg()[0], "util_pct": psutil.cpu_percent(interval=0.3)}


def snapshot(windows: bool = True) -> dict[str, Any]:
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "windows": windows_memory() if windows else None,
        "wsl": wsl_memory(),
        "gpu": gpu_state(),
        "cpu": cpu_state(),
    }
