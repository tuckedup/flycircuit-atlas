"""Live adaptive preflight: reads Windows + WSL + GPU free memory NOW and refuses or downsizes.

Uses the measured profile (``configs/hardware/adaptive_profile.yaml``) for calibration and
reserves; if it is absent the historical conservative profile is used and the result says so.
The Windows-growth rule is a documented HEURISTIC (see docs/RESOURCES.md), not a measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flycircuit.config import load_yaml, project_root
from flycircuit.resources import live
from flycircuit.resources.estimate import estimate_sparse_rnn

DEFAULTS = {  # used only when no measured profile exists
    "cuda_reserve_gib": 1.2, "windows_free_reserve_gib": 1.0, "wsl_ram_headroom_gib": 0.75,
    "cuda_ratio": 1.5, "cuda_context_gib": 0.4, "process_rss_gib": 1.5, "gpu_processes": 1,
}


def load_adaptive(root: Path | None = None) -> dict[str, Any] | None:
    p = (root or project_root()) / "configs" / "hardware" / "adaptive_profile.yaml"
    return load_yaml(p) if p.exists() else None


def calibration(profile: dict[str, Any] | None) -> dict[str, float]:
    c = dict(DEFAULTS)
    if not profile:
        return c
    pol, m = profile.get("policy", {}), profile.get("measured", {})
    c["cuda_reserve_gib"] = pol.get("cuda_reserve_gib", c["cuda_reserve_gib"])
    c["windows_free_reserve_gib"] = pol.get("windows_free_reserve_gib", c["windows_free_reserve_gib"])
    c["wsl_ram_headroom_gib"] = pol.get("wsl_ram_headroom_gib", c["wsl_ram_headroom_gib"])
    cal = m.get("calibration", {})
    if cal.get("cuda_ratio_max_theo_ge_0p2gib"):
        c["cuda_ratio"] = max(1.0, 1.15 * cal["cuda_ratio_max_theo_ge_0p2gib"])  # 15% margin
    c["cuda_fixed_gib"] = (cal.get("cuda_fixed_gib") or 0.0) * 1.25 if cal.get("cuda_fixed_gib") is not None \
        else 0.0
    if cal.get("cuda_process_rss_gib_max"):
        c["process_rss_gib"] = cal["cuda_process_rss_gib_max"]
    ctx = m.get("gpu", {}).get("cuda_context_gib")
    if ctx:
        c["cuda_context_gib"] = ctx
    c["gpu_processes"] = m.get("gpu", {}).get("gpu_processes", 1)
    return c


def _needs(est: dict[str, Any], cal: dict[str, float], use_cuda: bool) -> tuple[float, float]:
    ram = est["host_gib"] + (cal["process_rss_gib"] if use_cuda else 0.6)
    cuda = (est["cuda_gib"] * cal["cuda_ratio"] + cal.get("cuda_fixed_gib", 0.0)
            + cal["cuda_context_gib"]) if use_cuda else 0.0
    return ram, cuda


def check(est_args: dict[str, int], snap: dict[str, Any], cal: dict[str, float],
          use_cuda: bool = True) -> dict[str, Any]:
    """Pure function of (request, live snapshot, calibration) -> verdict. Unit-testable."""
    est = estimate_sparse_rnn(est_args["neurons"], est_args["edges"], est_args["batch"], est_args["tbptt"])
    ram_need, cuda_need = _needs(est, cal, use_cuda)
    fails: list[str] = []
    warns: list[str] = []
    wsl, win, gpu = snap["wsl"], snap.get("windows"), snap["gpu"]

    wsl_room = wsl["available_gib"] - cal["wsl_ram_headroom_gib"]
    if ram_need > wsl_room:
        fails.append(f"WSL RAM: need ~{ram_need:.2f} GiB (estimate + process baseline) but only "
                     f"{wsl_room:.2f} GiB available after {cal['wsl_ram_headroom_gib']} GiB headroom")
    if win is None:
        warns.append("Windows free RAM could not be read (powershell.exe unavailable); "
                     "Windows headroom NOT checked")
    else:
        # growth beyond what the VM already holds (reclaimable cache) must come from Windows
        from_windows = live.windows_growth_gib(ram_need, win, wsl)
        after = win["free_gib"] - from_windows
        if from_windows <= 0.05 and win["free_gib"] < cal["windows_free_reserve_gib"]:
            warns.append(f"Windows free RAM already low ({win['free_gib']:.2f} GiB < reserve "
                         f"{cal['windows_free_reserve_gib']} GiB) but this job fits inside the VM's "
                         "current footprint, so it should not worsen it")
        elif after < cal["windows_free_reserve_gib"]:
            fails.append(f"Windows RAM: free {win['free_gib']:.2f} GiB; this job may take up to "
                         f"{from_windows:.2f} GiB from Windows, leaving {after:.2f} GiB "
                         f"< reserve {cal['windows_free_reserve_gib']} GiB (risk of Windows paging)")
        if win["commit_total_gib"] and win["commit_free_gib"] < 2.0:
            warns.append(f"Windows commit headroom low ({win['commit_free_gib']:.1f} GiB free)")
    if wsl["swap_used_gib"] > 0.5:
        warns.append(f"WSL swap in use ({wsl['swap_used_gib']:.2f} GiB): memory pressure already present")
    if use_cuda:
        if not gpu.get("available"):
            fails.append(f"CUDA requested but unavailable: {gpu.get('error')}")
        else:
            room = gpu["free_gib"] - cal["cuda_reserve_gib"]
            if cuda_need > room:
                fails.append(f"CUDA: need ~{cuda_need:.2f} GiB (calibrated x{cal['cuda_ratio']:.2f} + "
                             f"context) but nvidia-smi free {gpu['free_gib']:.2f} GiB minus "
                             f"{cal['cuda_reserve_gib']} GiB reserve = {room:.2f} GiB")
            if gpu["temp_c"] >= 85:
                warns.append(f"GPU temperature {gpu['temp_c']:.0f} C")
            if gpu["util_pct"] >= 60:
                warns.append(f"GPU already {gpu['util_pct']:.0f}% utilised (competing workload?)")
    return {"ok": not fails, "failures": fails, "warnings": warns,
            "estimate": est, "need": {"ram_gib": ram_need, "cuda_gib": cuda_need},
            "request": est_args, "use_cuda": use_cuda}


def downsize(est_args: dict[str, int], snap: dict[str, Any], cal: dict[str, float],
             use_cuda: bool = True, min_batch: int = 1, min_tbptt: int = 8) -> dict[str, Any]:
    """Halve batch first, then tbptt, until it fits. Never changes neurons/edges (scientific
    config). Returns the verdict for the smaller request plus the original failure list."""
    first = check(est_args, snap, cal, use_cuda)
    if first["ok"]:
        return {**first, "downsized": False}
    a = dict(est_args)
    while True:
        if a["batch"] > min_batch:
            a["batch"] = max(min_batch, a["batch"] // 2)
        elif a["tbptt"] > min_tbptt:
            a["tbptt"] = max(min_tbptt, a["tbptt"] // 2)
        else:
            return {**first, "downsized": False,
                    "advice": "does not fit even at batch=1, tbptt=8; free memory, use CPU, or reduce "
                              "the circuit (a pilot subset must be named, not silent)"}
        v = check(a, snap, cal, use_cuda)
        if v["ok"]:
            return {**v, "downsized": True, "original_request": est_args, "original_failures":
                    first["failures"],
                    "advice": "Fits only after reducing batch/tbptt. Apply the SAME reduction to every "
                              "compared condition and record it; do not reuse the old run ID."}


def adaptive_preflight(est_args: dict[str, int], use_cuda: bool = True, allow_downsize: bool = False,
                       root: Path | None = None, snap: dict[str, Any] | None = None) -> dict[str, Any]:
    prof = load_adaptive(root)
    cal = calibration(prof)
    snap = snap or live.snapshot()
    res = downsize(est_args, snap, cal, use_cuda) if allow_downsize else check(est_args, snap, cal, use_cuda)
    res["profile"] = prof["profile"] if prof else "none (defaults; run `flycircuit resources profile`)"
    res["snapshot"] = snap
    return res


def hardware_profile_from_adaptive(root: Path | None = None):
    """Adapter for code that expects the old HardwareProfile type."""
    from flycircuit.config import HardwareProfile

    prof = load_adaptive(root)
    if not prof:
        raise FileNotFoundError("configs/hardware/adaptive_profile.yaml missing; run "
                                "`bash scripts/linux/bench_profile.sh`")
    return HardwareProfile(**prof["hardware_profile"])


__all__ = ["adaptive_preflight", "calibration", "check", "downsize", "load_adaptive",
           "hardware_profile_from_adaptive"]
