"""Theoretical memory arithmetic, preflight against budgets, and a small dry-run benchmark.

All estimates are THEORETICAL until compared with measured peaks from a pilot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from flycircuit.config import HardwareProfile
from flycircuit.resources.probe import GIB, peak_ram_mib

# Assumed multiplier for Arrow buffers / Python objects / temp conversion on the host side.
HOST_OVERHEAD_FACTOR = 1.5
SAVED_STATE_TENSORS_PER_STEP = 4


def estimate_sparse_rnn(
    n_neurons: int,
    n_edges: int,
    batch: int = 1,
    tbptt: int = 32,
    dtype_bytes: int = 4,
    index_bytes: int = 8,
) -> dict[str, Any]:
    """Theoretical memory for a COO/scatter sparse RNN with Adam, truncated BPTT."""
    params = n_edges * dtype_bytes
    indices = 2 * n_edges * index_bytes
    grads_opt = 3 * params  # grad + Adam m + v
    # scatter formulation saves gathered presynaptic activity and edge messages (B x E) per step
    edge_msgs = tbptt * batch * n_edges * dtype_bytes * 2
    states = tbptt * batch * n_neurons * dtype_bytes * SAVED_STATE_TENSORS_PER_STEP
    cuda = params + indices + grads_opt + edge_msgs + states
    host = int((n_edges * (2 * index_bytes + dtype_bytes) + n_neurons * 8) * HOST_OVERHEAD_FACTOR)
    return {
        "kind": "theoretical",
        "n_neurons": n_neurons, "n_edges": n_edges, "batch": batch, "tbptt": tbptt,
        "cuda_bytes": int(cuda), "host_bytes": host,
        "cuda_gib": cuda / GIB, "host_gib": host / GIB,
        "assumptions": {
            "host_overhead_factor": HOST_OVERHEAD_FACTOR,
            "saved_state_tensors_per_step": SAVED_STATE_TENSORS_PER_STEP,
            "fp32": dtype_bytes == 4,
        },
    }


@dataclass
class PreflightResult:
    ok: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "failures": self.failures, "warnings": self.warnings}


def preflight(
    est: dict[str, Any],
    hw: HardwareProfile,
    probe_result: dict[str, Any],
    new_storage_gib: float = 0.0,
    use_cuda: bool = True,
    walltime_minutes: float | None = None,
) -> PreflightResult:
    res = PreflightResult(ok=True)

    def fail(msg: str) -> None:
        res.ok = False
        res.failures.append(msg)

    if est["host_gib"] > hw.max_pipeline_ram_gib:
        fail(f"host RAM estimate {est['host_gib']:.2f} GiB > budget {hw.max_pipeline_ram_gib} GiB; "
             "reduce neurons/edges or stream in batches")
    avail = probe_result["wsl_ram_available_gib"] - hw.host_ram_headroom_gib
    if est["host_gib"] > avail:
        fail(f"host RAM estimate {est['host_gib']:.2f} GiB > WSL available minus headroom "
             f"{avail:.2f} GiB; close project processes or lower the budget")
    if use_cuda:
        cuda = probe_result["cuda"]
        if not cuda.get("available"):
            fail("CUDA requested but unavailable")
        else:
            if est["cuda_gib"] > hw.max_cuda_allocated_gib:
                fail(f"CUDA estimate {est['cuda_gib']:.2f} GiB > budget "
                     f"{hw.max_cuda_allocated_gib} GiB; reduce tbptt/batch or edges")
            free = cuda["free_gib"] - hw.cuda_headroom_gib
            if est["cuda_gib"] > free:
                fail(f"CUDA estimate {est['cuda_gib']:.2f} GiB > free minus headroom {free:.2f} GiB")
    if new_storage_gib > hw.max_new_project_storage_gib:
        fail(f"planned storage {new_storage_gib} GiB > max_new_project_storage_gib "
             f"{hw.max_new_project_storage_gib}")
    host = probe_result.get("host_disk")
    if host is None:
        res.warnings.append("host (Windows) free disk could not be measured from WSL")
    elif host["free_gib"] - new_storage_gib < hw.min_host_free_disk_gib:
        fail(f"host free disk {host['free_gib']:.1f} GiB minus planned {new_storage_gib} GiB "
             f"< minimum {hw.min_host_free_disk_gib} GiB")
    if walltime_minutes is not None and walltime_minutes > hw.max_default_job_walltime_minutes:
        fail(f"extrapolated walltime {walltime_minutes:.1f} min > "
             f"max_default_job_walltime_minutes {hw.max_default_job_walltime_minutes}")
    return res


def dry_run_benchmark(
    n_neurons: int, n_edges: int, tbptt: int, n_trials: int, n_conditions: int,
    hw: HardwareProfile, bench_edges: int = 200_000, reps: int = 3,
) -> dict[str, Any]:
    """Time a small CPU sparse-RNN forward/backward and linearly extrapolate runtime.

    The extrapolation is an assumption (time ~ linear in edges); it is NOT a measurement
    at the requested size.
    """
    import torch

    torch.set_num_threads(hw.cpu_threads)
    e = min(n_edges, bench_edges)
    n = max(2, min(n_neurons, max(64, e // 20)))
    g = torch.Generator().manual_seed(0)
    pre = torch.randint(0, n, (e,), generator=g)
    post = torch.randint(0, n, (e,), generator=g)
    w = (torch.rand(e, generator=g) * 0.01).requires_grad_(True)
    rss0 = peak_ram_mib()
    times = []
    for _ in range(reps):
        h = torch.zeros(1, n)
        t0 = time.perf_counter()
        for _ in range(tbptt):
            msg = w * h[:, pre] + 0.01
            agg = torch.zeros(1, n).index_add_(1, post, msg)
            h = 0.9 * h + 0.1 * torch.tanh(agg)
        h.sum().backward()
        times.append(time.perf_counter() - t0)
    t_trial_bench = min(times)
    per_edge_trial = t_trial_bench / e
    trial_s = per_edge_trial * n_edges
    total_min = trial_s * n_trials * n_conditions / 60.0
    return {
        "kind": "extrapolated_from_small_cpu_benchmark",
        "bench_edges": e, "bench_neurons": n, "bench_trial_seconds": t_trial_bench,
        "est_seconds_per_tbptt_window_at_target": trial_s,
        "est_total_minutes": total_min,
        "peak_ram_mib_after_bench": peak_ram_mib(),
        "peak_ram_mib_before_bench": rss0,
        "assumption": "runtime linear in edge count; CPU benchmark; no GPU speedup assumed",
    }
