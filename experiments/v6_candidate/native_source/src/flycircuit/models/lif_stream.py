"""Streaming-summary LIF inference for whole-brain-scale networks (CPU numpy or torch CPU/CUDA).

Same model, constants and step schedule as :mod:`flycircuit.models.lif` (the validated independent
engine; see its docstring): exact exponential integration per ``dt``, delay ``D`` steps, writes to
refractory neurons discarded, reset zeroes ``g``. The difference is the OUTPUT: the full spike list
is never kept. Only bounded summaries are retained:

* ``counts[n]``        spikes per neuron over the run
* ``first_step[n]``    first spike step (-1 if none)
* ``pop_per_step``     number of spikes at each step (population activity)
* ``watch_raster``     exact boolean raster ``[steps, n_watch]`` for a small watch list (<= 4096)
* ``psth``             per-watch-neuron spike counts per ``psth_bin_steps``

Memory is O(N + E + steps*n_watch), independent of the number of spikes. Inference only: this module
has no autograd and says nothing about whole-brain gradient training (see
:func:`estimate_whole_brain_training`).

Backends: ``numpy`` (float64, bit-compatible schedule with ``lif.simulate``), and ``torch`` on
``cpu``/``cuda`` (``dtype`` float64 default; float32 is a profiled option, not assumed correct).
"""

from __future__ import annotations

import resource
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from flycircuit.models.lif import LIFParams


@dataclass
class StreamSummary:
    n: int
    steps: int
    dt: float
    counts: np.ndarray
    first_step: np.ndarray
    pop_per_step: np.ndarray
    watch_idx: np.ndarray
    watch_raster: np.ndarray  # bool [steps, n_watch]
    psth_bin_steps: int
    psth: np.ndarray  # int32 [n_watch, n_bins]
    meta: dict[str, Any] = field(default_factory=dict)

    def rates_hz(self) -> np.ndarray:
        return self.counts / (self.steps * self.dt / 1000.0)

    def watch_trains(self) -> dict[int, np.ndarray]:
        return {
            int(self.watch_idx[j]): np.flatnonzero(self.watch_raster[:, j]).astype(np.int64)
            for j in range(len(self.watch_idx))
        }


def _csc_arrays(W: sp.spmatrix):
    Wc = sp.csc_matrix(W, dtype=np.float64)
    Wc.sort_indices()
    return Wc.indptr.astype(np.int64), Wc.indices.astype(np.int64), Wc.data


def _group_events(ext_steps, ext_targets, ext_weight, steps, w_ext):
    es = np.asarray(ext_steps if ext_steps is not None else [], dtype=np.int64)
    et = np.asarray(ext_targets if ext_targets is not None else [], dtype=np.int64)
    order = np.argsort(es, kind="stable")
    es, et = es[order], et[order]
    if ext_weight is None:
        ew = np.full(len(es), w_ext)
    elif np.isscalar(ext_weight):
        ew = np.full(len(es), float(ext_weight))
    else:
        ew = np.asarray(ext_weight, dtype=float)[order]
    return es, et, ew, np.searchsorted(es, np.arange(steps + 1))


def _maxrss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0  # KiB -> GiB


def simulate_stream(
    W_mV: sp.spmatrix,
    steps: int,
    params: LIFParams | None = None,
    ext_steps: np.ndarray | None = None,
    ext_targets: np.ndarray | None = None,
    ext_weight: np.ndarray | float | None = None,
    no_refractory: np.ndarray | None = None,
    watch: np.ndarray | None = None,
    psth_bin_ms: float = 10.0,
    backend: str = "numpy",
    device: str = "cpu",
    dtype: str = "float64",
) -> StreamSummary:
    """Run the network and return bounded streaming summaries. ``W_mV`` is ``[post, pre]`` in mV."""
    p = params or LIFParams()
    n = W_mV.shape[0]
    if W_mV.shape != (n, n):
        raise ValueError("W must be square [post, pre]")
    watch_idx = np.asarray(watch if watch is not None else [], dtype=np.int64)
    if len(watch_idx) > 4096:
        raise ValueError("watch list limited to 4096 neurons (bounded raster)")
    bin_steps = max(1, int(round(psth_bin_ms / p.dt)))
    n_bins = (steps + bin_steps - 1) // bin_steps
    indptr, rows, data = _csc_arrays(W_mV)
    es, et, ew, bounds = _group_events(ext_steps, ext_targets, ext_weight, steps, p.w_ext)
    rfc_len = np.full(n, p.n_refractory, dtype=np.int64)
    if no_refractory is not None and len(no_refractory):
        rfc_len[np.asarray(no_refractory, dtype=np.int64)] = 0
    t0 = time.perf_counter()
    if backend == "numpy":
        out = _run_numpy(p, n, steps, indptr, rows, data, es, et, ew, bounds, rfc_len,
                         watch_idx, bin_steps, n_bins)
        peak_cuda = None
    elif backend == "torch":
        out, peak_cuda = _run_torch(p, n, steps, indptr, rows, data, es, et, ew, bounds, rfc_len,
                                    watch_idx, bin_steps, n_bins, device, dtype)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    counts, first, pop, raster, psth = out
    meta = {
        "backend": backend, "device": device if backend == "torch" else "cpu",
        "dtype": dtype if backend == "torch" else "float64",
        "elapsed_s": time.perf_counter() - t0, "sim_ms": steps * p.dt,
        "peak_rss_gib": _maxrss_gib(), "peak_cuda_alloc_gib": peak_cuda,
        "n": int(n), "nnz": int(len(data)), "total_spikes": int(counts.sum()),
    }
    return StreamSummary(n, steps, p.dt, counts, first, pop, watch_idx, raster, bin_steps, psth, meta)


def _run_numpy(p, n, steps, indptr, rows, data, es, et, ew, bounds, rfc_len, watch_idx,
               bin_steps, n_bins):
    D = p.n_delay
    e_m, e_g = np.exp(-p.dt / p.t_mbr), np.exp(-p.dt / p.tau)
    k_a = p.tau / (p.tau - p.t_mbr)
    v = np.full(n, p.v_0)
    g = np.zeros(n)
    free_at = np.zeros(n, dtype=np.int64)
    inbox = np.zeros((D + 1, n))
    counts = np.zeros(n, dtype=np.int64)
    first = np.full(n, -1, dtype=np.int64)
    pop = np.zeros(steps, dtype=np.int64)
    raster = np.zeros((steps, len(watch_idx)), dtype=bool)
    psth = np.zeros((len(watch_idx), n_bins), dtype=np.int32)
    for k in range(steps):
        act = free_at <= k
        a = g * k_a
        v = np.where(act, p.v_0 + (v - p.v_0 - a) * e_m + a * e_g, v)
        g = np.where(act, g * e_g, g)
        spk = act & (v > p.v_th)
        slot = k % (D + 1)
        g = g + np.where(act, inbox[slot], 0.0)
        inbox[slot] = 0.0
        lo, hi = bounds[k], bounds[k + 1]
        if hi > lo:
            tg = et[lo:hi]
            okm = act[tg]
            np.add.at(v, tg[okm], ew[lo:hi][okm])
        idx = np.flatnonzero(spk)
        if len(idx):
            starts = indptr[idx]
            lens = indptr[idx + 1] - starts
            tot = int(lens.sum())
            if tot:
                off = np.repeat(starts - (np.cumsum(lens) - lens), lens) + np.arange(tot)
                inbox[(k + D) % (D + 1)] += np.bincount(rows[off], weights=data[off], minlength=n)
            if D == 0:
                g = g + np.where(act, inbox[slot], 0.0)
                inbox[slot] = 0.0
            v[idx] = p.v_rst
            g[idx] = 0.0
            free_at[idx] = k + rfc_len[idx]
            counts[idx] += 1
            f = first[idx] < 0
            first[idx[f]] = k
            pop[k] = len(idx)
            if len(watch_idx):
                w = spk[watch_idx]
                raster[k] = w
                psth[:, k // bin_steps] += w
    return counts, first, pop, raster, psth


def _run_torch(p, n, steps, indptr, rows, data, es, et, ew, bounds, rfc_len, watch_idx,
               bin_steps, n_bins, device, dtype):
    import torch

    dt_ = {"float64": torch.float64, "float32": torch.float32}[dtype]
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    D = p.n_delay
    e_m, e_g = float(np.exp(-p.dt / p.t_mbr)), float(np.exp(-p.dt / p.tau))
    k_a = p.tau / (p.tau - p.t_mbr)
    T = lambda x, dty=None: torch.as_tensor(x, device=dev, dtype=dty)  # noqa: E731
    indptr_t, rows_t, data_t = T(indptr), T(rows), T(data, dt_)
    et_t, ew_t = T(et), T(ew, dt_)
    rfc_t = T(rfc_len)
    watch_t = T(watch_idx, torch.int64)
    v = torch.full((n,), p.v_0, dtype=dt_, device=dev)
    g = torch.zeros(n, dtype=dt_, device=dev)
    free_at = torch.zeros(n, dtype=torch.int64, device=dev)
    inbox = torch.zeros((D + 1, n), dtype=dt_, device=dev)
    counts = torch.zeros(n, dtype=torch.int64, device=dev)
    first = torch.full((n,), -1, dtype=torch.int64, device=dev)
    pop = torch.zeros(steps, dtype=torch.int64, device=dev)
    raster = torch.zeros((steps, len(watch_idx)), dtype=torch.bool, device=dev)
    psth = torch.zeros((len(watch_idx), n_bins), dtype=torch.int32, device=dev)
    v0 = torch.tensor(p.v_0, dtype=dt_, device=dev)
    for k in range(steps):
        act = free_at <= k
        a = g * k_a
        v = torch.where(act, v0 + (v - v0 - a) * e_m + a * e_g, v)
        g = torch.where(act, g * e_g, g)
        spk = act & (v > p.v_th)
        slot = k % (D + 1)
        g = g + torch.where(act, inbox[slot], torch.zeros_like(g))
        inbox[slot] = 0.0
        lo, hi = int(bounds[k]), int(bounds[k + 1])
        if hi > lo:
            tg = et_t[lo:hi]
            v.index_add_(0, tg, ew_t[lo:hi] * act[tg].to(dt_))
        idx = torch.nonzero(spk).squeeze(1)
        if idx.numel():
            starts = indptr_t[idx]
            lens = indptr_t[idx + 1] - starts
            tot = int(lens.sum())
            if tot:
                csum = torch.cumsum(lens, 0) - lens
                off = torch.repeat_interleave(starts - csum, lens) + torch.arange(tot, device=dev)
                inbox[(k + D) % (D + 1)].index_add_(0, rows_t[off], data_t[off])
            if D == 0:
                g = g + torch.where(act, inbox[slot], torch.zeros_like(g))
                inbox[slot] = 0.0
            v[idx] = p.v_rst
            g[idx] = 0.0
            free_at[idx] = k + rfc_t[idx]
            counts[idx] += 1
            fm = first[idx] < 0
            first[idx[fm]] = k
            pop[k] = idx.numel()
            if len(watch_idx):
                w = spk[watch_t]
                raster[k] = w
                psth[:, k // bin_steps] += w.to(torch.int32)
    peak = None
    if dev.type == "cuda":
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**30
    to = lambda x: x.cpu().numpy()  # noqa: E731
    return (to(counts), to(first), to(pop), to(raster), to(psth)), peak


# ----------------------------------------------------------------------------------------------
# Honest resource arithmetic (THEORETICAL; measured numbers live in results/whole_brain/)
# ----------------------------------------------------------------------------------------------
def estimate_whole_brain_inference(n: int, nnz: int, steps: int, n_watch: int = 256,
                                   delay_steps: int = 18, dtype_bytes: int = 8) -> dict[str, Any]:
    """Host-RAM arithmetic for :func:`simulate_stream` (CSC int64 indices + float64 data + state)."""
    gib = 2**30
    csc = nnz * (8 + dtype_bytes) + (n + 1) * 8
    state = n * dtype_bytes * (3 + (delay_steps + 1)) + n * 8 * 3  # v,g,tmp + inbox + counters
    raster = steps * n_watch
    bincount_tmp = n * dtype_bytes * 2
    total = csc + state + raster + bincount_tmp
    return {"csc_gib": csc / gib, "state_gib": state / gib, "raster_gib": raster / gib,
            "total_gib": total / gib, "kind": "theoretical", "note": "does not include scipy "
            "conversion temporaries during construction (~2x CSC) or the Python/torch runtime"}


def estimate_whole_brain_training(n: int, nnz: int, tbptt: int, batch: int = 1,
                                  scatter_backend: bool = True) -> dict[str, Any]:
    """Theoretical GPU memory for sparse-RNN truncated BPTT (FP32) at whole-brain scale.

    Dominant term for the gather/scatter backend: the edge-aligned tensor ``h[pre] * w`` saved for
    backward at EVERY step: ``tbptt * batch * nnz * 4 B`` (x2 when both the gathered activity and
    the product are kept). Not launched here; see docs/whole_brain.md.
    """
    gib = 2**30
    params = nnz * 4 * (1 + 1 + 2)  # weight + grad + Adam m,v
    states = tbptt * batch * n * 4 * 4  # h, pre-activation, phi(.), misc
    edge_saved = tbptt * batch * nnz * 4 * (2 if scatter_backend else 0)
    idx = nnz * 8 * 2  # pre/post index (int64)
    total = params + states + edge_saved + idx
    return {"params_optim_gib": params / gib, "states_gib": states / gib,
            "edge_saved_gib": edge_saved / gib, "indices_gib": idx / gib,
            "total_gib": total / gib, "kind": "theoretical", "tbptt": tbptt, "batch": batch,
            "n": n, "nnz": nnz}
