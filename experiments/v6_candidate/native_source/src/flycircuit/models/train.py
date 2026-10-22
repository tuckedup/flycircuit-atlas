"""Truncated-BPTT training helper plus memory estimate/measurement.

Gradient horizon: loss on each chunk back-propagates at most ``tbptt_steps`` steps; the hidden
state is detached between chunks (no gradient flows to earlier chunks). This is an APPROXIMATION
of full BPTT. No activation checkpointing is used (checkpointing would keep exact gradients but
trade compute for memory; truncation changes the gradient itself).
"""

from __future__ import annotations

import resource
import time

import torch


def detach_state(state):
    return {k: v.detach() for k, v in state.items()}


def estimate_memory_bytes(n: int, e: int, batch: int, tbptt_steps: int, n_params: int,
                          dtype_bytes: int = 4, optimizer_states: int = 2) -> dict:
    """THEORETICAL estimate (not a measurement). Per step the scatter backend saves the gathered
    h[:, pre] * w product (B*E) and a few (B,N) tensors; params + grads + Adam moments."""
    per_step = (2 * batch * e + 6 * batch * n) * dtype_bytes
    act = per_step * tbptt_steps
    params = n_params * dtype_bytes * (2 + optimizer_states)
    idx = 2 * e * 8 + e * 4
    return {
        "activations_bytes": act, "params_optimizer_bytes": params, "index_bytes": idx,
        "total_bytes": act + params + idx, "total_gib": (act + params + idx) / 2**30,
        "kind": "theoretical",
    }


def train_tbptt(model, x, target, loss_fn, optimizer, tbptt_steps: int = 32, u=None):
    """One pass over a (T,B,*) sequence in chunks of ``tbptt_steps``. Returns mean loss."""
    state = None
    losses = []
    for s in range(0, x.shape[0], tbptt_steps):
        xs, ys = x[s:s + tbptt_steps], target[s:s + tbptt_steps]
        us = None if u is None else u[s:s + tbptt_steps]
        optimizer.zero_grad(set_to_none=True)
        y, state = model(xs, state, us) if us is not None else model(xs, state)
        loss = loss_fn(y, ys)
        loss.backward()
        optimizer.step()
        state = detach_state(state)
        losses.append(float(loss.detach()))
    return sum(losses) / max(1, len(losses))


def measure_peak(fn, device: str = "cpu") -> dict:
    """Run ``fn`` and record wall time and peak memory (CUDA allocated, or Python-heap on CPU)."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return {"seconds": time.perf_counter() - t,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "device": device}
    t = time.perf_counter()
    fn()
    sec = time.perf_counter() - t
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux: KiB
    return {"seconds": sec, "peak_process_rss_bytes": peak_rss, "device": device,
            "note": "process-lifetime peak RSS (monotonic), includes interpreter + torch"}
