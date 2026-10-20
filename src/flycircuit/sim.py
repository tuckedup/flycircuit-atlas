"""Simple discrete-time LIF dynamics for engineering validation.

This is NOT the published reference model (Shiu et al.); it only exercises the
graph -> dynamics -> manifest path. The reference engine is a later milestone.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from flycircuit.graph.versioned import VersionedGraph


def simulate_lif(
    g: VersionedGraph,
    input_idx: np.ndarray,
    steps: int = 500,
    dt_ms: float = 1.0,
    tau_m_ms: float = 20.0,
    v_rest: float = 0.0,
    v_th: float = 1.0,
    v_reset: float = 0.0,
    refractory_ms: float = 2.0,
    weight_scale: float = 0.06,
    input_rate_hz: float = 200.0,
    input_amp: float = 1.2,
    seed: int = 0,
    silenced: np.ndarray | None = None,
) -> dict[str, Any]:
    """Event-driven Poisson input to ``input_idx``; recurrent drive W @ spikes (W[post, pre])."""
    rng = np.random.default_rng(seed)
    n = g.n
    W = g.csr * weight_scale
    v = np.full(n, v_rest, dtype=np.float64)
    refr = np.zeros(n, dtype=np.int64)
    ref_steps = int(round(refractory_ms / dt_ms))
    alpha = dt_ms / tau_m_ms
    spikes_prev = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    p_in = input_rate_hz * dt_ms / 1000.0
    input_idx = np.asarray(input_idx, dtype=np.int64)
    sil = np.zeros(n, dtype=bool) if silenced is None else np.asarray(silenced, dtype=bool)
    for _ in range(steps):
        ext = np.zeros(n)
        ext[input_idx] = (rng.random(len(input_idx)) < p_in) * input_amp
        i_syn = W @ spikes_prev
        active = refr == 0
        v = np.where(active, v + alpha * (-(v - v_rest)) + i_syn + ext, v)
        spk = (v >= v_th) & active & ~sil
        v[spk] = v_reset
        v[sil] = v_rest
        refr = np.where(spk, ref_steps, np.maximum(refr - 1, 0))
        counts += spk
        spikes_prev = spk.astype(np.float64)
    secs = steps * dt_ms / 1000.0
    return {
        "spike_counts": counts,
        "rates_hz": counts / secs,
        "params": {
            "steps": steps, "dt_ms": dt_ms, "tau_m_ms": tau_m_ms, "v_th": v_th,
            "refractory_ms": refractory_ms, "weight_scale": weight_scale,
            "input_rate_hz": input_rate_hz, "input_amp": input_amp,
        },
    }
