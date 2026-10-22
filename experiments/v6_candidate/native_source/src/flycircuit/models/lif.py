"""Independent LIF reference engine for the Shiu et al. connectome model (spec section 10).

This is an independent re-implementation of the model *defined* by the original code
(``external/Drosophila_brain_model/model.py``, commit 91bdd1e, MIT). It shares no code with Brian2.

Model (all voltages in mV, times in ms):

    dv/dt = (v_0 - v + g) / t_mbr      (frozen while refractory)
    dg/dt = -g / tau                   (frozen while refractory)
    spike when v > v_th and not refractory  ->  v = v_rst, g = 0, refractory for t_rfc
    presynaptic spike at neuron j  ->  g_i += w_ij  after delay t_dly   (current-like alpha synapse:
                                       g is a voltage-valued kernel state, w_ij = sign*contacts*w_syn)
    external (Poisson) event       ->  v_i += w_syn * f_poi  instantly, target has no refractory

The ODE is linear, so each step is advanced with the EXACT solution over dt (not Euler):

    A = g tau / (tau - t_mbr);  v' = v_0 + (v - v_0 - A) e^{-dt/t_mbr} + A e^{-dt/tau};  g' = g e^{-dt/tau}

Step schedule (one clock tick = dt), mirroring the published code's semantics:
  1. record state (start of step)
  2. integrate non-refractory neurons by dt
  3. threshold: v > v_th and not refractory -> spikes of this step
  4. deliver synaptic input that arrives at this step (spikes emitted `D = round(t_dly/dt)` steps
     ago): ``g += w``; then apply external events: ``v += w_ext``. Writes to a REFRACTORY neuron
     are discarded (Brian2 conditional write for "unless refractory" variables)
  5. reset spiking neurons (v = v_rst, g = 0 - which also discards a ``g`` increment delivered in
     this same step -, refractory counter starts)
Whether this schedule equals Brian2's is NOT assumed: ``flycircuit refcheck run`` measures it.
Units, ``dt`` and the equations are not a "conceptual" simplification of the original model.

Release distinction: the repo's code defaults describe the paper's v630; ``release`` here is only
metadata - the v783 comparison is an adaptation, not an exact v630 reproduction.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

_UNITS = {"mV": 1.0, "ms": 1.0, "Hz": 1.0}  # target units: mV, ms, Hz (per second)


@dataclass(frozen=True)
class LIFParams:
    v_0: float = -52.0  # mV resting potential
    v_rst: float = -52.0  # mV reset potential
    v_th: float = -45.0  # mV threshold
    t_mbr: float = 20.0  # ms membrane time scale
    tau: float = 5.0  # ms synaptic kernel decay
    t_rfc: float = 2.2  # ms refractory period
    t_dly: float = 1.8  # ms synaptic delay
    w_syn: float = 0.275  # mV per synapse (contact)
    r_poi: float = 150.0  # Hz Poisson rate
    f_poi: float = 250.0  # Poisson weight factor
    dt: float = 0.1  # ms (Brian2 default clock, used by the published code)

    @property
    def w_ext(self) -> float:
        """mV added to v per external event (``w_syn * f_poi`` in the published code)."""
        return self.w_syn * self.f_poi

    @property
    def n_delay(self) -> int:
        return int(round(self.t_dly / self.dt))

    @property
    def n_refractory(self) -> int:
        return int(round(self.t_rfc / self.dt))

    def with_(self, **kw) -> LIFParams:
        return replace(self, **kw)

    @classmethod
    def from_shiu_source(cls, model_py: str | Path | None = None) -> LIFParams:
        """Read constants from ``model.py::default_params`` WITHOUT importing Brian2."""
        path = Path(model_py) if model_py else _default_model_py()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        vals: dict[str, float] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "default_params":
                for k, v in zip(node.value.keys, node.value.values, strict=True):
                    key = k.value
                    num = _eval_units(v)
                    if num is not None:
                        vals[key] = num
        need = {"v_0": "v_0", "v_rst": "v_rst", "v_th": "v_th", "t_mbr": "t_mbr", "tau": "tau",
                "t_rfc": "t_rfc", "t_dly": "t_dly", "w_syn": "w_syn", "r_poi": "r_poi",
                "f_poi": "f_poi"}
        missing = [k for k in need if k not in vals]
        if missing:
            raise ValueError(f"could not parse {missing} from {path}")
        return cls(**{k: vals[k] for k in need})


def _eval_units(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        x = _eval_units(node.operand)
        return None if x is None else -x
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        a, b = node.left, node.right
        if isinstance(b, ast.Name) and b.id in _UNITS:
            x = _eval_units(a)
            return None if x is None else x * _UNITS[b.id]
        x, y = _eval_units(a), _eval_units(b)
        return None if x is None or y is None else x * y
    return None


def _default_model_py() -> Path:
    from flycircuit.config import project_root

    return project_root() / "external" / "Drosophila_brain_model" / "model.py"


@dataclass
class LIFResult:
    spike_steps: np.ndarray  # int64 step index of every spike (time = step * dt)
    spike_neurons: np.ndarray  # int64 neuron index
    n_neurons: int
    steps: int
    dt: float
    rec_idx: np.ndarray | None = None
    v_trace: np.ndarray | None = None  # [steps, n_rec] state at START of each step (mV)
    g_trace: np.ndarray | None = None

    def counts(self) -> np.ndarray:
        return np.bincount(self.spike_neurons, minlength=self.n_neurons)

    def rates_hz(self) -> np.ndarray:
        return self.counts() / (self.steps * self.dt / 1000.0)

    def trains(self) -> dict[int, np.ndarray]:
        out: dict[int, list[int]] = {}
        for s, n in zip(self.spike_steps.tolist(), self.spike_neurons.tolist(), strict=True):
            out.setdefault(n, []).append(s)
        return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def poisson_events(
    targets: np.ndarray, steps: int, rate_hz: float, dt_ms: float, seed: int | np.random.SeedSequence
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit Bernoulli(p = rate*dt) event stream per target per step (shared by both engines).

    One independent stream per target NEURON ID (spawn_key = base key + neuron id), so a target's
    events do not change when other targets are added/removed or branches draw other numbers.
    Returns (event_steps, event_target_neuron) sorted.
    """
    targets = np.asarray(targets, dtype=np.int64)
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    p = rate_hz * dt_ms / 1000.0
    es, et = [], []
    for t in targets.tolist():
        kid = np.random.SeedSequence(ss.entropy, spawn_key=(*ss.spawn_key, int(t)))
        hit = np.flatnonzero(np.random.default_rng(kid).random(steps) < p)
        es.append(hit)
        et.append(np.full(len(hit), t, dtype=np.int64))
    if not es:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    s, t = np.concatenate(es).astype(np.int64), np.concatenate(et)
    o = np.lexsort((t, s))
    return s[o], t[o]


def silence_outgoing(W: sp.spmatrix, neurons) -> sp.csc_matrix:
    """Published lesion semantics: zero ALL synapses FROM the given presynaptic neurons."""
    Wc = sp.csc_matrix(W, dtype=np.float64, copy=True)
    for j in np.asarray(neurons, dtype=np.int64).tolist():
        Wc.data[Wc.indptr[j] : Wc.indptr[j + 1]] = 0.0
    Wc.eliminate_zeros()
    return Wc


def simulate(
    W_mV: sp.spmatrix,
    steps: int,
    params: LIFParams | None = None,
    ext_steps: np.ndarray | None = None,
    ext_targets: np.ndarray | None = None,
    no_refractory: np.ndarray | None = None,
    record: np.ndarray | None = None,
    ext_weight: np.ndarray | float | None = None,
    background: tuple[np.ndarray, np.ndarray, float] | None = None,
    background_seed: int = 0,
) -> LIFResult:
    """Run the LIF network.

    W_mV: sparse ``[post, pre]`` matrix of signed synaptic weights IN mV (sign*contacts*w_syn).
    ext_steps/ext_targets: explicit external events (step index, target neuron); each adds
        ``ext_weight`` (default ``params.w_ext``) to v.
    no_refractory: neurons with refractory period 0 (the published code does this for Poisson targets).
    record: neuron indices whose (v, g) at the START of every step are stored.
    background: (post_idx, weight_mV, rate_hz) - "background drive" boundary treatment: each entry
        is one cut upstream pair firing as an independent Poisson process at ``rate_hz`` that adds
        ``weight_mV`` (signed, = sign*contacts*w_syn) to ``g`` of ``post_idx`` (synaptic, like an
        in-circuit spike, without delay modelling). This is a stand-in for missing upstream
        dynamics, not an equivalent of them.
    """
    p = params or LIFParams()
    n = W_mV.shape[0]
    if W_mV.shape != (n, n):
        raise ValueError("W must be square [post, pre]")
    Wc = sp.csc_matrix(W_mV, dtype=np.float64)
    D = p.n_delay
    e_m, e_g = np.exp(-p.dt / p.t_mbr), np.exp(-p.dt / p.tau)
    k_a = p.tau / (p.tau - p.t_mbr)
    v = np.full(n, p.v_0)
    g = np.zeros(n)
    rfc_len = np.full(n, p.n_refractory, dtype=np.int64)
    if no_refractory is not None and len(no_refractory):
        rfc_len[np.asarray(no_refractory, dtype=np.int64)] = 0
    # refr_free_at[i]: first step index at which neuron i is no longer refractory
    free_at = np.zeros(n, dtype=np.int64)
    inbox = np.zeros((D + 1, n))
    # external events grouped by step
    if ext_steps is None:
        ext_steps = np.empty(0, np.int64)
        ext_targets = np.empty(0, np.int64)
    es = np.asarray(ext_steps, dtype=np.int64)
    et = np.asarray(ext_targets, dtype=np.int64)
    order = np.argsort(es, kind="stable")
    es, et = es[order], et[order]
    if ext_weight is None:
        ew = np.full(len(es), p.w_ext)
    elif np.isscalar(ext_weight):
        ew = np.full(len(es), float(ext_weight))
    else:
        ew = np.asarray(ext_weight, dtype=float)[order]
    bounds = np.searchsorted(es, np.arange(steps + 1))
    bg = None
    if background is not None and len(background[0]):
        bpost, bw, brate = background
        bg = (np.asarray(bpost, dtype=np.int64), np.asarray(bw, dtype=float),
              len(bpost) * brate * p.dt / 1000.0,
              np.random.default_rng(np.random.SeedSequence(background_seed)))
    rec = None if record is None else np.asarray(record, dtype=np.int64)
    v_tr = None if rec is None else np.zeros((steps, len(rec)))
    g_tr = None if rec is None else np.zeros((steps, len(rec)))
    sp_steps: list[np.ndarray] = []
    sp_neu: list[np.ndarray] = []
    for k in range(steps):
        if rec is not None:
            v_tr[k] = v[rec]
            g_tr[k] = g[rec]
        act = free_at <= k
        a = g * k_a
        v = np.where(act, p.v_0 + (v - p.v_0 - a) * e_m + a * e_g, v)
        g = np.where(act, g * e_g, g)
        spk = act & (v > p.v_th)
        # delivery at this step (synapses slot)
        slot = k % (D + 1)
        # conditional write: v and g are "unless refractory" variables, so Brian2 DISCARDS any
        # write (synaptic g += w, external v += w) to a refractory neuron (found by refcheck
        # tier 1 and confirmed in brian2/groups/neurongroup.py: set_conditional_write).
        g = g + np.where(act, inbox[slot], 0.0)
        inbox[slot] = 0.0
        lo, hi = bounds[k], bounds[k + 1]
        if hi > lo:
            tg_ = et[lo:hi]
            okm = act[tg_]
            np.add.at(v, tg_[okm], ew[lo:hi][okm])
        if bg is not None:
            bpost, bw, lam, rg = bg
            kk = rg.poisson(lam)  # expected events this step over all cut pairs
            if kk:
                pick = rg.integers(0, len(bpost), size=kk)
                okm = act[bpost[pick]]
                np.add.at(g, bpost[pick][okm], bw[pick][okm])
        idx = np.flatnonzero(spk)
        if len(idx):
            # schedule outgoing (arrives D steps later); D == 0 delivers within this step's slot
            col = Wc[:, idx].sum(axis=1).A1
            inbox[(k + D) % (D + 1)] += col
            if D == 0:
                g = g + np.where(act, inbox[slot], 0.0)
                inbox[slot] = 0.0
            v[idx] = p.v_rst
            g[idx] = 0.0
            free_at[idx] = k + rfc_len[idx]
            sp_steps.append(np.full(len(idx), k, dtype=np.int64))
            sp_neu.append(idx.astype(np.int64))
    st = np.concatenate(sp_steps) if sp_steps else np.empty(0, np.int64)
    ne = np.concatenate(sp_neu) if sp_neu else np.empty(0, np.int64)
    return LIFResult(st, ne, n, steps, p.dt, rec, v_tr, g_tr)


def weights_mV(signed_contacts: np.ndarray, pre: np.ndarray, post: np.ndarray, n: int,
               params: LIFParams | None = None) -> sp.csc_matrix:
    """Build ``W[post, pre]`` in mV from edge-aligned signed contact counts."""
    p = params or LIFParams()
    return sp.csc_matrix((np.asarray(signed_contacts, float) * p.w_syn, (post, pre)), shape=(n, n))


def describe() -> dict[str, Any]:
    return {"engine": "flycircuit.models.lif", "integrator": "exact exponential per dt",
            "schedule": ["record", "integrate", "threshold", "deliver", "reset"],
            "reference": "external/Drosophila_brain_model@91bdd1e"}
