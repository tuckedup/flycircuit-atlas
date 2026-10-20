"""Paired simulation of a SparseConnectomeRNN with optional intervention.

The engine reimplements the model's step using the model's own components (recurrent op,
encoder, decoder, bias, nonlinearity) so interventions can act between the update and the
readout. With ``spec=None`` it reproduces ``model.forward`` exactly (tested).

Pairing: baseline and intervention branches build ``RngStreams(seed, run)`` independently, so the
initial state, stimulus events and observation noise are identical draws; the intervention's own
randomness (``disruption``) is a separate stream and never shifts the shared ones.

Modes
-----
open_loop    inputs are exogenous replays; actions do not alter future observations.
closed_loop  previous readout y_{t-1} is fed back into the first input channels
             (``feedback_gain``), an ENGINEERED reafference loop: intervention-induced action
             changes alter later observations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from flycircuit.interventions.rng import RngStreams
from flycircuit.interventions.spec import (
    CLAMP_KINDS,
    InterventionSpec,
)


@dataclass(frozen=True)
class SimConfig:
    steps: int = 60
    batch: int = 1
    stim_prob: float = 0.3
    stim_amp: float = 1.0
    obs_noise_std: float = 0.05
    init_state_std: float = 0.05
    closed_loop: bool = False
    feedback_gain: float = 0.5


def make_events(model, streams: RngStreams, cfg: SimConfig) -> dict[str, torch.Tensor]:
    """Shared stochastic events, drawn from named streams (identical across branches)."""
    n_in = model.spec.n_in
    t, b = cfg.steps, cfg.batch
    rs = streams.stream("stimulus")
    stim = (rs.random((t, b, n_in)) < cfg.stim_prob).astype(np.float32) * cfg.stim_amp
    noise = streams.stream("obs_noise").standard_normal((t, b, n_in)).astype(np.float32)
    h0 = streams.stream("init").standard_normal((b, model.n)).astype(np.float32)
    return {
        "stim": torch.from_numpy(stim),
        "obs_noise": torch.from_numpy(noise) * cfg.obs_noise_std,
        "h0": torch.from_numpy(h0) * cfg.init_state_std,
    }


def edited_edge_values(model, w0: torch.Tensor, spec: InterventionSpec) -> torch.Tensor:
    """Edge-level editing (deletion / gain scaling). Node clamps and drive kinds leave w intact."""
    tgt = torch.as_tensor(np.asarray(spec.targets, dtype=np.int64))
    out_m = torch.isin(model.pre.cpu(), tgt).to(w0.device)
    in_m = torch.isin(model.post.cpu(), tgt).to(w0.device)
    d = float(spec.dose)
    k = spec.kind
    if k == "delete_outgoing":
        return w0 * (1.0 - d * out_m)
    if k == "delete_incoming":
        return w0 * (1.0 - d * in_m)
    if k == "disconnect_both":
        return w0 * (1.0 - d * (out_m | in_m))
    if k == "gain_scale_outgoing":
        return w0 * (1.0 + (d - 1.0) * out_m)
    if k == "gain_scale_incoming":
        return w0 * (1.0 + (d - 1.0) * in_m)
    return w0


@torch.no_grad()
def run_branch(
    model,
    streams: RngStreams,
    cfg: SimConfig,
    spec: InterventionSpec | None = None,
    plasticity: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor] | None = None,
) -> dict:
    """Simulate one branch. Returns readout ``y`` (T,B,n_out) and per-node mean |h| (N,)."""
    ev = make_events(model, streams, cfg)
    dev = model.bias.device
    h = ev["h0"].to(dev)
    z = torch.zeros(cfg.batch, model.spec.z_dim, device=dev) if model.M is not None else None
    w_base = model.edge_values().detach()
    t0, t1 = (0, 0) if spec is None else spec.window(cfg.steps)
    tgt = None if spec is None else torch.as_tensor(np.asarray(spec.targets), device=dev)
    w_edit = None
    if spec is not None and spec.kind not in CLAMP_KINDS + ("drive_excitation", "transient_disruption"):
        w_edit = edited_edge_values(model, w_base, spec)
    dis = streams.stream("disruption") if spec is not None else None  # separate stream
    y_prev = torch.zeros(cfg.batch, model.spec.n_out, device=dev)
    ys, abs_h = [], torch.zeros(model.n, device=dev)
    clamped_trace = []
    n_fb = min(model.spec.n_out, model.spec.n_in)
    for t in range(cfg.steps):
        x = ev["stim"][t].to(dev) + ev["obs_noise"][t].to(dev)
        if cfg.closed_loop:
            x = x.clone()
            x[:, :n_fb] += cfg.feedback_gain * y_prev[:, :n_fb]
        active = spec is not None and t0 <= t < t1
        w = w_base
        if plasticity is not None:
            w_base = plasticity(w_base, h, t)
            w = w_base
        if active and w_edit is not None:
            w = edited_edge_values(model, w_base, spec) if plasticity is not None else w_edit
        drive = model.recurrent(h, w) + model.bias
        drive = drive.index_add(1, model.sensory_idx, x @ model.encoder)
        if model.M is not None:
            drive = drive + z @ model.M
            z = (1 - model.spec.alpha_z) * z
        if active and spec.kind == "drive_excitation":
            drive[:, tgt] += float(spec.dose)
        if spec is not None and spec.kind == "transient_disruption":
            # draw every step (active or not) so the stream position is window-independent
            nz = torch.from_numpy(
                dis.standard_normal((cfg.batch, len(spec.targets))).astype(np.float32)
            ).to(dev)
            if active:
                drive[:, tgt] += float(spec.dose) * nz
        h = (1 - model.alpha) * h + model.alpha * model.phi(drive)
        if active and spec.kind in CLAMP_KINDS:
            h[:, tgt] = float(spec.dose)
            clamped_trace.append(h[:, tgt].clone())
        y = h[:, model.readout_idx] @ model.decoder
        y_prev = y
        ys.append(y)
        abs_h += h.abs().mean(0)
    out = {"y": torch.stack(ys).cpu(), "mean_abs_h": (abs_h / cfg.steps).cpu()}
    if clamped_trace:
        out["clamped_trace"] = torch.stack(clamped_trace).cpu()
    return out


def effect_metrics(base_y: torch.Tensor, int_y: torch.Tensor, start: int) -> dict[str, float]:
    """Per-run effect of an intervention on the readout over steps ``start:``.

    delta_readout: mean(y_int) - mean(y_base) (signed, readout units).
    deviation: ||y_int - y_base||_F / ||y_base||_F (relative, unitless).
    """
    yb, yi = base_y[start:], int_y[start:]
    nb = float(yb.norm())
    return {
        "delta_readout": float(yi.mean() - yb.mean()),
        "deviation": float((yi - yb).norm() / (nb + 1e-12)),
        "baseline_readout_mean": float(yb.mean()),
        "baseline_readout_norm": nb,
    }
