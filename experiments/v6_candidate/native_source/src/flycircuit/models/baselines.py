"""Baselines sharing the connectome model's interface: ``forward(x_seq, state) -> (y_seq, state)``.

* ``GRUBaseline``            practical dense GRU (hidden size chosen by matching, NOT 140k).
* ``RandomSparseRNN``        same class as the connectome model on a random simple digraph
                             (see ``flycircuit.graph.controls.random_sparsity``).
* ``EncoderDecoderOnly``     no recurrent core: per-step MLP of the input.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from flycircuit.graph.controls import random_sparsity
from flycircuit.models.sparse_rnn import ModelSpec, SparseConnectomeRNN


class GRUBaseline(nn.Module):
    kind = "gru"

    def __init__(self, n_in: int, n_out: int, hidden: int, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden = hidden
        self.n_in, self.n_out = n_in, n_out
        self.enc = nn.Linear(n_in, hidden)
        self.cell = nn.GRU(hidden, hidden)
        self.dec = nn.Linear(hidden, n_out)

    def init_state(self, batch, device=None):
        return {"h": torch.zeros(1, batch, self.hidden, device=device or self.enc.weight.device)}

    def forward(self, x_seq, state=None, u_seq=None):
        if state is None:
            state = self.init_state(x_seq.shape[1], x_seq.device)
        out, h = self.cell(torch.relu(self.enc(x_seq)), state["h"])
        return self.dec(out), {"h": h}

    def param_report(self) -> dict:
        h = self.hidden
        enc = sum(p.numel() for p in self.enc.parameters())
        dec = sum(p.numel() for p in self.dec.parameters())
        core = sum(p.numel() for p in self.cell.parameters())
        return {
            "kind": self.kind, "trainable_params": enc + dec + core,
            "recurrent_gain_params": core, "encoder_params": enc, "decoder_params": dec,
            "bias_params": 0, "active_state_size": h,
            "n_edges": 3 * h * h, "flops_per_step": 6 * h * (h + h) + 2 * (self.n_in * h + h * self.n_out),
        }


class EncoderDecoderOnly(nn.Module):
    kind = "encoder_decoder_only"

    def __init__(self, n_in: int, n_out: int, hidden: int, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden = hidden
        self.n_in, self.n_out = n_in, n_out
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, n_out))

    def init_state(self, batch, device=None):
        return {}

    def forward(self, x_seq, state=None, u_seq=None):
        return self.net(x_seq), {}

    def param_report(self) -> dict:
        p = sum(q.numel() for q in self.parameters())
        return {
            "kind": self.kind, "trainable_params": p, "recurrent_gain_params": 0,
            "encoder_params": self.n_in * self.hidden + self.hidden,
            "decoder_params": self.hidden * self.n_out + self.n_out, "bias_params": 0,
            "active_state_size": 0, "n_edges": 0,
            "flops_per_step": 2 * (self.n_in * self.hidden + self.hidden * self.n_out),
        }


def random_sparse_rnn(g, spec: ModelSpec, seed: int = 0, **kw) -> SparseConnectomeRNN:
    """Random sparse RNN with the connectome's N and E (uniform random simple digraph)."""
    ng = random_sparsity(g, seed=seed)
    m = SparseConnectomeRNN.sparse_from_graph(ng.graph, spec, **kw)
    m.kind = "random_sparse_rnn"
    return m


def match_gru_hidden(target_params: int, n_in: int, n_out: int) -> int:
    """Smallest GRU hidden size whose total trainable params >= target (parameter matching)."""
    h = 1
    while True:
        total = (n_in * h + h) + 3 * (2 * h * h + 2 * h) + (h * n_out + n_out)
        if total >= target_params:
            return h
        h += 1


def match_encdec_hidden(target_params: int, n_in: int, n_out: int) -> int:
    h = 1
    while (n_in * h + h) + (h * n_out + n_out) < target_params:
        h += 1
    return h


def match_gru_hidden_flops(target_flops: int, n_in: int, n_out: int) -> int:
    h = 1
    while 6 * h * (2 * h) + 2 * (n_in * h + h * n_out) < target_flops:
        h += 1
    return h


def matching_report(reference, n_in: int, n_out: int, seed: int = 0) -> dict:
    """Parameter- and compute-matched GRU / enc-dec vs a reference model; discrepancies exposed.
    Encoder/decoder counts and active state size are reported separately, never merged."""
    ref = reference.param_report()
    rows = {"reference": ref}
    hp = match_gru_hidden(ref["trainable_params"], n_in, n_out)
    hf = match_gru_hidden_flops(ref["flops_per_step"], n_in, n_out)
    he = match_encdec_hidden(ref["trainable_params"], n_in, n_out)
    rows["gru_param_matched"] = GRUBaseline(n_in, n_out, hp, seed).param_report()
    rows["gru_compute_matched"] = GRUBaseline(n_in, n_out, hf, seed).param_report()
    rows["encdec_param_matched"] = EncoderDecoderOnly(n_in, n_out, he, seed).param_report()
    disc = {}
    for name, r in rows.items():
        if name == "reference":
            continue
        disc[name] = {
            "param_ratio_vs_ref": r["trainable_params"] / max(1, ref["trainable_params"]),
            "flops_ratio_vs_ref": r["flops_per_step"] / max(1, ref["flops_per_step"]),
            "active_state_vs_ref": r["active_state_size"] - ref["active_state_size"],
        }
    rows["discrepancies"] = disc
    rows["note"] = (
        "Parameter-matched GRU generally has far lower compute/state than the connectome model; "
        "compute-matched GRU has more params. Neither is 'the' fair baseline: report both."
    )
    return rows


__all__ = [
    "EncoderDecoderOnly", "GRUBaseline", "match_encdec_hidden", "match_gru_hidden",
    "match_gru_hidden_flops", "matching_report", "np", "random_sparse_rnn",
]
