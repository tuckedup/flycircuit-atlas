"""Connectome-constrained sparse rate RNN.

Dynamics (per step, ``W[post, pre]`` convention)::

    h_next = (1 - alpha) * h + alpha * phi(W @ h + B @ x + M @ z + b)
    W[post, pre] = edge_mask * sign[pre] * gain * f(contact_count)
    z_next = (1 - alpha_z) * z + alpha_z * u        (engineered context state)

``alpha = dt / tau`` must satisfy ``0 < alpha <= 1`` (explicit-Euler stability of the leaky
integrator); anything else raises. ``f`` is the coupling prior (identity / log1p / sqrt).

W is NEVER materialised densely: only an edge-indexed value vector ``w[E]`` exists. Two verified
backends are provided: ``scatter`` (gather + index_add_) and ``coo`` (torch.sparse.mm with a
differentiable COO tensor). Both are tested against a tiny dense oracle (forward and grads).

``z`` and the encoder/decoder are ENGINEERED interface variables, not inferred biology.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

PRIORS = ("identity", "log1p", "sqrt")
SIGN_POLICIES = ("excitatory", "inhibitory", "zero", "random_fixed")
BACKENDS = ("scatter", "coo")
GAIN_MODES = ("pair", "class")


def coupling_prior(count: torch.Tensor, prior: str) -> torch.Tensor:
    if prior == "identity":
        return count
    if prior == "log1p":
        return torch.log1p(count)
    if prior == "sqrt":
        return torch.sqrt(count)
    raise ValueError(f"unknown prior {prior!r}; allowed {PRIORS}")


def resolve_signs(
    sign: np.ndarray | None, n: int, unknown_policy: str = "excitatory", seed: int = 0
) -> np.ndarray:
    """Per-presynaptic-node sign in {-1,+1,0}. Input ``sign``: +1/-1 known, 0 unknown, None = all
    unknown. ``unknown_policy`` decides unknown nodes (a sensitivity axis, not a biological claim):
    excitatory(+1) | inhibitory(-1) | zero (silent output) | random_fixed (seeded +/-1)."""
    if unknown_policy not in SIGN_POLICIES:
        raise ValueError(f"unknown_policy must be one of {SIGN_POLICIES}")
    s = np.zeros(n, dtype=np.float32) if sign is None else np.asarray(sign, dtype=np.float32).copy()
    if s.shape != (n,) or not np.isin(s, (-1, 0, 1)).all():
        raise ValueError("sign must be shape (n,) with values in {-1,0,+1}")
    unk = s == 0
    if unknown_policy == "excitatory":
        s[unk] = 1.0
    elif unknown_policy == "inhibitory":
        s[unk] = -1.0
    elif unknown_policy == "random_fixed":
        rng = np.random.default_rng(seed)
        s[unk] = rng.choice([-1.0, 1.0], size=int(unk.sum()))
    return s


@dataclass
class ModelSpec:
    n_in: int
    n_out: int
    alpha: float | None = None  # if None derived from dt/tau
    dt: float = 1.0
    tau: float = 5.0
    prior: str = "log1p"
    gain_mode: str = "pair"
    nonneg_gain: bool = True  # True: gain=softplus(raw)>=0 (sign fixed); False: free signed weight
    backend: str = "scatter"
    trainable_encoder: bool = False
    trainable_decoder: bool = True
    z_dim: int = 0
    alpha_z: float = 0.2
    nonlinearity: str = "tanh"
    init_gain: float = 0.1
    seed: int = 0
    extra: dict = field(default_factory=dict)

    def resolved_alpha(self) -> float:
        a = self.dt / self.tau if self.alpha is None else float(self.alpha)
        if not (0.0 < a <= 1.0):
            raise ValueError(f"alpha={a} outside (0,1]; need dt<=tau (dt={self.dt}, tau={self.tau})")
        return a


def _softplus_inv(y: float) -> float:
    return float(np.log(np.expm1(y)))


class SparseConnectomeRNN(nn.Module):
    """Sparse rate RNN whose recurrent support is fixed by (pre, post) edge indices."""

    kind = "connectome_sparse_rnn"

    def __init__(
        self,
        n: int,
        pre: np.ndarray,
        post: np.ndarray,
        count: np.ndarray,
        spec: ModelSpec,
        sign: np.ndarray | None = None,
        unknown_sign_policy: str = "excitatory",
        edge_class: np.ndarray | None = None,
        sensory_idx: np.ndarray | None = None,
        readout_idx: np.ndarray | None = None,
        trainable_edge_mask: np.ndarray | None = None,
    ):
        super().__init__()
        if spec.prior not in PRIORS:
            raise ValueError(f"prior must be in {PRIORS}")
        if spec.backend not in BACKENDS:
            raise ValueError(f"backend must be in {BACKENDS}")
        if spec.gain_mode not in GAIN_MODES:
            raise ValueError(f"gain_mode must be in {GAIN_MODES}")
        self.spec = spec
        self.n = int(n)
        self.alpha = spec.resolved_alpha()
        pre_t = torch.as_tensor(np.asarray(pre), dtype=torch.long)
        post_t = torch.as_tensor(np.asarray(post), dtype=torch.long)
        e = len(pre_t)
        self.n_edges = e
        self.register_buffer("pre", pre_t)
        self.register_buffer("post", post_t)
        cnt = torch.as_tensor(np.asarray(count), dtype=torch.float32)
        self.register_buffer("count", cnt)
        self.register_buffer("prior_val", coupling_prior(cnt, spec.prior))
        s = resolve_signs(sign, n, unknown_sign_policy, spec.seed)
        self.register_buffer("sign_node", torch.as_tensor(s))
        tm = np.ones(e, dtype=bool) if trainable_edge_mask is None else np.asarray(trainable_edge_mask)
        self.register_buffer("edge_trainable", torch.as_tensor(tm))

        # gains
        if spec.gain_mode == "pair":
            n_gain = e
            self.register_buffer("class_of_edge", torch.arange(e))
        else:
            if edge_class is None:
                raise ValueError("gain_mode='class' requires edge_class")
            ec = np.asarray(edge_class)
            _, inv = np.unique(ec, return_inverse=True)
            n_gain = int(inv.max()) + 1 if e else 0
            self.register_buffer("class_of_edge", torch.as_tensor(inv, dtype=torch.long))
        init = spec.init_gain
        raw0 = _softplus_inv(init) if spec.nonneg_gain else init
        self.raw_gain = nn.Parameter(torch.full((n_gain,), raw0))
        self.n_gain_params = n_gain

        self.register_buffer("sensory_idx", self._idx(sensory_idx, min(spec.n_in, n)))
        self.register_buffer("readout_idx", self._idx(readout_idx, min(spec.n_out, n), tail=True))
        # fixed encoder: x_k -> sensory node k (scaled identity); trainable: dense n_in x n_sens
        n_sens = len(self.sensory_idx)
        enc = torch.zeros(spec.n_in, n_sens)
        for k in range(min(spec.n_in, n_sens)):
            enc[k, k] = 1.0
        self.encoder = nn.Parameter(enc, requires_grad=spec.trainable_encoder)
        dec = torch.zeros(len(self.readout_idx), spec.n_out)
        g = torch.Generator().manual_seed(spec.seed)
        dec.copy_(torch.randn(dec.shape, generator=g) / max(1, dec.shape[0]) ** 0.5)
        self.decoder = nn.Parameter(dec, requires_grad=spec.trainable_decoder)
        self.bias = nn.Parameter(torch.zeros(n))
        if spec.z_dim > 0:
            gz = torch.randn(spec.z_dim, n, generator=g) * 0.1
            self.M = nn.Parameter(gz, requires_grad=False)  # engineered, fixed modulatory map
        else:
            self.M = None
        self.phi = {"tanh": torch.tanh, "relu": torch.relu, "identity": lambda v: v}[
            spec.nonlinearity
        ]

    def _idx(self, idx, k, tail=False):
        if idx is None:
            idx = np.arange(self.n - k, self.n) if tail else np.arange(k)
        return torch.as_tensor(np.asarray(idx), dtype=torch.long)

    # ---- weights ------------------------------------------------------
    def gain(self) -> torch.Tensor:
        g = self.raw_gain
        g = torch.nn.functional.softplus(g) if self.spec.nonneg_gain else g
        return g[self.class_of_edge]

    def edge_values(self) -> torch.Tensor:
        """w[E] = trainable? sign[pre]*gain*f(count). Non-trainable edges use frozen init gain."""
        g = self.gain()
        if not bool(self.edge_trainable.all()):
            frozen = torch.full_like(g, self.spec.init_gain)
            g = torch.where(self.edge_trainable, g, frozen.detach())
        return self.sign_node[self.pre] * g * self.prior_val

    def recurrent(self, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """(B,N) -> (B,N): out[b,post] = sum_e w[e]*h[b,pre[e]]."""
        if self.spec.backend == "scatter":
            out = torch.zeros_like(h)
            return out.index_add(1, self.post, h[:, self.pre] * w)
        idx = torch.stack([self.post, self.pre])
        w_sp = torch.sparse_coo_tensor(
            idx, w, (self.n, self.n), check_invariants=False
        ).coalesce()
        return torch.sparse.mm(w_sp, h.t()).t()

    # ---- dynamics -----------------------------------------------------
    def init_state(self, batch: int, device=None) -> dict[str, torch.Tensor]:
        dev = device or self.bias.device
        st = {"h": torch.zeros(batch, self.n, device=dev)}
        if self.spec.z_dim > 0:
            st["z"] = torch.zeros(batch, self.spec.z_dim, device=dev)
        return st

    def step(self, x, state, w, u=None):
        h = state["h"]
        drive = self.recurrent(h, w) + self.bias
        inp = x @ self.encoder  # (B, n_sens)
        drive = drive.index_add(1, self.sensory_idx, inp)
        new = {}
        if self.M is not None:
            z = state["z"]
            drive = drive + z @ self.M
            uu = torch.zeros_like(z) if u is None else u
            new["z"] = (1 - self.spec.alpha_z) * z + self.spec.alpha_z * uu
        new["h"] = (1 - self.alpha) * h + self.alpha * self.phi(drive)
        y = new["h"][:, self.readout_idx] @ self.decoder
        return y, new

    def forward(self, x_seq, state=None, u_seq=None):
        """x_seq: (T,B,n_in) -> y_seq (T,B,n_out), final state."""
        if state is None:
            state = self.init_state(x_seq.shape[1], x_seq.device)
        w = self.edge_values()
        ys = []
        for t in range(x_seq.shape[0]):
            y, state = self.step(x_seq[t], state, w, None if u_seq is None else u_seq[t])
            ys.append(y)
        return torch.stack(ys), state

    # ---- reports ------------------------------------------------------
    def param_report(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "kind": self.kind,
            "trainable_params": trainable,
            "recurrent_gain_params": self.n_gain_params if self.raw_gain.requires_grad else 0,
            "encoder_params": self.encoder.numel() if self.encoder.requires_grad else 0,
            "decoder_params": self.decoder.numel() if self.decoder.requires_grad else 0,
            "bias_params": self.bias.numel(),
            "active_state_size": self.n + self.spec.z_dim,
            "n_edges": self.n_edges,
            "flops_per_step": 2 * self.n_edges + 4 * self.n,
        }

    @staticmethod
    def sparse_from_graph(g, spec: ModelSpec, **kw) -> SparseConnectomeRNN:
        """Build from a VersionedGraph (edge arrays only; no dense conversion)."""
        sens = np.flatnonzero(g.node_masks["sensory_input"]) if g.node_masks["sensory_input"].any() else None
        read = np.flatnonzero(g.node_masks["action_readout"]) if g.node_masks["action_readout"].any() else None
        return SparseConnectomeRNN(
            g.n, g.pre, g.post, g.count, spec, sensory_idx=sens, readout_idx=read,
            trainable_edge_mask=g.edge_masks["trainable"] if g.edge_masks["trainable"].any() else None,
            **kw,
        )
