"""Three-factor eligibility-trace plasticity with a TD value function.

Indexing: plastic edges are ``e = 0..P-1`` (a subset of the model's edge list, in model order);
``pre[e]``/``post[e]`` are node indices; ``post_pos[e]`` indexes into the readout set.
Units: one time unit = one neural step (``dt`` of the model; default 1 ms with tau=5 ms).

Update order inside a trial (B = 1, state and eligibility reset to zero at trial start):
  for t in 0..T-1:
      h_{t+1} = model step(h_t, x_t)                      # (pre activity h_t, then post activity)
      elig   <- decay * elig + h_t[pre] * eps_t[post]   # local pre x postsynaptic perturbation
  after the trial (terminal, one decision): delta = r + gamma * V(next) - V(s), V(next) := 0
  gain <- clip(gain + lr * delta * elig, lo, hi)          # modulation reaches ONLY plastic edges
  value:  v <- v + lr_v * delta * phi(s)                  # linear TD(0) critic on [cue, ctx, 1]

``eps_t`` is Gaussian exploration noise injected on the (z-scored) readout nodes; the rule is
reward-modulated node-perturbation learning (Fiete & Seung style). It is a scalar-TD abstraction,
NOT a claim that each dopamine neuron carries a scalar TD error (compartment-specific or
multi-timescale variants are controlled extensions).
Gains, eligibility, decoder and optimizer state are separate objects so resets are separable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

RULES = ("np_v1", "np_window", "comp_feedback")


@dataclass
class PlasticityConfig:
    lr: float = 0.3
    decay: float = 0.9
    gain_lo: float = 0.0
    gain_hi: float = 1.0
    gamma: float = 0.9
    lr_value: float = 0.05
    modulation: bool = True  # False -> delta forced to 0 (modulation-disabled control)
    # ---- v2 (protocol learning-v2); defaults reproduce the v1 rule exactly -----------------
    rule: str = "np_v1"
    #   np_v1          v1 node-perturbation, eligibility accumulated over ALL steps (kept for v1
    #                  reproducibility; diagnosed as uncorrelated with the reward gradient)
    #   np_window      node-perturbation with eligibility accumulated in the response window only
    #   comp_feedback  compartment-specific three-factor rule: eligibility trace of local
    #                  pre x post-derivative signal (decay = 1 - alpha), third factor = scalar TD
    #                  delta, times a per-postsynaptic-compartment feedback signal derived from the
    #                  FIXED decoder (engineered 'valence' feedback; an e-prop-style variant)
    homeostasis: float = 0.0  # per-trial leak of plastic gains toward init (isolated variant)
    shuffle_feedback: bool = False  # control: compartment feedback permuted across compartments
    optimizer: str = "sgd"  # sgd | rmsprop: per-synapse RMS-normalised step (parity with Adam in BPTT)
    rms_beta: float = 0.99
    rms_eps: float = 1e-8

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError(f"rule must be in {RULES}")
        if self.optimizer not in ("sgd", "rmsprop"):
            raise ValueError("optimizer must be sgd|rmsprop")


class ValueFunction:
    """Linear critic V(s) = v . phi(s); phi = [mean cue input over time, ctx one-hot, 1]."""

    def __init__(self, n_in: int, lr: float = 0.05):
        self.v = np.zeros(n_in + 1, dtype=np.float64)
        self.lr = lr

    @staticmethod
    def features(x_seq: np.ndarray) -> np.ndarray:
        """x_seq (T,1,n_in) -> phi (n_in+1,)"""
        return np.concatenate([x_seq[:, 0].max(0), [1.0]]).astype(np.float64)

    def value(self, phi: np.ndarray) -> float:
        return float(self.v @ phi)

    def td_error(self, reward: float, phi: np.ndarray, gamma: float = 0.9,
                 next_phi: np.ndarray | None = None) -> float:
        nxt = 0.0 if next_phi is None else self.value(next_phi)
        return reward + gamma * nxt - self.value(phi)

    def update(self, delta: float, phi: np.ndarray) -> None:
        self.v += self.lr * delta * phi


class PlasticSynapses:
    """Plastic gains + eligibility for a subset of edges; mutated outside autograd."""

    def __init__(self, plastic_mask: np.ndarray, pre: torch.Tensor, post: torch.Tensor,
                 readout_idx: torch.Tensor, init_gain: float, cfg: PlasticityConfig, device=None,
                 sign_node: torch.Tensor | None = None):
        self.cfg = cfg
        self._sign_node = sign_node
        self.init_gain = init_gain
        self.idx = torch.as_tensor(np.flatnonzero(plastic_mask), dtype=torch.long, device=device)
        self.pre = pre[self.idx]
        self.post = post[self.idx]
        pos = {int(r): i for i, r in enumerate(readout_idx.tolist())}
        self.post_pos = torch.as_tensor([pos[int(p)] for p in self.post.tolist()],
                                        dtype=torch.long, device=device)
        self.n_plastic = len(self.idx)
        self.pre_sign = (torch.ones(self.n_plastic, device=device) if sign_node is None
                         else sign_node[self.pre].to(device))
        self.reset_gain()
        self.reset_eligibility()
        self.reset_optimizer()

    def reset_gain(self) -> None:
        self.gain = torch.full((self.n_plastic,), self.init_gain, device=self.idx.device)

    def reset_eligibility(self) -> None:
        self.elig = torch.zeros(self.n_plastic, device=self.idx.device)
        self.tr = torch.zeros(self.n_plastic, device=self.idx.device)

    def reset_optimizer(self) -> None:
        """Per-synapse second-moment estimate of the RMS-normalised update (optimizer state)."""
        self.sq = torch.zeros(self.n_plastic, device=self.idx.device)

    @torch.no_grad()
    def accumulate(self, h_prev: torch.Tensor, post_signal: torch.Tensor) -> None:
        """elig <- decay*elig + h_prev[pre] * post_signal[post]; B = 1. ``post_signal`` (B, n_readout)
        is the postsynaptic exploration perturbation eps (node perturbation), in readout z-units."""
        pre_eff = h_prev[0, self.pre] * self.pre_sign  # effective sign of the synaptic effect
        self.elig = self.cfg.decay * self.elig + pre_eff * post_signal[0, self.post_pos]

    @torch.no_grad()
    def accumulate_trace(self, h_prev: torch.Tensor, phi_prime_ro: torch.Tensor, alpha: float,
                         in_window: bool, n_window: int) -> None:
        """comp_feedback rule. Local trace of d h_post / d gain (recurrent terms dropped):
            tr <- (1-alpha)*tr + alpha * phi'(post) * sign(pre) * h_pre        (every step)
        and the response-window mean of the trace is the eligibility read by the update:
            elig += tr / n_window   if the step is inside the response window.
        ``phi_prime_ro`` (1, n_readout) is the post-synaptic transfer-function derivative."""
        if not hasattr(self, "tr") or self.tr.shape != self.elig.shape:
            self.tr = torch.zeros_like(self.elig)
        pre_eff = h_prev[0, self.pre] * self.pre_sign * self.prior_scale
        self.tr = (1.0 - alpha) * self.tr + alpha * phi_prime_ro[0, self.post_pos] * pre_eff
        if in_window:
            self.elig = self.elig + self.tr / n_window

    prior_scale = 1.0  # set by the agent: coupling-prior value of each plastic edge (d w / d gain)

    @torch.no_grad()
    def reset_trace(self) -> None:
        self.tr = torch.zeros_like(self.elig)

    @torch.no_grad()
    def apply(self, delta: float, comp_signal: torch.Tensor | None = None) -> None:
        """gain <- clip(gain + lr * delta * elig [* comp_signal[post]]); then optional homeostatic
        leak toward the initial gain. ``comp_signal`` (n_readout,) is the per-compartment feedback
        (comp_feedback rule only)."""
        if self.n_plastic == 0:
            return
        c = self.cfg
        if c.modulation:
            if c.optimizer == "rmsprop":  # bounded, scale-free step per synapse (v2 exploratory option)
                u = delta * self.elig
                if comp_signal is not None:
                    u = u * comp_signal[self.post_pos]
                self.sq = c.rms_beta * self.sq + (1.0 - c.rms_beta) * u * u
                step = c.lr * u / (torch.sqrt(self.sq / (1.0 - c.rms_beta)) + c.rms_eps)
            else:  # sgd: operation order IDENTICAL to the code that produced the v2 final records
                step = c.lr * delta * self.elig
                if comp_signal is not None:
                    step = step * comp_signal[self.post_pos]
            self.gain = torch.clamp(self.gain + step, c.gain_lo, c.gain_hi)
        if c.homeostasis > 0.0:
            self.gain = self.gain + c.homeostasis * (self.init_gain - self.gain)

    def override(self, w: torch.Tensor, sign_node: torch.Tensor, prior_val: torch.Tensor,
                 ) -> torch.Tensor:
        """Return a copy of edge values ``w`` with plastic edges replaced by sign*gain*prior."""
        w = w.detach().clone()
        w[self.idx] = sign_node[self.pre] * self.gain * prior_val[self.idx]
        return w
