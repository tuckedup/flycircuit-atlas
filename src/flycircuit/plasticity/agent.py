"""Learning agent: one trial = one decision with scalar reward, for several learner kinds.

Learners (all receive the SAME feedback: cue, chosen action, scalar reward; critic baseline V):
  bptt               policy gradient (REINFORCE with TD-critic baseline) through the trial dynamics;
                     trains recurrent gains + decoder + bias for sparse models, all weights for
                     GRU / encoder-decoder. Truncation: one trial (T steps) is the gradient horizon.
  decoder_only       same loss, only the decoder/readout layer is trainable.
  plastic_fixed_dec  three-factor plasticity on plastic edges; decoder frozen at random init.
  plastic_plus_dec   three-factor plasticity + decoder policy-gradient (Adam).
  plastic_mod_off    three-factor machinery with modulation disabled (delta forced 0): control.
Plasticity agents add Gaussian exploration noise to readout-node activity (needed for node
perturbation credit assignment); bptt-type learners use the softmax policy only.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from flycircuit.models.baselines import EncoderDecoderOnly, GRUBaseline
from flycircuit.models.sparse_rnn import SparseConnectomeRNN
from flycircuit.plasticity.three_factor import PlasticityConfig, PlasticSynapses, ValueFunction

LEARNERS = ("bptt", "decoder_only", "plastic_fixed_dec", "plastic_plus_dec", "plastic_mod_off")
PLASTIC = ("plastic_fixed_dec", "plastic_plus_dec", "plastic_mod_off")
RESET_KINDS = ("state", "gain", "decoder", "full")


class LearningAgent:
    def __init__(self, model, learner: str, task, lr: float, plastic_mask: np.ndarray | None = None,
                 explore_noise: float = 0.1, decoder_lr: float = 3e-3, seed: int = 0,
                 carry_state: bool = False, plast_cfg: PlasticityConfig | None = None,
                 logit_cap: float = 2.0, zscore: bool = True):
        if learner not in LEARNERS:
            raise ValueError(f"learner must be in {LEARNERS}")
        self.use_zscore = zscore  # False: raw readout activity (explicit interface-control arm)
        self.model, self.learner, self.task = model, learner, task
        self.sparse = isinstance(model, SparseConnectomeRNN)
        if learner in PLASTIC and not self.sparse:
            raise ValueError("plasticity learners need the sparse connectome-type model")
        self.lr, self.decoder_lr = lr, decoder_lr
        self.logit_cap = float(logit_cap)
        self.carry_state = carry_state
        self.rule = (plast_cfg or PlasticityConfig()).rule
        # comp_feedback needs no injected noise (policy sampling is the exploration)
        self.noise = explore_noise if (learner in PLASTIC and self.rule != "comp_feedback") else 0.0
        self.value = ValueFunction(task.spec.n_in)
        self.gamma = (plast_cfg or PlasticityConfig()).gamma
        self.torch_gen = torch.Generator().manual_seed(seed + 17)
        self.window = task.response_window()
        self._init_snapshot = copy.deepcopy(model.state_dict())
        self.state = None
        self.plastic: PlasticSynapses | None = None
        if learner in PLASTIC:
            cfg = copy.copy(plast_cfg or PlasticityConfig())
            cfg.lr = lr
            if learner == "plastic_mod_off":
                cfg.modulation = False
            if plastic_mask is None:
                raise ValueError("plastic_mask required for plasticity learners")
            self.plastic = PlasticSynapses(plastic_mask, model.pre, model.post, model.readout_idx,
                                           model.spec.init_gain, cfg, sign_node=model.sign_node)
            self.plastic.prior_scale = model.prior_val[self.plastic.idx]
            if cfg.rule == "comp_feedback" and model.spec.nonlinearity != "tanh":
                raise ValueError("comp_feedback derives phi' from tanh; set nonlinearity='tanh'")
            self._fb_perm = None
            if cfg.shuffle_feedback:
                self._fb_perm = torch.as_tensor(
                    np.random.default_rng(seed + 31).permutation(len(model.readout_idx)))
        if self.sparse:
            self.calibrate()
        self._set_trainable()
        self._make_optimizer()

    # ---- parameters -------------------------------------------------------------------
    def decoder_params(self):
        m = self.model
        if self.sparse:
            return [m.decoder]
        if isinstance(m, GRUBaseline):
            return list(m.dec.parameters())
        if isinstance(m, EncoderDecoderOnly):
            return list(m.net[2].parameters())
        raise TypeError(type(m))

    def _set_trainable(self):
        m = self.model
        if self.learner == "bptt":
            # sparse model: recurrent gains + decoder; the per-node bias stays fixed at 0 (Adam
            # random-walks ~N irrelevant bias parameters and collapses the readout; see learning.md)
            self.params = [m.raw_gain, m.decoder] if self.sparse else list(m.parameters())
        elif self.learner in ("decoder_only", "plastic_plus_dec"):
            self.params = self.decoder_params()
        else:
            self.params = []
        for p in m.parameters():
            p.requires_grad_(False)
        for p in self.params:
            p.requires_grad_(True)

    def _make_optimizer(self):
        if not self.params:
            self.opt = None
            return
        lr = self.decoder_lr if self.learner == "plastic_plus_dec" else self.lr
        self.opt = torch.optim.Adam(self.params, lr=lr)

    # ---- dynamics ---------------------------------------------------------------------
    def _forward(self, x: torch.Tensor, grad: bool, noise: float, track: bool, state=None):
        """x (T,B,n_in) -> logits (B,2), final state."""
        m = self.model
        win = self.window
        if not self.sparse:
            with torch.set_grad_enabled(grad):
                y, st = m(x, state)
            return y[win].mean(0), st
        ro = m.readout_idx
        with torch.set_grad_enabled(grad):
            w = m.edge_values()
            if self.plastic is not None:
                w = self.plastic.override(w, m.sign_node, m.prior_val)
            elif not grad:
                w = w.detach()
            st = state if state is not None else m.init_state(x.shape[1])
            ys = []
            for t in range(x.shape[0]):
                h_prev = st["h"]
                _, st = m.step(x[t], st, w)
                z = self._zscore(st["h"][:, ro])
                in_win = win.start <= t < win.stop
                if noise > 0:
                    eps = noise * torch.randn(x.shape[1], len(ro), generator=self.torch_gen)
                    z = z + eps
                    if track and (self.rule == "np_v1" or in_win):
                        self.plastic.accumulate(h_prev, eps)
                elif track and self.rule == "comp_feedback":
                    th = (st["h"][:, ro] - (1.0 - m.alpha) * h_prev[:, ro]) / m.alpha  # tanh(drive)
                    self.plastic.accumulate_trace(h_prev, (1.0 - th * th).clamp_min(0.0), m.alpha,
                                                  in_win, win.stop - win.start)
                ys.append(z @ m.decoder)
        return torch.stack(ys)[win].mean(0), st

    def _zscore(self, h_ro: torch.Tensor) -> torch.Tensor:
        """Readout standardisation with FIXED pre-learning calibration statistics (engineered
        interface; task-agnostic, label-free; same procedure for every sparse condition)."""
        return ((h_ro - self.f_mu) / self.f_sd).clamp(-5.0, 5.0)

    @torch.no_grad()
    def calibrate(self, n: int = 128) -> None:
        """Estimate per-readout-node mean/sd over random cue patterns (no labels, no reward)."""
        m = self.model
        sp = self.task.spec
        rng = np.random.default_rng(np.random.SeedSequence([self.task.seed, 777]))
        cues = np.zeros((n, sp.n_cue), dtype=np.float32)
        for i in range(n):
            cues[i, rng.permutation(sp.n_cue)[: sp.k_active]] = 1.0
        w = m.edge_values()
        if self.plastic is not None:
            w = self.plastic.override(w, m.sign_node, m.prior_val)
        x = torch.as_tensor(self.task.make_input(cues, 0, rng))
        st = m.init_state(n)
        acts = []
        for t in range(x.shape[0]):
            _, st = m.step(x[t], st, w)
            if t >= x.shape[0] - sp.resp_steps:
                acts.append(st["h"][:, m.readout_idx])
        a = torch.cat(acts, 0)
        self.f_mu = a.mean(0)
        self.f_sd = a.std(0) + 1e-3
        if not self.use_zscore:
            self.f_mu = torch.zeros_like(self.f_mu)
            self.f_sd = torch.ones_like(self.f_sd)

    def _x(self, cue: np.ndarray, ctx: int, rng_noise):
        return torch.as_tensor(self.task.make_input(cue, ctx, rng_noise))

    # ---- one training trial -----------------------------------------------------------
    def trial(self, cue: np.ndarray, ctx: int, label: int, noise_rng: np.random.Generator,
              act_rng: np.random.Generator) -> dict:
        """``noise_rng`` (input noise) and ``act_rng`` (action sampling) are separate streams so
        stimulus events are identical across compared conditions even when actions differ."""
        xn = self.task.make_input(cue, ctx, noise_rng)
        x = torch.as_tensor(xn)
        track = self.plastic is not None
        if track:
            self.plastic.reset_eligibility()
        grad = self.opt is not None
        st0 = self.state if self.carry_state else None
        logits, st = self._forward(x, grad, self.noise, track, st0)
        if self.carry_state:
            self.state = {k: v.detach() for k, v in st.items()}
        cap = self.logit_cap  # bounded policy keeps exploration alive (needed for reversal)
        logp = torch.log_softmax(cap * torch.tanh(logits[0] / cap), -1)
        p = logp.detach().exp().numpy().astype(np.float64)
        p /= p.sum()
        a = int(act_rng.choice(2, p=p))
        r = float(a == label)
        phi = ValueFunction.features(xn)
        delta = self.value.td_error(r, phi, self.gamma)
        if grad:
            self.opt.zero_grad(set_to_none=True)
            (-delta * logp[a]).backward()
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.opt.step()
        if track:
            comp = None
            if self.rule == "comp_feedback":
                onehot = torch.zeros(2)
                onehot[a] = 1.0
                pi = torch.as_tensor(p, dtype=torch.float32)
                capd = 1.0 - torch.tanh(logits[0].detach() / cap) ** 2
                g_l = (onehot - pi) * capd  # d log pi(a) / d raw logits
                comp = (self.model.decoder.detach() @ g_l) / self.f_sd
                if self._fb_perm is not None:
                    comp = comp[self._fb_perm]
            self.plastic.apply(delta, comp)
        self.value.update(delta, phi)
        return {"action": a, "reward": r, "delta": delta, "correct": a == label}

    # ---- evaluation (greedy, no learning, no exploration noise) -----------------------
    @torch.no_grad()
    def evaluate(self, split: str, ctx: int, reversed_: bool = False) -> float:
        cue = self.task.cues[(split, ctx)]
        y = self.task.labels_for(split, ctx, reversed_)
        rng = np.random.default_rng(np.random.SeedSequence([self.task.seed, 555, ctx]))
        x = torch.as_tensor(self.task.make_input(cue, ctx, rng))
        logits, _ = self._forward(x, False, 0.0, False, None)
        return float((logits.argmax(-1).numpy() == y).mean())

    # ---- resets (separable interventions) ---------------------------------------------
    def reset(self, kind: str) -> None:
        """state: carried neural state only | gain: plastic gains + eligibility (and recurrent
        gain parameters of BPTT models) | decoder: decoder weights + its optimizer state |
        full: everything incl. critic, post-baseline and all optimizer state."""
        if kind not in RESET_KINDS:
            raise ValueError(f"reset kind must be in {RESET_KINDS}")
        m = self.model
        snap = self._init_snapshot
        if kind in ("state", "full"):
            self.state = None
        if kind in ("gain", "full"):
            if self.plastic is not None:
                self.plastic.reset_gain()
                self.plastic.reset_eligibility()
            if self.sparse:
                with torch.no_grad():
                    m.raw_gain.copy_(snap["raw_gain"])
                    m.bias.copy_(snap["bias"])
        if kind in ("decoder", "full"):
            with torch.no_grad():
                for name, p in m.named_parameters():
                    if any(p is q for q in self.decoder_params()):
                        p.copy_(snap[name])
            self._reset_optimizer_state(only_decoder=kind == "decoder")
        if kind == "full":
            self._reset_optimizer_state(only_decoder=False)
            if self.plastic is not None:
                self.plastic.reset_optimizer()
            self.value = ValueFunction(self.task.spec.n_in)

    def _reset_optimizer_state(self, only_decoder: bool) -> None:
        if self.opt is None:
            return
        dec = {id(p) for p in self.decoder_params()}
        for p in list(self.opt.state):
            if not only_decoder or id(p) in dec:
                del self.opt.state[p]

    # ---- v2 additions ------------------------------------------------------------------
    def idle(self, n: int) -> None:
        """``n`` un-reinforced trials (a retention interval): no reward, no policy update; only the
        isolated homeostatic leak (if configured) acts on plastic gains."""
        if self.plastic is not None and self.plastic.cfg.homeostasis > 0.0:
            for _ in range(n):
                self.plastic.gain = self.plastic.gain + self.plastic.cfg.homeostasis * (
                    self.plastic.init_gain - self.plastic.gain)

    def checkpoint_payload(self) -> dict:
        """Owned tensor copies to restore the TRAINED learner (no optimizer state): model tensors
        (incl. fixed encoder / sign / graph buffers), plastic gains, critic, readout calibration."""
        out = {"model": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
               "critic_v": self.value.v.copy(), "learner": self.learner, "rule": self.rule,
               "use_zscore": self.use_zscore}
        if self.plastic is not None:
            out["plastic_idx"] = self.plastic.idx.cpu().clone()
            out["plastic_gain"] = self.plastic.gain.detach().cpu().clone()
        if self.sparse:
            out["f_mu"], out["f_sd"] = self.f_mu.cpu().clone(), self.f_sd.cpu().clone()
        return out

    # ---- bookkeeping ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Tensors that reset kinds should (not) touch; used by tests and audits."""
        m = self.model
        out = {"decoder": [p.detach().clone() for p in self.decoder_params()],
               "state": None if self.state is None else {k: v.clone() for k, v in self.state.items()}}
        if self.sparse:
            out["raw_gain"] = m.raw_gain.detach().clone()
        if self.plastic is not None:
            out["plastic_gain"] = self.plastic.gain.clone()
            out["elig"] = self.plastic.elig.clone()
        return out
