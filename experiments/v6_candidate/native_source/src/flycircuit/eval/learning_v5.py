"""Learning v5: replay-factor pilot (R0 standard / R1 episodic replay / C step-matched extra current trials).

NEW module (prospective revision). It re-implements ``learning.train_phase`` / ``learning.run_continual`` with
explicit mode handling and SEPARATELY DERIVED random-number streams, and reuses (does not edit) the reviewed
v2/v3/v4 construction + learner. Nothing here monkeypatches shared globals (the v4 module-level ``CUR`` /
``torch.nn.utils.clip_grad_norm_`` patches are NOT used): learning rates are explicit arguments, the clipping
and the optimizer are verified from the REAL gradients / REAL optimizer object.

RNG design (``RNG_VERSION``), all from numpy ``SeedSequence`` spawn:
  stream A  current-task ordering / input noise / action sampling = SeedSequence([seed, 4242, 1]).spawn(3)
            == ``learning._rngs(seed, 1)`` (the historical continual stream, so R0 reproduces history).
            Consumed ONLY by the current-task trial of each step. Extra trials never touch it.
  stream B  replay (R1): SeedSequence([seed, 5151, 1]).spawn(3) = (order, noise, act).
  stream C  extra current-task trials (C): SeedSequence([seed, 5151, 2]).spawn(3) = (order, noise, act).
Stimulus events of the current-task schedule are therefore identical across R0/R1/C for a task seed; the
SAMPLED ACTIONS differ because the learned policies differ.

Everything is a SYNTHETIC task evaluation; results are model-relative engineering evidence only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from flycircuit.config import project_root
from flycircuit.eval import learning as L
from flycircuit.eval import learning_v2 as V2
from flycircuit.eval import learning_v3 as V3
from flycircuit.eval import learning_v4_select as SEL
from flycircuit.eval import stats as ST
from flycircuit.models.baselines import GRUBaseline
from flycircuit.plasticity.agent import LearningAgent

COPY = Path(__file__).resolve().parents[3]
ORIG = Path("/home/atulp/projects/flycircuit")
V4DIR = COPY / "results_v4" / "learning_readout_v4_1"
OUT = COPY / "results_v5" / "learning"
CHOICES_PATH = V4DIR / "W3_trained_w76_choices.json"
W0_MANIFEST = V4DIR / "W0_width_probe_manifest.json"
REGISTRY_PATH = V4DIR / "seed_registry.json"

RNG_VERSION = ("learning_v5.rng.1|A=SeedSequence([seed,4242,1]).spawn(3)=learning._rngs(seed,1)|"
               "B(replay)=SeedSequence([seed,5151,1]).spawn(3)|C(extra current)=SeedSequence([seed,5151,2]).spawn(3)|"
               "spawn order (order,noise,act)")
STREAM_ENTROPY = {"A": [None, 4242, 1], "B": [None, 5151, 1], "C": [None, 5151, 2]}  # None = task seed

PRIMARY = ("connectome", "gru33")
SECONDARY = ("encdec", "random_reach")  # historical architectures, descriptive, R0/R1 only
FAMILIES = PRIMARY + SECONDARY
MODES = {"connectome": ("R0", "R1", "C"), "gru33": ("R0", "R1", "C"),
         "encdec": ("R0", "R1"), "random_reach": ("R0", "R1")}
MODE_NAMES = {"R0": "standard", "R1": "episodic replay (16 items / finished task)",
              "C": "extra current-task trials, no old-task buffer"}
RATIOS = (1 / 3, 1.0, 3.0)
SEARCH_SEEDS = (1601, 1602, 1603, 1604)
CHECK_SEEDS = (1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612)
ENG_SEEDS = (1701, 1702, 1703, 1704, 1705, 1706)
HELDOUT_SEEDS = tuple(range(100601, 100621))
ADEQUACY_LEARNED = 0.80  # mean over tasks of R[l,l] (diagonal right after learning), per condition mean over seeds
GRU_HIDDEN = 33
# optimizer-owned (core, decoder) parameter counts asserted at construction
EXPECTED_OWNED = {"connectome": (7399, 152), "gru33": (7392, 68), "encdec": (8140, 816),
                  "random_reach": (7399, 152)}
CLIP_NORM = 1.0  # hard-coded in LearningAgent.trial (clip_grad_norm_(params, 1.0)); verified from gradients
SOURCE_FILES = [
    "src/flycircuit/eval/learning_v5.py", "src/flycircuit/eval/learning.py", "src/flycircuit/eval/learning_v2.py",
    "src/flycircuit/eval/learning_v3.py", "src/flycircuit/eval/learning_v4.py",
    "src/flycircuit/eval/learning_v4_select.py", "src/flycircuit/eval/stats.py",
    "src/flycircuit/plasticity/agent.py", "src/flycircuit/plasticity/three_factor.py",
    "src/flycircuit/models/baselines.py", "src/flycircuit/models/sparse_rnn.py",
    "src/flycircuit/tasks/association.py", "src/flycircuit/tasks/graph_loader.py",
    "scripts_v5/learning_pilot.py", "scripts_v5/learning_analysis.py", "scripts_v5/learning_selftest.py",
    "scripts_v5/verify_unseen.py", "scripts_v5/launch_learning_pilot.sh",
    "configs/experiments/learning_v3_protocol.yaml",
    "results_v4/learning_readout_v4_1/seed_registry.json",
    "results_v4/learning_readout_v4_1/W3_trained_w76_choices.json",
    "results_v4/learning_readout_v4_1/W0_width_probe_manifest.json",
    "results_v4/learning_readout_v4_1/MEMORY_REDESIGN_PROPOSAL.md",
]


class StopRequested(Exception):
    """SIGTERM / wall-limit raised from inside a row; the row is discarded (never recorded as a result)."""


# ----------------------------------------------------------------------------------------------- utils
def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def canon(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=1, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path, rec: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path, repair: bool = False) -> list[dict]:
    """Strict reader. A missing final newline means a torn last write: with repair=True the torn tail is moved to
    ``<name>.torn_tail`` and the file truncated to the last complete line (that coordinate was never recorded)."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        if not repair:
            raise ValueError(f"truncated JSONL (no final newline): {path}")
        cut = data.rfind(b"\n") + 1
        path.with_name(path.name + ".torn_tail").write_bytes(data[cut:])
        with open(path, "r+b") as f:
            f.truncate(cut)
            f.flush()
            os.fsync(f.fileno())
        data = data[:cut]
    out = []
    for n, raw in enumerate(data.splitlines(), 1):
        if not raw.strip():
            raise ValueError(f"blank JSONL row {path}:{n}")
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSONL row {path}:{n}")
        out.append(row)
    return out


def mean_se(v) -> tuple[float, float]:
    a = np.asarray(list(v), float)
    if a.size == 0 or not np.isfinite(a).all():
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), float("inf")
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(a.size))


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ----------------------------------------------------------------------------------------------- RNG streams
class Stream:
    __slots__ = ("order", "noise", "act", "entropy")

    def __init__(self, entropy: list[int]):
        ss = np.random.SeedSequence(entropy)
        o, n, a = ss.spawn(3)
        self.order, self.noise, self.act = (np.random.default_rng(o), np.random.default_rng(n),
                                            np.random.default_rng(a))
        self.entropy = list(entropy)

    def states(self) -> list[str]:
        return [json.dumps(g.bit_generator.state, sort_keys=True, default=str)
                for g in (self.order, self.noise, self.act)]


class Streams:
    def __init__(self, seed: int):
        self.seed = int(seed)
        ent = {k: [self.seed if x is None else x for x in v] for k, v in STREAM_ENTROPY.items()}
        self.A, self.B, self.C = Stream(ent["A"]), Stream(ent["B"]), Stream(ent["C"])

    def info(self) -> dict:
        return {"rng_version": RNG_VERSION, "task_seed": self.seed,
                "entropy": {"A": self.A.entropy, "B": self.B.entropy, "C": self.C.entropy}}


# ----------------------------------------------------------------------------------------------- budgets
def expected_counts(proto: dict, mode: str) -> dict:
    """Trial / optimizer-step counts derived from the CONFIG (never hard-coded): R0 = n_ctx*trials_per_phase;
    R1/C add one extra trial per current trial in phases 1..n_ctx-1."""
    if mode not in ("R0", "R1", "C"):
        raise ValueError(mode)
    n_t = int(proto["task"]["n_ctx"])
    tpp = int(proto["training"]["trials_per_phase"])
    cur = n_t * tpp
    extra = (n_t - 1) * tpp if mode in ("R1", "C") else 0
    return {"current_trials": cur, "extra_trials": extra, "optimizer_steps": cur + extra}


# ----------------------------------------------------------------------------------------------- replay buffer
def _pattern_keys(x: np.ndarray) -> set[bytes]:
    return {np.packbits(xi.astype(np.uint8)).tobytes() for xi in x}


class ReplayBuffer:
    """Holds ONLY training items of COMPLETED tasks. ``add_completed_task`` refuses anything that is not (a prefix of)
    the task's training split or that matches any validation/test pattern."""

    def __init__(self, task, budget: int):
        self.task, self.budget = task, int(budget)
        self.items: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._heldout = set()
        for (split, _ctx), x in task.cues.items():
            if split != "train":
                self._heldout |= _pattern_keys(x)
        self.peak_payload: dict | None = None
        self.log: list[dict] = []

    @property
    def ctxs(self) -> tuple[int, ...]:
        return tuple(self.items)

    def add_completed_task(self, phase: int, cues: np.ndarray, labels: np.ndarray) -> None:
        tr = self.task.cues[("train", phase)]
        k = len(cues)
        assert k == self.budget and k <= len(tr), (k, self.budget)
        assert np.array_equal(cues, tr[:k]), "replay items must be the first items of the TRAIN split"
        assert np.array_equal(labels, self.task.labels_for("train", phase)[:k])
        assert not (_pattern_keys(cues) & self._heldout), "replay item matches a validation/test pattern"
        assert phase not in self.items
        self.items[phase] = (np.array(cues, copy=True), np.array(labels, copy=True))
        p = self.payload()
        if self.peak_payload is None or p["n_items"] >= self.peak_payload["n_items"]:
            self.peak_payload = p

    def audit(self, current_phase: int) -> dict:
        """Called at the start of each phase: every buffered context must be a COMPLETED earlier task."""
        assert all(c < current_phase for c in self.items), (tuple(self.items), current_phase)
        rec = {"phase": current_phase, "ctxs": list(self.items), "n_items": sum(len(v[1]) for v in self.items.values())}
        self.log.append(rec)
        return rec

    def payload(self) -> dict:
        n_items = sum(len(v[1]) for v in self.items.values())
        arr_bytes = sum(c.nbytes + l.nbytes for c, l in self.items.values())
        py = sys.getsizeof(self.items) + sum(sys.getsizeof(k) + sys.getsizeof(v) + sys.getsizeof(v[0])
                                              + sys.getsizeof(v[1]) for k, v in self.items.items())
        # getsizeof(owned ndarray) includes its data buffer; overhead = estimate - raw array bytes
        return {"n_items": n_items, "n_tasks": len(self.items), "array_nbytes": int(arr_bytes),
                "python_getsizeof_estimate": int(py), "python_overhead_estimate": int(py - arr_bytes),
                "logical_bits_estimate": int(n_items * (self.task.spec.n_cue + 1)),
                "note": "logical bits = items*(n_cue+1) (the 544-bit draft figure at 32 items); array_nbytes and "
                        "getsizeof are the actual in-memory sizes (estimates)"}


EMPTY_PAYLOAD = {"n_items": 0, "n_tasks": 0, "array_nbytes": 0, "python_getsizeof_estimate": 0,
                 "python_overhead_estimate": 0, "logical_bits_estimate": 0, "note": "no old-task buffer"}


# ----------------------------------------------------------------------------------------------- clip monitor
class ClipMonitor:
    """Records the REAL pre-clip gradient norm (tensor hooks fire on the accumulated leaf gradient before
    ``clip_grad_norm_``) and the post-clip norm (``p.grad`` after the trial). Per-tensor hooks: no global patch."""

    def __init__(self, params):
        self.params = list(params)
        self._sq: list[float] = []
        self.pre: list[float] = []
        self.post: list[float] = []
        self._handles = [p.register_hook(self._hook) for p in self.params]

    def _hook(self, g):
        self._sq.append(float(g.detach().pow(2).sum()))

    def after_trial(self):
        self.pre.append(math.sqrt(sum(self._sq)))
        self._sq.clear()
        self.post.append(math.sqrt(sum(float(p.grad.pow(2).sum()) for p in self.params if p.grad is not None)))

    def close(self):
        for h in self._handles:
            h.remove()

    def summary(self, max_norm: float = CLIP_NORM) -> dict:
        pre, post = np.asarray(self.pre), np.asarray(self.post)
        if pre.size == 0:
            return {"n": 0}
        fin = np.isfinite(pre) & np.isfinite(post)
        over = fin & (pre > max_norm * (1 + 1e-6))
        under = fin & ~over
        ok_over = bool(np.all(np.abs(post[over] - max_norm) <= 1e-3 * max_norm)) if over.any() else True
        ok_under = bool(np.all(np.abs(post[under] - pre[under]) <= 1e-4 * np.maximum(1.0, pre[under]))) if under.any() else True
        return {"n": int(pre.size), "nonfinite": int((~fin).sum()), "clip_max_norm": max_norm,
                "n_clipped": int(over.sum()), "clip_fraction": float(over.mean()),
                "pre_norm_median": float(np.median(pre[fin])) if fin.any() else None,
                "pre_norm_p95": float(np.percentile(pre[fin], 95)) if fin.any() else None,
                "pre_norm_max": float(pre[fin].max()) if fin.any() else None,
                "post_norm_max": float(post[fin].max()) if fin.any() else None,
                "clip_behaviour_consistent": bool(ok_over and ok_under)}


# ----------------------------------------------------------------------------------------------- environment
def build_w76(base, w0_path=W0_MANIFEST):
    """Same construction as scripts_v4/w3_pilot.build_w76 / width_probe; asserted equal to the W0 manifest."""
    g = base.graph
    n = g.n
    sens = set(map(int, base.sensory_idx))
    r25 = [int(i) for i in base.readout_idx]
    indeg = np.bincount(g.post, minlength=n)
    contacts = np.bincount(g.post, weights=g.count, minlength=n)
    s25 = set(r25)
    elig = [i for i in range(n) if i not in sens and i not in s25 and indeg[i] > 0]
    elig.sort(key=lambda i: (-contacts[i], i))
    r76 = r25 + elig
    assert len(r25) == 25 and len(r76) == 76 and r76[:25] == r25
    w0 = json.load(open(w0_path))
    assert [int(i) for i in w0["sets"]["W76"]] == r76, "W76 differs from W0 manifest"
    assert [int(i) for i in w0["sets"]["W25"]] == r25, "W25 differs from W0 manifest"
    return r25, r76, hashlib.sha256(json.dumps(r76).encode()).hexdigest()


def with_ro(base, ro):
    from flycircuit.tasks.graph_loader import CircuitBundle
    b = base
    return CircuitBundle(b.graph, b.sign, b.sensory_idx, np.asarray(ro, dtype=base.readout_idx.dtype),
                         b.label, b.notes, b.modulatory)


class Env:
    """Loaded once per process: protocol, original (W25) base for dense families, W76 base for sparse ones."""

    def __init__(self):
        if COPY not in Path(__file__).resolve().parents:
            raise RuntimeError("must import flycircuit from the isolated copy")
        if project_root().resolve() != ORIG.resolve():
            raise RuntimeError("set FLYCIRCUIT_HOME=/home/atulp/projects/flycircuit")
        self.proto = V3.load_protocol()
        self.protocol_hash = V2.protocol_hash(self.proto)
        self.base25 = V2.load_base(self.proto, "flywire")
        self.r25, self.r76, self.w76_sha = build_w76(self.base25)
        self.base76 = with_ro(self.base25, self.r76)
        self.n_in = int(self.proto["task"]["n_cue"]) + int(self.proto["task"]["n_ctx"])
        self.packs = {"plain": [], "group": []}  # null families are not part of this pilot

    def make_task(self, seed: int, proto: dict | None = None):
        return L.make_task(proto or self.proto, int(seed))

    def build_model(self, family: str, seed: int):
        """-> (model, zscore, info). 'gru37' (historical) is deliberately not constructible here."""
        if family in ("connectome", "random_reach"):
            m, _b, zs, info = V2.build_model_v2(self.proto, self.base76, self.packs, family, seed, self.n_in)
            assert len(m.readout_idx) == 76, family
            return m, zs, info
        if family == "encdec":
            m, _b, zs, info = V2.build_model_v2(self.proto, self.base25, self.packs, "encdec", seed, self.n_in)
            assert info == {"hidden": 407}, info
            return m, zs, info
        if family == "gru33":
            return GRUBaseline(self.n_in, 2, GRU_HIDDEN, seed), True, {"hidden": GRU_HIDDEN}
        raise ValueError(f"unknown family {family!r} (historical GRU37 is excluded from v5 grids)")

    def make_agent(self, family: str, core_lr: float, dec_lr: float, seed: int, task, proto: dict | None = None):
        """bptt learner with the v4 two-parameter-group Adam, built WITHOUT module-level state."""
        proto = proto or self.proto
        model, zs, info = self.build_model(family, seed)
        tr = proto["training"]
        agent = LearningAgent(model, "bptt", task, float(core_lr), plastic_mask=None,
                              explore_noise=tr["explore_noise"], decoder_lr=float(dec_lr), seed=seed,
                              plast_cfg=V2.plast_cfg_for(proto, "bptt"),
                              logit_cap=tr.get("policy_logit_cap", 2.0), zscore=zs)
        dec_ids = {id(p) for p in agent.decoder_params()}
        core = [p for p in agent.params if id(p) not in dec_ids]
        dec = [p for p in agent.params if id(p) in dec_ids]
        if not core or not dec:
            raise RuntimeError("empty parameter group")
        agent.opt = torch.optim.Adam([{"params": core, "lr": float(core_lr)}, {"params": dec, "lr": float(dec_lr)}])
        rep = param_report(agent, family)
        return agent, info, rep


def param_report(agent, family: str | None = None) -> dict:
    """Counts from the ACTUAL optimizer param groups (not labels). core/decoder/bias/critic, trainable vs frozen."""
    opt = agent.opt
    m = agent.model
    owned = {}
    for gi, g in enumerate(opt.param_groups):
        for p in g["params"]:
            owned[id(p)] = gi
    names = {id(p): n for n, p in m.named_parameters()}
    groups = []
    for gi, g in enumerate(opt.param_groups):
        groups.append({"group": gi, "lr": float(g["lr"]), "n_tensors": len(g["params"]),
                       "numel": int(sum(p.numel() for p in g["params"])),
                       "names": sorted(names.get(id(p), "?") for p in g["params"]),
                       "weight_decay": float(g.get("weight_decay", 0.0)), "betas": list(g.get("betas", ()))})
    dec_ids = {id(p) for p in agent.decoder_params()}
    core_n = int(sum(p.numel() for g in opt.param_groups for p in g["params"] if id(p) not in dec_ids))
    dec_n = int(sum(p.numel() for g in opt.param_groups for p in g["params"] if id(p) in dec_ids))
    bias_owned = int(sum(p.numel() for g in opt.param_groups for p in g["params"]
                         if "bias" in names.get(id(p), "")))
    frozen = {n: int(p.numel()) for n, p in m.named_parameters() if id(p) not in owned}
    out = {"optimizer": type(opt).__name__, "groups": groups, "owned_total": core_n + dec_n,
           "core_owned": core_n, "decoder_owned": dec_n, "bias_owned_within_core_or_decoder": bias_owned,
           "frozen_params": frozen, "frozen_total": int(sum(frozen.values())),
           "frozen_bias_nodes": int(m.bias.numel()) if agent.sparse and "bias" in frozen else 0,
           "model_buffers_numel": int(sum(b.numel() for b in m.buffers())),
           "critic_params_not_optimizer_owned": int(agent.value.v.size),
           "critic_note": "linear TD critic updated by a numpy rule (lr_value), not by the optimizer; not counted in owned_total",
           "clip_max_norm": CLIP_NORM, "requires_grad_total": int(sum(p.numel() for p in m.parameters() if p.requires_grad))}
    assert out["requires_grad_total"] == out["owned_total"], "requires_grad set differs from optimizer-owned set"
    if family is not None:
        want = EXPECTED_OWNED[family]
        assert (core_n, dec_n) == want, f"{family}: owned (core,dec)=({core_n},{dec_n}) != expected {want}"
    return out


def inspect_lr(agent, core_lr: float, dec_lr: float) -> None:
    gs = agent.opt.param_groups
    assert type(agent.opt) is torch.optim.Adam and len(gs) == 2
    assert float(gs[0]["lr"]) == float(core_lr) and float(gs[1]["lr"]) == float(dec_lr), (gs[0]["lr"], gs[1]["lr"])
    dec_ids = {id(p) for p in agent.decoder_params()}
    assert all(id(p) not in dec_ids for p in gs[0]["params"]) and all(id(p) in dec_ids for p in gs[1]["params"])


# ----------------------------------------------------------------------------------------------- training loops
def _do_trial(agent, mon, counts, kind, cue, ctx, label, rng_noise, rng_act, trace):
    if trace is not None:
        trace.append((kind, ctx))
    out = agent.trial(cue, ctx, label, rng_noise, rng_act)
    counts[kind] += 1
    if mon is not None:
        mon.after_trial()
    return out


def train_phase_v5(agent, task, phase: int, n: int, S: Streams, mode: str, buf: ReplayBuffer | None,
                   mon=None, counts=None, trace=None, stop=None) -> list[dict]:
    """Faithful to ``learning.train_phase`` for the current-task trial (stream A); extra trials use B (R1) or C (C)."""
    A = S.A
    counts = counts if counts is not None else {"cur": 0, "replay": 0, "extra": 0}
    hist = []
    extra_on = phase >= 1 and mode in ("R1", "C")
    if mode == "R1":
        assert (phase >= 1) == bool(buf is not None and buf.items), "buffer must be non-empty exactly in phases >= 1"
    for t in range(n):
        if stop is not None and t % 25 == 0:
            stop()
        cue, lab = task.sample_train(A.order, phase, False)
        hist.append(_do_trial(agent, mon, counts, "cur", cue, phase, int(lab[0]), A.noise, A.act, trace))
        if extra_on and mode == "R1":
            ctxs = buf.ctxs
            c2 = int(ctxs[S.B.order.integers(len(ctxs))])
            cues, labs = buf.items[c2]
            i = int(S.B.order.integers(len(labs)))
            _do_trial(agent, mon, counts, "replay", cues[i:i + 1], c2, int(labs[i]), S.B.noise, S.B.act, trace)
        elif extra_on and mode == "C":
            cue2, lab2 = task.sample_train(S.C.order, phase, False)
            _do_trial(agent, mon, counts, "extra", cue2, phase, int(lab2[0]), S.C.noise, S.C.act, trace)
    return hist


def adam_step_counts(agent) -> dict:
    steps = []
    for g in agent.opt.param_groups:
        for p in g["params"]:
            st = agent.opt.state.get(p)
            steps.append(int(st["step"]) if st and "step" in st else 0)
    return {"min": min(steps), "max": max(steps), "n_tensors": len(steps)}


def recompute_forgetting(R, policy: str = "keep") -> float:
    """Independent implementation of average forgetting (final task excluded)."""
    R = np.asarray(R, float)
    T = R.shape[0]
    f = [max(R[l, j] for l in range(j, T - 1)) - R[T - 1, j] for j in range(T - 1)]
    if policy == "clip":
        f = [max(x, 0.0) for x in f]
    return float(np.mean(f))


def learned_acc(R) -> float:
    """Mean learned-task accuracy = mean over tasks l of R[l, l] (diagonal: task l right after learning it)."""
    R = np.asarray(R, float)
    return float(np.mean([R[l, l] for l in range(R.shape[0])]))


def peak_per_task(R) -> list[float]:
    R = np.asarray(R, float)
    T = R.shape[0]
    return [float(max(R[l, j] for l in range(j, T))) for j in range(T)]


def run_continual_v5(agent, task, proto: dict, seed: int, mode: str, splits=("val",), trace=None,
                     stop=None, monitor: bool = True) -> dict:
    t = proto["training"]
    n_t = task.spec.n_ctx
    S = Streams(seed)
    budget = int(t["replay_budget"])
    buf = ReplayBuffer(task, budget) if mode == "R1" else None
    mon = ClipMonitor(agent.params) if monitor else None
    counts = {"cur": 0, "replay": 0, "extra": 0}
    mats = {sp: np.zeros((n_t, n_t)) for sp in splits}
    t0 = time.perf_counter()
    t_eval = 0.0
    phase_end = []
    try:
        for phase in range(n_t):
            if buf is not None:
                buf.audit(phase)
            train_phase_v5(agent, task, phase, int(t["trials_per_phase"]), S, mode, buf, mon, counts, trace, stop)
            if buf is not None and phase < n_t - 1:  # last task is never replayed (nothing follows it)
                buf.add_completed_task(phase, task.cues[("train", phase)][:budget],
                                       task.labels_for("train", phase)[:budget])
            te = time.perf_counter()
            for sp in splits:
                for j in range(n_t):
                    mats[sp][phase, j] = agent.evaluate(sp, j)
            t_eval += time.perf_counter() - te
            phase_end.append(time.perf_counter() - t0 - t_eval)
    finally:
        if mon is not None:
            mon.close()
    exp = expected_counts(proto, mode)
    adam = adam_step_counts(agent)
    n_calls = counts["cur"] + counts["replay"] + counts["extra"]
    assert counts["cur"] == exp["current_trials"], (counts, exp)
    assert counts["replay"] + counts["extra"] == exp["extra_trials"], (counts, exp)
    assert n_calls == exp["optimizer_steps"] and adam["min"] == adam["max"] == exp["optimizer_steps"], (adam, exp)
    assert (mode == "R1") == (counts["replay"] > 0) or exp["extra_trials"] == 0
    assert (mode == "C") == (counts["extra"] > 0) or exp["extra_trials"] == 0
    out: dict = {"mode": mode, "steps": {"expected_from_config": exp, "trial_calls": counts,
                                         "adam_state_steps": adam, "asserted": True},
                 "replay_payload_peak": (buf.peak_payload if buf is not None else EMPTY_PAYLOAD),
                 "replay_buffer_audit": (buf.log if buf is not None else []),
                 "old_task_buffer": mode == "R1",
                 "clip": mon.summary() if mon is not None else None,
                 "t_train_s": time.perf_counter() - t0 - t_eval, "t_eval_s": t_eval,
                 "t_phase_train_cum_s": phase_end,
                 "stream_states_end": {"A": S.A.states()} if trace is not None else None}
    for sp, R in mats.items():
        out[f"R_{sp}"] = R.tolist()
        out[f"final_mean_acc_{sp}"] = float(R[-1].mean())
        out[f"learned_acc_{sp}"] = learned_acc(R)
        out[f"learned_per_task_{sp}"] = [float(R[l, l]) for l in range(n_t)]
        out[f"peak_per_task_{sp}"] = peak_per_task(R)
        out[f"avg_forgetting_keep_{sp}"] = ST.average_forgetting(R, "keep")
        out[f"avg_forgetting_clip_{sp}"] = ST.average_forgetting(R, "clip")
        out[f"bwt_{sp}"] = ST.backward_transfer(R)
    return out


def run_association_acq_v5(agent, task, proto: dict, seed: int, splits=("val",), stop=None) -> dict:
    """Historical association ACQUISITION endpoint on a FRESH agent: ``trials_per_phase`` reward-only trials on
    context 0 with ``learning._rngs(seed, 0)``, then the greedy ``<split>_acc_acq``. The historical protocol continues
    with delay/reversal/reset probes AFTER this evaluation; those cannot influence ``*_acc_acq`` (evaluate() consumes no
    training stream), so stopping here yields the identical number at ~1/3 of the cost (checked in the selftest)."""
    t = proto["training"]
    A = Stream([int(seed), 4242, 0])  # == L._rngs(seed, 0)
    mon = ClipMonitor(agent.params)
    counts = {"cur": 0, "replay": 0, "extra": 0}
    t0 = time.perf_counter()
    try:
        hist = []
        for k in range(int(t["trials_per_phase"])):
            if stop is not None and k % 25 == 0:
                stop()
            cue, lab = task.sample_train(A.order, 0, False)
            hist.append(_do_trial(agent, mon, counts, "cur", cue, 0, int(lab[0]), A.noise, A.act, None))
    finally:
        mon.close()
    adam = adam_step_counts(agent)
    assert counts["cur"] == int(t["trials_per_phase"]) == adam["min"] == adam["max"], (counts, adam)
    correct = np.array([x["correct"] for x in hist], dtype=float)
    acq = L._acq_trials(correct, t["acquisition_window"], t["acquisition_criterion"])
    out = {"train_s": time.perf_counter() - t0, "steps": {"trial_calls": counts, "adam_state_steps": adam,
                                                          "expected": int(t["trials_per_phase"])},
           "clip": mon.summary(), "acquisition_trials": acq, "acquisition_censored": acq is None,
           "train_accuracy_last50": float(correct[-50:].mean())}
    for sp in splits:
        out[f"{sp}_acc_acq"] = agent.evaluate(sp, 0)
    return out


# ----------------------------------------------------------------------------------------------- rows
def coord_key(kind: str, family: str, mode: str, core: float, dec: float, seed: int) -> str:
    return json.dumps([kind, family, mode, float(core), float(dec), int(seed)], separators=(",", ":"))


def _row_header(env: Env, kind, phase, family, mode, core, dec, seed, splits, extra) -> dict:
    r = {"kind": kind, "phase": phase, "family": family, "mode": mode, "core_lr": float(core),
         "decoder_lr": float(dec), "seed": int(seed), "key": coord_key(kind, family, mode, core, dec, seed),
         "splits": list(splits), "protocol_hash": env.protocol_hash, "rng": Streams(seed).info(),
         "w76_sha256": env.w76_sha}
    r.update(extra or {})
    return r


def execute_row(env: Env, kind: str, phase: str, family: str, mode: str, core: float, dec: float, seed: int,
                splits=("val",), stop=None, extra: dict | None = None, proto: dict | None = None) -> dict:
    """One coordinate -> one auditable record. Exceptions other than StopRequested become a recorded failure."""
    proto = proto or env.proto
    rec = _row_header(env, kind, phase, family, mode, core, dec, seed, splits, extra)
    t0 = time.perf_counter()
    status = "ok"
    try:
        task = env.make_task(seed, proto)
        tb = time.perf_counter()
        agent, info, rep = env.make_agent(family, core, dec, seed, task, proto)
        inspect_lr(agent, core, dec)
        rec.update(model_info=info, params=rep, t_build_s=time.perf_counter() - tb)
        if kind == "assoc":
            assert mode == "-"
            rec["association"] = run_association_acq_v5(agent, task, proto, seed, splits, stop)
        else:
            rec["continual"] = run_continual_v5(agent, task, proto, seed, mode, splits, None, stop)
    except StopRequested:
        raise
    except Exception as e:  # recorded, never converted into a result
        import traceback
        status = f"failed: {e!r}"
        rec["traceback_tail"] = traceback.format_exc()[-1500:]
    rec.update(status=status, elapsed_s=time.perf_counter() - t0, peak_rss_mib=peak_rss_mib(),
               finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return rec


# ----------------------------------------------------------------------------------------------- design
def centers() -> dict:
    """(core_center, decoder_center) per family from W3_trained_w76_choices.json (GRU33 re-uses the W3 GRU37 choice as a
    CENTRE ONLY: it is a new architecture; random_reach/encdec/connectome use their own W3 choice)."""
    ch = json.load(open(CHOICES_PATH))
    src = {"connectome": "connectome", "gru33": "gru", "encdec": "encdec", "random_reach": "random_reach"}
    return {f: (float(ch[s]["core_lr"]), float(ch[s]["decoder_lr"])) for f, s in src.items()}


def axis(c: float) -> list[float]:
    return [float(f"{c * r:.6g}") for r in RATIOS]


def settings(family: str) -> list[dict]:
    """3x3 grid, core-major. ``ri``/``rj`` index RATIOS (0:1/3, 1:centre, 2:x3) -> simplicity relative to the centre."""
    cc, dc = centers()[family]
    gc, gd = axis(cc), axis(dc)
    return [{"core_lr": gc[i], "decoder_lr": gd[j], "ri": i, "rj": j} for i in range(3) for j in range(3)]


def simplicity(s: dict) -> tuple:
    """Deterministic simplicity order relative to the family centre (learning_v4_select.two_group_simplicity on the
    normalised multipliers): smaller max multiplier, then smaller geometric mean, then numeric rates."""
    return SEL.two_group_simplicity((RATIOS[s["ri"]], RATIOS[s["rj"]]), 1.0) + (s["core_lr"], s["decoder_lr"])


def search_coordinates(families=FAMILIES) -> list[dict]:
    """Every search coordinate in execution order: family priority, seed-outer, setting, then assoc, then modes."""
    out = []
    for f in families:
        for seed in SEARCH_SEEDS:
            for s in settings(f):
                out.append({"kind": "assoc", "phase": "search", "family": f, "mode": "-", "seed": seed, **s})
                for m in MODES[f]:
                    out.append({"kind": "cont", "phase": "search", "family": f, "mode": m, "seed": seed, **s})
    for c in out:
        c["key"] = coord_key(c["kind"], c["family"], c["mode"], c["core_lr"], c["decoder_lr"], c["seed"])
    return out


def check_coordinates(family: str, core: float, dec: float) -> list[dict]:
    out = []
    for seed in CHECK_SEEDS:
        base = {"phase": "check", "family": family, "seed": seed, "core_lr": float(core), "decoder_lr": float(dec)}
        out.append({"kind": "assoc", "mode": "-", **base})
        for m in MODES[family]:
            out.append({"kind": "cont", "mode": m, **base})
    for c in out:
        c["key"] = coord_key(c["kind"], c["family"], c["mode"], c["core_lr"], c["decoder_lr"], c["seed"])
    return out


def timing_coordinates() -> list[dict]:
    """<= 6 planned timing coordinates: connectome and gru33 x R0,R1,C on engineering seed 1701 at the family centre."""
    cen = centers()
    out = []
    for f in PRIMARY:
        for m in ("R0", "R1", "C"):
            out.append({"kind": "cont", "phase": "timing", "family": f, "mode": m, "seed": ENG_SEEDS[0],
                        "core_lr": cen[f][0], "decoder_lr": cen[f][1]})
    for c in out:
        c["key"] = coord_key(c["kind"], c["family"], c["mode"], c["core_lr"], c["decoder_lr"], c["seed"])
    assert len(out) <= 6
    return out


def heldout_coordinates(selection: dict) -> list[dict]:
    """Fixed order: seed-outer (100601..100620), family (connectome, gru33), assoc then R0,R1,C."""
    out = []
    for seed in HELDOUT_SEEDS:
        for f in PRIMARY:
            core, dec = selection["families"][f]["core_lr"], selection["families"][f]["decoder_lr"]
            base = {"phase": "heldout", "family": f, "seed": seed, "core_lr": float(core), "decoder_lr": float(dec)}
            out.append({"kind": "assoc", "mode": "-", **base})
            for m in ("R0", "R1", "C"):
                out.append({"kind": "cont", "mode": m, **base})
    for c in out:
        c["key"] = coord_key(c["kind"], c["family"], c["mode"], c["core_lr"], c["decoder_lr"], c["seed"])
    return out


# ----------------------------------------------------------------------------------------------- selection
def select_shared_pair(rows: list[dict], family: str, search_seeds=SEARCH_SEEDS, split: str = "val",
                       modes: tuple | None = None) -> dict:
    """ONE shared (core, decoder) pair per family across its conditions, from SEARCH-seed validation data only.

    adequate(setting)  : for EVERY condition, mean over search seeds of learned_acc (= mean_l R[l,l]) >= 0.80.
    score(setting)     : per seed, mean over the conditions of final_mean_acc; mean and SE (ddof=1) over the seeds.
    chosen             : among ADEQUATE settings, best = max mean (ties -> simplest); threshold = best.mean - best.SE;
                         chosen = simplest adequate setting with mean >= threshold (simplicity relative to the centre).
    none adequate      : the 'best diagnostic' = max over complete settings of the WORST condition's mean learned_acc
                         (ties -> simplicity); flagged inadequate. Forgetting is never used. Boundary picks are flagged
                         (tuning-range limitation) and the grid is not extended. A setting lacking any (condition, seed)
                         ok row is ineligible."""
    modes = tuple(modes) if modes else MODES[family]  # `modes` only for the secondary per-condition-tuned analysis
    stg = settings(family)
    table = {}
    for s in stg:
        k = (s["core_lr"], s["decoder_lr"])
        per = {m: {} for m in modes}
        for r in rows:
            if (r.get("kind") == "cont" and r.get("phase") == "search" and r["family"] == family
                    and (r["core_lr"], r["decoder_lr"]) == k and r["mode"] in per and r["status"] == "ok"
                    and r["seed"] in search_seeds):
                R = r["continual"][f"R_{split}"]
                per[r["mode"]][r["seed"]] = (learned_acc(R), float(np.mean(np.asarray(R)[-1])))
        complete = all(len(per[m]) == len(search_seeds) for m in modes)
        ent = {"core_lr": k[0], "decoder_lr": k[1], "ri": s["ri"], "rj": s["rj"], "complete": complete,
               "missing": {m: sorted(set(search_seeds) - set(per[m])) for m in modes if len(per[m]) != len(search_seeds)}}
        if complete:
            lm = {m: float(np.mean([per[m][sd][0] for sd in search_seeds])) for m in modes}
            fm = {m: float(np.mean([per[m][sd][1] for sd in search_seeds])) for m in modes}
            seed_scores = [float(np.mean([per[m][sd][1] for m in modes])) for sd in search_seeds]
            mu, se = mean_se(seed_scores)
            ent.update(learned_mean_by_condition=lm, final_mean_by_condition=fm, score_mean=mu, score_se=se,
                       seed_scores=seed_scores, adequate=bool(all(v >= ADEQUACY_LEARNED for v in lm.values())),
                       worst_condition_learned=min(lm.values()))
        table[k] = ent
    elig = [e for e in table.values() if e["complete"]]
    out = {"family": family, "rule": select_shared_pair.__doc__, "split": split, "adequacy_threshold": ADEQUACY_LEARNED,
           "adequacy_definition": "per condition: mean over search seeds of mean_l R[l,l]; adequate if all conditions >= threshold",
           "n_settings": len(stg), "n_complete": len(elig), "settings": [table[(s["core_lr"], s["decoder_lr"])] for s in stg]}
    if not elig:
        out.update(chosen=None, reason="no complete setting")
        return out

    def simp(e):
        return simplicity({"ri": e["ri"], "rj": e["rj"], "core_lr": e["core_lr"], "decoder_lr": e["decoder_lr"]})

    adequate = [e for e in elig if e["adequate"]]
    if adequate:
        best_mean = max(e["score_mean"] for e in adequate)
        best = min((e for e in adequate if e["score_mean"] == best_mean), key=simp)
        thr = best["score_mean"] - best["score_se"]
        cand = [e for e in adequate if e["score_mean"] >= thr]
        chosen = min(cand, key=simp)
        out.update(adequate_found=True, best=_pick(best), threshold=thr, n_candidates=len(cand),
                   candidates=[_pick(c) for c in sorted(cand, key=simp)], chosen=_pick(chosen),
                   chosen_equals_best=chosen is best, n_adequate=len(adequate))
        edge = {chosen["ri"], chosen["rj"], best["ri"], best["rj"]} & {0, 2}
    else:
        wbest = max(e["worst_condition_learned"] for e in elig)
        chosen = min((e for e in elig if e["worst_condition_learned"] == wbest), key=simp)
        out.update(adequate_found=False, n_adequate=0, chosen=_pick(chosen), best=_pick(chosen),
                   flag="INADEQUATE: no setting has every condition's mean learned-task accuracy >= threshold; "
                        "best diagnostic = max worst-condition learned accuracy; no valid target comparison")
        edge = {chosen["ri"], chosen["rj"]} & {0, 2}
    out["boundary_selection"] = bool(edge)
    out["tuning_range_limitation"] = ("chosen/best setting lies on the edge of the 3x3 grid in at least one axis; "
                                      "grid not extended" if edge else None)
    return out


def _pick(e: dict) -> dict:
    return {k: e[k] for k in ("core_lr", "decoder_lr", "ri", "rj", "score_mean", "score_se", "adequate",
                              "learned_mean_by_condition", "final_mean_by_condition") if k in e}


# ----------------------------------------------------------------------------------------------- forecast
# Performance-blind prior: seconds per optimizer step derived from the W3 per-run times (assoc + continual +
# oracle ~ 4000 optimizer steps): connectome 16 s, gru 10 s, encdec 4.4 s, random_reach 14.3 s; replaced per family
# by the timing runs when available. Fixed per-row overhead (build + calibration + evaluation) 0.5 s.
PRIOR_SEC_PER_STEP = {"connectome": 16.0 / 4000, "gru33": 10.0 / 4000, "encdec": 4.4 / 4000, "random_reach": 14.3 / 4000}
ROW_OVERHEAD_S = 0.5


def per_row_seconds(family: str, proto: dict, timing: dict | None = None) -> dict:
    sps = PRIOR_SEC_PER_STEP[family]
    if timing and family in timing:  # measured seconds per optimizer step from the timing runs (elapsed only)
        sps = timing[family]
    out = {m: expected_counts(proto, m)["optimizer_steps"] * sps + ROW_OVERHEAD_S for m in ("R0", "R1", "C")}
    out["assoc"] = int(proto["training"]["trials_per_phase"]) * sps + ROW_OVERHEAD_S
    return out


def timing_table_from_rows(rows: list[dict], proto: dict) -> dict:
    """seconds-per-step per family from timing JSONL rows: total elapsed / total optimizer steps (performance-blind)."""
    acc: dict = {}
    for r in rows:
        if r.get("kind") == "cont" and r.get("status") == "ok":
            n = r["continual"]["steps"]["adam_state_steps"]["max"]
            a = acc.setdefault(r["family"], [0.0, 0])
            a[0] += r["elapsed_s"] - ROW_OVERHEAD_S
            a[1] += n
    return {f: v[0] / v[1] for f, v in acc.items() if v[1] > 0}


def forecast(proto: dict, families=FAMILIES, timing: dict | None = None, n_search_seeds=len(SEARCH_SEEDS),
             n_check_seeds=len(CHECK_SEEDS), phase_budget_s: float = 150 * 60, timing_cap_s: float = 300.0,
             load_s: float = 15.0) -> dict:
    per_family = {}
    for f in families:
        t = per_row_seconds(f, proto, timing)
        n_units = 9 * n_search_seeds + n_check_seeds
        cont = sum(t[m] for m in MODES[f]) * n_units
        assoc = t["assoc"] * n_units
        per_family[f] = {"per_row_s": t, "n_units": n_units, "continual_s": cont, "assoc_s": assoc,
                         "total_s": cont + assoc, "n_continual": n_units * len(MODES[f]), "n_assoc": n_units}
    return {"per_family": per_family, "total_s": sum(v["total_s"] for v in per_family.values()),
            "n_continual": sum(v["n_continual"] for v in per_family.values()),
            "n_assoc": sum(v["n_assoc"] for v in per_family.values()), "load_s": load_s,
            "timing_cap_s": timing_cap_s, "phase_budget_s": phase_budget_s,
            "note": "prior per-step seconds from W3 per-run times; replaced by timing runs when a timing table is passed"}


def plan_with_omission(proto: dict, timing: dict | None = None, phase_budget_s: float = 150 * 60,
                       margin_s: float = 300.0, timing_cap_s: float = 300.0) -> dict:
    """Performance-blind: omit random_reach first, then encdec, ONLY if the forecast does not fit; primary never dropped."""
    fams = list(FAMILIES)
    avail = phase_budget_s - timing_cap_s - margin_s
    steps = []
    while True:
        fc = forecast(proto, tuple(fams), timing, phase_budget_s=phase_budget_s, timing_cap_s=timing_cap_s)
        need = fc["total_s"] + fc["load_s"]
        fits = need <= avail
        steps.append({"families": list(fams), "forecast_s": need, "available_s": avail, "fits": fits})
        if fits:
            break
        if "random_reach" in fams:
            fams.remove("random_reach")
        elif "encdec" in fams:
            fams.remove("encdec")
        else:
            break
    return {"families": fams, "omitted": [f for f in SECONDARY if f not in fams], "fits": steps[-1]["fits"],
            "steps": steps, "final_forecast": fc, "available_s": avail,
            "primary_never_dropped": all(f in fams for f in PRIMARY)}
