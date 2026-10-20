"""Learning study v2 (protocol ``learning-v2.0``): follow-up to the v1 benchmark (spec §4C, §5).

Relationship to v1 (never overwritten): v1 test splits were SEEN, so v2 uses disjoint seeds
(search 11-12, pilot 21-22, final 201-205; v1 used 1-2 and 101-105), widened equal-budget
validation-only learning-rate searches, a diagnosed + corrected local rule (``comp_feedback``), explicit
interface controls, v2 sign policy (``graph/signs.py``) and v2 null graphs (``graph/nulls_v2.py``).
Everything here is a SYNTHETIC cue-valence task on a (real pilot) circuit with ENGINEERED interfaces;
results are model-relative and not biological evidence.

Stages (results/learning_v2/<stage>/): ``search`` (val only) -> ``pilot`` (exploratory) ->
``confirm`` (only after ``learning2 freeze`` froze configs/prereg_v2.yaml).
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from flycircuit.config import project_root
from flycircuit.eval import learning as L
from flycircuit.eval import stats as S
from flycircuit.graph.controls import random_sparsity
from flycircuit.models.baselines import (
    EncoderDecoderOnly,
    GRUBaseline,
    match_encdec_hidden,
    match_gru_hidden,
    match_gru_hidden_flops,
)
from flycircuit.models.sparse_rnn import ModelSpec, SparseConnectomeRNN
from flycircuit.plasticity.agent import LearningAgent
from flycircuit.plasticity.three_factor import PlasticityConfig
from flycircuit.tasks.association import ContingencyTask
from flycircuit.tasks.graph_loader import CircuitBundle

PROTOCOL_VERSION = "learning-v2.0"


class TimeBudget(Exception):
    """Raised between runs when the wall-clock slice of a resumable stage is used up."""


_DEADLINE: list[float | None] = [None]


def set_time_budget(minutes: float | None) -> None:
    _DEADLINE[0] = None if not minutes else time.monotonic() + 60.0 * minutes


def _check_deadline() -> None:
    if _DEADLINE[0] is not None and time.monotonic() > _DEADLINE[0]:
        raise TimeBudget

# ---- arms (learner + rule) and conditions ------------------------------------------------------
ARMS: dict[str, dict] = {
    "bptt": {"learner": "bptt"},
    "decoder_only": {"learner": "decoder_only"},  # frozen core + trainable linear decoder
    "plast_fixed": {"learner": "plastic_fixed_dec", "rule": "comp_feedback"},
    "plast_dec": {"learner": "plastic_plus_dec", "rule": "comp_feedback"},
    "plast_modoff": {"learner": "plastic_mod_off", "rule": "comp_feedback"},  # modulation disabled
    "plast_shuffled": {"learner": "plastic_fixed_dec", "rule": "comp_feedback",
                       "shuffle_feedback": True},  # compartment feedback permuted
    "plast_np_scalar": {"learner": "plastic_fixed_dec", "rule": "np_window"},  # scalar TD only
    "plast_homeo": {"learner": "plastic_plus_dec", "rule": "comp_feedback", "homeostasis": True},
}
SPARSE_FAMS = ("connectome", "connectome_noz", "random_reach", "kc_static")
DENSE_FAMS = ("gru", "gru_c", "encdec", "gru_z", "encdec_z")
LR_FROM = {"plast_modoff": "plast_fixed", "plast_homeo": "plast_dec"}  # no own search
DECLR_FROM = {"plast_dec": "decoder_only", "plast_homeo": "decoder_only"}  # decoder lr source


def is_null(f: str) -> bool:
    return f.startswith("null") or f.startswith("gnull")


def kind_of(family: str) -> str:
    """Learning-rate search kind: all nulls share the kind of plain null0."""
    return "null" if is_null(family) else family


def protocol_path() -> Path:
    return project_root() / "configs" / "experiments" / "learning_v2_protocol.yaml"


def load_protocol(path: Path | None = None) -> dict:
    return yaml.safe_load((path or protocol_path()).read_text())


def protocol_hash(proto: dict) -> str:
    return hashlib.sha256(json.dumps(proto, sort_keys=True).encode()).hexdigest()[:16]


def conditions(proto: dict, small: bool = False) -> list[tuple[str, str, str, str]]:
    """(arm, family, mode, pairing) tuples. pairing: 'single' | 'paired' (null i with seed index i)
    | 'factorial' (every null x every seed). mode: 'std' | 'replay'."""
    kn = proto["seeds"]["null_graphs"]
    nulls = [f"null{i}" for i in range(kn)]
    gnulls = [f"gnull{i}" for i in range(kn)]
    out: list[tuple[str, str, str, str]] = []

    def add(arm, fams, mode="std", pairing=None):
        for f in fams:
            p = pairing or ("paired" if is_null(f) else "single")
            out.append((arm, f, mode, p))

    if small:  # pilot: pipeline sanity, connectome vs GRU
        for arm in ARMS:
            add(arm, ["connectome"])
        add("bptt", ["gru", "encdec"])
        return out
    add("bptt", ["connectome", "random_reach", "gru", "gru_c", "encdec", "connectome_noz", "gru_z",
                 "encdec_z"])
    add("bptt", nulls, pairing="factorial")
    add("bptt", gnulls)
    add("decoder_only", ["connectome", "random_reach", "gru", "encdec", "kc_static"] + nulls)
    add("plast_fixed", ["connectome", "random_reach"] + nulls)
    add("plast_dec", ["connectome", "random_reach"] + gnulls)
    add("plast_dec", nulls, pairing="factorial")
    for arm in ("plast_modoff", "plast_shuffled", "plast_np_scalar", "plast_homeo"):
        add(arm, ["connectome"])
    add("bptt", ["connectome", "gru", "encdec"] + nulls, mode="replay")
    add("plast_dec", ["connectome"], mode="replay")
    return out


# ---- reachability (random control with verified input-output reachability) -------------------------
def reachable_fraction(g, src: np.ndarray, dst: np.ndarray) -> float:
    """Fraction of ``dst`` nodes reachable from the set ``src`` along directed edges (BFS on CSR)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import breadth_first_order

    a = sp.csr_matrix((np.ones(g.n_edges), (g.pre, g.post)), shape=(g.n, g.n))
    seen = np.zeros(g.n, dtype=bool)
    seen[src] = True
    # multi-source BFS through a virtual super-source row appended to the matrix
    n = g.n
    rows = np.concatenate([np.full(len(src), n), a.nonzero()[0]])
    cols = np.concatenate([src, a.nonzero()[1]])
    b = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n + 1, n + 1))
    order = breadth_first_order(b, n, directed=True, return_predecessors=False)
    seen[order[order < n]] = True
    return float(seen[dst].mean())


def random_reach_graph(base: CircuitBundle, seed: int, max_tries: int = 200):
    """random_sparsity resampled until every readout node is reachable from the sensory set (the
    connectome's own reachable fraction is reported as the reference; never silently relaxed)."""
    ref = reachable_fraction(base.graph, base.sensory_idx, base.readout_idx)
    best, best_f = None, -1.0
    for k in range(max_tries):
        g = random_sparsity(base.graph, seed=seed * 1000 + k).graph
        f = reachable_fraction(g, base.sensory_idx, base.readout_idx)
        if f > best_f:
            best, best_f = g, f
        if f >= ref:
            return g, {"reachable_fraction": f, "connectome_reachable_fraction": ref, "tries": k + 1}
    return best, {"reachable_fraction": best_f, "connectome_reachable_fraction": ref,
                  "tries": max_tries, "warning": "reference reachability not reached"}


# ---- dense controls with matched calibration ----------------------------------------------------
class ZGRU(GRUBaseline):
    """GRU whose decoder reads label-free standardised hidden activity (same procedure as the
    sparse models' readout calibration): isolates the effect of the readout normalisation."""

    kind = "gru_zscored"

    def __init__(self, n_in, n_out, hidden, seed=0):
        super().__init__(n_in, n_out, hidden, seed)
        self.register_buffer("mu", torch.zeros(hidden))
        self.register_buffer("sd", torch.ones(hidden))

    def forward(self, x_seq, state=None, u_seq=None):
        if state is None:
            state = self.init_state(x_seq.shape[1], x_seq.device)
        out, h = self.cell(torch.relu(self.enc(x_seq)), state["h"])
        return self.dec(((out - self.mu) / self.sd).clamp(-5, 5)), {"h": h}


class ZEncDec(EncoderDecoderOnly):
    kind = "encdec_zscored"

    def __init__(self, n_in, n_out, hidden, seed=0):
        super().__init__(n_in, n_out, hidden, seed)
        self.register_buffer("mu", torch.zeros(hidden))
        self.register_buffer("sd", torch.ones(hidden))

    def forward(self, x_seq, state=None, u_seq=None):
        hid = torch.tanh(self.net[0](x_seq))
        return self.net[2](((hid - self.mu) / self.sd).clamp(-5, 5)), {}


@torch.no_grad()
def calibrate_dense(model, task: ContingencyTask, n: int = 128) -> None:
    sp = task.spec
    rng = np.random.default_rng(np.random.SeedSequence([task.seed, 777]))
    cues = np.zeros((n, sp.n_cue), dtype=np.float32)
    for i in range(n):
        cues[i, rng.permutation(sp.n_cue)[: sp.k_active]] = 1.0
    x = torch.as_tensor(task.make_input(cues, 0, rng))
    if isinstance(model, ZGRU):
        out, _ = model.cell(torch.relu(model.enc(x)), model.init_state(n)["h"])
    else:
        out = torch.tanh(model.net[0](x))
    a = out[-sp.resp_steps:].reshape(-1, out.shape[-1])
    model.mu.copy_(a.mean(0))
    model.sd.copy_(a.std(0) + 1e-3)


# ---- construction -------------------------------------------------------------------------------
def _spec(proto: dict, n_in: int, seed: int) -> ModelSpec:
    m = proto["model"]
    return ModelSpec(n_in=n_in, n_out=2, dt=m["dt"], tau=m["tau"], prior=m["prior"],
                     init_gain=m["init_gain"], nonneg_gain=m["nonneg_gain"], seed=seed)


def _sparse_model(proto, b: CircuitBundle, seed: int, n_in: int) -> SparseConnectomeRNN:
    return L.build_sparse_model(proto, b, seed, n_in)


def _kc_static(proto, base: CircuitBundle, seed: int, n_in: int) -> SparseConnectomeRNN:
    """Interface-only control: same engineered encoder -> KC-like units -> z-score -> linear decoder,
    NO recurrent anatomy at all (zero edges)."""
    n_kc = len(base.sensory_idx)
    z = np.zeros(0, dtype=np.int64)
    model = SparseConnectomeRNN(n_kc, z, z, np.zeros(0, dtype=np.float32), _spec(proto, n_in, seed),
                                sign=np.ones(n_kc, dtype=np.float32), unknown_sign_policy="zero",
                                sensory_idx=np.arange(n_kc), readout_idx=np.arange(n_kc))
    L._fixed_encoder(model, proto["model"]["encoder_fan_in"], seed, proto["task"]["n_cue"])
    return model


def ref_param_report(proto: dict, base: CircuitBundle, n_in: int) -> dict:
    return _sparse_model(proto, base, 0, n_in).param_report()


def build_model_v2(proto, base, packs, family, seed, n_in):
    """-> (model, bundle|None, zscore, info). packs: {'plain': [VersionedGraph], 'group': [...]}"""
    info: dict = {}
    if family in ("connectome", "connectome_noz"):
        return _sparse_model(proto, base, seed, n_in), base, family != "connectome_noz", info
    if family == "random_reach":
        g, info = random_reach_graph(base, seed)
        b = base.with_graph(g, "random_reach")
        return _sparse_model(proto, b, seed, n_in), b, True, info
    if family == "kc_static":
        return _kc_static(proto, base, seed, n_in), None, True, info
    if family.startswith("gnull"):
        b = base.with_graph(packs["group"][int(family[5:])], family)
        return _sparse_model(proto, b, seed, n_in), b, True, info
    if family.startswith("null"):
        b = base.with_graph(packs["plain"][int(family[4:])], family)
        return _sparse_model(proto, b, seed, n_in), b, True, info
    ref = ref_param_report(proto, base, n_in)
    if family in ("gru", "gru_z"):
        h = match_gru_hidden(ref["trainable_params"], n_in, 2)
        return (GRUBaseline if family == "gru" else ZGRU)(n_in, 2, h, seed), None, True, \
            {"hidden": h}
    if family == "gru_c":
        h = match_gru_hidden_flops(ref["flops_per_step"], n_in, 2)
        return GRUBaseline(n_in, 2, h, seed), None, True, {"hidden": h, "match": "flops"}
    if family in ("encdec", "encdec_z"):
        h = match_encdec_hidden(ref["trainable_params"], n_in, 2)
        return (EncoderDecoderOnly if family == "encdec" else ZEncDec)(n_in, 2, h, seed), None, \
            True, {"hidden": h}
    raise ValueError(f"unknown family {family!r}")


def plast_cfg_for(proto: dict, arm: str) -> PlasticityConfig:
    pl, a = proto["plasticity"], ARMS[arm]
    return PlasticityConfig(
        decay=pl["decay"], gain_lo=pl["gain_lo"], gain_hi=pl["gain_hi"], gamma=pl["gamma"],
        lr_value=pl["lr_value"], rule=a.get("rule", "np_v1"),
        optimizer=pl.get("optimizer", "sgd"),
        homeostasis=pl["homeostasis"] if a.get("homeostasis") else 0.0,
        shuffle_feedback=a.get("shuffle_feedback", False))


def make_agent_v2(proto, base, packs, arm, family, lr, seed, task, decoder_lr=None):
    model, bundle, zs, info = build_model_v2(proto, base, packs, family, seed, task.spec.n_in)
    if isinstance(model, (ZGRU, ZEncDec)):
        calibrate_dense(model, task)
    a = ARMS[arm]
    mask = bundle.plastic_edge_mask if (bundle is not None and a["learner"].startswith("plastic")) \
        else None
    tr = proto["training"]
    agent = LearningAgent(
        model, a["learner"], task, lr, plastic_mask=mask,
        explore_noise=tr["explore_noise"],
        decoder_lr=tr["decoder_lr"] if decoder_lr is None else decoder_lr, seed=seed,
        plast_cfg=plast_cfg_for(proto, arm), logit_cap=tr.get("policy_logit_cap", 2.0), zscore=zs)
    meta = {"n_params_trainable": int(sum(p.numel() for p in agent.params)),
            "n_plastic_edges": None if agent.plastic is None else int(agent.plastic.n_plastic),
            "arm": arm, "family": family, **info}
    if bundle is not None:
        meta.update(n_nodes=bundle.graph.n, n_edges=bundle.graph.n_edges,
                    edge_hash=bundle.graph.edge_hash())
    elif family == "kc_static":
        meta.update(n_nodes=model.n, n_edges=0)
    return agent, meta


# ---- oracle ceilings (no recurrence, no anatomy) ---------------------------------------------------
def _fit_logreg(x: torch.Tensor, y: torch.Tensor):
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(x @ w + b, y) + 1e-3 * (w**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def oracle_ceiling(task: ContingencyTask, splits=("val", "test")) -> dict:
    """Logistic regression on the RAW cue (per context), trained on the train split only; reports
    validation/test accuracy: what an encoder/decoder with no recurrent core can in principle do."""
    out = {sp: [] for sp in splits}
    for ctx in range(task.spec.n_ctx):
        w, b = _fit_logreg(torch.as_tensor(task.cues[("train", ctx)]),
                           torch.as_tensor(task.labels[("train", ctx)]).float())
        for sp in splits:
            xs = torch.as_tensor(task.cues[(sp, ctx)])
            ys = torch.as_tensor(task.labels[(sp, ctx)])
            out[sp].append(float((((xs @ w + b) > 0).long() == ys).float().mean()))
    return {k: float(np.mean(v)) for k, v in out.items()}


# ---- checkpoints --------------------------------------------------------------------------------
def ckpt_dir(stage: str) -> Path:
    d = project_root() / "runs" / "learning_v2_ckpt" / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_checkpoint(agent: LearningAgent, meta: dict, stage: str, tag: str) -> str:
    name = f"{meta['arm']}__{meta['family']}__{meta['mode']}__lr{meta['lr']:g}__s{meta['seed']}__{tag}.pt"
    path = ckpt_dir(stage) / name
    torch.save({"payload": agent.checkpoint_payload(), "meta": meta}, path)
    try:
        return str(path.relative_to(project_root()))
    except ValueError:  # tests redirect the directory
        return str(path)


def _restore(agent: LearningAgent, payload: dict) -> None:
    agent.model.load_state_dict(payload["model"])
    if agent.plastic is not None:
        agent.plastic.gain = payload["plastic_gain"].clone()
    agent.value.v = payload["critic_v"].copy()
    if agent.sparse:
        agent.f_mu, agent.f_sd = payload["f_mu"].clone(), payload["f_sd"].clone()


def load_checkpoint_agent(path: str | Path, base: CircuitBundle, packs: dict | None = None,
                          proto: dict | None = None):
    """Rebuild the exact trained learner from a checkpoint written by ``save_checkpoint``.
    Returns (agent, task, meta). ``base`` must be the circuit loaded with the same sign policy."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    meta = blob["meta"]
    proto = proto or load_protocol()
    task = L.make_task(proto, meta["task_seed"])
    agent, _ = make_agent_v2(proto, base, packs or {}, meta["arm"], meta["family"], meta["lr"],
                             meta["seed"], task, meta.get("decoder_lr"))
    _restore(agent, blob["payload"])
    return agent, task, meta


# ---- protocols ----------------------------------------------------------------------------------
def _eval_matrix_row(agent, splits, n_t):
    return {sp: [agent.evaluate(sp, j) for j in range(n_t)] for sp in splits}


def run_association_v2(agent, task, proto, seed, splits, ckpt_fn=None) -> dict:
    t = proto["training"]
    rngs = L._rngs(seed, 0)
    out: dict = {"acc_untrained": {sp: agent.evaluate(sp, 0) for sp in splits}}
    h = L.train_phase(agent, task, 0, t["trials_per_phase"], rngs)
    correct = np.array([x["correct"] for x in h], dtype=float)
    acq = L._acq_trials(correct, t["acquisition_window"], t["acquisition_criterion"])
    out.update(acquisition_trials=acq, acquisition_censored=acq is None,
               train_accuracy_last50=float(correct[-50:].mean()),
               accumulated_reward_acq=float(sum(x["reward"] for x in h)),
               n_training_trials_acq=len(h))
    for sp in splits:
        out[f"{sp}_acc_acq"] = agent.evaluate(sp, 0)
    if ckpt_fn:
        out["ckpt_post_acquisition"] = ckpt_fn(agent, "post_acq")
    # --- delayed retention probe: D un-reinforced trials, then re-test (isolates homeostasis) ---
    agent.idle(int(t["delay_trials"]))
    for sp in splits:
        out[f"{sp}_acc_after_delay"] = agent.evaluate(sp, 0)
    # --- reversal ---
    h2 = L.train_phase(agent, task, 0, t["reversal_trials"], rngs, reversed_=True)
    c2 = np.array([x["correct"] for x in h2], dtype=float)
    out["perseveration_rate"] = float(1.0 - c2[: t["perseveration_window"]].mean())
    out["accumulated_reward_rev"] = float(sum(x["reward"] for x in h2))
    out["reversal_acquisition_trials"] = L._acq_trials(c2, t["acquisition_window"],
                                                       t["acquisition_criterion"])
    out["n_training_trials_rev"] = len(h2)
    for sp in splits:
        out[f"{sp}_acc_reversed"] = agent.evaluate(sp, 0, reversed_=True)
        out[f"{sp}_acc_old_contingency_after_reversal"] = agent.evaluate(sp, 0, reversed_=False)
    # --- reset probe (non-destructive): which component stores the learned association? -----------
    payload = agent.checkpoint_payload()
    rp = {}
    for kind in ("state", "gain", "decoder", "full"):
        agent.reset(kind)
        rp[kind] = agent.evaluate(splits[0], 0, reversed_=True)  # accuracy on CURRENT contingency
        _restore(agent, payload)
        agent.reset("state")  # carried state was None anyway; keeps API honest
    rp["none"] = agent.evaluate(splits[0], 0, reversed_=True)
    out["reset_probe_" + splits[0] + "_acc_current_contingency"] = rp
    return out


def run_continual_v2(agent, task, proto, seed, splits, ckpt_fn=None) -> dict:
    t = proto["training"]
    n_t = task.spec.n_ctx
    rngs = L._rngs(seed, 1)
    budget = int(t.get("replay_per_task", 0))
    replay: dict = {}
    mats = {sp: np.zeros((n_t, n_t)) for sp in splits}
    init = {sp: np.array([agent.evaluate(sp, j) for j in range(n_t)]) for sp in splits}
    n_trials = 0
    for phase in range(n_t):
        L.train_phase(agent, task, phase, t["trials_per_phase"], rngs, replay=replay or None,
                      replay_ctxs=tuple(replay))
        n_trials += t["trials_per_phase"] * (2 if replay else 1)
        if budget > 0:
            replay[phase] = (task.cues[("train", phase)][:budget],
                             task.labels_for("train", phase)[:budget])
        for sp in splits:
            for j in range(n_t):
                mats[sp][phase, j] = agent.evaluate(sp, j)
    out: dict = {"n_training_trials": n_trials, "replay_items_stored": budget * (n_t - 1),
                 "replay_storage_bits": budget * (n_t - 1) * (task.spec.n_cue + 1)}
    for sp, R in mats.items():
        out[f"R_{sp}"] = R.tolist()
        out[f"R_init_{sp}"] = init[sp].tolist()
        out[f"final_mean_acc_{sp}"] = float(R[-1].mean())
        out[f"avg_forgetting_keep_{sp}"] = S.average_forgetting(R, "keep")
        out[f"avg_forgetting_clip_{sp}"] = S.average_forgetting(R, "clip")
        out[f"bwt_{sp}"] = S.backward_transfer(R)
        out[f"fwt_{sp}"] = S.forward_transfer(R, init[sp])  # reference = untrained accuracy on task j
        out[f"lag1_retention_{sp}"] = float(np.mean([R[j + 1, j] - R[j, j] for j in range(n_t - 1)]))
    if ckpt_fn:
        out["ckpt_final"] = ckpt_fn(agent, "continual_final")
    return out


def peak_rss_mib() -> float:
    return L.peak_rss_mib()


def run_one_v2(proto, base, packs, arm, family, mode, lr, seed, splits=("val", "test"),
               protocols=("association", "continual"), stage=None, save_ckpt=False,
               decoder_lr=None) -> dict:
    t0 = time.perf_counter()
    proto_run = copy.deepcopy(proto)
    if mode == "replay":
        proto_run["training"]["replay_per_task"] = proto["training"]["replay_budget"]
    else:
        proto_run["training"]["replay_per_task"] = 0
    task = L.make_task(proto_run, seed)
    rec = {"arm": arm, "family": family, "mode": mode, "lr": lr, "seed": seed,
           "decoder_lr": decoder_lr, "protocol": PROTOCOL_VERSION,
           "protocol_hash": protocol_hash(proto), "splits": list(splits), "stage": stage}
    status = "ok"
    try:
        rec["oracle_raw_cue_logreg"] = oracle_ceiling(task, tuple(splits))
        meta_base = {"arm": arm, "family": family, "mode": mode, "lr": lr, "seed": seed,
                     "task_seed": seed, "decoder_lr": decoder_lr,
                     "protocol_hash": protocol_hash(proto)}

        def mk(tag_prefix):
            if not save_ckpt:
                return None
            return lambda ag, tag: save_checkpoint(ag, {**meta_base, "tag": tag}, stage or "adhoc", tag)

        if "association" in protocols:
            agent, info = make_agent_v2(proto_run, base, packs, arm, family, lr, seed, task, decoder_lr)
            rec.update(info)
            rec["association"] = run_association_v2(agent, task, proto_run, seed, list(splits),
                                                     mk("assoc"))
        if "continual" in protocols:
            agent, info = make_agent_v2(proto_run, base, packs, arm, family, lr, seed, task, decoder_lr)
            rec.update(info)
            rec["continual"] = run_continual_v2(agent, task, proto_run, seed, list(splits), mk("cont"))
    except Exception as e:  # recorded, never converted into a result
        status = f"failed: {e!r}"
    rec.update(status=status, elapsed_s=time.perf_counter() - t0, peak_rss_mib=peak_rss_mib())
    return rec


# ---- stage storage -------------------------------------------------------------------------------
def stage_dir(stage: str) -> Path:
    d = project_root() / "results" / "learning_v2" / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(r) -> str:
    return f"{r['arm']}|{r['family']}|{r['mode']}|{r['lr']:g}|{r['seed']}"


def read_records(stage: str) -> list[dict]:
    p = stage_dir(stage) / "records.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def _append(stage: str, rec: dict) -> None:
    with open(stage_dir(stage) / "records.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---- equal-budget adaptive validation search ------------------------------------------------------
def initial_grid(proto: dict, arm: str) -> list[float]:
    return list(proto["search"]["initial_grid"]["plastic" if arm.startswith("plast") else "gradient"])


def search_kinds(proto: dict) -> list[tuple[str, str]]:
    """(arm, kind) pairs that get their own search; others borrow (LR_FROM)."""
    seen, out = set(), []
    for arm, fam, mode, _ in conditions(proto):
        if arm in LR_FROM or mode != "std":
            continue
        k = (arm, kind_of(fam))
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _fam_for_kind(kind: str) -> str:
    return "null0" if kind == "null" else kind


def _eval_lr(ctx, lr: float) -> float:
    proto, base, packs, stage, arm, fam, seeds, dec_lr, done, log, name = ctx
    vals = []
    for sd in seeds:
        probe = {"arm": arm, "family": fam, "mode": "std", "lr": lr, "seed": sd}
        r = done.get(_key(probe))
        if r is None:
            _check_deadline()
            r = run_one_v2(proto, base, packs, arm, fam, "std", lr, sd, splits=("val",),
                           protocols=("continual",), stage=stage, decoder_lr=dec_lr)
            _append(stage, r)
            done[_key(r)] = r
            log(f"search {name:30s} lr={lr:<8g} seed={sd} {r['status']} {r['elapsed_s']:.1f}s")
        if r["status"] == "ok":
            vals.append(r["continual"]["final_mean_acc_val"])
    return float(np.mean(vals)) if vals else float("-inf")


def run_search(proto, base, packs, stage="search", log=print, only: set | None = None) -> dict:
    """For every (arm, kind): exactly ``budget`` learning-rate evaluations x ``seeds`` on VALIDATION
    only (continual protocol, criterion = mean final-phase val accuracy). Starts from a 5-point grid
    and spends the remaining evaluations extending toward the better edge or refining around an
    interior optimum, so every condition gets the same budget. Edge optima are flagged."""
    s = proto["search"]
    budget, seeds = s["budget_lr_evals"], s["seeds"]
    done = {_key(r): r for r in read_records(stage)}
    sel_path = stage_dir(stage) / "lr_selected.json"
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    order = sorted(search_kinds(proto), key=lambda ak: (ak[0] != "decoder_only", ak[0], ak[1]))
    for arm, kind in order:  # decoder_only first: its lr is the decoder lr of plastic+decoder arms
        if only and f"{arm}|{kind}" not in only:
            continue
        name = f"{arm}|{kind}"
        if name in sel and sel[name].get("complete"):
            continue
        fam = _fam_for_kind(kind)
        dec_lr = None
        if arm in DECLR_FROM and f"{DECLR_FROM[arm]}|{kind}" in sel:
            dec_lr = sel[f"{DECLR_FROM[arm]}|{kind}"]["lr"]
        scores: dict[float, float] = {}
        ctx = (proto, base, packs, stage, arm, fam, seeds, dec_lr, done, log, name)
        for lr in initial_grid(proto, arm):
            scores[lr] = _eval_lr(ctx, lr)
        while len(scores) < budget:
            lrs = sorted(scores)
            best = max(lrs, key=lambda k: (scores[k], -k))
            i = lrs.index(best)
            if i == 0:
                nxt = best / 3.0
            elif i == len(lrs) - 1:
                nxt = best * 3.0
            else:  # interior: refine on the side of the better neighbour (geometric midpoint)
                nb = lrs[i - 1] if scores[lrs[i - 1]] >= scores[lrs[i + 1]] else lrs[i + 1]
                nxt = float(np.sqrt(best * nb))
            scores[nxt] = _eval_lr(ctx, nxt)
        lrs = sorted(scores)
        best = max(lrs, key=lambda k: (scores[k], -k))
        sel[name] = {"lr": best, "decoder_lr": dec_lr, "complete": True,
                     "n_lr_evals": len(scores), "n_seeds": len(seeds),
                     "optimum_at_edge": bool(best in (lrs[0], lrs[-1])),
                     "val_scores": {f"{k:g}": scores[k] for k in lrs}}
        sel_path.write_text(json.dumps(sel, indent=2))
    sel_path.write_text(json.dumps(sel, indent=2))
    return sel


def lr_for(sel: dict, arm: str, family: str) -> tuple[float, float | None]:
    src = LR_FROM.get(arm, arm)
    kind = kind_of(family)
    e = sel[f"{src}|{kind}"]
    decl = None
    if arm in DECLR_FROM:
        d = sel.get(f"{DECLR_FROM[arm]}|{kind}")
        decl = d["lr"] if d else None
    return e["lr"], decl


def build_jobs(proto, sel, seeds, small=False):
    kn = proto["seeds"]["null_graphs"]
    jobs = []
    for arm, fam, mode, pairing in conditions(proto, small):
        lr, dlr = lr_for(sel, arm, fam)
        for si, s in enumerate(seeds):
            if is_null(fam) and pairing == "paired" and int(fam.lstrip("gnul")) != si % kn:
                continue
            jobs.append((arm, fam, mode, lr, dlr, s))
    return jobs


def run_stage(stage, proto, base, packs, sel, seeds, small=False, log=print, save_ckpt=False):
    done = {_key(r) for r in read_records(stage)}
    jobs = build_jobs(proto, sel, seeds, small)
    for n, (arm, fam, mode, lr, dlr, s) in enumerate(jobs):
        probe = {"arm": arm, "family": fam, "mode": mode, "lr": lr, "seed": s}
        if _key(probe) in done:
            continue
        _check_deadline()
        r = run_one_v2(proto, base, packs, arm, fam, mode, lr, s, stage=stage, save_ckpt=save_ckpt,
                       decoder_lr=dlr)
        _append(stage, r)
        a = r.get("association", {}).get("test_acc_acq")
        log(f"[{n + 1}/{len(jobs)}] {arm:15s} {fam:14s} {mode:6s} seed={s} {r['status']} "
            f"{r['elapsed_s']:.1f}s test_acc_acq={a}")
    return read_records(stage)


# ---- circuit + nulls loading ------------------------------------------------------------------------
def load_base(proto: dict, circuit: str = "flywire") -> CircuitBundle:
    from flycircuit.graph.signs import SignPolicy
    from flycircuit.tasks.graph_loader import load_flywire_circuit, make_learning_synthetic

    if circuit == "synthetic":
        return make_learning_synthetic(seed=0)
    sp = proto["sign_policy"]
    return load_flywire_circuit(proto["circuit"]["name"], min_contacts=proto["circuit"]["min_contacts"],
                                sign_policy=SignPolicy(**sp))


def load_packs(base: CircuitBundle, proto: dict, circuit: str = "flywire") -> dict:
    """Cached v2 nulls (hash-verified). Synthetic circuits get on-the-fly v1 nulls (engineering only)."""
    kn = proto["seeds"]["null_graphs"]
    if circuit == "synthetic":
        from flycircuit.graph.controls import null_ensemble

        ns = [n.graph for n in null_ensemble(base.graph, k=kn, base_seed=0)]
        return {"plain": ns, "group": ns}
    from flycircuit.graph.nulls_v2 import load_cached_nulls

    d = project_root() / "results" / "nulls_v2" / proto["circuit"]["name"]
    plain = load_cached_nulls(d / "plain", base.graph)
    group = load_cached_nulls(d / "group_constrained", base.graph)
    if len(plain) < kn or len(group) < kn:
        raise ValueError(f"need >= {kn} cached v2 nulls of each kind in {d}")
    return {"plain": plain[:kn], "group": group[:kn]}


# ---- summaries ---------------------------------------------------------------------------------------
def _cond(r) -> str:
    fam = "nulls" if is_null(r["family"]) and r["family"].startswith("null") else \
        ("gnulls" if is_null(r["family"]) else r["family"])
    return f"{r['arm']}|{fam}|{r['mode']}"


def _vals(recs, arm, family, getter, mode="std"):
    out = {}
    for r in recs:
        if r["arm"] == arm and r["family"] == family and r["mode"] == mode and r["status"] == "ok":
            try:
                out[r["seed"]] = float(getter(r))
            except (KeyError, TypeError):
                pass
    return out


ACC = lambda r: r["association"]["test_acc_acq"]  # noqa: E731
FOR = lambda r: r["continual"]["avg_forgetting_keep_test"]  # noqa: E731


def _paired(a: dict, b: dict):
    ss = sorted(set(a) & set(b))
    return ss, [a[s] for s in ss], [b[s] for s in ss]


def _col(rows, f):
    return [f(r) for r in rows]


def summarize(records: list[dict], proto: dict) -> dict:
    ok = [r for r in records if r["status"] == "ok"]
    groups: dict[str, list] = {}
    for r in ok:
        groups.setdefault(_cond(r), []).append(r)
    table = []
    for key, rows in sorted(groups.items()):
        table.append({
            "condition": key, "n_runs": len(rows), "seeds": sorted({r["seed"] for r in rows}),
            "test_acc_acq": S.bootstrap_ci(_col(rows, ACC), seed=1),
            "test_acc_reversed": S.bootstrap_ci(_col(rows, lambda r: r["association"]["test_acc_reversed"]), seed=2),
            "test_acc_after_delay": S.bootstrap_ci(_col(rows, lambda r: r["association"]["test_acc_after_delay"]), seed=3),
            "final_mean_acc": S.bootstrap_ci(_col(rows, lambda r: r["continual"]["final_mean_acc_test"]), seed=4),
            "forgetting_keep": S.bootstrap_ci(_col(rows, FOR), seed=5),
            "forgetting_clip": S.bootstrap_ci(_col(rows, lambda r: r["continual"]["avg_forgetting_clip_test"]), seed=6),
            "bwt": S.bootstrap_ci(_col(rows, lambda r: r["continual"]["bwt_test"]), seed=7),
            "fwt": S.bootstrap_ci(_col(rows, lambda r: r["continual"]["fwt_test"]), seed=8),
            "perseveration": float(np.mean(_col(rows, lambda r: r["association"]["perseveration_rate"]))),
            "acq_censored_frac": float(np.mean(_col(rows, lambda r: r["association"]["acquisition_censored"]))),
            "oracle_logreg_test": float(np.mean(_col(rows, lambda r: r["oracle_raw_cue_logreg"]["test"]))),
            "lr": rows[0]["lr"], "n_params_trainable": rows[0].get("n_params_trainable"),
            "n_plastic_edges": rows[0].get("n_plastic_edges"),
            "n_training_trials": rows[0]["continual"]["n_training_trials"],
            "replay_storage_bits": rows[0]["continual"]["replay_storage_bits"],
            "mean_elapsed_s": float(np.mean(_col(rows, lambda r: r["elapsed_s"]))),
        })
    out: dict = {"table": table, "protocol": PROTOCOL_VERSION, "protocol_hash": protocol_hash(proto),
                 "n_ok": len(ok), "n_failed": len(records) - len(ok),
                 "failures": [{"cond": _cond(r), "seed": r["seed"], "status": r["status"]}
                              for r in records if r["status"] != "ok"]}
    out["primary"] = primary_contrasts(ok, proto)
    return out


def _holm(ps: dict) -> dict:
    return L.holm(ps)


def primary_contrasts(ok: list[dict], proto: dict) -> dict:
    """The family of 6 prereg tests (Holm). All paired by training seed; null contrasts are nested."""
    res: dict = {}
    ps: dict = {}
    # E1/E2: connectome vs parameter-matched GRU (bptt)
    ca, ga = _vals(ok, "bptt", "connectome", ACC), _vals(ok, "bptt", "gru", ACC)
    cf, gf = _vals(ok, "bptt", "connectome", FOR), _vals(ok, "bptt", "gru", FOR)
    ss, a1, a2 = _paired(ca, ga)
    if ss:
        res["E1_acc_connectome_minus_gru"] = {**S.paired_bootstrap_diff(a1, a2, seed=11), "seeds": ss}
        ps["E1"] = L.signflip_p([x - y for x, y in zip(a1, a2, strict=True)])
    ss, f1, f2 = _paired(gf, cf)
    if ss:
        res["E2_forgetting_improvement_gru_minus_connectome"] = {
            **S.paired_bootstrap_diff(f1, f2, seed=12), "seeds": ss,
            "detail": S.forgetting_improvement(float(np.mean(f1)), float(np.mean(f2)),
                                               proto["targets"]["forgetting_improvement_abs"])}
        ps["E2"] = L.signflip_p([x - y for x, y in zip(f1, f2, strict=True)])
    # E3/E4: topology (nested bootstrap over nulls x seeds) for bptt and plast_dec
    for tag, arm in (("E3_topology_bptt", "bptt"), ("E4_topology_plast_dec", "plast_dec")):
        c = _vals(ok, arm, "connectome", ACC)
        groups = {}
        for fam in sorted({r["family"] for r in ok if r["arm"] == arm and r["family"].startswith("null")}):
            nv = _vals(ok, arm, fam, ACC)
            sss = sorted(set(c) & set(nv))
            if sss:
                groups[fam] = [c[s] - nv[s] for s in sss]
        if len(groups) >= 2:
            res[tag] = S.nested_bootstrap(groups, n_boot=2000, seed=13)
            allv = [x for v in groups.values() for x in v]
            # sign-flip over per-null mean differences (units = nulls) is conservative
            ps[tag[:2]] = L.signflip_p([float(np.mean(v)) for v in groups.values()])
            res[tag]["n_nulls"] = len(groups)
            res[tag]["mean_diff"] = float(np.mean(allv))
    # E5/E6: local plasticity vs modulation-disabled and vs shuffled-feedback (fixed decoder)
    pf = _vals(ok, "plast_fixed", "connectome", ACC)
    for tag, arm in (("E5_plast_fixed_minus_modoff", "plast_modoff"),
                     ("E6_plast_fixed_minus_shuffled", "plast_shuffled")):
        o = _vals(ok, arm, "connectome", ACC)
        ss, x, y = _paired(pf, o)
        if ss:
            res[tag] = {**S.paired_bootstrap_diff(x, y, seed=14), "seeds": ss}
            ps[tag[:2]] = L.signflip_p([a - b for a, b in zip(x, y, strict=True)])
    res["p_exact_signflip"] = ps
    res["p_holm_family"] = _holm(ps) if ps else {}
    res["family_note"] = ("Holm over the prereg family E1-E6 (p-values from exact paired sign-flip; "
                          "E3/E4 use per-null means as units). n=5 -> smallest two-sided p = 0.0625.")
    return res


def target_rows(summ: dict, ok: list[dict], proto: dict) -> list[dict]:
    t = proto["targets"]
    ca = _vals(ok, "bptt", "connectome", ACC)
    n = len(ca)
    m = float(np.mean(list(ca.values()))) if ca else None
    expl = n < proto["seeds"]["confirmatory_min_units"]
    rows = [{"target": "held_out_associative_learning_accuracy>=0.82",
             "condition": "bptt|connectome (declared primary)", "measured": m, "n_seeds": n,
             "status": S.target_status(m, t["accuracy"], exploratory=expl)}]
    e2 = summ["primary"].get("E2_forgetting_improvement_gru_minus_connectome", {}).get("detail")
    rows.append({"target": "forgetting_improvement_abs>=0.07_vs_param_matched_GRU_bptt",
                 "measured_absolute": None if not e2 else e2["absolute"],
                 "relative": None if not e2 else e2["relative"],
                 "status": S.target_status(None if not e2 else e2["absolute"],
                                           t["forgetting_improvement_abs"], exploratory=expl)})
    return rows


def write_summary(stage: str, proto: dict, extra: dict | None = None) -> dict:
    recs = read_records(stage)
    summ = summarize(recs, proto)
    ok = [r for r in recs if r["status"] == "ok"]
    summ["targets"] = target_rows(summ, ok, proto)
    summ["stage"], summ["n_records"] = stage, len(recs)
    if extra:
        summ.update(extra)
    (stage_dir(stage) / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
    return summ


# ---- sign-policy sensitivity (exploratory, VALIDATION split only) ------------------------------------
def run_sensitivity(proto, sel, stage="sensitivity", seeds=None, log=print) -> dict:
    """Re-run bptt|connectome and plast_dec|connectome under every named sign policy
    (graph.signs.policy_grid: unknown-sign fills, glutamate sign, monoamine channel, predicted-only ...)
    on validation data of the search seeds. Descriptive only; nothing here feeds the confirmatory stage."""
    from flycircuit.graph.signs import policy_grid
    from flycircuit.tasks.graph_loader import load_flywire_circuit

    seeds = seeds or proto["search"]["seeds"]
    done = {_key(r) + "|" + r.get("policy", "") for r in read_records(stage)}
    out = {}
    for pname, pol in policy_grid().items():
        base = load_flywire_circuit(proto["circuit"]["name"], min_contacts=proto["circuit"]["min_contacts"],
                                    sign_policy=pol)
        for arm in ("bptt", "plast_dec"):
            lr, dlr = lr_for(sel, arm, "connectome")
            for sd in seeds:
                probe = {"arm": arm, "family": "connectome", "mode": "std", "lr": lr, "seed": sd}
                if _key(probe) + "|" + pname in done:
                    continue
                _check_deadline()
                r = run_one_v2(proto, base, {}, arm, "connectome", "std", lr, sd, splits=("val",),
                               protocols=("continual",), stage=stage, decoder_lr=dlr)
                r["policy"] = pname
                r["sign_summary"] = base.notes.get("sign_summary", {}).get("sign_counts")
                _append(stage, r)
                log(f"sens {pname:28s} {arm:10s} seed={sd} {r['status']} {r['elapsed_s']:.1f}s")
    for r in read_records(stage):
        if r["status"] == "ok":
            out.setdefault(f"{r['policy']}|{r['arm']}", []).append(r["continual"]["final_mean_acc_val"])
    summ = {k: {"mean_val_final_acc": float(np.mean(v)), "n": len(v), "values": v} for k, v in out.items()}
    (stage_dir(stage) / "summary.json").write_text(json.dumps(summ, indent=2))
    return summ
