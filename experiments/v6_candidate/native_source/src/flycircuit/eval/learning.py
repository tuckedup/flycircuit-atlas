"""Learning benchmark runners: association / reversal and continual learning (spec §12).

Everything here is SYNTHETIC-task evaluation on a (real or synthetic) circuit with ENGINEERED
interfaces; results are model-relative engineering/exploratory evidence, never biological evidence.

Stages:  search (validation only) -> pilot (2 seeds, exploratory) -> confirm (frozen prereg).
Records are appended to ``results/learning/<stage>/records.jsonl`` (resume-safe, one line per run).
"""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import yaml

from flycircuit.config import project_root
from flycircuit.eval import stats as S
from flycircuit.graph.controls import null_ensemble, random_sparsity
from flycircuit.models.baselines import EncoderDecoderOnly, GRUBaseline, match_gru_hidden
from flycircuit.models.sparse_rnn import ModelSpec, SparseConnectomeRNN
from flycircuit.plasticity.agent import LEARNERS, LearningAgent
from flycircuit.plasticity.three_factor import PlasticityConfig
from flycircuit.tasks.association import ContingencyTask, TaskSpec
from flycircuit.tasks.graph_loader import CircuitBundle

SPARSE_FAMILIES = ("connectome", "random")  # plus null0..null4
DENSE_FAMILIES = ("gru", "encdec")
DENSE_LEARNERS = ("bptt", "decoder_only")
NULL_FULL_FACTORIAL_LEARNERS = ("bptt", "plastic_plus_dec")


def protocol_path() -> Path:
    return project_root() / "configs" / "experiments" / "learning_protocol.yaml"


def load_protocol(path: Path | None = None) -> dict:
    return yaml.safe_load((path or protocol_path()).read_text())


def protocol_hash(proto: dict) -> str:
    return hashlib.sha256(json.dumps(proto, sort_keys=True).encode()).hexdigest()[:16]


def is_sparse_family(f: str) -> bool:
    return f in SPARSE_FAMILIES or f.startswith("null")


# ---- construction -----------------------------------------------------------------------
def make_task(proto: dict, seed: int) -> ContingencyTask:
    return ContingencyTask(TaskSpec(**proto["task"]), seed)


def family_bundle(base: CircuitBundle, family: str, nulls: list | None, seed: int) -> CircuitBundle:
    if family == "connectome":
        return base
    if family == "random":
        return base.with_graph(random_sparsity(base.graph, seed=seed).graph, "random_sparse")
    if family.startswith("null"):
        i = int(family[4:])
        return base.with_graph(nulls[i].graph, f"degree_preserving_null_{i}")
    raise ValueError(family)


def _fixed_encoder(model: SparseConnectomeRNN, fan_in: int, seed: int, n_cue: int) -> None:
    """Fixed random sparse projection input -> sensory (KC) nodes: ENGINEERED interface."""
    import torch

    n_in, n_sens = model.encoder.shape
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    enc = np.zeros((n_in, n_sens), dtype=np.float32)
    for j in range(n_sens):
        enc[rng.choice(n_in, size=min(fan_in, n_in), replace=False), j] = 1.0
    with torch.no_grad():
        model.encoder.copy_(torch.as_tensor(enc))
    model.encoder.requires_grad_(False)


def build_sparse_model(proto: dict, b: CircuitBundle, seed: int, n_in: int) -> SparseConnectomeRNN:
    m = proto["model"]
    spec = ModelSpec(n_in=n_in, n_out=2, dt=m["dt"], tau=m["tau"], prior=m["prior"],
                     init_gain=m["init_gain"], nonneg_gain=m["nonneg_gain"], seed=seed)
    model = SparseConnectomeRNN(
        b.graph.n, b.graph.pre, b.graph.post, b.graph.count, spec, sign=b.sign,
        unknown_sign_policy=m["unknown_sign_policy"], sensory_idx=b.sensory_idx,
        readout_idx=b.readout_idx)
    _fixed_encoder(model, m["encoder_fan_in"], seed, proto["task"]["n_cue"])
    return model


def reference_trainable_params(proto: dict, base: CircuitBundle, n_in: int) -> int:
    """Trainable parameters of the connectome model under BPTT (recurrent gains + decoder)."""
    m = build_sparse_model(proto, base, 0, n_in)
    return m.n_gain_params + m.decoder.numel()


def build_model(proto, base: CircuitBundle, family: str, nulls, seed: int, n_in: int):
    if is_sparse_family(family):
        return build_sparse_model(proto, family_bundle(base, family, nulls, seed), seed, n_in), \
            family_bundle(base, family, nulls, seed)
    ref = reference_trainable_params(proto, base, n_in)
    h = match_gru_hidden(ref, n_in, 2)
    if family == "gru":
        return GRUBaseline(n_in, 2, h, seed), None
    if family == "encdec":
        from flycircuit.models.baselines import match_encdec_hidden

        return EncoderDecoderOnly(n_in, 2, match_encdec_hidden(ref, n_in, 2), seed), None
    raise ValueError(family)


def make_agent(proto, base, family, nulls, learner, lr, seed, task) -> tuple[LearningAgent, dict]:
    model, bundle = build_model(proto, base, family, nulls, seed, task.spec.n_in)
    pl = proto["plasticity"]
    cfg = PlasticityConfig(decay=pl["decay"], gain_lo=pl["gain_lo"], gain_hi=pl["gain_hi"],
                           gamma=pl["gamma"], lr_value=pl["lr_value"],
                           rule=pl.get("rule", "np_v1"), homeostasis=pl.get("homeostasis", 0.0),
                           shuffle_feedback=pl.get("shuffle_feedback", False))
    mask = bundle.plastic_edge_mask if bundle is not None else None
    agent = LearningAgent(model, learner, task, lr, plastic_mask=mask,
                          explore_noise=proto["training"]["explore_noise"],
                          decoder_lr=proto["training"]["decoder_lr"], seed=seed, plast_cfg=cfg,
                          logit_cap=proto["training"].get("policy_logit_cap", 2.0))
    info = {"n_params_trainable": int(sum(p.numel() for p in agent.params)),
            "n_plastic_edges": None if agent.plastic is None else int(agent.plastic.n_plastic)}
    if bundle is not None:
        info.update(n_nodes=bundle.graph.n, n_edges=bundle.graph.n_edges,
                    edge_hash=bundle.graph.edge_hash())
    return agent, info


# ---- training protocols -----------------------------------------------------------------
def _rngs(seed: int, stream: int):
    ss = np.random.SeedSequence([seed, 4242, stream])
    o, n, a = ss.spawn(3)
    return np.random.default_rng(o), np.random.default_rng(n), np.random.default_rng(a)


def train_phase(agent, task, ctx, n, rngs, reversed_=False, replay=None, replay_ctxs=()):
    order, noise, act = rngs
    hist = []
    for _ in range(n):
        cue, lab = task.sample_train(order, ctx, reversed_)
        hist.append(agent.trial(cue, ctx, int(lab[0]), noise, act))
        if replay:
            for _r in range(1):
                c2 = int(replay_ctxs[order.integers(len(replay_ctxs))])
                cues, labs = replay[c2]
                i = int(order.integers(len(labs)))
                agent.trial(cues[i:i + 1], c2, int(labs[i]), noise, act)
    return hist


def _acq_trials(correct: np.ndarray, window: int, crit: float):
    if len(correct) < window:
        return None
    cs = np.convolve(correct, np.ones(window) / window, mode="valid")
    hit = np.flatnonzero(cs >= crit)
    return int(hit[0] + window) if len(hit) else None  # trials needed; None = censored


def run_association(agent, task, proto, seed, splits=("val", "test")) -> dict:
    t = proto["training"]
    rngs = _rngs(seed, 0)
    h = train_phase(agent, task, 0, t["trials_per_phase"], rngs)
    correct = np.array([x["correct"] for x in h], dtype=float)
    out = {"acquisition_trials": _acq_trials(correct, t["acquisition_window"],
                                              t["acquisition_criterion"]),
           "acquisition_censored": _acq_trials(correct, t["acquisition_window"],
                                               t["acquisition_criterion"]) is None,
           "train_accuracy_last50": float(correct[-50:].mean()),
           "accumulated_reward_acq": float(sum(x["reward"] for x in h))}
    for sp in splits:
        out[f"{sp}_acc_acq"] = agent.evaluate(sp, 0)
    h2 = train_phase(agent, task, 0, t["reversal_trials"], rngs, reversed_=True)
    c2 = np.array([x["correct"] for x in h2], dtype=float)
    w = t["perseveration_window"]
    out["perseveration_rate"] = float(1.0 - c2[:w].mean())  # actions following the OLD contingency
    out["accumulated_reward_rev"] = float(sum(x["reward"] for x in h2))
    out["reversal_acquisition_trials"] = _acq_trials(c2, t["acquisition_window"],
                                                     t["acquisition_criterion"])
    for sp in splits:
        out[f"{sp}_acc_reversed"] = agent.evaluate(sp, 0, reversed_=True)
        out[f"{sp}_acc_old_contingency_after_reversal"] = agent.evaluate(sp, 0, reversed_=False)
    return out


def run_continual(agent, task, proto, seed, splits=("val", "test")) -> dict:
    t = proto["training"]
    n_t = task.spec.n_ctx
    rngs = _rngs(seed, 1)
    budget = int(t.get("replay_per_task", 0))
    replay: dict = {}
    mats = {sp: np.zeros((n_t, n_t)) for sp in splits}
    for phase in range(n_t):
        train_phase(agent, task, phase, t["trials_per_phase"], rngs,
                    replay=replay or None, replay_ctxs=tuple(replay))
        if budget > 0:
            replay[phase] = (task.cues[("train", phase)][:budget],
                             task.labels_for("train", phase)[:budget])
        for sp in splits:
            for j in range(n_t):
                mats[sp][phase, j] = agent.evaluate(sp, j)
    out = {}
    for sp, R in mats.items():
        out[f"R_{sp}"] = R.tolist()
        out[f"final_mean_acc_{sp}"] = float(R[-1].mean())
        out[f"avg_forgetting_keep_{sp}"] = S.average_forgetting(R, "keep")
        out[f"avg_forgetting_clip_{sp}"] = S.average_forgetting(R, "clip")
        out[f"bwt_{sp}"] = S.backward_transfer(R)
    out["forward_transfer"] = "not computed: needs per-task reference models trained without prior tasks"
    return out


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_one(proto, base, nulls, family, learner, lr, seed, splits=("val", "test"),
            protocols=("association", "continual")) -> dict:
    """One (condition, learner, seed) run on a FRESH model; returns an auditable record."""
    t0 = time.perf_counter()
    task = make_task(proto, seed)
    rec = {"family": family, "learner": learner, "lr": lr, "seed": seed,
           "protocol_hash": protocol_hash(proto), "splits": list(splits)}
    status = "ok"
    try:
        if "association" in protocols:
            agent, info = make_agent(proto, base, family, nulls, learner, lr, seed, task)
            rec.update(info)
            rec["association"] = run_association(agent, task, proto, seed, splits)
        if "continual" in protocols:
            agent, info = make_agent(proto, base, family, nulls, learner, lr, seed, task)
            rec.update(info)
            rec["continual"] = run_continual(agent, task, proto, seed, splits)
    except Exception as e:  # recorded, never converted into a fake result
        status = f"failed: {e!r}"
    rec.update(status=status, elapsed_s=time.perf_counter() - t0, peak_rss_mib=peak_rss_mib())
    return rec


# ---- stage orchestration ----------------------------------------------------------------
def stage_dir(stage: str) -> Path:
    d = project_root() / "results" / "learning" / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(r) -> str:
    return f"{r['family']}|{r['learner']}|{r['lr']}|{r['seed']}"


def read_records(stage: str) -> list[dict]:
    p = stage_dir(stage) / "records.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def _append(stage: str, rec: dict) -> None:
    with open(stage_dir(stage) / "records.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def families_for(learner: str, include_nulls: int, include_random=True) -> list[str]:
    fams = ["connectome"] + [f"null{i}" for i in range(include_nulls)]
    if include_random:
        fams.append("random")
    if learner in DENSE_LEARNERS:
        fams += list(DENSE_FAMILIES)
    return fams


def run_search(proto, base, nulls, stage: str = "search", log=print) -> dict:
    """Validation-only LR selection per (learner, family-kind) with a fixed, identical grid/budget.
    Null graphs use the lr selected on null0; test splits are never evaluated here."""
    grid, seeds = proto["search"]["lr_grid"], proto["search"]["seeds"]
    done = {_key(r): r for r in read_records(stage)}
    kinds = ["connectome", "null0", "random", "gru", "encdec"]
    for learner in LEARNERS:
        for fam in kinds:
            if fam in DENSE_FAMILIES and learner not in DENSE_LEARNERS:
                continue
            for lr in grid[learner]:
                for s in seeds:
                    probe = {"family": fam, "learner": learner, "lr": lr, "seed": s}
                    if _key(probe) in done:
                        continue
                    r = run_one(proto, base, nulls, fam, learner, lr, s, splits=("val",),
                                protocols=("continual",))
                    _append(stage, r)
                    done[_key(r)] = r
                    log(f"search {fam:10s} {learner:18s} lr={lr:<6} seed={s} "
                        f"{r['status']} {r['elapsed_s']:.1f}s")
    sel: dict = {}
    for learner in LEARNERS:
        for fam in kinds:
            if fam in DENSE_FAMILIES and learner not in DENSE_LEARNERS:
                continue
            sc = {}
            for lr in grid[learner]:
                vals = [done[_key({"family": fam, "learner": learner, "lr": lr, "seed": s})]
                        for s in seeds]
                vals = [v["continual"]["final_mean_acc_val"] for v in vals if v["status"] == "ok"]
                sc[lr] = float(np.mean(vals)) if vals else float("-inf")
            best = max(sc, key=lambda k: (sc[k], -k))  # ties -> smaller lr
            sel[f"{learner}|{fam}"] = {"lr": best, "val_scores": {str(k): v for k, v in sc.items()}}
    (stage_dir(stage) / "lr_selected.json").write_text(json.dumps(sel, indent=2))
    return sel


def lr_for(sel: dict, learner: str, family: str) -> float:
    fam = "null0" if family.startswith("null") else family
    return sel[f"{learner}|{fam}"]["lr"]


def run_stage(stage: str, proto, base, nulls, sel, seeds, n_nulls: int, log=print,
              learners=LEARNERS) -> list[dict]:
    done = {_key(r) for r in read_records(stage)}
    jobs = []
    for learner in learners:
        for fam in families_for(learner, n_nulls):
            for si, s in enumerate(seeds):
                if fam.startswith("null") and learner not in NULL_FULL_FACTORIAL_LEARNERS:
                    if int(fam[4:]) != si % n_nulls:
                        continue  # paired design: null i with seed index i
                jobs.append((fam, learner, lr_for(sel, learner, fam), s))
    for n, (fam, learner, lr, s) in enumerate(jobs):
        probe = {"family": fam, "learner": learner, "lr": lr, "seed": s}
        if _key(probe) in done:
            continue
        r = run_one(proto, base, nulls, fam, learner, lr, s)
        r["stage"] = stage
        _append(stage, r)
        a = r.get("association", {}).get("test_acc_acq")
        log(f"[{n + 1}/{len(jobs)}] {fam:10s} {learner:18s} seed={s} {r['status']} "
            f"{r['elapsed_s']:.1f}s test_acc_acq={a}")
    return read_records(stage)


def make_nulls(base: CircuitBundle, proto: dict):
    """k degree-preserving nulls; cached on disk keyed by (edge hash, k, base seed) because the
    swap is deterministic but slow for the real circuit."""
    from flycircuit.graph.controls import ControlResult, _rebuild

    k, bs = proto["seeds"]["null_graphs"], proto["seeds"]["null_graph_base_seed"]
    g = base.graph
    path = project_root() / "results" / "learning" / f"nulls_{g.edge_hash()[:16]}_{k}_{bs}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        stats = json.loads(str(z["stats"]))
        return [ControlResult(_rebuild(g, z[f"pre{i}"], z[f"post{i}"], z[f"count{i}"],
                                       "double_edge_swap"), "double_edge_swap", stats[i])
                for i in range(k)]
    nulls = null_ensemble(g, k=k, base_seed=bs)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = {}
    for i, n in enumerate(nulls):
        arrs[f"pre{i}"], arrs[f"post{i}"], arrs[f"count{i}"] = n.graph.pre, n.graph.post, n.graph.count
    np.savez(path, stats=json.dumps([n.stats for n in nulls], default=str), **arrs)
    return nulls


# ---- summaries --------------------------------------------------------------------------
def signflip_p(diffs) -> float:
    """Exact two-sided paired sign-flip permutation p-value for mean(diffs) == 0 (n <= 16)."""
    d = np.asarray(diffs, float)
    n = len(d)
    if n == 0 or n > 16:
        return float("nan")
    signs = np.array([[1 if (m >> i) & 1 else -1 for i in range(n)] for m in range(2**n)])
    means = np.abs((signs * d).mean(1))
    return float((means >= abs(d.mean()) - 1e-12).mean())


def holm(ps: dict) -> dict:
    """Holm step-down adjustment over a defined family of p-values."""
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def _vals(recs, family, learner, getter):
    out = {}
    for r in recs:
        if r["family"] == family and r["learner"] == learner and r["status"] == "ok":
            try:
                out[r["seed"]] = float(getter(r))
            except (KeyError, TypeError):
                pass
    return out


def summarize(records: list[dict], proto: dict, neg_policy: str = "keep") -> dict:
    tgt = yaml.safe_load((project_root() / "configs" / "targets.yaml").read_text())
    acc = lambda r: r["association"]["test_acc_acq"]  # noqa: E731
    fg = lambda r: r["continual"][f"avg_forgetting_{neg_policy}_test"]  # noqa: E731
    fams = sorted({r["family"] for r in records if not r["family"].startswith("null")})
    table = []
    for learner in LEARNERS:
        for fam in fams + ["nulls(all)"]:
            rows = [r for r in records if r["learner"] == learner and r["status"] == "ok"
                    and (r["family"].startswith("null") if fam == "nulls(all)" else r["family"] == fam)]
            if not rows:
                continue
            a = [acc(r) for r in rows]
            f = [fg(r) for r in rows]
            table.append({
                "learner": learner, "condition": fam, "n_runs": len(rows),
                "test_acc_acq": S.bootstrap_ci(a, seed=1),
                "avg_forgetting": S.bootstrap_ci(f, seed=2),
                "final_mean_acc": S.bootstrap_ci([r["continual"]["final_mean_acc_test"] for r in rows], seed=3),
                "perseveration": float(np.mean([r["association"]["perseveration_rate"] for r in rows])),
                "acq_censored_frac": float(np.mean([r["association"]["acquisition_censored"] for r in rows])),
                "mean_elapsed_s": float(np.mean([r["elapsed_s"] for r in rows])),
                "n_plastic_edges": rows[0].get("n_plastic_edges"),
                "n_params_trainable": rows[0].get("n_params_trainable"),
            })
    out = {"table": table, "negative_forgetting_policy": neg_policy,
           "protocol_hash": protocol_hash(proto)}
    # primary contrasts (learner = bptt): connectome vs param-matched GRU (paired by seed)
    comp = {}
    for learner, comparator in (("bptt", "gru"),):
        ca, ga = _vals(records, "connectome", learner, acc), _vals(records, comparator, learner, acc)
        cf, gf = _vals(records, "connectome", learner, fg), _vals(records, comparator, learner, fg)
        seeds = sorted(set(ca) & set(ga) & set(cf) & set(gf))
        if len(seeds) >= 1:
            comp["accuracy_connectome_minus_gru"] = S.paired_bootstrap_diff(
                [ca[s] for s in seeds], [ga[s] for s in seeds], seed=5)
            comp["forgetting_improvement_gru_minus_connectome"] = S.paired_bootstrap_diff(
                [gf[s] for s in seeds], [cf[s] for s in seeds], seed=6)
            comp["forgetting_improvement_detail"] = S.forgetting_improvement(
                float(np.mean([gf[s] for s in seeds])), float(np.mean([cf[s] for s in seeds])),
                _threshold(tgt, "forgetting_improvement", 0.07))
            comp["seeds_used"] = seeds
            pa = signflip_p([ca[s] - ga[s] for s in seeds])
            pf = signflip_p([gf[s] - cf[s] for s in seeds])
            comp["p_exact_signflip"] = {"accuracy": pa, "forgetting_improvement": pf}
            comp["p_holm_family_of_2"] = holm({"accuracy": pa, "forgetting_improvement": pf})
            comp["p_note"] = ("exact paired sign-flip test on per-seed differences; with n=5 the "
                              "smallest attainable two-sided p is 0.0625 (before Holm)")
    # topology contrast (nested): connectome minus null_i, per training seed, learner-wise
    topo = {}
    for learner in NULL_FULL_FACTORIAL_LEARNERS:
        ca = _vals(records, "connectome", learner, acc)
        groups: dict = {}
        for fam in sorted({r["family"] for r in records if r["family"].startswith("null")}):
            nv = _vals(records, fam, learner, acc)
            ss = sorted(set(ca) & set(nv))
            if ss:
                groups[fam] = [ca[s] - nv[s] for s in ss]
        if len(groups) >= 2:
            topo[learner] = S.nested_bootstrap(groups, n_boot=2000, seed=7)
    out["primary_contrasts"] = comp
    out["topology_contrast_acc_connectome_minus_null"] = topo
    # target status (primary learner bptt, connectome vs matched GRU)
    out["targets"] = _target_rows(records, comp, tgt, acc)
    return out


def _threshold(tgt: dict, tid: str, default: float) -> float:
    for t in tgt.get("targets", []):
        if t.get("id") == tid:
            return float(t["threshold"])
    return default


def _target_rows(records, comp, tgt, acc) -> list[dict]:
    rows = []
    ca = _vals(records, "connectome", "bptt", acc)
    n = len(ca)
    exploratory = n < 5
    m = float(np.mean(list(ca.values()))) if ca else None
    rows.append({"target": "held_out_associative_learning_accuracy>=0.82",
                 "measured_connectome_bptt": m, "n_seeds": n,
                 "status": S.target_status(m, 0.82, exploratory=exploratory, blocked=False)})
    d = comp.get("forgetting_improvement_detail")
    rows.append({"target": "forgetting_improvement_abs>=0.07_vs_param_matched_GRU_bptt",
                 "measured_absolute": None if d is None else d["absolute"],
                 "relative": None if d is None else d["relative"],
                 "status": S.target_status(None if d is None else d["absolute"], 0.07,
                                           exploratory=exploratory)})
    rows.append({"target": "causal_screen_100_targets", "status": "not run (other agent/stage)"})
    rows.append({"target": "navigation_success_25pct_dropout>=0.68",
                 "status": "not run (other agent/stage)"})
    return rows


def write_summary(stage: str, proto: dict, extra: dict | None = None) -> dict:
    recs = read_records(stage)
    summ = summarize(recs, proto, proto.get("negative_forgetting_policy", "keep"))
    summ["stage"] = stage
    summ["n_records"] = len(recs)
    summ["n_failed"] = sum(r["status"] != "ok" for r in recs)
    if extra:
        summ.update(extra)
    (stage_dir(stage) / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
    return summ
