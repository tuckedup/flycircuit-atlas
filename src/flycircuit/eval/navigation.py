"""Tier-1 navigation benchmark: model construction, matched DAgger imitation, frozen-protocol evaluation.

Everything here is an ENGINEERED benchmark on a lightweight 2-D arena. The 'connectome' condition is
a right-hemisphere mushroom-body PILOT SUBSET (not the whole circuit); inputs/outputs are engineered
interfaces (sensors -> 10 Kenyon cells, MBONs -> linear action decoder). Results say nothing about
biomechanical navigation or a living fly.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from flycircuit.envs.arena import (
    MAX_STEPS,
    N_ACT,
    N_OBS,
    DropoutSpec,
    EvalCondition,
    make_layout,
)
from flycircuit.envs.controllers import (
    NeuralController,
    PrivilegedTeacher,
    RandomWalk,
    SensorScripted,
    aggregate,
    run_episodes,
)

PROTOCOL_REL = "configs/experiments/nav_protocol.yaml"
HASH_REL = "configs/experiments/nav_protocol.sha256"


def _root() -> Path:
    from flycircuit.config import project_root

    return project_root()


# ---------------------------------------------------------------- protocol (freeze + verify)
def load_protocol(root: Path | None = None) -> dict:
    root = root or _root()
    return yaml.safe_load((root / PROTOCOL_REL).read_text(encoding="utf-8"))


def protocol_hash(proto: dict) -> str:
    """Hash of the protocol content excluding the amendments log (amendments carry old hashes)."""
    body = {k: v for k, v in proto.items() if k != "amendments"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def freeze_protocol(root: Path | None = None) -> str:
    root = root or _root()
    h = protocol_hash(load_protocol(root))
    (root / HASH_REL).write_text(h + "\n", encoding="utf-8")
    return h


def verify_protocol(root: Path | None = None) -> dict:
    """Returns {'frozen': bool, 'match': bool, 'hash': str}. A mismatch means the protocol was
    edited after freezing without an amendment entry + re-freeze."""
    root = root or _root()
    proto = load_protocol(root)
    h = protocol_hash(proto)
    p = root / HASH_REL
    if not p.exists():
        return {"frozen": False, "match": False, "hash": h}
    stored = p.read_text().strip()
    return {"frozen": True, "match": stored == h or any(
        a.get("new_hash") == h for a in proto.get("amendments", [])), "hash": h, "stored": stored}


# ---------------------------------------------------------------- real circuit -> graph
SIGN_CONF = 0.6
_SIGN_MAP = {"acetylcholine": 1, "ach": 1, "gaba": -1, "glutamate": -1, "glut": -1}


def load_mb_graph(circuit: str = "mushroom_body_right_pilot", threshold: int = 5,
                  n_sensory: int = N_OBS, release: tuple[str, str] = ("flywire", "783")):
    """Build the VersionedGraph for a selected circuit. Returns (graph, sign[n], meta).

    Interface (engineered, identical for all graph conditions because it depends only on node
    identity/degree): sensory_input = ``n_sensory`` Kenyon cells with the highest out-degree
    (ties by lower node index); action_readout = all MBONs. Edges: directed pairs between
    internal nodes with contact count >= ``threshold`` (a model choice).
    """
    import duckdb

    from flycircuit.data.paths import processed_dir
    from flycircuit.graph.versioned import VersionedGraph

    proc = processed_dir(*release)
    cdir = proc / "circuits" / circuit
    ids_p, nq, ep = cdir / "internal_ids.parquet", proc / "neurons.parquet", proc / "edges_pairs.parquet"
    for p in (ids_p, nq, ep):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing; run `flycircuit data ingest` then `flycircuit data select "
                f"mushroom_body ...` first (real-data prerequisite, not fabricated)")
    con = duckdb.connect()
    con.execute("SET memory_limit='1GiB'")
    nodes = con.execute(
        f"SELECT i.root_id, n.cell_class, n.top_nt, n.top_nt_conf FROM '{ids_p}' i "
        f"LEFT JOIN '{nq}' n USING (root_id) ORDER BY i.root_id").fetchall()
    root_ids = np.array([r[0] for r in nodes], dtype=np.int64)
    edges = con.execute(
        f"SELECT e.pre, e.post, e.count FROM '{ep}' e "
        f"WHERE e.count >= {int(threshold)} AND e.pre IN (SELECT root_id FROM '{ids_p}') "
        f"AND e.post IN (SELECT root_id FROM '{ids_p}')").fetchnumpy()
    cell_class = {int(r[0]): r[1] for r in nodes}
    g = VersionedGraph.from_edges(release[0], release[1], edges["pre"], edges["post"],
                                  edges["count"], root_ids=root_ids,
                                  node_attrs={"cell_class": cell_class})
    cls = np.array([cell_class.get(int(r)) or "" for r in g.root_ids])
    kc = np.flatnonzero(cls == "Kenyon_Cell")
    mbon = np.flatnonzero(cls == "MBON")
    if len(kc) < n_sensory or len(mbon) < N_ACT:
        raise ValueError(f"selection lacks Kenyon cells ({len(kc)}) or MBONs ({len(mbon)})")
    dout = np.bincount(g.pre, minlength=g.n)
    order = kc[np.lexsort((kc, -dout[kc]))]  # highest out-degree first, ties by index
    sens = np.sort(order[:n_sensory])
    m_sens = np.zeros(g.n, bool)
    m_sens[sens] = True
    m_read = np.zeros(g.n, bool)
    m_read[mbon] = True
    g.set_node_mask("sensory_input", m_sens)
    g.set_node_mask("action_readout", m_read)
    g.set_edge_mask("trainable", np.ones(g.n_edges, bool))
    sign = np.zeros(g.n, dtype=np.int8)
    nt = {int(r[0]): (str(r[2]).lower() if r[2] else None, r[3]) for r in nodes}
    for i, rid in enumerate(g.root_ids):
        name, conf = nt.get(int(rid), (None, None))
        if name in _SIGN_MAP and conf is not None and conf >= SIGN_CONF:
            sign[i] = _SIGN_MAP[name]
    meta = {
        "circuit": circuit, "release": f"{release[0]}:{release[1]}", "threshold": threshold,
        "n_nodes": g.n, "n_edges": g.n_edges, "n_kenyon": int(len(kc)), "n_mbon": int(len(mbon)),
        "sensory_nodes": [str(int(g.root_ids[i])) for i in sens],
        "mapping_hash": g.mapping_hash(), "edge_hash": g.edge_hash(),
        "known_sign_fraction": float((sign != 0).mean()),
        "internal_ids_sha256": hashlib.sha256(ids_p.read_bytes()).hexdigest(),
        "scope_note": "right-hemisphere mushroom-body PILOT subset, not the whole circuit",
    }
    return g, sign, meta


def reach_fraction(g) -> float:
    """Fraction of readout nodes reachable from the sensory nodes along directed edges (diagnostic:
    a null graph can disconnect the engineered input from the readout)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import breadth_first_order

    a = sp.csr_matrix((np.ones(g.n_edges), (g.pre, g.post)), shape=(g.n, g.n))
    seen = np.zeros(g.n, bool)
    for s in np.flatnonzero(g.node_masks["sensory_input"]):
        seen[breadth_first_order(a, s, directed=True, return_predecessors=False)] = True
    read = g.node_masks["action_readout"]
    return float(seen[read].mean())


# ---------------------------------------------------------------- models
def build_model(condition: str, seed: int, graph, sign, proto: dict):
    import torch

    from flycircuit.graph.controls import directed_double_edge_swap
    from flycircuit.models.baselines import (
        EncoderDecoderOnly,
        GRUBaseline,
        match_encdec_hidden,
        match_gru_hidden,
        match_gru_hidden_flops,
        random_sparse_rnn,
    )
    from flycircuit.models.sparse_rnn import ModelSpec, SparseConnectomeRNN

    mi = proto["model_interface"]
    torch.manual_seed(seed)
    spec = ModelSpec(
        n_in=N_OBS, n_out=N_ACT, dt=mi["dt"], tau=mi["tau"], prior=mi["prior"],
        trainable_encoder=mi["trainable_encoder"], trainable_decoder=mi["trainable_decoder"],
        init_gain=mi["init_gain"], seed=seed)
    pol = mi["unknown_sign_policy"]
    if "@signs=" in condition:  # sensitivity variant, e.g. connectome_pilot_mb@signs=random_fixed
        condition, pol = condition.split("@signs=")
    ref =SparseConnectomeRNN.sparse_from_graph(graph, spec, sign=sign, unknown_sign_policy=pol)
    rep = ref.param_report()
    info: dict[str, Any] = {"condition": condition, "seed": seed, "unknown_sign_policy": pol}
    if condition == "connectome_pilot_mb":
        model = ref
    elif condition == "null_double_edge_swap":
        res = directed_double_edge_swap(graph, seed=seed)
        info["null_stats"] = {k: res.stats[k] for k in
                              ("acceptance_rate", "mixing_ratio", "mixing_adequate") if k in res.stats}
        info["null_reach_readout_fraction"] = reach_fraction(res.graph)
        model = SparseConnectomeRNN.sparse_from_graph(
            res.graph, spec, sign=sign, unknown_sign_policy=pol)
        model.kind = "null_double_edge_swap"
    elif condition == "random_sparse":
        model = random_sparse_rnn(graph, spec, seed=seed, sign=sign, unknown_sign_policy=pol)
        info["random_reach_readout_fraction"] = reach_fraction(
            _rand_graph(graph, seed))
    elif condition == "gru_param_matched":
        model = GRUBaseline(N_OBS, N_ACT, match_gru_hidden(rep["trainable_params"], N_OBS, N_ACT), seed)
    elif condition == "gru_compute_matched":
        model = GRUBaseline(N_OBS, N_ACT, match_gru_hidden_flops(rep["flops_per_step"], N_OBS, N_ACT), seed)
    elif condition == "encdec_param_matched":
        model = EncoderDecoderOnly(N_OBS, N_ACT, match_encdec_hidden(rep["trainable_params"], N_OBS, N_ACT), seed)
    else:
        raise ValueError(f"unknown condition {condition!r}")
    info["param_report"] = model.param_report()
    info["reference_param_report"] = rep
    return model, info


def _rand_graph(graph, seed):
    from flycircuit.graph.controls import random_sparsity

    return random_sparsity(graph, seed=seed).graph


# ---------------------------------------------------------------- training (matched DAgger)
def _pad(arr: np.ndarray, t: int) -> np.ndarray:
    if arr.shape[0] == t:
        return arr
    pad = np.zeros((t - arr.shape[0], *arr.shape[1:]), dtype=arr.dtype)
    return np.concatenate([arr, pad], 0)


def train_student(model, proto: dict, seed: int, log=print) -> dict:
    """Matched imitation schedule: identical layouts/noise/mask seeds/rounds/epochs/optimizer for
    every model with the same ``seed``. Returns a training summary (never selects checkpoints)."""
    import torch

    from flycircuit.models.train import detach_state

    tr = proto["training"]
    torch.set_num_threads(int(tr.get("cpu_threads", 4)))
    torch.manual_seed(seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=tr["lr"])
    ctrl = NeuralController(model)
    teacher = PrivilegedTeacher()
    levels = tr["dropout_levels"]
    per_level = tr["episodes_per_round"] // len(levels)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ms: list[np.ndarray] = []
    t0 = time.perf_counter()
    losses_by_round: list[float] = []
    diverged = False
    peak_rss = 0.0
    n_env_episodes = 0
    for r, beta in enumerate(tr["dagger_betas"]):
        rng_mix = np.random.default_rng(seed * 1000 + r)
        for li, lvl in enumerate(levels):
            base = seed * 2000 + r * tr["episodes_per_round"] + li * per_level
            lays = [make_layout(base + j, "train") for j in range(per_level)]
            cond = EvalCondition(f"train_dr{lvl}", dropout=DropoutSpec(lvl))
            _, _, (x, y, m) = run_episodes(
                ctrl, lays, cond, noise_seed=seed * 100 + r, mask_seed=seed * 100 + r + 50,
                beta=beta, record=True, teacher=teacher, rng_mix=rng_mix)
            xs.append(_pad(x, MAX_STEPS))
            ys.append(_pad(y, MAX_STEPS))
            ms.append(_pad(m, MAX_STEPS))
            n_env_episodes += per_level
        X = torch.as_tensor(np.concatenate(xs, 1))
        Y = torch.as_tensor(np.concatenate(ys, 1))
        M = torch.as_tensor(np.concatenate(ms, 1))
        nseq = X.shape[1]
        g = torch.Generator().manual_seed(seed * 7919 + r)
        ep_losses = []
        for _ in range(tr["epochs_per_round"]):
            perm = torch.randperm(nseq, generator=g)
            for b0 in range(0, nseq, tr["batch_sequences"]):
                idx = perm[b0:b0 + tr["batch_sequences"]]
                xb, yb, mb = X[:, idx], Y[:, idx], M[:, idx]
                state = None
                for s in range(0, MAX_STEPS, tr["tbptt_steps"]):
                    xc, yc, mc = (a[s:s + tr["tbptt_steps"]] for a in (xb, yb, mb))
                    if float(mc.sum()) == 0:
                        break
                    opt.zero_grad(set_to_none=True)
                    out, state = model(xc, state)
                    err = ((torch.tanh(out) - yc) ** 2).sum(-1)
                    loss = (err * mc).sum() / mc.sum()
                    if not torch.isfinite(loss):
                        diverged = True
                        break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
                    opt.step()
                    state = detach_state(state)
                    ep_losses.append(float(loss.detach()))
                if diverged:
                    break
            if diverged:
                break
        losses_by_round.append(float(np.mean(ep_losses)) if ep_losses else float("nan"))
        log(f"  round {r} beta={beta} seqs={nseq} loss={losses_by_round[-1]:.4f}")
        if diverged:
            break
    try:
        import resource

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # pragma: no cover
        pass
    return {"diverged": diverged, "loss_by_round": losses_by_round,
            "train_seconds": time.perf_counter() - t0, "train_env_episodes": n_env_episodes,
            "dataset_sequences": int(sum(a.shape[1] for a in xs)), "peak_rss_mib_process": peak_rss}


# ---------------------------------------------------------------- evaluation
def _condition(spec: dict) -> EvalCondition:
    return EvalCondition(
        spec["name"], contrast=float(spec.get("contrast", 1.0)), conflict=bool(spec.get("conflict", False)),
        dropout=DropoutSpec(float(spec.get("dropout", 0.0)), spec.get("kind", "channel"), "episode"))


def eval_layouts(proto: dict, split: str = "test") -> list:
    lo, hi = proto["test_layouts" if split == "test" else "val_layouts"]["indices"]
    return [make_layout(i, split) for i in range(lo, hi)]


def assert_no_layout_leakage(proto: dict, n_train: int = 2000) -> dict:
    """Test/val geometry must not occur in (a sample of) the train split."""
    tr = {make_layout(i, "train").hash() for i in range(n_train)}
    te = {lay.hash() for lay in eval_layouts(proto, "test")}
    va = {lay.hash() for lay in eval_layouts(proto, "val")}
    inter = (tr & te) | (tr & va) | (te & va)
    if inter:
        raise AssertionError(f"layout leakage: {len(inter)} shared geometries")
    return {"train_checked": len(tr), "test": len(te), "val": len(va), "shared": 0}


def evaluate_controller(controller, proto: dict, conditions: list[dict] | None = None,
                        split: str = "test") -> dict:
    out = {}
    lays = eval_layouts(proto, split)
    nseed = proto["dropout_protocol_25"]["noise_seed"]
    mseed = proto["dropout_protocol_25"]["mask_seed"]
    for cs in conditions or proto["evaluation_conditions"]:
        cond = _condition(cs)
        metrics, prov = run_episodes(controller, lays, cond, noise_seed=nseed, mask_seed=mseed)
        agg = aggregate(metrics)
        agg["success_per_episode"] = [int(m["success"]) for m in metrics]
        agg["provenance"] = prov
        out[cs["name"]] = agg
    return out


def _slim(res: dict) -> dict:
    return {c: {k: v for k, v in a.items() if k != "success_per_episode"} for c, a in res.items()}


# ---------------------------------------------------------------- orchestration
def summarize(models: list[dict], proto: dict, exploratory: bool) -> dict:
    """Bootstrap over trained models (unit of replication). Primary: connectome mean at 25% protocol."""
    from flycircuit.eval.stats import bootstrap_ci, paired_bootstrap_diff, target_status

    prim = proto["primary_endpoint"]
    cond = prim["condition"]
    by = {}
    for m in models:
        by.setdefault(m["condition"], {})[m["seed"]] = m
    table: dict[str, Any] = {}
    for c, d in by.items():
        live = [d[s] for s in sorted(d) if not d[s]["training"]["diverged"]]
        row = {"n_models": len(d), "n_diverged": len(d) - len(live)}
        for cs in proto["evaluation_conditions"]:
            vals = [m["eval"][cs["name"]]["success_rate"] for m in live]
            row[cs["name"]] = bootstrap_ci(vals, n_boot=5000) if vals else None
        table[c] = row
    contrasts = {}
    pop = prim["population"]
    for comp in [prim["comparator_primary"], *prim["comparators_secondary"]]:
        if comp in by and pop in by:
            seeds = sorted(set(by[pop]) & set(by[comp]))
            a = [by[pop][s]["eval"][cond]["success_rate"] for s in seeds]
            b = [by[comp][s]["eval"][cond]["success_rate"] for s in seeds]
            contrasts[f"{pop}_minus_{comp}"] = paired_bootstrap_diff(a, b, n_boot=5000)
    est = None
    if pop in table and table[pop][cond]:
        est = table[pop][cond]["estimate"]
    return {
        "per_condition": table, "contrasts_exploratory": contrasts,
        "primary": {
            "metric": prim["metric"], "condition": cond, "population": pop,
            "threshold": prim["target_threshold"], "measured_mean": est,
            "status": target_status(est, prim["target_threshold"], exploratory=exploratory),
            "n_trained_models": len(by.get(pop, {})),
        },
    }


def run_benchmark(seeds: list[int], conditions: list[str] | None = None, label: str = "pilot",
                  exploratory: bool = True, root: Path | None = None, log=print,
                  require_frozen: bool = True) -> dict:
    """Train + evaluate every (condition, seed). Sequential (one experiment at a time)."""
    from flycircuit.tracking.manifest import RunRecorder

    root = root or _root()
    v = verify_protocol(root)
    if require_frozen and not (v["frozen"] and v["match"]):
        raise RuntimeError(
            f"navigation protocol not frozen / hash mismatch ({v}); run `flycircuit nav freeze` "
            "(or record an amendment) before training reportable models")
    proto = load_protocol(root)
    leak = assert_no_layout_leakage(proto)
    graph, sign, gmeta = load_mb_graph()
    conds = conditions or proto["conditions"]
    cfg = {"protocol_hash": v["hash"], "seeds": seeds, "conditions": conds, "label": label,
           "graph": gmeta, "leakage_check": leak}
    models: list[dict] = []
    with RunRecorder(f"nav-{label}", cfg, seeds, "matched DAgger imitation, sequential",
                     proto["training"]["stopping_rule"], data_release=gmeta["release"],
                     data_hashes={"internal_ids_sha256": gmeta["internal_ids_sha256"],
                                  "edge_hash": gmeta["edge_hash"]},
                     mapping_hash=gmeta["mapping_hash"], root=root, label=label) as rec:
        refs = {}
        for name, ctl in (("privileged_teacher", PrivilegedTeacher()),
                          ("sensor_scripted", SensorScripted()), ("random_action", RandomWalk(0))):
            refs[name] = _slim(evaluate_controller(ctl, proto))
            log(f"reference {name}: 25% success="
                f"{refs[name][proto['primary_endpoint']['condition']]['success_rate']:.3f}")
        for seed in seeds:
            for c in conds:
                log(f"[seed {seed}] {c}")
                model, info = build_model(c, seed, graph, sign, proto)
                trn = train_student(model, proto, seed, log=log)
                ev = evaluate_controller(NeuralController(model, name=c), proto)
                models.append({"condition": c, "seed": seed, "info": info, "training": trn,
                               "eval": ev})
                log(f"  -> 25%-dropout success {ev[proto['primary_endpoint']['condition']]['success_rate']:.3f}"
                    f" ({trn['train_seconds']:.0f}s)")
        summ = summarize(models, proto, exploratory)
        out_dir = root / "results" / "navigation" / rec.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"run_id": rec.run_id, "protocol_hash": v["hash"], "graph": gmeta,
                   "references": refs, "models": models, "summary": summ,
                   "exploratory": exploratory,
                   "caveats": ["engineered interface and arena; not biomechanics",
                               "connectome condition = right-hemisphere MB pilot subset",
                               "episodes within a model are not independent replicates"]}
        (out_dir / "results.json").write_text(json.dumps(payload, indent=1, default=float))
        (out_dir / "summary.md").write_text(render_markdown(payload, proto))
        rec.add_artifact(out_dir / "results.json")
        rec.add_artifact(out_dir / "summary.md")
        if summ["primary"]["measured_mean"] is not None:
            rec.metrics["primary_success_25pct"] = summ["primary"]["measured_mean"]
    payload["out_dir"] = str(out_dir)
    return payload


def render_markdown(payload: dict, proto: dict) -> str:
    cond = [c["name"] for c in proto["evaluation_conditions"]]
    lines = [f"# Navigation run {payload['run_id']}", "",
             f"Protocol hash `{payload['protocol_hash'][:12]}`; "
             f"{'EXPLORATORY' if payload['exploratory'] else 'confirmatory'}; "
             f"{payload['summary']['primary']['n_trained_models']} trained connectome models.", "",
             "Success rate (mean over trained models, 95% bootstrap CI over models):", "",
             "| condition | " + " | ".join(cond) + " |", "|---|" + "---|" * len(cond)]
    for c, row in payload["summary"]["per_condition"].items():
        cells = []
        for k in cond:
            b = row[k]
            if not b:
                cells.append("n/a")
            else:
                ci = "" if math.isnan(b["lo"]) else f" [{b['lo']:.2f},{b['hi']:.2f}]"
                cells.append(f"{b['estimate']:.3f}{ci}")
        lines.append(f"| {c} (n={row['n_models']}) | " + " | ".join(cells) + " |")
    lines += ["", "Reference controllers (no learning; 200 test episodes, not replicates):", "",
              "| controller | " + " | ".join(cond) + " |", "|---|" + "---|" * len(cond)]
    for c, row in payload["references"].items():
        lines.append(f"| {c} | " + " | ".join(f"{row[k]['success_rate']:.3f}" for k in cond) + " |")
    p = payload["summary"]["primary"]
    lines += ["", f"Primary: {p['population']} @ {p['condition']} mean="
              f"{p['measured_mean']} vs threshold {p['threshold']} -> status **{p['status']}**.", ""]
    return "\n".join(lines)
