"""Navigation v2 benchmark: protocol freeze, graphs/signs/nulls, models, matched DAgger, search, final run.

ENGINEERED benchmark on a lightweight 2-D arena; the 'connectome' is the right-hemisphere mushroom-body
PILOT subset with ENGINEERED sensory/readout interfaces. See docs/navigation_v2.md. v1 code/results are untouched.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from flycircuit.envs.arena import DropoutSpec
from flycircuit.envs.arena_v2 import (
    N_OBS_V2,
    EvalCondV2,
    NavEnvV2,
    layout_set,
)
from flycircuit.envs.controllers import (
    NeuralController,
    PrivilegedTeacher,
    RandomWalk,
    aggregate,
    run_episodes,
)
from flycircuit.envs.controllers_v2 import SCRIPTED

PROTOCOL_REL = "configs/experiments/nav_v2_protocol.yaml"
HASH_REL = "configs/experiments/nav_v2_protocol.sha256"
OUT_REL = "results/navigation_v2"
MAX_STEPS = 250
N_ACT = 2


def _root() -> Path:
    from flycircuit.config import project_root

    return project_root()


# ------------------------------------------------------------------ protocol freeze / verify
def load_protocol(root: Path | None = None) -> dict:
    return yaml.safe_load(((root or _root()) / PROTOCOL_REL).read_text(encoding="utf-8"))


def protocol_hash(proto: dict) -> str:
    body = {k: v for k, v in proto.items() if k != "amendments"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def freeze_protocol(root: Path | None = None) -> str:
    root = root or _root()
    h = protocol_hash(load_protocol(root))
    (root / HASH_REL).write_text(h + "\n", encoding="utf-8")
    return h


def verify_protocol(root: Path | None = None) -> dict:
    root = root or _root()
    proto = load_protocol(root)
    h = protocol_hash(proto)
    p = root / HASH_REL
    if not p.exists():
        return {"frozen": False, "match": False, "hash": h}
    stored = p.read_text().strip()
    ok = stored == h or any(a.get("new_hash") == h for a in proto.get("amendments", []))
    return {"frozen": True, "match": ok, "hash": h, "stored": stored}


def _require_frozen(root: Path) -> dict:
    v = verify_protocol(root)
    if not (v["frozen"] and v["match"]):
        raise RuntimeError(f"nav v2 protocol not frozen / hash mismatch ({v}); run `flycircuit nav2 freeze`")
    return v


# ------------------------------------------------------------------ conditions & layouts
def cond_from_spec(spec: dict) -> EvalCondV2:
    d = float(spec.get("dropout", 0.0))
    return EvalCondV2(
        spec["name"], conflict=bool(spec.get("conflict", True)), ctx=spec.get("ctx", "pulse"),
        contrast=float(spec.get("contrast", 1.0)),
        odor_noise_mult=float(spec.get("odor_noise_mult", 1.0)),
        vision_noise_mult=float(spec.get("vision_noise_mult", 1.0)),
        plume_r=float(spec.get("plume_r", 10.0)),
        dropout=DropoutSpec(d, "channel", "episode"))


def eval_layouts(proto: dict, split: str, ood: bool = False) -> list:
    ev = proto["evaluation"]
    if ood:
        c = ev["test_layouts_ood_obstacles"]
        return layout_set("test", c["indices"][0], c["indices"][1], tuple(c["n_obstacles"]))
    c = ev["val_layouts"] if split == "val" else ev["test_layouts"]
    return layout_set(c["split"], c["indices"][0], c["indices"][1])


def train_layout_indices(seeds: list[int], n_rounds: int = 5, per_round: int = 48) -> list[int]:
    return [s * 1000 + r * 100 + k for s in seeds for r in range(n_rounds) for k in range(per_round)]


def assert_no_leakage(proto: dict, seeds: list[int]) -> dict:
    from flycircuit.envs.arena_v2 import geometry_hashes, make_layout_v2

    tr = geometry_hashes([make_layout_v2(i, "train") for i in train_layout_indices(seeds)])
    va = geometry_hashes(eval_layouts(proto, "val"))
    te = geometry_hashes(eval_layouts(proto, "test"))
    oo = geometry_hashes(eval_layouts(proto, "test", ood=True))
    sets = {"train": tr, "val": va, "test": te, "ood": oo}
    shared = {}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n = len(sets[a] & sets[b])
            if n:
                shared[f"{a}&{b}"] = n
    if shared:
        raise AssertionError(f"layout geometry leakage: {shared}")
    return {k: len(v) for k, v in sets.items()} | {"shared": 0}


# ------------------------------------------------------------------ graph, signs, nulls
def load_graph_v2(n_sensory: int = N_OBS_V2, circuit: str = "mushroom_body_right_pilot"):
    """MB pilot graph (v1 loader, 11 sensory KCs) + annotation_v2 node signs (modulatory -> 0)."""
    from flycircuit.eval import navigation as nv1
    from flycircuit.graph.signs import SignPolicy, load_assignment_for_ids

    g, _sign_v1, meta = nv1.load_mb_graph(circuit, n_sensory=n_sensory)
    pol = SignPolicy()
    sa = load_assignment_for_ids(g.root_ids, pol)
    sign = sa.sign.astype(np.int8)
    meta["sign_policy"] = pol.tag()
    meta["sign_assignment_hash"] = sa.hash()
    meta["sign_summary"] = {"plus": int((sign > 0).sum()), "minus": int((sign < 0).sum()),
                            "zero": int((sign == 0).sum()), "modulatory": int(sa.modulatory.sum())}
    meta["reach_readout_fraction"] = nv1.reach_fraction(g)
    return g, sign, meta


def load_nulls(g, kind: str, circuit: str = "mushroom_body_right_pilot") -> list:
    from flycircuit.graph.nulls_v2 import load_cached_nulls

    d = _root() / "results" / "nulls_v2" / circuit / kind
    if not (d / "report.json").exists():
        raise FileNotFoundError(f"{d}/report.json missing; run scripts/linux/make_nulls_v2.py {circuit}")
    return load_cached_nulls(d, g)


def random_reachable(g, seed: int, max_tries: int = 50):
    """Uniform random simple digraph (same N, E); resampled until sensory->readout reachability >= the
    connectome's. Returns (graph, info)."""
    from flycircuit.eval import navigation as nv1
    from flycircuit.graph.controls import random_sparsity

    target = nv1.reach_fraction(g)
    last = None
    for k in range(max_tries):
        rg = random_sparsity(g, seed=seed + 100_003 * k).graph
        r = nv1.reach_fraction(rg)
        last = (rg, r, k)
        if r >= target:
            return rg, {"reach_fraction": r, "target_reach": target, "attempts": k + 1, "verified": True}
    return last[0], {"reach_fraction": last[1], "target_reach": target, "attempts": max_tries,
                     "verified": False}


# ------------------------------------------------------------------ models
def build_model_v2(condition: str, seed: int, graph, sign, proto: dict, nulls: dict[str, list]):
    import torch

    from flycircuit.eval import navigation as nv1
    from flycircuit.models.baselines import (
        EncoderDecoderOnly,
        GRUBaseline,
        match_encdec_hidden,
        match_gru_hidden,
        match_gru_hidden_flops,
    )
    from flycircuit.models.sparse_rnn import ModelSpec, SparseConnectomeRNN

    mi = proto["model_interface"]
    torch.manual_seed(seed)
    spec = ModelSpec(n_in=N_OBS_V2, n_out=N_ACT, dt=mi["dt"], tau=mi["tau"], prior=mi["prior"],
                     trainable_encoder=mi["trainable_encoder"], trainable_decoder=mi["trainable_decoder"],
                     init_gain=mi["init_gain"], seed=seed)
    pol = mi["unknown_sign_policy"]

    def sparse(gr):
        return SparseConnectomeRNN.sparse_from_graph(gr, spec, sign=sign, unknown_sign_policy=pol)

    ref = sparse(graph)
    rep = ref.param_report()
    info: dict[str, Any] = {"condition": condition, "seed": seed, "unknown_sign_policy": pol}
    if condition == "connectome_pilot_mb":
        model = ref
    elif condition in ("null_v2_swap", "null_v2_group"):
        lst = nulls[condition]
        idx = proto["seeds"]["final"].index(seed) if seed in proto["seeds"]["final"] else seed % len(lst)
        ng = lst[idx % len(lst)]
        info["null_index"] = idx % len(lst)
        info["null_reach_readout_fraction"] = nv1.reach_fraction(ng)
        model = sparse(ng)
        model.kind = condition
    elif condition == "random_sparse_reachable":
        rg, rinfo = random_reachable(graph, seed)
        info["random"] = rinfo
        model = sparse(rg)
        model.kind = "random_sparse_reachable"
    elif condition == "gru_param_matched":
        model = GRUBaseline(N_OBS_V2, N_ACT, match_gru_hidden(rep["trainable_params"], N_OBS_V2, N_ACT), seed)
    elif condition == "gru_compute_matched":
        model = GRUBaseline(N_OBS_V2, N_ACT,
                            match_gru_hidden_flops(rep["flops_per_step"], N_OBS_V2, N_ACT), seed)
    elif condition == "encdec_param_matched":
        model = EncoderDecoderOnly(N_OBS_V2, N_ACT,
                                   match_encdec_hidden(rep["trainable_params"], N_OBS_V2, N_ACT), seed)
    else:
        raise ValueError(f"unknown condition {condition!r}")
    info["param_report"] = model.param_report()
    info["reference_param_report"] = rep
    return model, info


# ------------------------------------------------------------------ training (matched DAgger)
def _pad(arr: np.ndarray, t: int) -> np.ndarray:
    if arr.shape[0] == t:
        return arr
    return np.concatenate([arr, np.zeros((t - arr.shape[0], *arr.shape[1:]), dtype=arr.dtype)], 0)


def _recipe_cond(level: float, r: dict) -> EvalCondV2:
    kind = r["kind"]
    return EvalCondV2(
        f"train_{kind}", conflict=(kind != "noconflict"),
        ctx="persistent" if kind == "conflict_persistent" else "pulse",
        contrast=float(r["contrast"]), odor_noise_mult=float(r["odor_noise_mult"]),
        vision_noise_mult=float(r["vision_noise_mult"]), dropout=DropoutSpec(level, "channel", "episode"))


def train_student_v2(model, proto: dict, seed: int, lr: float, log=print) -> dict:
    import torch

    from flycircuit.envs.arena_v2 import make_layout_v2
    from flycircuit.models.train import detach_state

    tr = proto["training"]
    import os

    # thread count is a throughput setting (measured in configs/hardware/adaptive_profile.yaml), not science
    torch.set_num_threads(int(os.environ.get("FLYCIRCUIT_NAV2_THREADS", tr.get("cpu_threads", 4))))
    torch.manual_seed(seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    ctrl = NeuralController(model)
    teacher = PrivilegedTeacher()
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ms: list[np.ndarray] = []
    losses_by_round: list[float] = []
    diverged = False
    n_eps = 0
    t0 = time.perf_counter()
    for r, beta in enumerate(tr["dagger_betas"]):
        rng_mix = np.random.default_rng(seed * 1000 + r)
        slot = 0
        for lvl in tr["dropout_levels"]:
            for rec in tr["per_dropout_level_recipe"]:
                lays = [make_layout_v2(seed * 1000 + r * 100 + slot + j, "train") for j in range(rec["n"])]
                slot += rec["n"]
                cond = _recipe_cond(lvl, rec)
                _, _, (x, y, m) = run_episodes(
                    ctrl, lays, cond, noise_seed=seed * 100 + r, mask_seed=seed * 100 + r + 50,
                    beta=beta, record=True, teacher=teacher, rng_mix=rng_mix, env_cls=NavEnvV2)
                xs.append(_pad(x, MAX_STEPS))
                ys.append(_pad(y, MAX_STEPS))
                ms.append(_pad(m, MAX_STEPS))
                n_eps += rec["n"]
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
                    loss = ((((torch.tanh(out) - yc) ** 2).sum(-1)) * mc).sum() / mc.sum()
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
        log(f"    round {r} beta={beta} seqs={nseq} loss={losses_by_round[-1]:.4f}")
        if diverged:
            break
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # pragma: no cover
        rss = float("nan")
    return {"diverged": diverged, "lr": lr, "loss_by_round": losses_by_round,
            "train_seconds": time.perf_counter() - t0, "train_env_episodes": n_eps,
            "peak_rss_mib_process": rss}


# ------------------------------------------------------------------ evaluation
def evaluate_v2(controller, proto: dict, split: str = "test", names: list[str] | None = None) -> dict:
    ev = proto["evaluation"]
    out = {}
    cache: dict[bool, list] = {}
    for spec in ev["conditions"]:
        if names and spec["name"] not in names:
            continue
        ood = spec.get("layouts") == "ood_obstacles"
        if split == "val" and ood:
            continue
        if ood not in cache:
            cache[ood] = eval_layouts(proto, split, ood=ood)
        cond = cond_from_spec(spec)
        metrics, prov = run_episodes(controller, cache[ood], cond, noise_seed=ev["eval_noise_seed"],
                                     mask_seed=ev["eval_mask_seed"], env_cls=NavEnvV2)
        agg = aggregate(metrics)
        agg["success_per_episode"] = [int(m["success"]) for m in metrics]
        agg["ambiguous"] = bool(spec.get("ambiguous", False))
        agg["provenance"] = prov
        out[spec["name"]] = agg
    return out


def scripted_references(proto: dict, split: str = "test") -> dict:
    refs = {}
    for name in proto["evaluation"]["scripted_references"]:
        if name == "privileged_teacher":
            ctl = PrivilegedTeacher()
        elif name == "random_action":
            ctl = RandomWalk(0)
        else:
            ctl = SCRIPTED[name]()
        r = evaluate_v2(ctl, proto, split)
        refs[name] = {c: {k: v for k, v in a.items() if k != "success_per_episode"} for c, a in r.items()}
    return refs


def _slim(a: dict) -> dict:
    return {c: {k: v for k, v in x.items() if k != "success_per_episode"} for c, x in a.items()}


# ------------------------------------------------------------------ lr search (validation only)
def _out_dir(root: Path, label: str) -> Path:
    d = root / OUT_REL / label
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_search(label: str = "search", conditions: list[str] | None = None, root: Path | None = None,
               log=print, grid: list[float] | None = None, seeds: list[int] | None = None) -> dict:
    """Equal-budget lr search for every model condition on the VALIDATION split (conflict_pulse)."""
    from flycircuit.tracking.manifest import RunRecorder

    root = root or _root()
    v = _require_frozen(root)
    proto = load_protocol(root)
    sel = proto["selection"]
    grid = grid or sel["grid"]
    seeds = seeds or sel["search_seeds"]
    conds = conditions or proto["conditions"]
    graph, sign, gmeta = load_graph_v2()
    nulls = {k: load_nulls(graph, "plain" if k == "null_v2_swap" else "group_constrained")
             for k in ("null_v2_swap", "null_v2_group") if k in conds}
    out = _out_dir(root, label)
    part = out / "partial.json"
    done = json.loads(part.read_text()) if part.exists() else {}
    cfg = {"protocol_hash": v["hash"], "grid": grid, "seeds": seeds, "conditions": conds, "graph": gmeta,
           "stage": "lr_search_validation_only"}
    with RunRecorder("nav2-search", cfg, seeds, "lr search, matched DAgger", "fixed grid",
                     data_release=gmeta["release"], data_hashes={"edge_hash": gmeta["edge_hash"]},
                     mapping_hash=gmeta["mapping_hash"], root=root, label=label) as rec:
        for c in conds:
            for lr in grid:
                for s in seeds:
                    key = f"{c}|{lr}|{s}"
                    if key in done:
                        continue
                    log(f"[search] {c} lr={lr} seed={s}")
                    model, info = build_model_v2(c, s, graph, sign, proto, nulls)
                    trn = train_student_v2(model, proto, s, lr, log=log)
                    ev = evaluate_v2(NeuralController(model, name=c), proto, "val", ["conflict_pulse"])
                    sr = ev["conflict_pulse"]["success_rate"]
                    done[key] = {"condition": c, "lr": lr, "seed": s, "val_success": sr,
                                 "diverged": trn["diverged"], "train_seconds": trn["train_seconds"],
                                 "loss_final": trn["loss_by_round"][-1]}
                    part.write_text(json.dumps(done, indent=1))
                    log(f"    val success {sr:.3f} ({trn['train_seconds']:.0f}s)")
        payload = write_selection(out, done, conds, grid, seeds, v["hash"], gmeta)
        rec.add_artifact(out / "selection.json")
    return payload


def choose_lrs(done: dict, conds: list[str], grid: list[float], seeds: list[int]) -> dict:
    """Pick, per condition, the lr with the best mean validation success (tie -> smaller lr).
    Raises if a condition lacks any (lr, seed) trial: the budget must be equal for every condition."""
    chosen = {}
    for c in conds:
        rows = {}
        for lr in grid:
            keys = [f"{c}|{lr}|{s}" for s in seeds]
            missing = [k for k in keys if k not in done]
            if missing:
                raise RuntimeError(f"incomplete search for {c}: missing {missing[:3]}...")
            v_ = [done[k]["val_success"] for k in keys if not done[k]["diverged"]]
            rows[lr] = float(np.mean(v_)) if v_ else float("nan")
        best = max((lr for lr in grid if not np.isnan(rows[lr])), key=lambda lr: (rows[lr], -lr))
        chosen[c] = {"lr": best, "val_by_lr": rows, "grid_edge": best in (min(grid), max(grid))}
    return chosen


def write_selection(out: Path, done: dict, conds, grid, seeds, proto_hash: str, gmeta: dict) -> dict:
    payload = {"protocol_hash": proto_hash, "graph": gmeta, "grid": grid, "seeds": seeds,
               "chosen": choose_lrs(done, conds, grid, seeds), "trials": done, "split_used": "val"}
    sp = out / "selection.json"
    sp.write_text(json.dumps(payload, indent=1, default=float))
    (out / "selection.sha256").write_text(hashlib.sha256(sp.read_bytes()).hexdigest() + "\n")
    return payload


def merge_search(labels: list[str], out_label: str = "search", root: Path | None = None) -> dict:
    """Merge partial lr-search trials from several parallel processes into one selection.json."""
    root = root or _root()
    v = _require_frozen(root)
    proto = load_protocol(root)
    done: dict = {}
    for lb in labels:
        done.update(json.loads((root / OUT_REL / lb / "partial.json").read_text()))
    graph, _, gmeta = load_graph_v2()
    return write_selection(_out_dir(root, out_label), done, proto["conditions"],
                           proto["selection"]["grid"], proto["selection"]["search_seeds"], v["hash"], gmeta)


# ------------------------------------------------------------------ final run
def load_selection(root: Path, label: str = "search") -> tuple[dict, str]:
    p = root / OUT_REL / label / "selection.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run `flycircuit nav2 search` first")
    return json.loads(p.read_text()), hashlib.sha256(p.read_bytes()).hexdigest()


def run_final(seeds: list[int] | None = None, label: str = "final", search_label: str = "search",
              conditions: list[str] | None = None, root: Path | None = None, log=print,
              exploratory: bool = False) -> dict:
    from flycircuit.tracking.manifest import RunRecorder

    root = root or _root()
    v = _require_frozen(root)
    proto = load_protocol(root)
    seeds = seeds or proto["seeds"]["final"]
    conds = conditions or proto["conditions"]
    selection, sel_hash = load_selection(root, search_label)
    leak = assert_no_leakage(proto, sorted(set(seeds) | set(proto["seeds"]["search"])))
    graph, sign, gmeta = load_graph_v2()
    nulls = {k: load_nulls(graph, "plain" if k == "null_v2_swap" else "group_constrained")
             for k in ("null_v2_swap", "null_v2_group") if k in conds}
    out = _out_dir(root, label)
    pdir = out / "models"
    pdir.mkdir(exist_ok=True)
    cfg = {"protocol_hash": v["hash"], "selection_sha256": sel_hash, "seeds": seeds, "conditions": conds,
           "graph": gmeta, "leakage_check": leak, "exploratory": exploratory}
    models: list[dict] = []
    with RunRecorder("nav2-final", cfg, seeds, "matched DAgger imitation, sequential",
                     proto["training"]["stopping_rule"], data_release=gmeta["release"],
                     data_hashes={"edge_hash": gmeta["edge_hash"], "selection_sha256": sel_hash},
                     mapping_hash=gmeta["mapping_hash"], root=root, label=label) as rec:
        rp = out / "refs.json"
        if rp.exists():
            refs = json.loads(rp.read_text())
        else:
            refs = scripted_references(proto, "test")
            rp.write_text(json.dumps(refs, default=float))
        for k, row in refs.items():
            log(f"reference {k}: conflict_pulse={row['conflict_pulse']['success_rate']:.3f} "
                f"hidden={row['conflict_hidden']['success_rate']:.3f}")
        for s in seeds:
            for c in conds:
                f = pdir / f"{c}_{s}.json"
                if f.exists():
                    models.append(json.loads(f.read_text()))
                    continue
                lr = selection["chosen"][c]["lr"]
                log(f"[seed {s}] {c} lr={lr}")
                model, info = build_model_v2(c, s, graph, sign, proto, nulls)
                trn = train_student_v2(model, proto, s, lr, log=log)
                ev = evaluate_v2(NeuralController(model, name=c), proto, "test")
                res = {"condition": c, "seed": s, "lr": lr, "info": info, "training": trn,
                       "eval": ev, "grid_edge": selection["chosen"][c]["grid_edge"]}
                f.write_text(json.dumps(res, default=float))
                models.append(res)
                log(f"  -> conflict_pulse {ev['conflict_pulse']['success_rate']:.3f} "
                    f"({trn['train_seconds']:.0f}s)")
        summ = summarize_v2(models, proto, refs)
        payload = {"run_id": rec.run_id, "protocol_hash": v["hash"], "selection_sha256": sel_hash,
                   "graph": gmeta, "references": refs, "summary": summ, "exploratory": exploratory,
                   "models": [{k: (_slim(m[k]) if k == "eval" else m[k]) for k in m} for m in models],
                   "caveats": ["engineered interface and arena; not biomechanics",
                               "connectome condition = right-hemisphere MB pilot subset",
                               "n=5 trained models per condition: exploratory-grade intervals",
                               "episodes within a model are not independent replicates",
                               "conflict_hidden is intrinsically ambiguous and is not scored as failure"]}
        (out / "results.json").write_text(json.dumps(payload, indent=1, default=float))
        (out / "summary.md").write_text(render_markdown(payload, proto))
        rec.add_artifact(out / "results.json")
        rec.add_artifact(out / "summary.md")
        pm = summ["question_1_task_success"].get("connectome_pilot_mb")
        if pm:
            rec.metrics["connectome_conflict_pulse_success"] = pm["estimate"]
    payload["out_dir"] = str(out)
    return payload


def summarize_v2(models: list[dict], proto: dict, refs: dict) -> dict:
    from flycircuit.eval.stats import bootstrap_ci, paired_bootstrap_diff

    by: dict[str, dict[int, dict]] = {}
    for m in models:
        by.setdefault(m["condition"], {})[m["seed"]] = m
    cnames = [c["name"] for c in proto["evaluation"]["conditions"]]
    table = {}
    for c, d in by.items():
        live = [d[s] for s in sorted(d) if not d[s]["training"]["diverged"]]
        row = {"n_models": len(d), "n_diverged": len(d) - len(live),
               "lr": sorted({m["lr"] for m in d.values()}),
               "grid_edge": any(m.get("grid_edge") for m in d.values())}
        for cn in cnames:
            vals = [m["eval"][cn]["success_rate"] for m in live if cn in m["eval"]]
            row[cn] = bootstrap_ci(vals, n_boot=5000) if vals else None
        table[c] = row
    primary = "conflict_pulse"
    oracle = refs.get("ctx_memory", {}).get(primary, {}).get("success_rate")
    q1 = {}
    for c, row in table.items():
        b = row[primary]
        q1[c] = (dict(b, fraction_of_scripted_oracle=(b["estimate"] / oracle if oracle else None))
                 if b else None)
    contr = {}
    pop = "connectome_pilot_mb"
    for comp in ["null_v2_swap", *proto["question_2_anatomical_advantage"]["secondary_contrasts"]]:
        if comp in by and pop in by:
            seeds = sorted(set(by[pop]) & set(by[comp]))
            per_cond = {}
            for cn in cnames:
                if proto["evaluation"]["conditions"][cnames.index(cn)].get("ambiguous"):
                    continue
                a = [by[pop][s]["eval"][cn]["success_rate"] for s in seeds]
                b = [by[comp][s]["eval"][cn]["success_rate"] for s in seeds]
                per_cond[cn] = paired_bootstrap_diff(a, b, n_boot=5000)
            contr[comp] = per_cond
    prim = contr.get("null_v2_swap", {}).get(primary)
    if prim is None:
        verdict = "not evaluated"
    elif prim["lo"] != prim["lo"]:  # nan
        verdict = "not evaluated (n<2 units)"
    elif prim["lo"] > 0:
        verdict = "advantage supported (CI excludes 0, positive)"
    elif prim["hi"] < 0:
        verdict = "connectome worse than its null (CI excludes 0, negative)"
    else:
        verdict = "no advantage demonstrated (CI includes 0)"
    amb = {c: (table[c]["conflict_hidden"] if table[c].get("conflict_hidden") else None) for c in table}
    return {"per_condition": table, "question_1_task_success": q1,
            "question_2_anatomical_advantage": {"primary_contrast": "connectome - null_v2_swap on "
                                                + primary, "primary_result": prim, "verdict": verdict,
                                                "contrasts_exploratory": contr},
            "ambiguous_condition_not_scored": {"conflict_hidden": amb,
                                               "scripted_ceiling": refs.get("ctx_memory", {}).get(
                                                   "conflict_hidden", {}).get("success_rate")},
            "scripted_oracle_conflict_pulse": oracle}


def render_markdown(payload: dict, proto: dict) -> str:
    cn = [c["name"] for c in proto["evaluation"]["conditions"]]
    lines = [f"# Navigation v2 run {payload['run_id']}", "",
             f"Protocol `{payload['protocol_hash'][:12]}`, selection `{payload['selection_sha256'][:12]}`, "
             f"{'EXPLORATORY' if payload['exploratory'] else 'frozen protocol, n=5 (exploratory-grade CIs)'}.",
             "", "Success rate, mean over trained models [95% bootstrap CI over models]:", "",
             "| model | lr | " + " | ".join(cn) + " |", "|---|---|" + "---|" * len(cn)]
    for c, row in payload["summary"]["per_condition"].items():
        cells = []
        for k in cn:
            b = row.get(k)
            if not b:
                cells.append("n/a")
            else:
                ci = "" if b["lo"] != b["lo"] else f" [{b['lo']:.2f},{b['hi']:.2f}]"
                cells.append(f"{b['estimate']:.3f}{ci}")
        flag = " (grid edge)" if row["grid_edge"] else ""
        lines.append(f"| {c} (n={row['n_models']}) | {row['lr']}{flag} | " + " | ".join(cells) + " |")
    lines += ["", "Scripted references (no learning, 200 test episodes each, not replicates):", "",
              "| controller | " + " | ".join(cn) + " |", "|---|" + "---|" * len(cn)]
    for c, row in payload["references"].items():
        lines.append(f"| {c} | " + " | ".join(f"{row[k]['success_rate']:.3f}" for k in cn) + " |")
    q2 = payload["summary"]["question_2_anatomical_advantage"]
    lines += ["", f"**Q2 (anatomical advantage), primary contrast {q2['primary_contrast']}:** "
              f"{q2['primary_result']} -> **{q2['verdict']}**.", "",
              "`conflict_hidden` is intrinsically ambiguous (no information identifies the goal); it is "
              "reported but never scored as failure.", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ v1 protocol under annotation_v2 (exploratory)
def run_v1_sensitivity(seeds: list[int] | None = None, label: str = "v1_sens_annotation_v2",
                       conditions: list[str] | None = None, root: Path | None = None, log=print) -> dict:
    """Re-run the UNCHANGED v1 protocol with annotation_v2 signs (unknown/modulatory -> 0).
    EXPLORATORY: the v1 test layouts were already viewed in v1. New run id; v1 results untouched."""
    from flycircuit.eval import navigation as nv1
    from flycircuit.tracking.manifest import RunRecorder

    root = root or _root()
    proto1 = nv1.load_protocol(root)
    v1v = nv1.verify_protocol(root)
    if not (v1v["frozen"] and v1v["match"]):
        raise RuntimeError(f"v1 protocol hash mismatch {v1v}; refusing (v1 must stay untouched)")
    p2 = load_protocol(root)
    seeds = seeds or p2["sensitivity_v1_under_annotation_v2"]["seeds"]
    proto = copy.deepcopy(proto1)
    proto["model_interface"]["unknown_sign_policy"] = "zero"
    graph, sign, gmeta = load_graph_v2(n_sensory=proto1["interface"]["n_obs"])
    conds = conditions or proto1["conditions"]
    out = _out_dir(root, label)
    cfg = {"v1_protocol_hash": v1v["hash"], "seeds": seeds, "conditions": conds, "graph": gmeta,
           "change_vs_v1": "sign policy annotation_v2, unknown/modulatory=0 (v1: top_nt-only, unknown=excitatory)"}
    models = []
    with RunRecorder("nav1-sens-annv2", cfg, seeds, "v1 matched DAgger", proto1["training"]["stopping_rule"],
                     data_release=gmeta["release"], data_hashes={"edge_hash": gmeta["edge_hash"]},
                     mapping_hash=gmeta["mapping_hash"], root=root, label=label) as rec:
        for s in seeds:
            for c in conds:
                f = out / f"{c}_{s}.json"
                if f.exists():
                    models.append(json.loads(f.read_text()))
                    continue
                log(f"[v1-sens seed {s}] {c}")
                model, info = nv1.build_model(c, s, graph, sign, proto)
                trn = nv1.train_student(model, proto, s, log=log)
                ev = nv1.evaluate_controller(NeuralController(model, name=c), proto)
                res = {"condition": c, "seed": s, "info": info, "training": trn, "eval": ev}
                f.write_text(json.dumps(res, default=float))
                models.append(res)
                log(f"  -> 25% success {ev[proto['primary_endpoint']['condition']]['success_rate']:.3f}")
        summ = nv1.summarize(models, proto, exploratory=True)
        prior_tab = None
        cands = [p for p in sorted((root / "results" / "navigation").glob("*/results.json"))
                 if json.loads(p.read_text()).get("exploratory") is False]
        prior = cands[0] if cands else None
        if prior:
            prior_tab = json.loads(prior.read_text())["summary"]["per_condition"]
        payload = {"run_id": rec.run_id, "v1_protocol_hash": v1v["hash"], "graph": gmeta,
                   "summary": summ, "v1_confirmatory_per_condition_for_comparison": prior_tab,
                   "exploratory": True, "models": [{**m, "eval": _slim(m["eval"])} for m in models],
                   "caveat": "EXPLORATORY sensitivity; v1 test layouts previously viewed; the only change is "
                             "the sign policy (graph, interfaces, training, seeds 100-104 identical to v1)."}
        (out / "results.json").write_text(json.dumps(payload, indent=1, default=float))
        rec.add_artifact(out / "results.json")
    payload["out_dir"] = str(out)
    return payload
