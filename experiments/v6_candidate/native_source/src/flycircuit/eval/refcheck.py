"""Reference-engine validation: flycircuit LIF vs the published Brian2 model (spec section 10).

Protocol and tolerances live in ``configs/experiments/refcheck.yaml`` and were written BEFORE any
comparison ran; the sha256 of that file is stored in every result. This is simulator-vs-simulator
ENGINEERING validation. It says nothing about biological validity.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from flycircuit.config import project_root
from flycircuit.models.lif import (
    LIFParams,
    LIFResult,
    poisson_events,
    silence_outgoing,
    simulate,
    weights_mV,
)

PROTOCOL = "configs/experiments/refcheck.yaml"
BRIAN_PY = "external/brian_env/.venv/bin/python"
BRIAN_RUNNER = "external/brian_ref/run_brian.py"
SHIU_DIR = "external/Drosophila_brain_model"


# ----------------------------------------------------------------------------- protocol
def load_protocol(root: Path | None = None) -> tuple[dict, str]:
    root = root or project_root()
    p = root / PROTOCOL
    raw = p.read_bytes()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def _crit(name: str, value, tol, op: str, note: str = "") -> dict:
    """One tolerance check. value None => 'undefined' (never silently a pass)."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        status = "undefined"
    elif op == ">=":
        status = "pass" if value >= tol else "fail"
    elif op == "<=":
        status = "pass" if value <= tol else "fail"
    elif op == "==":
        status = "pass" if value == tol else "fail"
    else:
        raise ValueError(op)
    return {"criterion": name, "value": value, "tolerance": tol, "op": op, "status": status,
            "note": note}


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson r, or None when undefined (constant vector / < 2 points)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# ----------------------------------------------------------------------------- cases
@dataclass
class Case:
    name: str
    description: str
    n: int
    pre: np.ndarray
    post: np.ndarray
    signed: np.ndarray  # signed integer contact counts
    ext_steps: np.ndarray
    ext_targets: np.ndarray
    no_rfc: np.ndarray
    silenced: np.ndarray
    steps: int
    rec: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    mode: str = "shared"
    rate_hz: float = 150.0
    seed: int = 0

    def W(self, p: LIFParams):
        return weights_mV(self.signed, self.pre, self.post, self.n, p)


def _case(name, desc, n, edges, events, no_rfc, silenced=(), steps=10000) -> Case:
    pre = np.array([e[0] for e in edges], dtype=np.int64)
    post = np.array([e[1] for e in edges], dtype=np.int64)
    signed = np.array([e[2] for e in edges], dtype=np.int64)
    es = np.array([e[0] for e in events], dtype=np.int64)
    et = np.array([e[1] for e in events], dtype=np.int64)
    return Case(name, desc, n, pre, post, signed, es, et, np.array(no_rfc, np.int64),
                np.array(silenced, np.int64), steps, np.arange(n, dtype=np.int64))


def microcircuit_cases() -> list[Case]:
    """Deterministic SYNTHETIC microcircuits with explicit shared input events (not biology)."""
    cases = []
    chain = [(0, 1, 250), (1, 2, 250), (2, 3, 250)]  # strong enough to propagate (see amendments)
    ev = [(10, 0), (400, 0), (405, 0), (900, 0), (2000, 0), (2001, 0), (2002, 0), (6000, 0)]
    cases.append(_case("chain_exc", "0->1->2->3 excitatory; events drive neuron 0 (incl. a "
                       "burst inside the refractory window)", 4, chain, ev, [0]))
    cases.append(_case("chain_inh", "0->1 exc, 3->2 sub-threshold exc, 1->2 inhibitory",
                       4, [(0, 1, 250), (1, 2, -250), (3, 2, 60)],
                       [(s, 0) for s in (10, 3000, 3004)] + [(s, 3) for s in (20, 25, 30, 3010, 3012)],
                       [0, 3]))
    conv = [(i, 6, 40) for i in range(6)]
    cases.append(_case("convergence", "6 sources -> 1 target, 40 contacts each; coincident and "
                       "staggered events", 7, conv,
                       [(100, 0), (100, 1), (100, 2), (100, 3), (100, 4), (100, 5),
                        (1000, 0), (1030, 1), (1060, 2), (5000, 0), (5000, 1), (5000, 2)],
                       list(range(6))))
    cases.append(_case("refractory_burst", "pacemaker 1 (no refractory) fires every step, driving "
                       "neuron 0 through a strong synapse: tests refractory freeze, g reset, "
                       "delays", 2, [(1, 0, 150)],
                       [(s, 1) for s in range(100, 600, 4)], [1]))
    cases.append(_case("lesion_chain", "chain_exc with neuron 1 silenced (published lesion = zero "
                       "outgoing synapses)", 4, chain, ev, [0], silenced=[1]))
    return cases


# ----------------------------------------------------------------------------- engines
def run_lif(case: Case, p: LIFParams, background=None, bg_seed: int = 0) -> LIFResult:
    W = case.W(p)
    if len(case.silenced):
        W = silence_outgoing(W, case.silenced)
    return simulate(W, case.steps, p, case.ext_steps, case.ext_targets,
                    no_refractory=case.no_rfc, record=case.rec if len(case.rec) else None,
                    background=background, background_seed=bg_seed)


def run_brian(cases: list[Case], p: LIFParams, root: Path | None = None) -> tuple[dict, dict]:
    """Run jobs on the published Brian2 model in its separate environment."""
    root = root or project_root()
    py, runner = root / BRIAN_PY, root / BRIAN_RUNNER
    if not py.exists():
        raise FileNotFoundError(f"{py} (run scripts/linux/setup_brian_env.sh)")
    model_py = root / SHIU_DIR / "model.py"
    if not model_py.exists():
        raise FileNotFoundError(f"{model_py} (clone philshiu/Drosophila_brain_model into external/)")
    arrays, jobs = {}, []
    for c in cases:
        for k in ("pre", "post", "signed", "silenced", "ext_steps", "ext_targets", "rec"):
            arrays[f"{c.name}__{k}"] = np.asarray(getattr(c, k))
        arrays[f"{c.name}__no_rfc"] = c.no_rfc
        jobs.append({"name": c.name, "n": c.n, "steps": c.steps, "dt_ms": p.dt, "mode": c.mode,
                     "w_ext_mV": p.w_ext, "rate_hz": c.rate_hz, "seed": c.seed})
    with tempfile.TemporaryDirectory(prefix="refcheck_") as td:
        tdp = Path(td)
        np.savez(tdp / "arrays.npz", **arrays)
        (tdp / "jobs.json").write_text(json.dumps({"jobs": jobs}))
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        env["PYTHONNOUSERSITE"] = "1"
        t0 = time.time()
        r = subprocess.run([str(py), "-I", str(runner), str(tdp / "jobs.json"),
                            str(tdp / "arrays.npz"), str(tdp / "out.npz"), str(model_py)],
                           cwd=td, capture_output=True, text=True, env=env, timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(f"Brian2 reference failed (rc={r.returncode}):\n{r.stderr[-3000:]}")
        out = dict(np.load(tdp / "out.npz"))
        meta = json.loads((tdp / "out.meta.json").read_text())
    meta["wall_s"] = time.time() - t0
    return out, meta


def brian_result(out: dict, c: Case, dt: float) -> LIFResult:
    return LIFResult(out[f"{c.name}__spike_steps"], out[f"{c.name}__spike_neurons"], c.n,
                     c.steps, dt, c.rec if len(c.rec) else None, out.get(f"{c.name}__v"),
                     out.get(f"{c.name}__g"))


# ----------------------------------------------------------------------------- tier 1
def compare_numerical(a: LIFResult, b: LIFResult) -> dict[str, Any]:
    """Per-neuron exact spike-train comparison plus state-trace error (a = ours, b = Brian2)."""
    ta, tb = a.trains(), b.trains()
    neurons = sorted(set(ta) | set(tb))
    identical = 0
    within1 = 0
    max_shift = 0
    count_equal = True
    for n_ in neurons:
        x, y = ta.get(n_, np.empty(0, np.int64)), tb.get(n_, np.empty(0, np.int64))
        if len(x) != len(y):
            count_equal = False
            continue
        s = int(np.abs(np.sort(x) - np.sort(y)).max()) if len(x) else 0
        max_shift = max(max_shift, s)
        identical += s == 0
        within1 += s <= 1
    out: dict[str, Any] = {
        "n_spiking_neurons_union": len(neurons),
        "spike_counts_ours": int(len(a.spike_steps)),
        "spike_counts_brian": int(len(b.spike_steps)),
        "counts_equal_per_neuron": count_equal,
        "neurons_identical_trains": int(identical),
        "neurons_within_1_step": int(within1),
        "max_spike_shift_steps": int(max_shift),
    }
    if a.v_trace is not None and b.v_trace is not None:
        out["v_max_abs_mV"] = float(np.abs(a.v_trace - b.v_trace).max())
        out["g_max_abs_mV"] = float(np.abs(a.g_trace - b.g_trace).max())
    return out


def tier1(proto: dict, p: LIFParams, brian_out: dict | None) -> dict:
    tol = proto["tiers"]["tier1_microcircuits"]["tolerances"]
    res = {}
    for c in microcircuit_cases():
        ours = run_lif(c, p)
        entry: dict[str, Any] = {"description": c.description, "spikes_ours": int(len(ours.spike_steps))}
        if brian_out is None:
            entry["status"] = "blocked: Brian2 reference not run"
        else:
            cmp_ = compare_numerical(ours, brian_result(brian_out, c, p.dt))
            crit = [
                _crit("spike_counts_per_neuron_equal", cmp_["counts_equal_per_neuron"], True, "=="),
                _crit("spike_time_max_abs_steps", cmp_["max_spike_shift_steps"],
                      tol["spike_time_max_abs_steps"], "<="),
                _crit("spike_time_within_acceptable_steps", cmp_["max_spike_shift_steps"],
                      tol["spike_time_acceptable_steps"], "<=", "secondary"),
                _crit("v_trace_max_abs_mV", cmp_.get("v_max_abs_mV"), tol["state_trace_max_abs_mV"], "<="),
                _crit("g_trace_max_abs_mV", cmp_.get("g_max_abs_mV"), tol["state_trace_max_abs_mV"], "<="),
            ]
            entry.update({"comparison": cmp_, "criteria": crit,
                          "status": "pass" if all(x["status"] == "pass" for x in crit[:2] + crit[3:])
                          else "fail"})
            if cmp_["spike_counts_brian"] == 0 and cmp_["spike_counts_ours"] == 0:
                entry["warning"] = "no spikes in either engine: case is vacuous"
        res[c.name] = entry
    return res


# ----------------------------------------------------------------------------- tier 2 / 3
def rate_metrics(ra: np.ndarray, rb: np.ndarray, tol: dict, stim_mask: np.ndarray) -> dict:
    """ra/rb: per-neuron mean rates (Hz) of ours / Brian2."""
    nz = (ra > 0) | (rb > 0)
    out = {
        "rate_pearson": pearson(ra[nz], rb[nz]),
        "rate_pearson_downstream_only": pearson(ra[nz & ~stim_mask], rb[nz & ~stim_mask]),
        "n_neurons_active_either": int(nz.sum()),
        "n_downstream_active_either": int((nz & ~stim_mask).sum()),
        "mean_rate_ours_hz": float(ra.mean()), "mean_rate_brian_hz": float(rb.mean()),
    }
    out["mean_rate_rel_err"] = (abs(ra.mean() - rb.mean()) / rb.mean()) if rb.mean() > 0 else None
    return out


def psth(res: list[LIFResult], bin_steps: int, steps: int) -> np.ndarray:
    nb = steps // bin_steps
    h = np.zeros(nb)
    for r in res:
        h += np.bincount(np.minimum(r.spike_steps // bin_steps, nb - 1), minlength=nb)
    return h


def _pilot(proto):
    from flycircuit.graph.bridge import load_circuit_graph

    t2 = proto["tiers"]["tier2_real_pilot"]
    return load_circuit_graph(t2["circuit"], threshold=t2["threshold_contacts"],
                              sign_policy="shiu_reported", unknown_sign="zero")


def pilot_cases(proto: dict, p: LIFParams, cg, mode="shared", lesion=False, trials=None):
    t2 = proto["tiers"]["tier2_real_pilot"]
    g = cg.graph
    signed = cg.signed_contacts().astype(np.int64)
    kc = cg.indices_of_group("kenyon")
    rng = np.random.default_rng(0)
    stim = np.sort(rng.choice(kc, size=t2["stimulus"]["n"], replace=False))
    lesion_idx = cg.indices_of_type("APL") if lesion else np.empty(0, np.int64)
    cases = []
    for k in range(trials or t2["n_trials"]):
        es, et = poisson_events(stim, 10000, t2["stimulus"]["rate_hz"], p.dt,
                                np.random.SeedSequence([t2["event_seed_base"], k]))
        nm = f"pilot_{'lesion' if lesion else 'ctrl'}_{mode}_{k}"
        cases.append(Case(nm, "mushroom_body_right_pilot", g.n, g.pre, g.post, signed,
                          es, et, stim, lesion_idx, 10000, np.empty(0, np.int64), mode,
                          float(t2["stimulus"]["rate_hz"]), 5000 + k))
    return cases, stim, lesion_idx


def tier2(proto: dict, p: LIFParams, skip_brian: bool, cg=None) -> dict:
    t2 = proto["tiers"]["tier2_real_pilot"]
    tol = t2["tolerances"]
    cg = cg or _pilot(proto)
    ctrl, stim, _ = pilot_cases(proto, p, cg, "shared", False)
    les, _, lesion_idx = pilot_cases(proto, p, cg, "shared", True)
    if len(lesion_idx) == 0:
        return {"status": "blocked: lesion target APL not in the selected circuit"}
    ours_c = [run_lif(c, p) for c in ctrl]
    ours_l = [run_lif(c, p) for c in les]
    info = {"circuit": cg.provenance, "stimulated": len(stim), "lesion_neurons": lesion_idx.tolist(),
            "ours_total_spikes_ctrl": int(sum(len(r.spike_steps) for r in ours_c)),
            "ours_active_neurons_ctrl": int(np.count_nonzero(sum(r.counts() for r in ours_c)))}
    if skip_brian:
        return {**info, "status": "blocked: Brian2 reference not run (--skip-brian)"}
    bo, meta = run_brian(ctrl + les, p)
    b_c = [brian_result(bo, c, p.dt) for c in ctrl]
    b_l = [brian_result(bo, c, p.dt) for c in les]
    stim_mask = np.zeros(cg.graph.n, bool)
    stim_mask[stim] = True

    def mean_rate(rs):
        return np.mean([r.rates_hz() for r in rs], axis=0)

    ra, rb = mean_rate(ours_c), mean_rate(b_c)
    m = rate_metrics(ra, rb, tol, stim_mask)
    thr = t2["responder_rate_hz"]
    sa, sb = ra >= thr, rb >= thr
    jac = float((sa & sb).sum() / (sa | sb).sum()) if (sa | sb).any() else None
    bins = int(round(tol["psth_bin_ms"] / p.dt))
    ha, hb = psth(ours_c, bins, 10000), psth(b_c, bins, 10000)
    ps_r = pearson(ha, hb)
    l1 = float(np.abs(ha - hb).sum() / hb.sum()) if hb.sum() > 0 else None
    dA = mean_rate(ours_l) - ra
    dB = mean_rate(b_l) - rb
    lesion_r = pearson(dA, dB)
    sign_ok = None if (dA.mean() == 0 and dB.mean() == 0) else bool(np.sign(dA.mean()) == np.sign(dB.mean()))
    exact = sum(
        1 for a, b in zip(ours_c, b_c, strict=True)
        if len(a.spike_steps) == len(b.spike_steps)
        and np.array_equal(np.sort(a.spike_steps * 100000 + a.spike_neurons),
                           np.sort(b.spike_steps * 100000 + b.spike_neurons)))
    crit = [
        _crit("rate_pearson", m["rate_pearson"], tol["rate_pearson_min"], ">="),
        _crit("mean_rate_rel_err", m["mean_rate_rel_err"], tol["mean_rate_rel_err_max"], "<="),
        _crit("responder_jaccard", jac, tol["responder_jaccard_min"], ">="),
        _crit("psth_pearson", ps_r, tol["psth_pearson_min"], ">="),
        _crit("psth_norm_l1", l1, tol["psth_norm_l1_max"], "<="),
        _crit("lesion_delta_rate_pearson", lesion_r, tol["lesion_delta_rate_pearson_min"], ">=",
              "undefined if lesion changes nothing / constant"),
        _crit("lesion_population_sign_agrees", sign_ok, True, "==",
              "undefined if both mean deltas are 0"),
    ]
    desc = {
        "n_responders_ours": int(sa.sum()), "n_responders_brian": int(sb.sum()),
        "responders_downstream_ours": int((sa & ~stim_mask).sum()),
        "responders_downstream_brian": int((sb & ~stim_mask).sum()),
        "lesion_mean_delta_ours_hz": float(dA.mean()), "lesion_mean_delta_brian_hz": float(dB.mean()),
        "lesion_max_abs_delta_ours_hz": float(np.abs(dA).max()),
        "trials_with_identical_spike_sets": int(exact), "trials": len(ctrl),
        "brian_total_spikes_ctrl": int(sum(len(r.spike_steps) for r in b_c)),
    }
    return {**info, "metrics": m, "criteria": crit, "descriptive": desc, "brian_meta": meta,
            "status": "pass" if all(x["status"] == "pass" for x in crit) else
            ("fail" if any(x["status"] == "fail" for x in crit) else "undefined")}


def tier3(proto: dict, p: LIFParams, skip_brian: bool, cg=None) -> dict:
    t2 = proto["tiers"]["tier2_real_pilot"]
    t3 = proto["tiers"]["tier3_native_poisson"]
    if skip_brian:
        return {"status": "blocked: Brian2 reference not run (--skip-brian)"}
    cg = cg or _pilot(proto)
    cases, stim, _ = pilot_cases(proto, p, cg, "native", False, trials=t3["n_trials"])
    ours = []
    for k, c in enumerate(cases):  # our engine: independent RNG streams, same rate
        es, et = poisson_events(stim, 10000, c.rate_hz, p.dt, np.random.SeedSequence([777, k]))
        ours.append(run_lif(Case(**{**c.__dict__, "mode": "shared", "ext_steps": es,
                                    "ext_targets": et}), p))
    bo, meta = run_brian(cases, p)
    b = [brian_result(bo, c, p.dt) for c in cases]
    stim_mask = np.zeros(cg.graph.n, bool)
    stim_mask[stim] = True
    ra = np.mean([r.rates_hz() for r in ours], axis=0)
    rb = np.mean([r.rates_hz() for r in b], axis=0)
    m = rate_metrics(ra, rb, t3["tolerances"], stim_mask)
    crit = [_crit("rate_pearson", m["rate_pearson"], t3["tolerances"]["rate_pearson_min"], ">="),
            _crit("mean_rate_rel_err", m["mean_rate_rel_err"],
                  t3["tolerances"]["mean_rate_rel_err_max"], "<=")]
    return {"metrics": m, "criteria": crit, "brian_meta": meta, "stimulus_rate_hz":
            t2["stimulus"]["rate_hz"], "status": "pass" if all(x["status"] == "pass" for x in crit)
            else ("fail" if any(x["status"] == "fail" for x in crit) else "undefined")}


# ----------------------------------------------------------------------------- boundary
def boundary_sensitivity(proto: dict, p: LIFParams, cg=None, trials: int = 3) -> dict:
    """Descriptive: cut vs background-drive boundary on our engine (no pass/fail)."""
    from flycircuit.graph.bridge import boundary_background_pairs

    bs = proto["boundary_sensitivity"]
    cg = cg or _pilot(proto)
    ctrl, stim, _ = pilot_cases(proto, p, cg, "shared", False, trials=trials)
    pairs = boundary_background_pairs(cg, "background_drive")
    rate = bs["background_drive"]["rate_hz"]
    bg = (pairs[0], pairs[1] * p.w_syn, rate) if pairs is not None else None
    cut = [run_lif(c, p) for c in ctrl]
    bgr = [run_lif(c, p, background=bg, bg_seed=100 + k) for k, c in enumerate(ctrl)]
    ra = np.mean([r.rates_hz() for r in cut], axis=0)
    rb = np.mean([r.rates_hz() for r in bgr], axis=0)
    stim_mask = np.zeros(cg.graph.n, bool)
    stim_mask[stim] = True
    thr = proto["tiers"]["tier2_real_pilot"]["responder_rate_hz"]
    sa, sb = ra >= thr, rb >= thr
    return {
        "kind": "descriptive, our engine only; Poisson background != upstream dynamics",
        "boundary": cg.boundary.summary(), "background_rate_hz": rate,
        "cut_pairs_used_as_background": int(len(pairs[0])) if pairs else 0,
        "mean_rate_cut_hz": float(ra.mean()), "mean_rate_background_hz": float(rb.mean()),
        "rate_pearson_all": pearson(ra, rb),
        "rate_pearson_downstream": pearson(ra[~stim_mask], rb[~stim_mask]),
        "responders_cut": int(sa.sum()), "responders_background": int(sb.sum()),
        "responder_jaccard": float((sa & sb).sum() / (sa | sb).sum()) if (sa | sb).any() else None,
        "n_trials": trials,
    }


# ----------------------------------------------------------------------------- integrity
def integrity(proto: dict | None = None) -> dict:
    """Real-data integrity checks on the actual FlyWire v783 selection graphs."""
    import duckdb
    import pyarrow as pa

    from flycircuit.data.selectors import _hash_ids
    from flycircuit.graph.bridge import (
        DEFAULT_NT_SIGN,
        SHIU_SIGN_TABLE,
        induced_subgraph_check,
        load_circuit_graph,
        processed_release_dir,
    )

    proto = proto or load_protocol()[0]
    ic = proto["integrity_real_graph"]
    proc = processed_release_dir()
    shiu = project_root() / SHIU_DIR / "Connectivity_783.parquet"
    out: list[dict] = []

    def add(name, ok, detail):
        out.append({"check": name, "status": "pass" if ok else "fail", "detail": detail})

    def info(name, detail):
        out.append({"check": name, "status": "info", "detail": detail})

    for circuit in ("mushroom_body_right_pilot", ic["circuit_for_pathways"]):
        cg = load_circuit_graph(circuit, threshold=proto["tiers"]["tier2_real_pilot"]["threshold_contacts"])
        sel, g = cg.selection, cg.graph
        con = duckdb.connect()
        con.execute("SET memory_limit='2GiB'")
        add(f"{circuit}: ids exact int64 & hash == selection.json",
            g.root_ids.dtype == np.int64 and _hash_ids(g.root_ids) == sel["ids_hash_sha256"],
            f"n={g.n}")
        add(f"{circuit}: neuron/edge/contact counts == selection.json",
            (g.n, g.n_edges, int(g.count.sum())) ==
            (sel["n_neurons"], sel["internal_pairs_retained"], sel["internal_contacts_retained"]),
            f"graph=({g.n},{g.n_edges},{int(g.count.sum())}) selection=({sel['n_neurons']},"
            f"{sel['internal_pairs_retained']},{sel['internal_contacts_retained']})")
        pre_id, post_id = g.root_ids[g.pre], g.root_ids[g.post]
        o = np.lexsort((post_id, pre_id))
        eh = hashlib.sha256(np.stack([pre_id[o], post_id[o], g.count[o].astype(np.int64)], 1)
                            .astype("<i8").tobytes()).hexdigest()
        add(f"{circuit}: edge hash == selection.json", eh == sel["edges_hash_sha256"], eh[:16])
        add(f"{circuit}: self-loops == selection.json", g.n_self_loops == sel["internal_self_loops"],
            f"{g.n_self_loops}")
        s = induced_subgraph_check(cg)
        add(f"{circuit}: induced graph endpoints valid & ids sorted unique",
            s["endpoints_valid"] and s["sorted_unique_ids"], s)
        bsum = cg.boundary.summary()
        info(f"{circuit}: boundary at threshold", bsum)
        info(f"{circuit}: node sign counts (shiu_reported)", cg.provenance["nodes_by_sign"])

        # orientation vs the published connectivity file: same (pre,post) must carry same count,
        # and the reversed pair must NOT systematically equal it
        if shiu.exists():
            con.register("e", pa.table({"pre": pre_id[o], "post": post_id[o],
                                        "count": g.count[o].astype(np.int64)}))
            fwd = con.execute(f"""SELECT count(*), count(*) FILTER (WHERE s.Connectivity = e.count)
                 FROM e JOIN '{shiu}' s ON s.Presynaptic_ID=e.pre AND s.Postsynaptic_ID=e.post"""
                              ).fetchone()
            rev = con.execute(f"""SELECT count(*) FILTER (WHERE s.Connectivity = e.count) FROM e
                 JOIN '{shiu}' s ON s.Presynaptic_ID=e.post AND s.Postsynaptic_ID=e.pre""").fetchone()
            nonrecip = con.execute("""SELECT count(*) FROM e WHERE NOT EXISTS (
                 SELECT 1 FROM e r WHERE r.pre=e.post AND r.post=e.pre)""").fetchone()[0]
            add(f"{circuit}: (pre,post,count) rows present with equal count in Shiu Connectivity_783",
                fwd[0] == fwd[1] == g.n_edges,
                f"forward matched {fwd[1]}/{g.n_edges}; reversed-orientation equal-count matches "
                f"{rev[0]} (reciprocal pairs may legitimately coincide; non-reciprocal pairs "
                f"{nonrecip}/{g.n_edges})")
            if circuit == ic["circuit_for_pathways"]:
                # global sign-rule agreement (report only)
                tbl = proc / SHIU_SIGN_TABLE
                if tbl.exists():
                    nt = (proc / "neurons.parquet")
                    rows = con.execute(f"""SELECT n.top_nt, s.sign, count(*) FROM '{tbl}' s
                         JOIN '{nt}' n USING (root_id) WHERE n.top_nt IS NOT NULL
                         GROUP BY 1,2""").fetchall()
                    tot = sum(r[2] for r in rows)
                    agree = sum(r[2] for r in rows if DEFAULT_NT_SIGN.get(r[0]) == r[1])
                    info("sign rule (ACh/DA/5HT/OA=+1, GABA/Glu=-1) vs Shiu per-neuron sign",
                         {"agreement_fraction": agree / tot, "neurons_compared": tot,
                          "note": "report only; Shiu's sign is not a pure function of top_nt"})
        else:
            info(f"{circuit}: Shiu connectivity absent", "orientation cross-check skipped")

        if circuit == ic["circuit_for_pathways"]:
            grp = cg.group

            def pairs_between(a, b_, grp=grp, g=g):
                m = (grp[g.pre] == a) & (grp[g.post] == b_)
                return int(m.sum()), int(g.count[m].sum())

            kc_mb, mb_kc = pairs_between("kenyon", "mbon"), pairs_between("mbon", "kenyon")
            dan_kc, kc_dan = pairs_between("dan", "kenyon"), pairs_between("kenyon", "dan")
            r1 = kc_mb[0] / max(mb_kc[0], 1)
            r2 = dan_kc[0] / max(kc_dan[0], 1)
            rmin = ic["kc_mbon_pair_ratio_min"]
            add(f"pathway KC->MBON pairs >= {rmin:g}x MBON->KC pairs (documented direction)",
                r1 >= rmin, f"KC->MBON pairs/contacts {kc_mb}, MBON->KC {mb_kc}, ratio {r1:.1f}")
            # demoted to report-only (see refcheck.yaml amendments): the "DAN->KC dominates"
            # expectation was mis-specified; KC->DAN pairs outnumber DAN->KC in this graph.
            info("pathway DAN->KC vs KC->DAN (report only; not a pass criterion)",
                 f"DAN->KC pairs/contacts {dan_kc}, KC->DAN {kc_dan}, pair ratio {r2:.2f}; "
                 "literature basis for the KC->DAN count unverified here")
            bnd = con.execute(
                f"SELECT cell_class, sum(contacts) c FROM '{proc / 'circuits' / circuit / 'boundary_ids.parquet'}' "
                "WHERE direction='incoming' GROUP BY 1 ORDER BY c DESC LIMIT 5").fetchall()
            info("top incoming boundary cell classes (contacts)", bnd)
    return {"checks": out, "n_fail": sum(c["status"] == "fail" for c in out),
            "n_pass": sum(c["status"] == "pass" for c in out)}


# ----------------------------------------------------------------------------- prepare / run
def prepare() -> dict:
    from flycircuit.graph.bridge import (
        SHIU_SIGN_TABLE,
        build_shiu_sign_table,
        processed_release_dir,
    )

    root = project_root()
    out = processed_release_dir() / SHIU_SIGN_TABLE
    return build_shiu_sign_table(root / SHIU_DIR / "Connectivity_783.parquet", out)


def shiu_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root / SHIU_DIR), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


def run_all(tiers: tuple[int, ...] = (1, 2, 3), skip_brian: bool = False, boundary: bool = True,
            do_integrity: bool = True) -> dict:
    from flycircuit.tracking.manifest import RunRecorder

    root = project_root()
    proto, sha = load_protocol(root)
    p = LIFParams.from_shiu_source()
    results: dict[str, Any] = {
        "protocol_sha256": sha, "timestamp_utc": datetime.now(UTC).isoformat(),
        "reference_commit": shiu_commit(root), "lif_params": p.__dict__,
        "release_used": "flywire:783 (adaptation; the repo's code defaults are v630)",
        "disclaimer": "simulator-vs-simulator engineering validation, not biological validation",
    }
    with RunRecorder("refcheck", {"tiers": list(tiers), "skip_brian": skip_brian,
                                  "protocol_sha256": sha}, [0, 1000], "refcheck LIF vs Brian2",
                     "fixed protocol; tolerances predeclared", "flywire:783", label="refcheck",
                     use_mlflow=False) as rec:
        if 1 in tiers:
            brian_out = None
            if not skip_brian:
                cases = microcircuit_cases()
                brian_out, meta = run_brian(cases, p)
                results["tier1_brian_meta"] = meta
            results["tier1"] = tier1(proto, p, brian_out)
        cg = None
        if 2 in tiers or 3 in tiers or boundary:
            cg = _pilot(proto)
        if 2 in tiers:
            results["tier2"] = tier2(proto, p, skip_brian, cg)
        if 3 in tiers:
            results["tier3"] = tier3(proto, p, skip_brian, cg)
        if boundary:
            results["boundary_sensitivity"] = boundary_sensitivity(proto, p, cg)
        if do_integrity:
            results["integrity"] = integrity(proto)
        outdir = root / "results" / "refcheck"
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"{rec.run_id}.json"
        results["run_id"] = rec.run_id
        path.write_text(json.dumps(results, indent=1, default=_json_default))
        (outdir / "latest.json").write_text(json.dumps(results, indent=1, default=_json_default))
        rec.add_artifact(path)
    return results


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def render_markdown(results: dict) -> str:
    """Tables for docs/refcheck.md from a results dict (no values are invented)."""
    L = ["## Latest measured run", "",
         f"- run: `{results.get('run_id')}`  at {results.get('timestamp_utc')}",
         f"- protocol sha256: `{results.get('protocol_sha256')}`",
         f"- reference commit: `{results.get('reference_commit')}`",
         f"- release: {results.get('release_used')}", ""]
    t1 = results.get("tier1")
    if t1:
        L += ["### Tier 1 - deterministic microcircuits (numerical agreement)", "",
              "| case | ours spikes | Brian spikes | max shift (steps) | v max abs (mV) | g max abs (mV) | status |",
              "|---|---|---|---|---|---|---|"]
        for k, v in t1.items():
            c = v.get("comparison", {})
            L.append(f"| {k} | {v['spikes_ours']} | {c.get('spike_counts_brian','-')} | "
                     f"{c.get('max_spike_shift_steps','-')} | {c.get('v_max_abs_mV','-')} | "
                     f"{c.get('g_max_abs_mV','-')} | {v['status']} |")
        L.append("")
    for key, title in (("tier2", "Tier 2 - mushroom_body_right_pilot, shared Poisson events"),
                       ("tier3", "Tier 3 - native Brian2 PoissonInput (exploratory)")):
        t = results.get(key)
        if not t:
            continue
        L += [f"### {title}", "", f"status: **{t['status']}**", ""]
        if "criteria" in t:
            L += ["| criterion | value | tolerance | status |", "|---|---|---|---|"]
            for c in t["criteria"]:
                L.append(f"| {c['criterion']} | {c['value']} | {c['op']} {c['tolerance']} | {c['status']} |")
            L.append("")
        if "descriptive" in t:
            L += ["Descriptive: `" + json.dumps(t["descriptive"], default=_json_default) + "`", ""]
        if "metrics" in t:
            L += ["Metrics: `" + json.dumps(t["metrics"], default=_json_default) + "`", ""]
    b = results.get("boundary_sensitivity")
    if b:
        L += ["### Boundary sensitivity (descriptive)", "", "`" + json.dumps(b, default=_json_default) + "`", ""]
    i = results.get("integrity")
    if i:
        L += ["### Real-data integrity", "", f"pass {i['n_pass']} / fail {i['n_fail']}", "",
              "| check | status | detail |", "|---|---|---|"]
        for c in i["checks"]:
            L.append(f"| {c['check']} | {c['status']} | {str(c['detail']).replace('|','/')} |")
        L.append("")
    return "\n".join(L)
