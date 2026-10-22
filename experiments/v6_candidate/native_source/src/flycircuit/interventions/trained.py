"""Causal screens on TRAINED model instances (screen v2). Everything is model-relative.

Unit of replication (kept separate on purpose):
  * trained instance  = an independently seeded training run (own seed, own task instance, own
                        decoder/encoder/gains). Between-instance variation is the headline uncertainty.
  * noise repetition  = an independent draw of input noise with the instance FIXED. It only measures
                        how precisely one instance's effect is estimated; it never counts as replicates
                        of "the model".
Per intervention the instance-level effect is the mean over noise repetitions; inference (bootstrap CI,
t-test, sign consistency) is across instances. Holm/BH is applied to p-values within a declared family.

Metric (primary, declared): change in held-out cue ACCURACY (pp) of the greedy 2-action decision; secondary:
change in the logit margin. Held-out cues = the task instance's val+test splits; nothing in this screen
selects, tunes or trains on them. Instances whose baseline held-out accuracy < ``min_baseline_acc`` did not
learn the task and are EXCLUDED from the primary analysis (listed in ``instances.parquet``).

Closed-loop note: in the cue-valence task actions never change later observations, so the task is
open-loop by construction. A closed-loop contrast needs an embodied task (navigation); here it is recorded
as not applicable. The engineered reafference loop is contrasted on the structural model instead
(``sensitivity.py``).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy import stats as sps

from flycircuit.config import config_hash, project_root
from flycircuit.eval.stats import bootstrap_ci
from flycircuit.interventions.engine import edited_edge_values
from flycircuit.interventions.inference import adjust_families
from flycircuit.interventions.rng import RngStreams
from flycircuit.interventions.spec import CLAMP_KINDS, DOSE_UNITS, InterventionSpec
from flycircuit.interventions.targets import (
    degrees,
    eligible_units,
    matched_random_spec,
    plan_screen,
)

INSTANCE_FORMAT = "flycircuit.instance/1"
PRIMARY_METRIC = "delta_accuracy"


# ---------------------------------------------------------------------------------------------
# instances
# ---------------------------------------------------------------------------------------------
@dataclass
class Instance:
    path: str
    seed: int
    learner: str
    sign_variant: str
    model: object
    f_mu: torch.Tensor
    f_sd: torch.Tensor
    w_eff: torch.Tensor
    task: object
    window: slice
    baseline_val_acc: float
    meta: dict


def sign_vector(root_ids: np.ndarray, variant: str) -> tuple[np.ndarray, dict]:
    """annotation_v2 sign vector for a named policy variant (graph/signs.policy_grid)."""
    from flycircuit.graph.signs import load_assignment_for_ids, policy_grid

    grid = policy_grid()
    if variant not in grid:
        raise ValueError(f"unknown sign variant {variant!r}; choose {sorted(grid)}")
    asg = load_assignment_for_ids(root_ids, grid[variant])
    return asg.sign.astype(np.float32), {"tag": grid[variant].tag(), "hash": asg.hash(),
                                         "summary": asg.summary}


def build_bundle(circuit: str, min_contacts: int, variant: str):
    from dataclasses import replace

    from flycircuit.tasks.graph_loader import load_flywire_circuit

    base = load_flywire_circuit(circuit, min_contacts=min_contacts)
    sign, meta = sign_vector(base.graph.root_ids, variant)
    notes = {**base.notes, "sign_policy": meta["tag"], "sign_assignment_hash": meta["hash"]}
    return replace(base, sign=sign, notes=notes)


def _acc_margin(logits: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    pred = logits.argmax(-1)
    marg = logits.gather(1, y[:, None])[:, 0] - logits.gather(1, (1 - y)[:, None])[:, 0]
    return float((pred == y).float().mean()), float(marg.mean())


def heldout_cues(task, ctx: int = 0, splits=("val", "test")):
    cues = np.concatenate([task.cues[(s, ctx)] for s in splits])
    labels = np.concatenate([task.labels[(s, ctx)] for s in splits])
    return cues, labels


def train_instance(proto: dict, bundle, learner: str, lr: float, seed: int, variant: str,
                   out_path: Path, circuit: str, min_contacts: int) -> dict:
    """Train ONE instance on the acquisition phase (ctx 0) and write a self-describing checkpoint."""
    from flycircuit.eval.learning import _rngs, make_agent, make_task, protocol_hash, train_phase

    t0 = time.perf_counter()
    task = make_task(proto, seed)
    agent, info = make_agent(proto, bundle, "connectome", None, learner, lr, seed, task)
    train_phase(agent, task, 0, proto["training"]["trials_per_phase"], _rngs(seed, 0))
    m = agent.model
    w_eff = m.edge_values().detach()
    if agent.plastic is not None:
        w_eff = agent.plastic.override(w_eff, m.sign_node, m.prior_val).detach()
    val_acc, test_acc = agent.evaluate("val", 0), agent.evaluate("test", 0)
    ck = {
        "format": INSTANCE_FORMAT, "kind": "sparse_rnn", "seed": seed, "learner": learner,
        "lr": lr, "sign_variant": variant, "sign_tag": bundle.notes.get("sign_policy"),
        "circuit": circuit, "min_contacts": min_contacts,
        "edge_hash": bundle.graph.edge_hash(), "mapping_hash": bundle.graph.mapping_hash(),
        "protocol_hash": protocol_hash(proto), "proto": proto,
        "model_spec": {k: v for k, v in vars(m.spec).items()},
        "state_dict": {k: v.detach().cpu() for k, v in m.state_dict().items()},
        "f_mu": agent.f_mu.detach().cpu(), "f_sd": agent.f_sd.detach().cpu(),
        "w_eff": w_eff.cpu(),
        "train_summary": {"val_acc": val_acc, "test_acc": test_acc,
                          "n_trainable": info["n_params_trainable"],
                          "train_seconds": time.perf_counter() - t0},
        "origin": "flycircuit.interventions.trained.train_instance (v1 acquisition protocol, "
                  "annotation_v2 signs; NOT the frozen learning-v2 study)",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out_path)
    return {"path": str(out_path), "seed": seed, "learner": learner, "variant": variant,
            "val_acc": val_acc, "test_acc": test_acc,
            "seconds": ck["train_summary"]["train_seconds"]}


_BUNDLES: dict = {}


def load_instance(path: str | Path, device: str = "cpu") -> Instance:
    """Rebuild a model from a checkpoint (any producer following ``INSTANCE_FORMAT``)."""
    from flycircuit.eval.learning import build_sparse_model, make_task

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != INSTANCE_FORMAT:
        raise ValueError(f"{path}: unsupported checkpoint format {ck.get('format')!r}")
    key = (ck["circuit"], ck["min_contacts"], ck["sign_variant"])
    if key not in _BUNDLES:
        _BUNDLES[key] = build_bundle(*key)
    b = _BUNDLES[key]
    if b.graph.edge_hash() != ck["edge_hash"] or b.graph.mapping_hash() != ck["mapping_hash"]:
        raise ValueError(f"{path}: graph hash mismatch (checkpoint trained on a different graph)")
    task = make_task(ck["proto"], ck["seed"])
    model = build_sparse_model(ck["proto"], b, ck["seed"], task.spec.n_in)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    inst = Instance(str(path), ck["seed"], ck["learner"], ck["sign_variant"], model,
                    ck["f_mu"].to(device), ck["f_sd"].to(device), ck["w_eff"].to(device), task,
                    task.response_window(), float("nan"),
                    {k: ck[k] for k in ("sign_tag", "edge_hash", "mapping_hash", "protocol_hash",
                                        "lr", "train_summary", "origin")})
    cues, labels = heldout_cues(task)
    x = torch.as_tensor(task.make_input(cues, 0, np.random.default_rng(0))).to(device)
    logits = branch_logits(inst, x, None, RngStreams(0, 0))
    inst.baseline_val_acc = _acc_margin(logits, torch.as_tensor(labels).to(device))[0]
    return inst


# ---------------------------------------------------------------------------------------------
# paired task forward with interventions
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def branch_logits(inst: Instance, x: torch.Tensor, spec: InterventionSpec | None,
                  streams: RngStreams, return_activity: bool = False):
    """(T,B,n_in) -> logits (B,2). Same update as ``LearningAgent._forward`` (noise=0, no plasticity
    in the screening pass: effective gains are frozen at their end-of-training value)."""
    m = inst.model
    T, B = x.shape[0], x.shape[1]
    steps = T
    dev = m.bias.device
    h = torch.zeros(B, m.n, device=dev)
    w_base = inst.w_eff
    t0, t1 = (0, 0) if spec is None else spec.window(steps)
    tgt = (None if spec is None else
           torch.as_tensor(np.asarray(spec.targets), dtype=torch.long, device=dev))
    w_edit = None
    if spec is not None and spec.kind not in CLAMP_KINDS + ("drive_excitation",
                                                           "transient_disruption"):
        w_edit = edited_edge_values(m, w_base, spec)
    dis = streams.stream("disruption") if spec is not None else None
    ys = []
    act = torch.zeros(m.n, device=dev)
    for t in range(steps):
        active = spec is not None and t0 <= t < t1
        w = w_edit if (active and w_edit is not None) else w_base
        drive = m.recurrent(h, w) + m.bias
        drive = drive.index_add(1, m.sensory_idx, x[t] @ m.encoder)
        if spec is not None and spec.kind == "drive_excitation" and active:
            drive[:, tgt] += float(spec.dose)
        if spec is not None and spec.kind == "transient_disruption":
            nz = torch.from_numpy(
                dis.standard_normal((B, len(spec.targets))).astype(np.float32)).to(dev)
            if active:
                drive[:, tgt] += float(spec.dose) * nz
        h = (1 - m.alpha) * h + m.alpha * m.phi(drive)
        if active and spec.kind in CLAMP_KINDS:
            h[:, tgt] = float(spec.dose)
        z = ((h[:, m.readout_idx] - inst.f_mu) / inst.f_sd).clamp(-5.0, 5.0)
        ys.append(z @ m.decoder)
        if return_activity:
            act += h.abs().mean(0)
    logits = torch.stack(ys)[inst.window].mean(0)
    return (logits, act / steps) if return_activity else logits


def instance_effects(inst: Instance, specs: list[InterventionSpec], controls: dict,
                     n_rep: int, seed: int, ctx: int = 0) -> dict:
    """Per-intervention effect of ONE instance: arrays over noise repetitions, for target and its
    matched random controls. Baseline and every branch of a repetition share the same input draw."""
    cues, labels = heldout_cues(inst.task, ctx)
    dev = inst.model.bias.device
    y = torch.as_tensor(labels).to(dev)
    n_ctrl = max((len(v) for v in controls.values()), default=0)
    out = {s.intervention_id: {"d_acc": np.zeros(n_rep), "d_margin": np.zeros(n_rep),
                               "c_acc": np.zeros((n_ctrl, n_rep)), "c_margin": np.zeros((n_ctrl, n_rep))}
           for s in specs}
    base = {"acc": np.zeros(n_rep), "margin": np.zeros(n_rep), "act": None}
    acts = []
    for r in range(n_rep):
        streams = RngStreams(seed + 10_000 * inst.seed, r)
        rng = streams.stream("obs_noise")
        x = torch.as_tensor(inst.task.make_input(cues, ctx, rng)).to(dev)
        lb, act = branch_logits(inst, x, None, streams, return_activity=True)
        acts.append(act)
        ba, bm = _acc_margin(lb, y)
        base["acc"][r], base["margin"][r] = ba, bm
        for s in specs:
            la = branch_logits(inst, x, s, RngStreams(seed + 10_000 * inst.seed, r))
            a, mg = _acc_margin(la, y)
            o = out[s.intervention_id]
            o["d_acc"][r], o["d_margin"][r] = a - ba, mg - bm
            for k, c in enumerate(controls.get(s.intervention_id, [])):
                lc = branch_logits(inst, x, c, RngStreams(seed + 10_000 * inst.seed, r))
                ca, cm = _acc_margin(lc, y)
                o["c_acc"][k, r], o["c_margin"][k, r] = ca - ba, cm - bm
    base["act"] = torch.stack(acts).mean(0).cpu().numpy()
    return {"effects": out, "baseline": base}


# ---------------------------------------------------------------------------------------------
# aggregation across instances
# ---------------------------------------------------------------------------------------------
def _across(values: np.ndarray, seed: int) -> dict:
    """values: per-instance effects. Bootstrap CI over INSTANCES, t-test over instances."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    out = {"mean": float(v.mean()) if n else float("nan"),
           "sd": float(v.std(ddof=1)) if n > 1 else float("nan"), "n_instances": n}
    ci = bootstrap_ci(v, np.mean, n_boot=4000, seed=seed) if n >= 2 else {"lo": np.nan, "hi": np.nan}
    out["ci_lo"], out["ci_hi"] = ci["lo"], ci["hi"]
    if n < 3:
        out["p"] = float("nan")
    elif np.allclose(v, v[0]):
        out["p"] = 1.0 if abs(v[0]) < 1e-12 else float("nan")
    else:
        out["p"] = float(sps.ttest_1samp(v, 0.0).pvalue)
    nz = v[np.abs(v) > 1e-12]
    out["sign_consistency"] = (float(max((nz > 0).mean(), (nz < 0).mean())) if len(nz) else float("nan"))
    return out


def aggregate(specs: list[InterventionSpec], per_inst: list[dict], graph, seed: int,
              n_controls: int) -> list[dict]:
    """per_inst[i] = instance_effects(...) for the i-th (non-excluded) instance."""
    if not specs:
        return []
    ind, outd = degrees(graph)
    rows = []
    for si, s in enumerate(specs):
        iid = s.intervention_id
        d_acc = np.array([p["effects"][iid]["d_acc"].mean() for p in per_inst])
        d_mar = np.array([p["effects"][iid]["d_margin"].mean() for p in per_inst])
        within = np.array([p["effects"][iid]["d_acc"].std(ddof=1) for p in per_inst])
        c_acc = (np.array([p["effects"][iid]["c_acc"].mean() for p in per_inst])
                 if n_controls else None)
        base_act = np.mean([p["baseline"]["act"][list(s.targets)].mean() for p in per_inst])
        a = _across(d_acc, seed + si)
        m = _across(d_mar, seed + si + 1)
        row = {
            "intervention_id": iid, "kind": s.kind, "level": s.level, "target_label": s.label,
            "n_targets": len(s.targets), "dose": float(s.dose), "dose_unit": DOSE_UNITS[s.kind],
            "t_start": s.t_start, "t_end": -1 if s.t_end is None else s.t_end, "family": s.family,
            "mean_in_degree": float(ind[list(s.targets)].mean()),
            "mean_out_degree": float(outd[list(s.targets)].mean()),
            "target_baseline_activity": float(base_act),
            "n_instances": a["n_instances"],
            "effect_acc_mean": a["mean"], "effect_acc_sd_between": a["sd"],
            "effect_acc_ci_lo": a["ci_lo"], "effect_acc_ci_hi": a["ci_hi"], "p_value": a["p"],
            "sign_consistency": a["sign_consistency"],
            "within_instance_sd_acc": float(within.mean()),
            "effect_margin_mean": m["mean"], "effect_margin_ci_lo": m["ci_lo"],
            "effect_margin_ci_hi": m["ci_hi"], "margin_p_value": m["p"],
            "per_instance_effect_acc": json.dumps([float(x) for x in d_acc]),
            "baseline_acc_mean": float(np.mean([p["baseline"]["acc"].mean() for p in per_inst])),
        }
        if c_acc is not None:
            c = _across(d_acc - c_acc, seed + si + 2)
            row.update(control_effect_acc_mean=float(c_acc.mean()), vs_control_acc_mean=c["mean"],
                       vs_control_ci_lo=c["ci_lo"], vs_control_ci_hi=c["ci_hi"],
                       vs_control_p=c["p"], n_controls=n_controls)
        rows.append(row)
    for col, out_cols in (("p_value", ("p_holm", "p_bh")),
                          ("vs_control_p", ("vs_control_p_holm", "vs_control_p_bh"))):
        if col in rows[0]:
            hh, bb = adjust_families([r["family"] for r in rows], [r[col] for r in rows])
            for r, h_, b_ in zip(rows, hh, bb, strict=True):
                r[out_cols[0]], r[out_cols[1]] = float(h_), float(b_)
    return rows


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(sps.spearmanr(a, b).statistic)


# ---------------------------------------------------------------------------------------------
# study driver
# ---------------------------------------------------------------------------------------------
def _protocol(cfg: dict) -> dict:
    from flycircuit.config import load_yaml

    p = Path(cfg.get("protocol", "configs/experiments/learning_protocol.yaml"))
    return load_yaml(p if p.is_absolute() else project_root() / p)


def _lr(cfg: dict, learner: str) -> float:
    if learner in cfg.get("lr", {}):
        return float(cfg["lr"][learner])
    from flycircuit.eval.learning import lr_for, stage_dir

    sel = json.loads((stage_dir(cfg.get("lr_stage", "flywire_search")) / "lr_selected.json").read_text())
    return float(lr_for(sel, learner, "connectome"))


def train_stage(cfg: dict, run_dir: Path) -> list[dict]:
    proto = _protocol(cfg)
    log = []
    for variant in cfg["sign_variants"]:
        bundle = build_bundle(cfg["circuit"], cfg["min_contacts"], variant)
        for learner in cfg["learners"]:
            lr = _lr(cfg, learner)
            for seed in cfg["instance_seeds"]:
                p = run_dir / "instances" / variant / f"{learner}_s{seed}.pt"
                if p.exists():
                    ck = torch.load(p, map_location="cpu", weights_only=False)
                    log.append({"path": str(p), "seed": seed, "learner": learner, "variant": variant,
                                "val_acc": ck["train_summary"]["val_acc"],
                                "test_acc": ck["train_summary"]["test_acc"],
                                "seconds": ck["train_summary"]["train_seconds"], "reused": True})
                    continue
                r = train_instance(proto, bundle, learner, lr, seed, variant, p, cfg["circuit"],
                                   cfg["min_contacts"])
                print(f"  trained {variant}/{learner}/s{seed}: val={r['val_acc']:.3f} "
                      f"test={r['test_acc']:.3f} {r['seconds']:.0f}s", flush=True)
                log.append(r)
    return log


def _plan(graph, cfg: dict):
    # Kenyon cells receive the engineered cue projection but are legitimate circuit targets here
    # (v1 excluded sensory sites); readout (MBON) nodes stay excluded: clamping them rewrites the
    # decoder input directly rather than testing circuit function.
    units, erep = eligible_units(graph, cfg.get("level", "neuron"),
                                 min_degree=cfg.get("min_degree", 1),
                                 exclude_sensory=cfg.get("exclude_sensory", False))
    specs, plan = plan_screen(graph, units, cfg["n_interventions"], cfg["min_distinct_targets"],
                              t_start=cfg["t_start"], doses=cfg.get("doses"), seed=cfg["seed"])
    plan["eligibility"] = erep
    plan["exclude_sensory"] = cfg.get("exclude_sensory", False)
    pool = np.array(sorted({n for u in units for n in u.nodes}), dtype=np.int64)
    controls = {s.intervention_id: [matched_random_spec(graph, s, pool, cfg["seed"], k)
                                    for k in range(cfg["n_controls"])] for s in specs}
    return specs, controls, plan


def _sweep_specs(graph, specs: list[InterventionSpec], cfg: dict) -> list[InterventionSpec]:
    """Dose x timing sweep on a seeded random subset of the primary targets (no outcome-based
    selection, so no selection bias)."""
    sw = cfg["sweep"]
    prim = [s for s in specs if s.kind == "clamp_silence"]
    rng = np.random.default_rng(cfg["seed"] + 77)
    pick = [prim[i] for i in rng.permutation(len(prim))[: sw["n_targets"]]]
    out = []
    for s in pick:
        for kind, doses in sw["doses"].items():
            for d in doses:
                for (t0, t1) in sw["windows"]:
                    out.append(InterventionSpec(
                        kind=kind, targets=s.targets, dose=float(d), t_start=int(t0),
                        t_end=None if t1 is None else int(t1), level=s.level, label=s.label,
                        family=f"sweep:{kind}", intervention_id=f"sweep:{kind}:d{d}:w{t0}-{t1}:{s.label}"))
    return out


def screen_stage(cfg: dict, run_dir: Path, out: Path) -> dict:
    torch.set_num_threads(cfg.get("torch_threads", 2))
    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is unavailable")
    out.mkdir(parents=True, exist_ok=True)
    min_acc = cfg["min_baseline_acc"]
    results: dict = {}
    inst_rows = []
    cache: dict = {}
    base_graph = None
    for variant in cfg["sign_variants"]:
        for learner in cfg["learners"]:
            files = sorted((run_dir / "instances" / variant).glob(f"{learner}_s*.pt"))
            insts = [load_instance(f, device) for f in files]
            expected_hash = cfg.get("instance_protocol_hash")
            if expected_hash and any(i.meta["protocol_hash"] != expected_hash for i in insts):
                raise ValueError("instance protocol hash differs from the declared screen source")
            if not insts:
                continue
            graph = _BUNDLES[(cfg["circuit"], cfg["min_contacts"], variant)].graph
            base_graph = base_graph or graph
            assert graph.edge_hash() == base_graph.edge_hash(), "graph differs between variants"
            used = [i for i in insts if i.baseline_val_acc >= min_acc]
            for i in insts:
                inst_rows.append({"variant": variant, "learner": learner, "seed": i.seed,
                                  "baseline_heldout_acc": i.baseline_val_acc,
                                  "included_in_primary": i.baseline_val_acc >= min_acc,
                                  "path": i.path, "checkpoint_val_acc": i.meta["train_summary"]["val_acc"],
                                  "checkpoint_test_acc": i.meta["train_summary"]["test_acc"]})
            if "plan" not in cache:
                cache["plan"] = _plan(graph, cfg)
                cache["sweep"] = _sweep_specs(graph, cache["plan"][0], cfg)
            specs, controls, plan = cache["plan"]
            if not specs:
                results[f"{variant}|{learner}"] = {
                    "status": "no_eligible_targets", "n_used": len(used),
                    "eligibility": plan["eligibility"], "note": plan["note"],
                }
                continue
            if len(used) < 2:
                results[f"{variant}|{learner}"] = {"status": "insufficient_instances",
                                                   "n_used": len(used)}
                continue
            t0 = time.perf_counter()
            per = [instance_effects(i, specs, controls, cfg["n_rep"], cfg["seed"]) for i in used]
            rows = aggregate(specs, per, graph, cfg["seed"], cfg["n_controls"])
            sw_specs = cache["sweep"]
            per_sw = [instance_effects(i, sw_specs, {}, cfg["sweep"]["n_rep"], cfg["seed"] + 1) for i in used]
            sw_rows = aggregate(sw_specs, per_sw, graph, cfg["seed"], 0)
            tag = f"{variant}__{learner}"
            _write(out / f"screen__{tag}.parquet", rows, cfg, plan)
            _write(out / f"sweep__{tag}.parquet", sw_rows, cfg, plan)
            sw_tab, sw_dr = summarize_sweep(sw_rows, cfg["seed"])
            pq.write_table(pa.Table.from_pylist(sw_tab), out / f"sweep_summary__{tag}.parquet")
            results[f"{variant}|{learner}"] = {
                "status": "ok", "n_instances_total": len(insts), "n_used": len(used),
                "excluded_seeds": [i.seed for i in insts if i not in used],
                "primary_family_n": sum(r["family"] == "clamp_silence" for r in rows),
                "primary_holm_sig": sum(r["family"] == "clamp_silence" and r["p_holm"] < 0.05
                                        for r in rows),
                "primary_bh_sig": sum(r["family"] == "clamp_silence" and r["p_bh"] < 0.05 for r in rows),
                "primary_beats_control_holm": sum(
                    r["family"] == "clamp_silence" and r.get("vs_control_p_holm", 1) < 0.05 for r in rows),
                "mean_baseline_acc": float(np.mean([i.baseline_val_acc for i in used])),
                "seconds": time.perf_counter() - t0,
                "uncertainty_split": uncertainty_split(rows),
                "dose_response": sw_dr,
            }
            results[f"{variant}|{learner}"]["_rows"] = rows
            results[f"{variant}|{learner}"]["_per"] = per
            res_k = results[f"{variant}|{learner}"]
            print(f"  screened {tag}: n_used={res_k['n_used']} holm={res_k['primary_holm_sig']} "
                  f"bh={res_k['primary_bh_sig']} base_acc={res_k['mean_baseline_acc']:.3f} "
                  f"{res_k['seconds']:.0f}s", flush=True)
    pq.write_table(pa.Table.from_pylist(inst_rows), out / "instances.parquet")
    robustness = _robustness(results, cfg, out)
    summary = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}
    plan = cache["plan"][2] if "plan" in cache else {}
    (out / "plan.json").write_text(json.dumps(plan, indent=2, default=str))
    (out / "summary.json").write_text(json.dumps(
        {"screens": summary, "robustness": robustness,
         "instance_study": cfg.get("instance_study", "v1 acquisition protocol, annotation_v2 signs"),
         "instance_protocol_hash": cfg.get("instance_protocol_hash"),
         "closed_loop": {"trained_task": "not applicable (cue-valence task has no action->observation "
                                         "feedback)",
                         "structural_model": "see results/interventions_v2/structural (row "
                                             "'v2_default|closed_loop')"}},
        indent=2, default=str))
    try:
        _plots(out, results, cfg)
    except Exception as e:
        (out / "plot_error.txt").write_text(repr(e))
    return {"screens": summary, "robustness": robustness, "plan": {
        k: plan.get(k) for k in ("planned_interventions", "planned_distinct_targets",
                                 "meets_distinct_target_goal", "eligible_distinct_units")}}


def summarize_sweep(rows: list[dict], seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Dose x timing summary of the sweep (seeded random subset of primary targets).

    Per (kind, dose, window): mean |delta accuracy| over targets, computed PER INSTANCE first, then a
    bootstrap CI over instances. Dose-response: Spearman(dose, mean |effect|) within kind x window
    (only 3 doses -> descriptive)."""
    groups: dict = {}
    for r in rows:
        key = (r["kind"], r["dose"], r["t_start"], r["t_end"])
        groups.setdefault(key, []).append(np.asarray(json.loads(r["per_instance_effect_acc"])))
    table = []
    for (kind, dose, t0, t1), vecs in sorted(groups.items()):
        m = np.stack(vecs)  # targets x instances
        absm = np.abs(m).mean(0) * 100
        ci = bootstrap_ci(absm, np.mean, n_boot=2000, seed=seed) if len(absm) >= 2 else {"lo": np.nan, "hi": np.nan}
        table.append({"kind": kind, "dose": dose, "t_start": t0, "t_end": t1, "n_targets": m.shape[0],
                      "n_instances": m.shape[1], "mean_abs_effect_pp": float(absm.mean()),
                      "ci_lo_pp": ci["lo"], "ci_hi_pp": ci["hi"],
                      "mean_signed_effect_pp": float(m.mean() * 100)})
    dr = []
    for kind in sorted({t["kind"] for t in table}):
        for w in sorted({(t["t_start"], t["t_end"]) for t in table if t["kind"] == kind}):
            sub = sorted((t for t in table if t["kind"] == kind and (t["t_start"], t["t_end"]) == w),
                         key=lambda t: t["dose"])
            if len(sub) >= 3:
                dr.append({"kind": kind, "t_start": w[0], "t_end": w[1], "n_doses": len(sub),
                           "spearman_dose_vs_abs_effect": spearman([t["dose"] for t in sub],
                                                                   [t["mean_abs_effect_pp"] for t in sub])})
    return table, dr


def uncertainty_split(rows: list[dict], family: str = "clamp_silence") -> dict:
    """How much of the spread of an instance-level effect is between trained instances vs input
    noise: median over primary targets of between-instance SD and of the noise-repetition SD."""
    prim = [r for r in rows if r["family"] == family]
    b = np.array([r["effect_acc_sd_between"] for r in prim], float)
    w = np.array([r["within_instance_sd_acc"] for r in prim], float)
    return {"n_targets": len(prim), "median_between_instance_sd_pp": float(np.nanmedian(b) * 100),
            "median_noise_repetition_sd_pp": float(np.nanmedian(w) * 100),
            "median_ratio_between_over_noise": float(np.nanmedian(b / np.where(w > 0, w, np.nan)))
            if np.isfinite(w).any() else float("nan"),
            "note": "noise repetitions measure per-instance precision only; they are not replicates"}


def _write(path: Path, rows: list[dict], cfg: dict, plan: dict) -> None:
    source = cfg.get("instance_study", "v1 acquisition protocol, annotation_v2 signs; "
                     "NOT the frozen learning-v2 study")
    prov = {"label": f"model-relative; trained instances ({source}); "
                     "held-out cues val+test, no selection on them",
            "instance_study": source, "instance_protocol_hash": cfg.get("instance_protocol_hash"),
            "config_hash": config_hash(cfg), "primary_metric": PRIMARY_METRIC,
            "uncertainty": "across independently trained instances (noise repetitions only enter "
                           "the within-instance SD)",
            "multiplicity": "Holm/BH within `family`, p-values only", "plan": plan,
            "closed_loop": "not applicable: cue-valence task has no action->observation feedback"}
    t = pa.Table.from_pylist(rows).replace_schema_metadata(
        {b"flycircuit_provenance": json.dumps(prov, default=str, sort_keys=True).encode()})
    pq.write_table(t, path)


def _prim(rows, kind="clamp_silence"):
    d = {r["intervention_id"]: r for r in rows if r["family"] == kind}
    return d


def _robustness(results: dict, cfg: dict, out: Path) -> dict:
    """Do conclusions survive sign policy, learner (fitted recurrent weights), training seed, and
    baseline activity? All descriptive/exploratory (n instances small)."""
    ref_key = f"{cfg['sign_variants'][0]}|{cfg['learners'][0]}"
    rob: dict = {"reference": ref_key}
    ok = {k: v for k, v in results.items() if v.get("status") == "ok"}
    if ref_key not in ok:
        return {**rob, "status": "reference screen missing"}
    ref = _prim(ok[ref_key]["_rows"])
    ids = list(ref)
    refv = [ref[i]["effect_acc_mean"] for i in ids]
    rob["vs_reference"] = {}
    for k, v in ok.items():
        cur = _prim(v["_rows"])
        cv = [cur[i]["effect_acc_mean"] for i in ids]
        sr = {i for i in ids if ref[i]["p_holm"] < 0.05}
        sc = {i for i in ids if cur[i]["p_holm"] < 0.05}
        rob["vs_reference"][k] = {
            "spearman_effect": spearman(refv, cv),
            "holm_set_jaccard": len(sr & sc) / len(sr | sc) if (sr | sc) else None,
            "n_holm_sig": len(sc),
            "mean_abs_effect_pp": float(np.mean(np.abs(cv)) * 100)}
    # training-seed robustness: split-half over instances within the reference screen
    per = ok[ref_key]["_per"]
    if len(per) >= 4:
        half = len(per) // 2
        specs_ids = ids
        a = [np.mean([p["effects"][i]["d_acc"].mean() for p in per[:half]]) for i in specs_ids]
        b = [np.mean([p["effects"][i]["d_acc"].mean() for p in per[half:]]) for i in specs_ids]
        rob["seed_split_half_spearman"] = spearman(a, b)
    rob["effect_vs_target_baseline_activity_spearman"] = spearman(
        [abs(ref[i]["effect_acc_mean"]) for i in ids], [ref[i]["target_baseline_activity"] for i in ids])
    rob["effect_vs_total_degree_spearman"] = spearman(
        [abs(ref[i]["effect_acc_mean"]) for i in ids],
        [ref[i]["mean_in_degree"] + ref[i]["mean_out_degree"] for i in ids])
    rob["note"] = ("signs are policy-specific AND trained into each instance: each sign variant has its "
                   "own independently trained instances; learner 'decoder_only' keeps untrained "
                   "recurrent gains, so contrasting it with 'bptt' separates fitted-parameter effects")
    return rob


def _plots(out: Path, results: dict, cfg: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for key, v in results.items():
        if v.get("status") != "ok":
            continue
        prim = [r for r in v["_rows"] if r["family"] == "clamp_silence"]
        order = np.argsort([r["effect_acc_mean"] for r in prim])
        fig, ax = plt.subplots(figsize=(7, max(4, 0.11 * len(prim))))
        y = np.arange(len(prim))
        e = np.array([prim[i]["effect_acc_mean"] for i in order]) * 100
        lo = e - np.array([prim[i]["effect_acc_ci_lo"] for i in order]) * 100
        hi = np.array([prim[i]["effect_acc_ci_hi"] for i in order]) * 100 - e
        sig = np.array([prim[i]["p_holm"] < 0.05 for i in order])
        ax.errorbar(e, y, xerr=[lo, hi], fmt="none", ecolor="0.7", lw=0.7)
        ax.scatter(e, y, s=8, c=np.where(sig, "tab:red", "tab:blue"))
        ax.axvline(0, color="k", lw=0.5)
        ax.set_yticks([])
        ax.set_xlabel("change in held-out accuracy (pp); 95% bootstrap CI over trained instances")
        ax.set_title(f"clamp-silence screen, trained instances [{key}] model-relative\n"
                     f"n={v['n_used']} instances; red = Holm p<0.05", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"forest__{key.replace('|', '__')}.png", dpi=130)
        plt.close(fig)


def run_trained_study(cfg: dict, stage: str = "all", out_dir: Path | None = None,
                      use_mlflow: bool = False) -> dict:
    from flycircuit.tracking.manifest import RunRecorder

    root = project_root()
    with RunRecorder("screen-v2-trained", cfg, list(cfg["instance_seeds"]),
                     "train instances -> paired intervention screens (trained, model-relative)",
                     "fixed instance set; no adaptive stopping",
                     data_release="flywire:783", use_mlflow=use_mlflow, label="real-trained-v2") as rec:
        run_dir = Path(cfg["instances_run_dir"]) if cfg.get("instances_run_dir") else rec.dir
        if not run_dir.is_absolute():
            run_dir = root / run_dir
        out = out_dir or root / "results" / "interventions_v2" / "trained" / rec.run_id
        res: dict = {"run_id": rec.run_id, "instances_dir": str(run_dir), "out": str(out)}
        if stage in ("train", "all"):
            res["training"] = train_stage(cfg, run_dir)
        if stage in ("screen", "all"):
            res.update(screen_stage(cfg, run_dir, out))
        rec.add_artifact(out)
    return res
