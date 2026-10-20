"""``flycircuit interventions eligibility|screen``."""

from __future__ import annotations

import json
from pathlib import Path


def _load(args, cfg):
    from flycircuit.interventions.real import load_circuit_graph, synthetic_screen_graph

    if cfg["source"] == "synthetic":
        g = synthetic_screen_graph(cfg.get("n_nodes", 400), cfg.get("degree", 12), cfg.get("graph_seed", 0))
        import numpy as np

        sign = np.where(np.random.default_rng(1).random(g.n) < 0.2, -1.0, 1.0).astype("float32")
        return g, {"source": "synthetic"}, sign
    return load_circuit_graph(cfg["circuit"], min_contacts=cfg.get("min_contacts", 5),
                              n_sensory=cfg.get("n_sensory", 64), seed=cfg.get("graph_seed", 0),
                              sign_policy=resolve_sign_policy(cfg))


def resolve_sign_policy(cfg: dict):
    """Config -> ``"legacy_top_nt"`` | ``"annotation_v2"`` | a named ``SignPolicy`` variant."""
    from flycircuit.graph.signs import policy_grid

    name = cfg.get("sign_policy", "annotation_v2")
    if name in ("legacy_top_nt", "annotation_v2") and not cfg.get("sign_variant"):
        return name
    grid = policy_grid()
    v = cfg.get("sign_variant", "v2_default")
    if v not in grid:
        raise SystemExit(f"unknown sign_variant {v!r}; choose {sorted(grid)}")
    return grid[v]


def model_unknown_policy(cfg: dict) -> str:
    """v2 policies already fill unknowns inside the sign layer, so remaining zeros stay silent."""
    if cfg["source"] == "circuit" and cfg.get("sign_policy", "annotation_v2") != "legacy_top_nt":
        return "zero"
    return cfg["unknown_sign"]


def _cfg(args) -> dict:
    from flycircuit.config import load_yaml, project_root

    cfg = {"source": "synthetic", "seed": 0, "n_runs": 6, "n_interventions": 120,
           "min_distinct_targets": 100, "steps": 60, "level": "neuron", "closed_loop": False,
           "n_controls": 1, "unknown_sign": "excitatory", "t_start": 10, "min_degree": 1}
    if args.config:
        p = Path(args.config)
        cfg.update(load_yaml(p if p.is_absolute() else project_root() / p))
    for k in ("source", "circuit", "seed", "n_runs", "n_interventions", "steps", "level", "n_nodes"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    if args.closed_loop:
        cfg["closed_loop"] = True
    if cfg["source"] == "circuit" and "circuit" not in cfg:
        raise SystemExit("--circuit NAME required with --source circuit")
    return cfg


def _cmd_elig(args) -> int:
    from flycircuit.interventions.targets import dedupe_units, eligible_units

    cfg = _cfg(args)
    g, info, _ = _load(args, cfg)
    units, rep = eligible_units(g, cfg["level"], min_degree=cfg["min_degree"])
    units, dup = dedupe_units(units)
    rep.update(duplicate_units_dropped=dup, distinct_units=len(units), graph=info,
               synthetic=bool(g.synthetic))
    print(json.dumps(rep, indent=2, default=str))
    return 0 if units else 2


def _cmd_screen(args) -> int:
    from flycircuit.config import project_root
    from flycircuit.interventions.engine import SimConfig
    from flycircuit.interventions.screen import (
        build_model,
        run_screen,
        screen_provenance,
        write_artifacts,
    )
    from flycircuit.interventions.targets import eligible_units, plan_screen
    from flycircuit.tracking.manifest import RunRecorder

    cfg = _cfg(args)
    g, info, sign = _load(args, cfg)
    sim = SimConfig(steps=cfg["steps"], closed_loop=cfg["closed_loop"])
    cfg["unknown_sign"] = model_unknown_policy(cfg)
    model = build_model(g, sign, cfg.get("model"), cfg["unknown_sign"])
    units, erep = eligible_units(g, cfg["level"], min_degree=cfg["min_degree"])
    if not units:
        print(json.dumps({"error": "no eligible targets", "report": erep}, indent=2))
        return 2
    specs, plan = plan_screen(
        g, units, cfg["n_interventions"], cfg["min_distinct_targets"], t_start=cfg["t_start"],
        doses=cfg.get("doses"), seed=cfg["seed"])
    plan["eligibility"] = erep
    primary = plan["primary_kind"]
    prov = screen_provenance(g, info, cfg, sim, model, plan, cfg["seed"], cfg["unknown_sign"], primary)
    name = "screen-" + ("synthetic" if g.synthetic else cfg.get("circuit", "real"))
    with RunRecorder(name, cfg, [cfg["seed"]], "paired intervention screen",
                     f"fixed {cfg['n_runs']} runs x {len(specs)} interventions",
                     data_release=f"{g.dataset}:{g.version}", mapping_hash=g.mapping_hash(),
                     use_mlflow=not args.no_mlflow, label="synthetic" if g.synthetic else "real") as rec:
        per_run, summ = run_screen(g, model, specs, cfg["n_runs"], cfg["seed"], sim,
                                   cfg["n_controls"], progress=True)
        out = Path(args.out) if args.out else project_root() / "results" / "interventions" / rec.run_id
        paths = write_artifacts(out, per_run, summ, prov, plan, specs)
        for p in paths.values():
            rec.add_artifact(p)
        prim = [d for d in summ if d["family"] == primary]
        rec.metrics.update(n_interventions=len(summ), distinct_targets=plan["planned_distinct_targets"],
                           primary_sig_holm=sum(d["p_holm"] < 0.05 for d in prim))
    print(json.dumps({"out": str(out), "plan": {k: v for k, v in plan.items() if k != "eligibility"},
                      "primary_family_n": len(prim), "primary_holm_sig": rec.metrics["primary_sig_holm"],
                      "synthetic": bool(g.synthetic), "run_id": rec.run_id}, indent=2, default=str))
    return 0


def _cmd_sign_sens(args) -> int:
    from flycircuit.interventions.sensitivity import run_sign_sensitivity

    cfg = _cfg(args)
    res = run_sign_sensitivity(cfg, Path(args.out) if args.out else None,
                               use_mlflow=not args.no_mlflow)
    print(json.dumps({k: res[k] for k in ("out", "run_id", "distinct_targets")}, indent=2))
    for r in res["variants"]:
        print(f"{r['variant']:<42} signed={r['n_nonzero_sign']:>5} holm={r['n_holm_sig']:>3} "
              f"bh={r['n_bh_sig']:>3} rho_vs_default={r.get('vs_v2_default_spearman', float('nan')):.3f}")
    return 0


def _cmd_trained(args) -> int:
    from flycircuit.config import load_yaml, project_root
    from flycircuit.interventions.trained import run_trained_study

    p = Path(args.config)
    cfg = load_yaml(p if p.is_absolute() else project_root() / p)
    res = run_trained_study(cfg, stage=args.stage, out_dir=Path(args.out) if args.out else None,
                            use_mlflow=not args.no_mlflow)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res.get("status", "ok") in ("ok", "partial") else 2


def register(sub) -> None:
    p = sub.add_parser("interventions", help="causal intervention workbench (model-relative)")
    s = p.add_subparsers(dest="icmd", required=True)
    q = s.add_parser("sign-sensitivity", help="screen v2: structural model x sign-policy grid")
    q.add_argument("--config", required=True)
    q.add_argument("--out")
    q.add_argument("--no-mlflow", action="store_true")
    q.set_defaults(func=_cmd_sign_sens)
    q.set_defaults(closed_loop=False)
    t = s.add_parser("trained", help="trained-model screens (stage: train|screen|all)")
    t.add_argument("--config", required=True)
    t.add_argument("--stage", choices=["train", "screen", "all"], default="all")
    t.add_argument("--out")
    t.add_argument("--no-mlflow", action="store_true")
    t.set_defaults(func=_cmd_trained)
    for name, fn in (("eligibility", _cmd_elig), ("screen", _cmd_screen)):
        q = s.add_parser(name)
        q.add_argument("--config")
        q.add_argument("--source", choices=["synthetic", "circuit"])
        q.add_argument("--circuit")
        q.add_argument("--level", choices=["neuron", "type", "class", "region"])
        q.add_argument("--seed", type=int)
        q.add_argument("--n-runs", dest="n_runs", type=int)
        q.add_argument("--n-interventions", dest="n_interventions", type=int)
        q.add_argument("--n-nodes", dest="n_nodes", type=int)
        q.add_argument("--steps", type=int)
        q.add_argument("--closed-loop", dest="closed_loop", action="store_true")
        q.add_argument("--out")
        q.add_argument("--no-mlflow", action="store_true")
        q.set_defaults(func=fn)
