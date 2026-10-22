"""Screen v2 (structural, UNTRAINED model): sign-policy sensitivity on the real pilot circuit.

One graph, one planned set of interventions, one model seed; ONLY the presynaptic sign vector (and
the loop mode, for one extra row) changes between variants. So differences between rows are
attributable to the sign assumption alone. The v1 screens measured "sign sensitivity" while ~98% of
signs were unknown (docs/SIGN_AUDIT.md); the legacy rows below reproduce that regime for contrast, so
coverage and intrinsic sign dependence can be told apart.

All effects are model-relative; nothing here is evidence about living neurons.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats as sps

from flycircuit.config import config_hash, project_root
from flycircuit.graph.signs import SignAssignment, load_assignment_for_ids, policy_grid
from flycircuit.interventions.engine import SimConfig
from flycircuit.interventions.real import NT_SIGN, load_circuit_graph
from flycircuit.interventions.screen import build_model, run_screen, screen_provenance
from flycircuit.interventions.targets import eligible_units, plan_screen

REFERENCE = "v2_default"


def legacy_signs(g, unknown: str) -> tuple[np.ndarray, str]:
    nt = [str(x).lower() for x in g.node_attrs.get("top_nt", [""] * g.n)]
    return np.array([NT_SIGN.get(x, 0) for x in nt], dtype=np.float32), unknown


def variant_table(g, grid_names: list[str] | None = None) -> dict[str, dict]:
    """name -> {sign, unknown_policy (for the model), modulatory, meta}."""
    grid = policy_grid()
    names = grid_names or list(grid)
    out: dict[str, dict] = {}
    for name in names:
        asg: SignAssignment = load_assignment_for_ids(g.root_ids, grid[name])
        out[name] = {"sign": asg.sign.astype(np.float32), "unknown": "zero",
                     "modulatory": int(asg.modulatory.sum()), "tag": grid[name].tag(),
                     "evidence": asg.summary["evidence_counts"], "assignment_hash": asg.hash()}
    for fill in ("excitatory", "zero"):
        s, u = legacy_signs(g, fill)
        out[f"legacy_top_nt+unknown_{fill}"] = {
            "sign": s, "unknown": u, "modulatory": 0, "tag": f"v1-legacy top_nt only, unknown={fill}",
            "evidence": {"known": 0, "predicted": int((s != 0).sum()), "prior": 0,
                         "unknown": int((s == 0).sum())}, "assignment_hash": "legacy"}
    return out


def _agreement(ref: list[dict], cur: list[dict]) -> dict:
    r = {d["intervention_id"]: d for d in ref}
    ids = [d["intervention_id"] for d in cur if d["intervention_id"] in r]
    a = np.array([r[i]["effect_mean"] for i in ids])
    b = np.array([next(d for d in cur if d["intervention_id"] == i)["effect_mean"] for i in ids])
    out = {"n_compared": len(ids)}
    if len(ids) >= 3 and np.ptp(a) > 0 and np.ptp(b) > 0:
        out["spearman"] = float(sps.spearmanr(a, b).statistic)
        out["pearson"] = float(np.corrcoef(a, b)[0, 1])
    else:
        out["spearman"] = out["pearson"] = float("nan")
    tol = 1e-6
    both = (np.abs(a) > tol) & (np.abs(b) > tol)
    out["sign_agreement_among_nonzero"] = (float((np.sign(a[both]) == np.sign(b[both])).mean())
                                           if both.any() else float("nan"))
    top = lambda x, k=10: set(np.argsort(-np.abs(x))[:k].tolist())  # noqa: E731
    out["top10_abs_overlap"] = len(top(a) & top(b)) / 10 if len(ids) >= 10 else float("nan")
    sr = {i for i in ids if r[i]["p_holm"] < 0.05}
    sc = {d["intervention_id"] for d in cur if d["intervention_id"] in r and d["p_holm"] < 0.05}
    u = sr | sc
    out["holm_set_jaccard"] = len(sr & sc) / len(u) if u else float("nan")
    return out


def run_sign_sensitivity(cfg: dict, out_dir: Path | None = None, use_mlflow: bool = False,
                         variants: list[str] | None = None) -> dict:
    from flycircuit.tracking.manifest import RunRecorder

    g, info, _ = load_circuit_graph(
        cfg["circuit"], min_contacts=cfg.get("min_contacts", 5), n_sensory=cfg.get("n_sensory", 64),
        seed=cfg.get("graph_seed", 0), sign_policy="annotation_v2")
    table = variant_table(g, variants)
    sim = SimConfig(steps=cfg["steps"], closed_loop=False)
    units, erep = eligible_units(g, cfg["level"], min_degree=cfg.get("min_degree", 1))
    specs, plan = plan_screen(g, units, cfg["n_interventions"], cfg["min_distinct_targets"],
                              t_start=cfg["t_start"], doses=cfg.get("doses"), seed=cfg["seed"])
    plan["eligibility"] = erep
    primary = plan["primary_kind"]
    rows: dict[str, list[dict]] = {}
    meta: dict[str, dict] = {}
    with RunRecorder("screen-v2-sign-sensitivity", cfg, [cfg["seed"]],
                     "paired screen x sign-policy variants (untrained structural model)",
                     f"{len(table)}+1 variants x {len(specs)} interventions x {cfg['n_runs']} runs",
                     data_release=f"{g.dataset}:{g.version}", mapping_hash=g.mapping_hash(),
                     use_mlflow=use_mlflow, label="real-structural-v2") as rec:
        for name, v in table.items():
            model = build_model(g, v["sign"], cfg.get("model"), v["unknown"])
            _, summ = run_screen(g, model, specs, cfg["n_runs"], cfg["seed"], sim, cfg["n_controls"])
            rows[name] = summ
            meta[name] = {k: v[k] for k in ("tag", "unknown", "modulatory", "evidence",
                                             "assignment_hash")}
            meta[name]["n_nonzero_sign"] = int((v["sign"] != 0).sum())
            print(f"  variant {name}: done", flush=True)
        # open- vs closed-loop contrast for the reference policy (engineered reafference loop)
        ref = table[REFERENCE]
        model = build_model(g, ref["sign"], cfg.get("model"), ref["unknown"])
        _, summ = run_screen(g, model, specs, cfg["n_runs"], cfg["seed"],
                             SimConfig(steps=cfg["steps"], closed_loop=True), cfg["n_controls"])
        rows[f"{REFERENCE}|closed_loop"] = summ
        meta[f"{REFERENCE}|closed_loop"] = {**meta[REFERENCE], "tag": meta[REFERENCE]["tag"] + " | closed_loop"}

        base = rows[REFERENCE]
        vs_rows = []
        for name, summ in rows.items():
            prim = [d for d in summ if d["family"] == primary]
            refp = [d for d in base if d["family"] == primary]
            ag = _agreement(refp, prim)
            vs_rows.append({
                "variant": name, **meta[name], "evidence": json.dumps(meta[name]["evidence"]),
                "primary_family": primary, "n_primary": len(prim),
                "n_holm_sig": int(sum(d["p_holm"] < 0.05 for d in prim)),
                "n_bh_sig": int(sum(d["p_bh"] < 0.05 for d in prim)),
                "n_beats_control_holm": int(sum(d.get("vs_control_p_holm", 1.0) < 0.05 for d in prim)),
                "mean_abs_effect": float(np.mean([abs(d["effect_mean"]) for d in prim])),
                "mean_baseline_readout_norm": float(np.mean([d["baseline_readout_norm"] for d in prim])),
                **{f"vs_{REFERENCE}_{k}": v for k, v in ag.items()},
            })
        pair = []
        names = [n for n in rows if "closed_loop" not in n]
        for a in names:
            for b in names:
                pa_ = [d for d in rows[a] if d["family"] == primary]
                pb_ = [d for d in rows[b] if d["family"] == primary]
                pair.append({"a": a, "b": b, "spearman": _agreement(pa_, pb_)["spearman"]})
        out = out_dir or project_root() / "results" / "interventions_v2" / "structural" / rec.run_id
        out.mkdir(parents=True, exist_ok=True)
        prov = screen_provenance(g, info, cfg, sim, model, plan, cfg["seed"], "sign-policy grid",
                                 primary)
        prov.update(
            label="model-relative; UNTRAINED structural model; sign-policy sensitivity (screen v2)",
            supersedes_note="v1 screens (runs/*screen-mushroom_body_right_pilot*) stay as historical; "
                            "they used legacy top_nt signs (~1.8% usable) so their sign sensitivity "
                            "mostly reflects coverage",
            sign_policy_version=meta[REFERENCE]["tag"], config_hash=config_hash(cfg))
        md = {b"flycircuit_provenance": json.dumps(prov, default=str, sort_keys=True).encode()}
        pq.write_table(pa.Table.from_pylist(vs_rows).replace_schema_metadata(md),
                       out / "variants.parquet")
        flat = [{"variant": n, **{k: d[k] for k in (
            "intervention_id", "kind", "family", "target_label", "effect_mean", "effect_ci_lo",
            "effect_ci_hi", "p_value", "p_holm", "p_bh", "baseline_readout_norm",
            "target_baseline_activity", "mean_in_degree", "mean_out_degree")}}
            for n, summ in rows.items() for d in summ]
        pq.write_table(pa.Table.from_pylist(flat).replace_schema_metadata(md),
                       out / "per_intervention.parquet")
        pq.write_table(pa.Table.from_pylist(pair), out / "pairwise_rank_corr.parquet")
        (out / "plan.json").write_text(json.dumps(plan, indent=2, default=str))
        (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
        try:
            _plots(out, vs_rows, pair, names, primary)
        except Exception as e:
            (out / "plot_error.txt").write_text(repr(e))
        rec.add_artifact(out)
        rec.metrics.update(n_variants=len(rows), n_interventions=len(specs),
                           distinct_targets=plan["planned_distinct_targets"])
    return {"out": str(out), "run_id": rec.run_id, "variants": vs_rows,
            "distinct_targets": plan["planned_distinct_targets"]}


def _plots(out: Path, vs_rows: list[dict], pair: list[dict], names: list[str], primary: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(vs_rows) + 1.5))
    y = np.arange(len(vs_rows))
    ax.barh(y, [r["n_holm_sig"] for r in vs_rows], color="tab:red", label="Holm p<0.05")
    ax.barh(y, [r["n_bh_sig"] - r["n_holm_sig"] for r in vs_rows],
            left=[r["n_holm_sig"] for r in vs_rows], color="tab:orange", label="BH only")
    ax.set_yticks(y)
    ax.set_yticklabels([r["variant"] for r in vs_rows], fontsize=7)
    ax.set_xlabel(f"# significant of {vs_rows[0]['n_primary']} {primary} targets (family-wise)")
    ax.set_title("Screen v2 (untrained structural model, model-relative): significant targets by "
                 "sign policy", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "significant_by_policy.png", dpi=130)
    plt.close(fig)
    m = np.full((len(names), len(names)), np.nan)
    ix = {n: i for i, n in enumerate(names)}
    for p in pair:
        m[ix[p["a"]], ix[p["b"]]] = p["spearman"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(m, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    fig.colorbar(im, label="Spearman of target effects")
    ax.set_title("Agreement of per-target effects across sign policies (model-relative)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "rank_agreement_heatmap.png", dpi=130)
    plt.close(fig)
