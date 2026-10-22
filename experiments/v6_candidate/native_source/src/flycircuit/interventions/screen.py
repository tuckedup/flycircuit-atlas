"""Intervention screen: paired runs, matched random controls, effects, corrected tests, artifacts.

All effects are MODEL-RELATIVE (a fixed, structurally-constrained, by default UNTRAINED sparse RNN
unless a checkpoint is supplied). Lesion sensitivity here is not evidence of biological necessity.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from flycircuit.config import config_hash
from flycircuit.eval.stats import bootstrap_ci
from flycircuit.interventions.engine import SimConfig, effect_metrics, run_branch
from flycircuit.interventions.inference import adjust_families, effect_summary
from flycircuit.interventions.rng import RngStreams
from flycircuit.interventions.spec import DOSE_UNITS, InterventionSpec
from flycircuit.interventions.targets import degrees, matched_random_spec


def build_model(graph, sign, model_kwargs: dict | None = None, unknown_sign_policy="excitatory"):
    from flycircuit.models.sparse_rnn import ModelSpec, SparseConnectomeRNN

    mk = {"n_in": int(graph.node_masks["sensory_input"].sum()),
          "n_out": 2, "init_gain": 0.1, "tau": 5.0, "dt": 1.0, "seed": 0}
    mk.update(model_kwargs or {})
    return SparseConnectomeRNN.sparse_from_graph(
        graph, ModelSpec(**mk), sign=sign, unknown_sign_policy=unknown_sign_policy)


def run_screen(
    graph, model, specs: list[InterventionSpec], n_runs: int = 6, seed: int = 0,
    sim: SimConfig | None = None, n_controls: int = 1, progress: bool = False,
    pool: np.ndarray | None = None,
) -> tuple[list[dict], list[dict]]:
    """Returns (per-run rows, per-intervention summary rows) before multiplicity adjustment.

    Per run the baseline is simulated once; every intervention and its matched controls reuse the
    same shared events (stimulus/init/obs_noise) via a fresh RngStreams(seed, run).
    """
    sim = sim or SimConfig()
    ind, outd = degrees(graph)
    if pool is None:
        ok = ~(graph.node_masks["sensory_input"] | graph.node_masks["action_readout"])
        pool = np.flatnonzero(ok & ((ind + outd) > 0))
    controls: dict[str, list[InterventionSpec]] = {
        s.intervention_id: [matched_random_spec(graph, s, pool, seed, k) for k in range(n_controls)]
        for s in specs} if n_controls else {}

    per_run: list[dict] = []
    base_cache: dict[int, dict] = {}
    t_wall = time.perf_counter()
    for r in range(n_runs):
        base_cache[r] = run_branch(model, RngStreams(seed, r), sim, None)
    for si, s in enumerate(specs):
        for r in range(n_runs):
            base = base_cache[r]
            br = run_branch(model, RngStreams(seed, r), sim, s)
            m = effect_metrics(base["y"], br["y"], s.t_start)
            tgt_act = float(base["mean_abs_h"][list(s.targets)].mean())
            row = {"intervention_id": s.intervention_id, "run": r, "seed": seed, "role": "target",
                   "target_baseline_activity": tgt_act, **m}
            if s.kind in ("clamp_silence", "clamp_excite") and "clamped_trace" in br:
                row["clamp_max_abs_dev"] = float((br["clamped_trace"] - float(s.dose)).abs().max())
            per_run.append(row)
            for k, c in enumerate(controls.get(s.intervention_id, [])):
                bc = run_branch(model, RngStreams(seed, r), sim, c)
                mc = effect_metrics(base["y"], bc["y"], c.t_start)
                per_run.append({"intervention_id": s.intervention_id, "run": r, "seed": seed,
                                "role": f"control{k}", "target_baseline_activity": float("nan"),
                                **mc})
        if progress and (si + 1) % 20 == 0:
            print(f"  {si + 1}/{len(specs)} interventions ({time.perf_counter() - t_wall:.1f}s)",
                  flush=True)

    summaries = []
    by_id: dict[str, list[dict]] = {}
    for row in per_run:
        by_id.setdefault(row["intervention_id"], []).append(row)
    for si, s in enumerate(specs):
        rows = by_id[s.intervention_id]
        tr = sorted((x for x in rows if x["role"] == "target"), key=lambda x: x["run"])
        eff = np.array([x["delta_readout"] for x in tr])
        dev = np.array([x["deviation"] for x in tr])
        sm = effect_summary(eff, seed=seed + si)
        ctrl = [sorted((x for x in rows if x["role"] == f"control{k}"), key=lambda x: x["run"])
                for k in range(n_controls)]
        ctrl_eff = (np.mean([[x["delta_readout"] for x in c] for c in ctrl], axis=0)
                    if ctrl else None)
        d = {
            "intervention_id": s.intervention_id, "kind": s.kind, "level": s.level,
            "target_label": s.label, "n_targets": len(s.targets),
            "target_ids": [graph.qualified_id(int(i)) for i in s.targets[:50]],
            "target_ids_truncated": len(s.targets) > 50,
            "dose": float(s.dose), "dose_unit": DOSE_UNITS[s.kind], "t_start": s.t_start,
            "t_end": -1 if s.t_end is None else s.t_end, "mode": "closed_loop" if sim.closed_loop else "open_loop",
            "family": s.family, "mean_in_degree": float(ind[list(s.targets)].mean()),
            "mean_out_degree": float(outd[list(s.targets)].mean()),
            "effect_metric": "delta_readout = mean(y_int)-mean(y_base) over [t_start,T)",
            "effect_mean": sm["mean"], "effect_sd": sm["sd"], "effect_ci_lo": sm["ci_lo"],
            "effect_ci_hi": sm["ci_hi"], "n_runs": sm["n_runs"], "p_value": sm["p_value"],
            "p_note": sm["p_note"], "deviation_mean": float(dev.mean()),
            "baseline_readout_mean": float(np.mean([x["baseline_readout_mean"] for x in tr])),
            "baseline_readout_norm": float(np.mean([x["baseline_readout_norm"] for x in tr])),
            "target_baseline_activity": float(np.mean([x["target_baseline_activity"] for x in tr])),
            "synthetic": bool(graph.synthetic),
            "dataset": graph.dataset, "version": graph.version,
        }
        if ctrl_eff is not None:
            diff = eff - ctrl_eff
            dd = effect_summary(diff, seed=seed + 10_000 + si)
            d.update(control_effect_mean=float(ctrl_eff.mean()), vs_control_mean=dd["mean"],
                     vs_control_ci_lo=dd["ci_lo"], vs_control_ci_hi=dd["ci_hi"],
                     vs_control_p=dd["p_value"], n_controls=n_controls)
        summaries.append(d)
    h, b = adjust_families([d["family"] for d in summaries], [d["p_value"] for d in summaries])
    for d, hh, bb in zip(summaries, h, b, strict=True):
        d["p_holm"], d["p_bh"] = float(hh), float(bb)
    if n_controls:
        hc, bc = adjust_families([d["family"] for d in summaries], [d["vs_control_p"] for d in summaries])
        for d, hh, bb in zip(summaries, hc, bc, strict=True):
            d["vs_control_p_holm"], d["vs_control_p_bh"] = float(hh), float(bb)
    return per_run, summaries


def write_artifacts(
    out_dir: Path, per_run: list[dict], summaries: list[dict], provenance: dict, plan: dict,
    specs: list[InterventionSpec],
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {b"flycircuit_provenance": json.dumps(provenance, default=str, sort_keys=True).encode()}
    paths = {}
    t = pa.Table.from_pylist(summaries).replace_schema_metadata(meta)
    pq.write_table(t, out_dir / "screen.parquet")
    paths["screen"] = str(out_dir / "screen.parquet")
    t2 = pa.Table.from_pylist(per_run).replace_schema_metadata(meta)
    pq.write_table(t2, out_dir / "runs.parquet")
    paths["runs"] = str(out_dir / "runs.parquet")
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2, default=str))
    (out_dir / "interventions.json").write_text(
        json.dumps([s.metadata() for s in specs], indent=1, default=str))
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    paths["plan"] = str(out_dir / "plan.json")
    try:
        paths.update(make_plots(out_dir, summaries, provenance))
    except Exception as e:  # plots must not mask the scientific artifacts
        (out_dir / "plot_error.txt").write_text(repr(e))
    return paths


def make_plots(out_dir: Path, summaries: list[dict], provenance: dict) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = "SYNTHETIC GRAPH - not real neurons" if provenance.get("synthetic") else "model-relative"
    primary = [d for d in summaries if d["family"] == provenance.get("primary_family")]
    paths = {}
    if primary:
        order = np.argsort([d["effect_mean"] for d in primary])
        fig, ax = plt.subplots(figsize=(7, max(4, 0.12 * len(primary))))
        y = np.arange(len(primary))
        eff = np.array([primary[i]["effect_mean"] for i in order])
        lo = eff - np.array([primary[i]["effect_ci_lo"] for i in order])
        hi = np.array([primary[i]["effect_ci_hi"] for i in order]) - eff
        sig = np.array([primary[i]["p_holm"] < 0.05 for i in order])
        ax.errorbar(eff, y, xerr=[lo, hi], fmt="none", ecolor="0.6", lw=0.8)
        ax.scatter(eff, y, c=np.where(sig, "tab:red", "tab:blue"), s=10)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_yticks([])
        ax.set_xlabel("delta readout (a.u.), 95% bootstrap CI over runs")
        ax.set_title(f"{provenance.get('primary_family')} screen [{tag}]\nred = Holm p<0.05 (family)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "effects_forest.png", dpi=130)
        plt.close(fig)
        paths["forest"] = str(out_dir / "effects_forest.png")
    if summaries and "control_effect_mean" in summaries[0]:
        fig, ax = plt.subplots(figsize=(5, 5))
        x = [d["control_effect_mean"] for d in summaries]
        y = [d["effect_mean"] for d in summaries]
        ax.scatter(x, y, s=10)
        lim = max(1e-9, np.nanmax(np.abs(x + y)))
        ax.plot([-lim, lim], [-lim, lim], "k:", lw=0.7)
        ax.set_xlabel("matched random control effect"), ax.set_ylabel("target effect")
        ax.set_title(f"target vs degree-matched random control [{tag}]", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "target_vs_control.png", dpi=130)
        plt.close(fig)
        paths["control"] = str(out_dir / "target_vs_control.png")
    return paths


def screen_provenance(graph, info: dict, cfg: dict, sim: SimConfig, model, plan: dict, seed: int,
                      unknown_policy: str, primary_family: str) -> dict:
    return {
        "synthetic": bool(graph.synthetic),
        "label": ("SYNTHETIC screen - not a real-neuron screen" if graph.synthetic
                  else "model-relative causal screen on real circuit selection; UNTRAINED structural model unless stated"),
        "dataset": graph.dataset, "version": graph.version,
        "mapping_hash": graph.mapping_hash(), "edge_hash": graph.edge_hash(),
        "graph_info": info, "config": cfg, "config_hash": config_hash(cfg),
        "sim_config": asdict(sim), "seed": seed, "unknown_sign_policy": unknown_policy,
        "model_spec": {k: v for k, v in vars(model.spec).items() if k != "extra"},
        "plan": plan, "primary_family": primary_family,
        "uncertainty": "across independent stimulus/init/noise runs of ONE fixed model; not across trained models",
        "multiplicity": "Holm and BH applied within `family` (p-values only); CIs/effects uncorrected",
    }


__all__ = ["run_screen", "write_artifacts", "build_model", "screen_provenance", "bootstrap_ci"]
