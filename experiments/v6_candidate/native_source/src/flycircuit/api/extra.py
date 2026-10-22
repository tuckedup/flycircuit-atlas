"""Additive API pieces: real-circuit loading/switching, summaries, spatial cloud, results listing,
intervention comparison, replay playback and bounded CSV exports.

Real data is served only from locally ingested FlyWire files. Fields the source does not have
(morphology) are never invented. Soma positions come from the FlyWire v783 annotation table; the
source README (flyconnectome/flywire_annotations, supplemental_files/README.md) states
``soma_x/y/z`` are "in 4x4x40nm voxel space". Neurons without a soma entry report it as unknown.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse

from flycircuit.graph import VersionedGraph

REAL_ATTRS = ("cell_type", "cell_class", "super_class", "side", "flow", "top_nt", "top_nt_conf",
              "known_nt", "hemibrain_type")
DEFAULT_CIRCUIT = "mushroom_body"
RESULT_KINDS = ("interventions", "learning", "navigation", "refcheck", "audit")
SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")
MAX_JSON_BYTES = 2_000_000
MAX_EXPORT = 2000
MAX_CLOUD = 6000
COORD_SPACE = {
    "name": "FlyWire v783 annotation voxel space", "unit": "voxel", "voxel_size_nm": [4, 4, 40],
    "source": "flyconnectome/flywire_annotations supplemental_files/README.md "
              "('soma_x/y/z are soma coordinates in 4x4x40nm voxel space')",
    "display_transform": "micrometres = voxel * [4, 4, 40] / 1000 (display only; stored values are "
                         "voxels). Axes are shown as stored (image-space, y increases downward); "
                         "no anatomical re-orientation is assumed.",
}


def _py(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


def real_available(root: Path, circuit: str = DEFAULT_CIRCUIT, dataset="flywire", version="783"):
    base = root / "data" / "processed" / dataset / version
    return (base / "circuits" / circuit / "selection.json").exists() and (
        base / "edges_pairs.parquet").exists()


def list_circuits(root: Path, dataset="flywire", version="783") -> list[str]:
    cdir = root / "data" / "processed" / dataset / version / "circuits"
    if not cdir.is_dir():
        return []
    return sorted(p.name for p in cdir.iterdir() if (p / "selection.json").exists())


def load_real_graph(root: Path, circuit: str = DEFAULT_CIRCUIT, threshold: int = 5):
    """Return (VersionedGraph, info) for a saved selection, with annotation attributes."""
    from flycircuit.graph.bridge import load_circuit_graph

    base = root / "data" / "processed" / "flywire" / "783"
    cg = load_circuit_graph(circuit, threshold=threshold, processed=base)
    g = cg.graph
    con = duckdb.connect()
    con.execute("SET memory_limit='1GiB'")
    import pyarrow as pa

    con.register("ids", pa.table({"root_id": g.root_ids}))
    cols = ", ".join(REAL_ATTRS)
    rows = con.execute(
        f"SELECT root_id, {cols}, soma_x, soma_y, soma_z FROM '{base / 'neurons.parquet'}' "
        "WHERE root_id IN (SELECT root_id FROM ids)").fetchall()
    by = {int(r[0]): r[1:] for r in rows}
    k = len(REAL_ATTRS)
    attrs: dict[str, list[Any]] = {name: [] for name in REAL_ATTRS}
    attrs["group"] = [_py(x) for x in cg.group]
    attrs["soma_position"] = []
    for rid in g.root_ids.tolist():
        vals = by.get(rid)
        for i, name in enumerate(REAL_ATTRS):
            attrs[name].append(_py(vals[i]) if vals else None)
        xyz = vals[k:k + 3] if vals else None
        ok = xyz is not None and all(v is not None and v == v for v in xyz)
        attrs["soma_position"].append([float(v) for v in xyz] if ok else None)
    # circuit-selection labels take precedence for type/class (same source, bridge-validated)
    attrs["cell_type"] = [a if a is not None else _py(b) for a, b in
                          zip(attrs["cell_type"], cg.cell_type, strict=True)]
    g.node_attrs = {name: v for name, v in attrs.items()}
    n_soma = sum(1 for p in attrs["soma_position"] if p)
    info = {
        "mode": "real", "circuit": circuit, "release": "flywire:783",
        "threshold_contacts": threshold, "is_pilot_subset": bool(cg.selection.get("is_pilot_subset")),
        "mapping_hash": g.mapping_hash(), "edge_hash": g.edge_hash(),
        "boundary": cg.boundary.summary(),
        "n_with_soma_position": n_soma, "coordinate_space": COORD_SPACE,
        "note": ("REAL FlyWire v783 data: induced internal graph of the saved selection, pairs with "
                 f">= {threshold} contacts (a model choice). Boundary partners are cut and not shown. "
                 f"Soma positions known for {n_soma} of {g.n} neurons (voxel space 4x4x40 nm); "
                 "morphology is not served (unknown)."),
    }
    return g, info


def _count_by(values: list[Any], top: int = 40) -> list[dict[str, Any]]:
    c = Counter("unknown" if v is None else str(v) for v in values)
    return [{"value": k, "n": n} for k, n in c.most_common(top)]


def _read_json(p: Path) -> Any:
    if not p.exists() or p.stat().st_size > MAX_JSON_BYTES:
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _result_dirs(results_root: Path, kind: str) -> list[Path]:
    if kind == "audit":
        d = results_root / "audit"
        return sorted((p for p in d.iterdir() if p.is_dir() and p.name.startswith("screen_")),
                      reverse=True) if d.is_dir() else []
    d = results_root / kind
    if not d.is_dir():
        return []
    return sorted((p for p in d.iterdir() if not p.name.startswith(".") and p.name != "latest.json"
                   and p.suffix not in (".png", ".txt", ".npz")), reverse=True)[:100]


def _summarize_result(kind: str, d: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": kind, "id": d.name if d.is_dir() else d.stem}
    if kind in ("interventions", "audit"):
        prov = _read_json(d / "provenance.json") or {}
        cfg = prov.get("config") or {}
        out.update({"synthetic": prov.get("synthetic"), "label": prov.get("label"),
                    "circuit": (prov.get("graph_info") or {}).get("circuit"),
                    "unknown_sign": cfg.get("unknown_sign"),
                    "has_effects": (d / "screen.parquet").exists()})
    elif kind == "navigation":
        res = _read_json(d / "results.json") or {}
        out.update({"label": "navigation benchmark (synthetic arena)",
                    "circuit": (res.get("graph") or {}).get("circuit"),
                    "protocol_hash": res.get("protocol_hash")})
    elif kind == "refcheck":
        res = _read_json(d) or {}
        out.update({"label": "simulator-vs-simulator engineering validation",
                    "keys": sorted(res)[:8]})
    elif kind == "learning":
        out["label"] = "learning protocol stage (synthetic task on real-circuit subset)"
    return out


def _screen_path(results_root: Path, kind: str, rid: str) -> Path:
    if kind not in ("interventions", "audit") or not SAFE.match(rid):
        raise HTTPException(400, "invalid intervention result reference")
    p = results_root / kind / rid / "screen.parquet"
    if not p.exists():
        raise HTTPException(404, "no effects table for this result")
    return p


def _csv_response(header: str, cols: list[str], rows: list[list[Any]]) -> PlainTextResponse:
    buf = io.StringIO()
    buf.write(header.rstrip("\n") + "\n")
    w = csv.writer(buf)
    w.writerow(cols)
    for r in rows:
        w.writerow(["" if v is None else v for v in r])
    return PlainTextResponse(buf.getvalue(), media_type="text/csv")


def register_extra(app: FastAPI, S, root: Path, wrap, partner_edges) -> None:  # noqa: N803
    results_root = root / "results"

    @app.get("/api/mode")
    def mode():
        return wrap({"data": S.info, "switchable": S.switchable,
                     "circuits": list_circuits(root) if S.switchable else []})

    @app.get("/api/circuits")
    def circuits():
        cur = S.info.get("circuit")
        out = []
        for name in list_circuits(root):
            sel = _read_json(root / "data/processed/flywire/783/circuits" / name / "selection.json") or {}
            out.append({"name": name, "current": name == cur,
                        "is_pilot_subset": bool(sel.get("is_pilot_subset")),
                        "n_internal": (sel.get("counts") or {}).get("n_internal")
                        or sel.get("n_internal")})
        return wrap({"circuits": out, "switchable": S.switchable, "current": cur})

    @app.get("/api/circuit/select")
    def select_circuit(name: str):
        if not S.switchable:
            raise HTTPException(409, "circuit switching is available only in real-data mode")
        if not SAFE.match(name) or name not in list_circuits(root):
            raise HTTPException(404, f"unknown circuit {name!r}; see /api/circuits")
        if name not in S.cache:
            if len(S.cache) >= 2:  # keep memory bounded: current + one more
                S.cache.pop(next(k for k in S.cache if k != S.info.get("circuit")))
            S.cache[name] = load_real_graph(root, name)
        S.g, S.info = S.cache[name]
        return wrap({"circuit": name, "graph": S.g.summary()})

    @app.get("/api/summary")
    def summary():
        g = S.g
        a = g.node_attrs
        out = {"graph": g.summary(), "total_contacts": int(g.count.sum()),
               "type_summary": _count_by(a.get("cell_type", []), 60),
               "class_summary": _count_by(a.get("cell_class", [])),
               "super_class_summary": _count_by(a.get("super_class", [])),
               "side_summary": _count_by(a.get("side", [])),
               "group_summary": _count_by(a.get("group", [])),
               "transmitter_summary": _count_by(a.get("top_nt", [])),
               "n_with_soma_position": S.info.get("n_with_soma_position", 0),
               "region_note": ("Neuron-level region = annotation side/super_class. Per-neuropil "
                               "contact breakdown is available per pair via /api/pair.")}
        return wrap(out)

    @app.get("/api/cloud")
    def cloud(limit: int = Query(1500, ge=1, le=MAX_CLOUD)):
        """Soma points with genuine coordinates only. Deterministic stride downsampling."""
        g = S.g
        sp = g.node_attrs.get("soma_position")
        if not sp:
            return wrap({"available": False, "points": [], "total_with_soma": 0,
                         "reason": "this graph carries no soma coordinates"})
        have = [i for i in range(g.n) if sp[i]]
        step = max(1, -(-len(have) // limit))
        pick = have[::step][:limit]
        grp = g.node_attrs.get("group") or [None] * g.n
        ct = g.node_attrs.get("cell_type") or [None] * g.n
        vs = np.array([4.0, 4.0, 40.0]) / 1000.0
        pts = [{"id": g.qualified_id(i), "x": round(sp[i][0] * vs[0], 2),
                "y": round(sp[i][1] * vs[1], 2), "z": round(sp[i][2] * vs[2], 2),
                "group": grp[i], "cell_type": ct[i]} for i in pick]
        return wrap({"available": True, "points": pts, "total_with_soma": len(have),
                     "n_neurons": g.n, "returned": len(pts), "downsample_stride": step,
                     "unit": "micrometre (display transform of 4x4x40 nm voxels)",
                     "coordinate_space": COORD_SPACE})

    @app.get("/api/pair")
    def pair(pre: str, post: str):
        if S.info.get("mode") != "real":
            raise HTTPException(404, "per-neuropil breakdown exists only for real data")
        g = S.g
        try:
            a, b = g.root_ids[g.resolve(pre)], g.root_ids[g.resolve(post)]
        except (ValueError, KeyError) as exc:
            raise HTTPException(404, str(exc)) from exc
        f = root / "data" / "processed" / "flywire" / "783" / "edges_by_neuropil.parquet"
        con = duckdb.connect()
        rows = con.execute(f"SELECT neuropil, count FROM '{f}' WHERE pre = ? AND post = ? "
                           "ORDER BY count DESC LIMIT 100", [int(a), int(b)]).fetchall()
        return wrap({"pre": pre, "post": post,
                     "neuropils": [{"neuropil": r[0], "contacts": int(r[1])} for r in rows]})

    # ---- results & interventions -----------------------------------------------------------
    @app.get("/api/results")
    def results():
        items = []
        for kind in RESULT_KINDS:
            for p in _result_dirs(results_root, kind):
                items.append(_summarize_result(kind, p))
        return wrap({"results": items,
                     "note": "Listing only; results are model-relative, see docs/RESULTS.md."})

    @app.get("/api/results/{kind}/{rid}")
    def result_detail(kind: str, rid: str):
        if kind not in RESULT_KINDS or not SAFE.match(rid):
            raise HTTPException(400, "invalid result reference")
        p = results_root / kind / rid
        if p.is_dir():
            files = sorted(f.name for f in p.iterdir() if f.is_file())
            summ = None
            for cand in ("summary.json", "results.json", "provenance.json", "interventions.json"):
                if cand in files:
                    summ = _read_json(p / cand)
                    break
            return wrap({"kind": kind, "id": rid, "files": files, "summary": summ,
                         "summary_truncated_or_missing": summ is None})
        pj = p if p.suffix == ".json" else p.with_suffix(".json")
        if pj.exists():
            return wrap({"kind": kind, "id": rid, "files": [pj.name], "summary": _read_json(pj),
                         "summary_truncated_or_missing": False})
        raise HTTPException(404, "result not found")

    @app.get("/api/interventions/{kind}/{rid}/effects")
    def effects(kind: str, rid: str, family: str = "", sort: str = Query("abs_effect",
                pattern="^(abs_effect|p_holm|effect)$"), limit: int = Query(120, ge=1, le=500)):
        f = _screen_path(results_root, kind, rid)
        prov = _read_json(results_root / kind / rid / "provenance.json") or {}
        con = duckdb.connect()
        order = {"abs_effect": "abs(effect_mean) DESC", "p_holm": "p_holm ASC NULLS LAST",
                 "effect": "effect_mean ASC"}[sort]
        where, params = ("WHERE family = ?", [family]) if family else ("", [])
        rows = con.execute(
            f"SELECT intervention_id, kind, level, target_label, n_targets, dose, dose_unit, t_start, "
            f"t_end, mode, family, effect_metric, effect_mean, effect_ci_lo, effect_ci_hi, n_runs, "
            f"p_value, p_holm, p_bh, vs_control_mean, vs_control_ci_lo, vs_control_ci_hi, "
            f"vs_control_p_holm FROM '{f}' {where} ORDER BY {order} LIMIT {int(limit)}",
            params).fetchall()
        cols = ["intervention_id", "kind", "level", "target_label", "n_targets", "dose", "dose_unit",
                "t_start", "t_end", "mode", "family", "effect_metric", "effect_mean", "effect_ci_lo",
                "effect_ci_hi", "n_runs", "p_value", "p_holm", "p_bh", "vs_control_mean",
                "vs_control_ci_lo", "vs_control_ci_hi", "vs_control_p_holm"]
        fams = con.execute(f"SELECT family, count(*), sum((p_holm < 0.05)::int), "
                           f"sum((p_bh < 0.05)::int) FROM '{f}' GROUP BY 1").fetchall()
        return wrap({
            "kind": kind, "id": rid, "columns": cols,
            "rows": [[_py(v) for v in r] for r in rows],
            "families": [{"family": a, "n": int(b), "holm_lt_0_05": int(c or 0),
                          "bh_lt_0_05": int(d or 0)} for a, b, c, d in fams],
            "synthetic": prov.get("synthetic"), "label": prov.get("label"),
            "unknown_sign_policy": (prov.get("config") or {}).get("unknown_sign"),
            "graph_info": prov.get("graph_info"), "multiplicity": prov.get("multiplicity"),
            "caveat": ("Model-relative. Intervals reflect independent stimulus/initial-state/noise "
                       "runs of ONE fixed (untrained unless stated) model, not trained-model "
                       "replicates or biological replicates. Holm/BH are applied within a family "
                       "of p-values only."),
        })

    @app.get("/api/interventions/compare")
    def compare(a: str, b: str, limit: int = Query(200, ge=1, le=500)):
        """a, b = '<kind>/<result id>'. Joins two screens on intervention_id (same targets)."""
        def split(x: str):
            k, _, r = x.partition("/")
            return k, r
        pa, pb = _screen_path(results_root, *split(a)), _screen_path(results_root, *split(b))
        con = duckdb.connect()
        sel = ("SELECT a.intervention_id, a.target_label, a.family, a.effect_mean, a.effect_ci_lo, "
               "a.effect_ci_hi, a.p_holm, b.effect_mean, b.effect_ci_lo, b.effect_ci_hi, b.p_holm "
               f"FROM '{pa}' a JOIN '{pb}' b USING (intervention_id)")
        rows = con.execute(f"{sel} ORDER BY abs(a.effect_mean - b.effect_mean) DESC "
                           f"LIMIT {int(limit)}").fetchall()
        agg = con.execute(
            f"SELECT count(*), sum((sign(a.effect_mean) = sign(b.effect_mean))::int), "
            f"sum((a.p_holm < 0.05)::int), sum((b.p_holm < 0.05)::int), "
            f"sum(((a.p_holm < 0.05) AND (b.p_holm < 0.05))::int) "
            f"FROM '{pa}' a JOIN '{pb}' b USING (intervention_id)").fetchone()
        pro = [(_read_json(results_root / s[0] / s[1] / "provenance.json") or {})
               for s in (split(a), split(b))]
        return wrap({
            "a": a, "b": b,
            "a_sign_policy": (pro[0].get("config") or {}).get("unknown_sign"),
            "b_sign_policy": (pro[1].get("config") or {}).get("unknown_sign"),
            "n_shared_interventions": int(agg[0] or 0), "n_same_effect_sign": int(agg[1] or 0),
            "n_holm_sig_a": int(agg[2] or 0), "n_holm_sig_b": int(agg[3] or 0),
            "n_holm_sig_both": int(agg[4] or 0),
            "columns": ["intervention_id", "target_label", "family", "effect_a", "ci_lo_a", "ci_hi_a",
                        "p_holm_a", "effect_b", "ci_lo_b", "ci_hi_b", "p_holm_b"],
            "rows": [[_py(v) for v in r] for r in rows],
            "caveat": "Descriptive comparison of two model-relative screens; not a hypothesis test.",
        })

    # ---- replay ---------------------------------------------------------------------------
    def _replay_doc(run_id: str):
        if not SAFE.match(run_id):
            raise HTTPException(400, "invalid run id")
        d = root / "runs" / run_id
        if not d.is_dir():
            raise HTTPException(404, "run not found")
        p = d / "replay.json"
        if not p.exists() or p.stat().st_size > 4 * MAX_JSON_BYTES:
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    @app.get("/api/replays")
    def replays():
        out = []
        for p in sorted((root / "runs").glob("*/replay.json"), reverse=True)[:50]:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({"run_id": p.parent.name, "label": d.get("label"), "kind": d.get("kind"),
                        "n_steps": d.get("n_steps"), "n_neurons": len(d.get("neurons", [])),
                        "structural_untrained": d.get("structural_untrained"),
                        "synthetic": d.get("synthetic"),
                        "n_trials": len(d.get("trials", [])),
                        "has_gains": bool(d.get("gains"))})
        return wrap({"replays": out})

    @app.get("/api/runs/{run_id}/replay")
    def replay(run_id: str):
        doc = _replay_doc(run_id)
        if doc is None:
            return wrap({"available": False,
                         "reason": "this run saved no activity replay (no replay.json); "
                                   "nothing is simulated or invented"})
        return wrap({"available": True, "mode": "prerecorded replay (not live simulation)",
                     "data": doc})

    # ---- exports (bounded, with provenance header) ----------------------------------------
    def head(what: str, extra: str = "") -> str:
        g = S.g
        return (f"# FlyCircuit Atlas export: {what} (Atul Pandey); release={g.dataset}:{g.version}; "
                f"mode={S.info.get('mode')}; circuit={S.info.get('circuit')}; synthetic={g.synthetic}; "
                f"mapping_hash={g.mapping_hash()}; {extra}")

    @app.get("/api/export/neurons.csv", response_class=PlainTextResponse)
    def export_neurons(q: str = "", limit: int = Query(500, ge=1, le=MAX_EXPORT)):
        g = S.g
        ct = g.node_attrs.get("cell_type") or [None] * g.n
        sp = g.node_attrs.get("soma_position") or [None] * g.n
        sel = [i for i in range(g.n)
               if not q or q in str(int(g.root_ids[i])) or (ct[i] and q.lower() in str(ct[i]).lower())
               ][:limit]
        rows = []
        for i in sel:
            _, ci = g.in_partners(i)
            _, co = g.out_partners(i)
            row = [g.qualified_id(i), str(int(g.root_ids[i]))]
            for k in (*REAL_ATTRS, "group"):
                v = g.node_attrs.get(k)
                row.append(None if v is None else v[i])
            p = sp[i]
            row += [None if not p else p[0], None if not p else p[1], None if not p else p[2],
                    int(ci.sum()), int(co.sum())]
            rows.append(row)
        return _csv_response(
            head("neurons", f"rows<={limit}; query={q!r}; soma in 4x4x40nm voxels (blank=unknown)"),
            ["id", "root_id", *REAL_ATTRS, "group", "soma_x_vox", "soma_y_vox", "soma_z_vox",
             "in_contacts", "out_contacts"], rows)

    @app.get("/api/export/partners.csv", response_class=PlainTextResponse)
    def export_partners(id: str, direction: str = Query("both", pattern="^(in|out|both)$"),
                        hops: int = Query(1, ge=1, le=2), limit: int = Query(500, ge=1, le=MAX_EXPORT)):
        g = S.g
        _, items, trunc = partner_edges(id, direction, hops, limit)
        rows = [[g.qualified_id(a), g.qualified_id(b), int(c)] for (a, b), c in items]
        return _csv_response(head("partners", f"center={id}; direction={direction}; hops={hops}; "
                                              f"truncated={trunc}"),
                             ["pre", "post", "contacts"], rows)

    @app.get("/api/export/interventions.csv", response_class=PlainTextResponse)
    def export_interventions(kind: str, rid: str, limit: int = Query(500, ge=1, le=MAX_EXPORT)):
        f = _screen_path(results_root, kind, rid)
        prov = _read_json(results_root / kind / rid / "provenance.json") or {}
        con = duckdb.connect()
        cur = con.execute(
            f"SELECT * EXCLUDE (target_ids) FROM '{f}' LIMIT {int(limit)}")
        cols = [d[0] for d in cur.description]
        rows = [[_py(v) for v in r] for r in cur.fetchall()]
        return _csv_response(
            head("interventions", f"result={kind}/{rid}; synthetic={prov.get('synthetic')}; "
                                  f"unknown_sign={(prov.get('config') or {}).get('unknown_sign')}; "
                                  "model-relative; Holm/BH within family only"), cols, rows)

    @app.get("/api/export/replay.csv", response_class=PlainTextResponse)
    def export_replay(run: str, limit: int = Query(5000, ge=1, le=20000)):
        doc = _replay_doc(run)
        if doc is None:
            raise HTTPException(404, "this run has no replay.json")
        prov = doc.get("provenance", {})
        ids = doc["neurons"]
        rows = [[t, doc["stimulus"][t], *doc["activity"][t]] for t in range(min(doc["n_steps"], limit))]
        return _csv_response(
            head("replay", f"run={run}; kind={doc.get('kind')}; recorded=true; "
                           f"structural_untrained={doc.get('structural_untrained')}; "
                           f"seed={prov.get('seed')}; sign_policy={prov.get('sign_policy')}"),
            ["step", "stimulus_drive", *ids], rows)


def mode_from_env() -> tuple[str, str]:
    return (os.environ.get("FLYCIRCUIT_API_MODE", "auto").lower(),
            os.environ.get("FLYCIRCUIT_CIRCUIT", DEFAULT_CIRCUIT))


_ = VersionedGraph  # re-exported type for callers
