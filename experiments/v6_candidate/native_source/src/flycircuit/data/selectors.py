"""Annotation-aware circuit selectors with boundary accounting.

Labels were verified against flywire_annotations (commit a83b277, Supplemental_file1) on
2026-10-15: cell_class in {Kenyon_Cell, MBON, DAN, MBIN, CX, gustatory, ...} and cell_type
names such as KC*, MBON*, PAM*, PPL*, APL, DPM, T4*/T5*/Mi1/Tm*. A selector whose required group
matches no neuron raises SelectionError listing near-miss labels - never an empty circuit.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np

from .paths import processed_dir


class SelectionError(ValueError):
    pass


@dataclass(frozen=True)
class Group:
    name: str
    field: str  # annotation column
    op: str  # eq | in | prefix | regex
    values: tuple[str, ...]
    also: tuple[tuple[str, str, tuple[str, ...]], ...] = ()  # extra AND conditions (field, op, vals)
    required: bool = True


def _g(name, fld, op, *vals, also=(), required=True):
    return Group(name, fld, op, tuple(vals), tuple(also), required)


SELECTORS: dict[str, dict] = {
    "mushroom_body": {
        "description": "Mushroom-body learning circuitry: KCs, MBONs, DANs (PAM/PPL), MBIN, APL, DPM.",
        "groups": [
            _g("kenyon", "cell_class", "eq", "Kenyon_Cell"),
            _g("mbon", "cell_class", "eq", "MBON"),
            _g("dan", "cell_class", "eq", "DAN", also=[("cell_type", "regex", ("^(PAM|PPL)",))]),
            _g("apl", "cell_type", "eq", "APL"),
            _g("dpm", "cell_type", "eq", "DPM"),
            _g("mbin_other", "cell_class", "eq", "MBIN", required=False),
        ],
        "notes": "Upstream partners (e.g. PNs) are boundary partners, not auto-included.",
    },
    "central_complex": {
        "description": "Central complex (cell_class == CX).",
        "groups": [_g("cx", "cell_class", "eq", "CX")],
        "notes": "cell_class CX as labelled in the 783 annotations.",
    },
    "visual_motion": {
        "description": "Optic-lobe motion-pathway types (T4/T5/Mi/Tm/T2/T3/LPi/LPLC).",
        "groups": [_g("motion", "cell_type", "regex",
                      r"^(T4|T5|Mi1$|Mi4$|Mi9$|Tm1$|Tm2$|Tm3$|Tm4$|Tm9$|T2|T3$|LPi|LPLC)")],
        "notes": "A heuristic type list, not the full flyvis network; document before use.",
    },
    "feeding_reference": {
        "description": "Gustatory receptor neurons + brain motor neurons (feeding assay scaffold).",
        "groups": [_g("gustatory", "cell_class", "eq", "gustatory"),
                   _g("brain_motor", "cell_class", "eq", "brain_motor_neuron")],
        "notes": ("APPROXIMATE: the Shiu et al. assay uses specific sugar-GRN / MN9 root IDs from "
                  "its repo; labels alone do not reproduce it. Verify before claiming parity."),
    },
    "whole_brain": {
        "description": "All proofread neurons. Inference/profiling only; no gradient training.",
        "groups": [_g("all", "annotated", "eq", "true", required=False)],
        "notes": "whole_brain_gradient_training=false on the laptop profile.",
    },
}


def _cond(fld: str, op: str, vals: tuple[str, ...]) -> str:
    def q(s):
        return "'" + s.replace("'", "''") + "'"
    if fld == "annotated":
        return "TRUE"
    if op == "eq":
        return f"{fld} = {q(vals[0])}"
    if op == "in":
        return f"{fld} IN ({', '.join(map(q, vals))})"
    if op == "prefix":
        return f"starts_with({fld}, {q(vals[0])})"
    if op == "regex":
        return f"regexp_matches({fld}, {q(vals[0])})"
    raise ValueError(op)


def group_sql(g: Group) -> str:
    parts = [_cond(g.field, g.op, g.values)] + [_cond(*a) for a in g.also]
    return " AND ".join(parts)


def _hash_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.sort(ids.astype("<i8")).tobytes()).hexdigest()


def select_circuit(name: str, *, dataset: str = "flywire", version: str = "783",
                   side: str | None = None, pilot_max: int | None = None, seed: int = 0,
                   threshold: int = 5, out_name: str | None = None,
                   processed: Path | None = None) -> dict:
    if name not in SELECTORS:
        raise SelectionError(f"unknown circuit {name!r}; known: {sorted(SELECTORS)}")
    proc = processed or processed_dir(dataset, version)
    npq, epq = proc / "neurons.parquet", proc / "edges_pairs.parquet"
    for p in (npq, epq):
        if not p.exists():
            raise SelectionError(f"missing {p}; run `flycircuit data ingest` first")
    spec = SELECTORS[name]
    con = duckdb.connect()
    con.execute("SET memory_limit='1GiB'")
    cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM '{npq}'").fetchall()}
    sidecond = f" AND side = '{side}'" if side else ""
    members = []
    counts: dict[str, int] = {}
    for g in spec["groups"]:
        missing = {g.field} | {a[0] for a in g.also}
        missing = [c for c in missing if c not in cols and c != "annotated"]
        if missing:
            raise SelectionError(f"annotation columns missing for group {g.name}: {missing}")
        n = con.execute(
            f"SELECT count(*) FROM '{npq}' WHERE {group_sql(g)}{sidecond}").fetchone()[0]
        counts[g.name] = n
        if n == 0 and g.required:
            near = []
            if g.field in cols and g.field != "annotated":
                needle = re.sub(r"[^A-Za-z]", "", g.values[0])[:3].lower()
                near = [r[0] for r in con.execute(
                    f"SELECT DISTINCT {g.field} FROM '{npq}' WHERE {g.field} IS NOT NULL "
                    f"AND lower({g.field}) LIKE '%{needle}%' LIMIT 15").fetchall()]
            raise SelectionError(
                f"selector {name!r}: required group {g.name!r} ({g.field} {g.op} {g.values}) "
                f"matched 0 neurons. Similar labels present: {near}. Check the annotation "
                "release/field names (docs/DATA.md) or edit SELECTORS.")
        members.append(f"WHEN {group_sql(g)} THEN '{g.name}'")
    case = "CASE " + " ".join(members) + " ELSE NULL END"
    con.execute(f"""CREATE TABLE sel AS SELECT root_id, uid, root_id_str, {case} AS grp,
                    cell_class, cell_type, side, top_nt, top_nt_conf
                    FROM '{npq}' WHERE TRUE{sidecond}""")
    con.execute("DELETE FROM sel WHERE grp IS NULL")
    total_before = con.execute("SELECT count(*) FROM sel").fetchone()[0]
    pilot = None
    if pilot_max is not None and total_before > pilot_max:
        # deterministic proportional subsample per group, ordered by md5(seed|root_id)
        con.execute(f"""CREATE TABLE sel2 AS SELECT * FROM (
            SELECT *, row_number() OVER (PARTITION BY grp ORDER BY md5('{seed}|' || root_id_str))
                      AS rk,
                      count(*) OVER (PARTITION BY grp) AS gn FROM sel)
            WHERE rk <= greatest(1, floor(gn * {pilot_max} / {total_before}))""")
        con.execute("DROP TABLE sel")
        con.execute("ALTER TABLE sel2 RENAME TO sel")
        pilot = {"requested_max": pilot_max, "selected_before": total_before, "seed": seed,
                 "method": "per-group proportional, ordered by md5(seed|root_id)"}
    ids = np.array([r[0] for r in con.execute(
        "SELECT root_id FROM sel ORDER BY root_id").fetchall()], dtype=np.int64)
    if len(ids) == 0:
        raise SelectionError("selection empty after side/pilot filters")
    group_counts = dict(con.execute("SELECT grp, count(*) FROM sel GROUP BY grp ORDER BY grp")
                        .fetchall())
    con.register("ids", __import__("pyarrow").table({"root_id": ids}))
    E = f"'{epq}'"
    internal = con.execute(
        f"""SELECT count(*), coalesce(sum(count),0),
            count(*) FILTER (WHERE count>={threshold}),
            coalesce(sum(count) FILTER (WHERE count>={threshold}),0),
            count(*) FILTER (WHERE pre=post)
            FROM {E} WHERE pre IN (SELECT root_id FROM ids) AND post IN (SELECT root_id FROM ids)"""
    ).fetchone()
    con.execute(f"""CREATE TABLE bnd AS
        SELECT 'incoming' AS direction, pre AS boundary_id, count(*) AS n_internal_partners,
               sum(count) AS contacts, max(count) AS max_pair_contacts
        FROM {E} WHERE post IN (SELECT root_id FROM ids) AND pre NOT IN (SELECT root_id FROM ids)
          AND count>={threshold} GROUP BY pre
        UNION ALL
        SELECT 'outgoing', post, count(*), sum(count), max(count)
        FROM {E} WHERE pre IN (SELECT root_id FROM ids) AND post NOT IN (SELECT root_id FROM ids)
          AND count>={threshold} GROUP BY post""")
    cut_all = con.execute(
        f"""SELECT count(*) FILTER (WHERE post IN (SELECT root_id FROM ids)),
                   count(*) FILTER (WHERE pre IN (SELECT root_id FROM ids))
            FROM {E} WHERE (pre IN (SELECT root_id FROM ids)) != (post IN (SELECT root_id FROM ids))"""
    ).fetchone()
    cut_thr = con.execute(
        "SELECT direction, count(*), coalesce(sum(contacts),0) FROM bnd GROUP BY 1").fetchall()
    # internal edge hash (retained at threshold), orientation pre->post
    ed = con.execute(
        f"""SELECT pre, post, count FROM {E} WHERE count>={threshold}
            AND pre IN (SELECT root_id FROM ids) AND post IN (SELECT root_id FROM ids)
            ORDER BY pre, post""").fetchnumpy()
    edge_hash = hashlib.sha256(
        np.stack([ed["pre"], ed["post"], ed["count"]], 1).astype("<i8").tobytes()).hexdigest()

    cname = out_name or (name + (f"_{side}" if side else "") + ("_pilot" if pilot else ""))
    out = proc / "circuits" / cname
    out.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT * FROM sel ORDER BY root_id) TO '{out / 'internal_ids.parquet'}' "
                "(FORMAT parquet)")
    con.execute(f"""COPY (SELECT b.*, n.cell_class, n.cell_type, n.super_class
        FROM bnd b LEFT JOIN '{npq}' n ON n.root_id=b.boundary_id
        ORDER BY direction, contacts DESC) TO '{out / 'boundary_ids.parquet'}' (FORMAT parquet)""")
    n_nodes = len(ids)
    n_edges = int(internal[2])
    sel_json = {
        "circuit": cname, "selector": name, "description": spec["description"],
        "dataset": dataset, "version": version, "side_filter": side, "pilot": pilot,
        "is_pilot_subset": pilot is not None,
        "algorithm": "group predicates over annotations (see groups); union; optional pilot",
        "groups": [{"name": g.name, "field": g.field, "op": g.op, "values": list(g.values),
                    "also": [list(a) for a in g.also], "required": g.required}
                   for g in spec["groups"]],
        "group_counts_selected": group_counts, "group_counts_pre_pilot": counts,
        "n_neurons": n_nodes, "threshold": threshold,
        "internal_pairs_all": int(internal[0]), "internal_contacts_all": int(internal[1]),
        "internal_pairs_retained": n_edges, "internal_contacts_retained": int(internal[3]),
        "internal_self_loops": int(internal[4]),
        "cut_pairs_incoming_all_thresholds": int(cut_all[0]),
        "cut_pairs_outgoing_all_thresholds": int(cut_all[1]),
        "boundary_at_threshold": {d: {"partners": int(n), "contacts": int(c)}
                                  for d, n, c in cut_thr},
        "ids_hash_sha256": _hash_ids(ids), "edges_hash_sha256": edge_hash,
        "size_estimate": {
            "csr_bytes_fp32_int32": int(n_edges * 8 + (n_nodes + 1) * 4),
            "rnn_state_bytes_per_batch1": int(n_nodes * 4),
            "note": "theoretical arithmetic, not a measured peak"},
        "notes": spec["notes"],
    }
    (out / "selection.json").write_text(json.dumps(sel_json, indent=2))
    (out / "BOUNDARY.md").write_text(_boundary_md(sel_json))
    con.close()
    return sel_json


def _boundary_md(s: dict) -> str:
    b = s["boundary_at_threshold"]
    inc, outg = b.get("incoming", {}), b.get("outgoing", {})
    pil = "**PILOT SUBSET** - " + json.dumps(s["pilot"]) if s["pilot"] else "full selection"
    return f"""# Boundary report: {s['circuit']}

Release: {s['dataset']} v{s['version']} | selector `{s['selector']}` | {pil}
Neurons: {s['n_neurons']} | threshold: >= {s['threshold']} contacts per directed pair
ids hash: `{s['ids_hash_sha256']}` | edges hash: `{s['edges_hash_sha256']}`

| item | pairs | contacts |
|---|---|---|
| internal pairs (all) | {s['internal_pairs_all']} | {s['internal_contacts_all']} |
| internal pairs retained | {s['internal_pairs_retained']} | {s['internal_contacts_retained']} |
| incoming boundary partners (cut) | {inc.get('partners', 0)} neurons | {inc.get('contacts', 0)} |
| outgoing boundary partners (cut) | {outg.get('partners', 0)} neurons | {outg.get('contacts', 0)} |

Boundary neurons: `boundary_ids.parquet` (one hop only, no recursion). Self-loops internal:
{s['internal_self_loops']}.

## Treatments (choose explicitly per experiment; compare for sensitivity)
1. `cut`: remove all boundary edges (inputs from outside the circuit are lost).
2. `background`: constant/Poisson drive onto neurons that lose incoming contacts; this is NOT
   equivalent to the missing upstream dynamics.
3. `reduced_boundary`: model top incoming partners as explicit input channels (justify choice).
Whatever is used must be recorded in the run manifest and results labelled model-relative.
"""
