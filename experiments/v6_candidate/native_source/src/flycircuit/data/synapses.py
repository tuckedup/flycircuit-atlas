"""Bounded processing of the FULL FlyWire synapse export (flywire_synapses_783.feather, ~9.5 GB).

Never loads the file at once: record batches with column projection; scalar tallies per batch;
cross-batch pair aggregation in an on-disk DuckDB stage with an explicit memory cap and a
project-owned spill directory. Keeps four DISTINCT quantities apart (source README: the export
contains synapses whose partners are not all proofread neurons):

  broad_export_synapses     every row of the export (~130 M by the source description)
  both_endpoints_proofread  rows whose pre AND post root ID are in the proofread root array
  contacts                  == both_endpoints_proofread rows (one row = one synaptic contact)
  directed_pairs            distinct (pre, post) among both-proofread rows

Coordinates are documented by the SOURCE as nanometres (pre_pt_position_*, post_pt_position_*);
that statement is carried through as ``source_documented`` metadata, not independently verified here.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from flycircuit.data.feather_io import SchemaError, open_reader
from flycircuit.data.paths import scratch_dir

DOCUMENTED_COLUMNS = [
    "id", "pre_pt_root_id", "post_pt_root_id", "connection_score", "cleft_score",
    "gaba", "ach", "glut", "oct", "ser", "da", "neuropil",
    "post_pt_position_x", "post_pt_position_y", "post_pt_position_z",
    "pre_pt_position_x", "pre_pt_position_y", "pre_pt_position_z",
]
REQUIRED = ["pre_pt_root_id", "post_pt_root_id", "neuropil"]
COORD_COLS = [f"{s}_pt_position_{a}" for s in ("pre", "post") for a in "xyz"]
COORD_METADATA = {
    "units": "nanometers", "units_status": "source_documented (Zenodo 10676866 description), "
    "not independently verified", "space": "FlyWire v783 coordinate frame as released; "
    "transform to any viewer frame is NOT defined here",
}
MAX_GIB_DEFAULT = 2.0


def check_schema(path: str | Path) -> dict[str, Any]:
    """Compare the file schema with the source-documented columns; raise if essentials are absent."""
    r = open_reader(path)
    names = r.schema.names
    missing_req = [c for c in REQUIRED if c not in names]
    if missing_req:
        raise SchemaError(f"synapse export lacks required columns {missing_req}; has {names}")
    return {
        "columns": [(f.name, str(f.type)) for f in r.schema],
        "record_batches": r.num_record_batches,
        "missing_documented": [c for c in DOCUMENTED_COLUMNS if c not in names],
        "undocumented_extra": [c for c in names if c not in DOCUMENTED_COLUMNS],
        "has_coordinates": all(c in names for c in COORD_COLS),
    }


def _iter(path, cols, max_batches=None):
    r = open_reader(path)
    n = r.num_record_batches if max_batches is None else min(max_batches, r.num_record_batches)
    for i in range(n):
        b = r.get_batch(i)
        yield pa.Table.from_arrays([b.column(c) for c in cols], names=cols)


def scan_counts(path: str | Path, roots: np.ndarray, *, mem_gib: float = MAX_GIB_DEFAULT,
                threads: int = 4, max_batches: int | None = None, keep_stage: bool = False,
                pairs_out: Path | None = None) -> dict[str, Any]:
    """One streaming pass: classify every synapse against the proofread root set and aggregate pairs."""
    roots = np.asarray(roots, dtype=np.int64)
    vset = pa.array(roots)
    tmp = scratch_dir() / "synapse_scan"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "spill").mkdir(parents=True)
    con = duckdb.connect(str(tmp / "stage.duckdb"))
    con.execute(f"SET memory_limit='{mem_gib}GiB'")
    con.execute(f"SET threads={threads}")
    con.execute(f"SET temp_directory='{tmp / 'spill'}'")
    con.execute("CREATE TABLE pstage(pre BIGINT, post BIGINT, cnt BIGINT)")
    con.execute("CREATE TABLE nstage(neuropil VARCHAR, cnt BIGINT)")
    con.execute("CREATE TABLE rstage(root BIGINT)")
    tally = dict.fromkeys(("total", "pre_null", "post_null", "both", "pre_only", "post_only",
                           "neither", "self_loop_both"), 0)
    nb = 0
    for t in _iter(path, REQUIRED, max_batches):
        pre, post = t["pre_pt_root_id"], t["post_pt_root_id"]
        n = t.num_rows
        pre_null = pc.sum(pc.or_(pc.is_null(pre), pc.equal(pre, 0))).as_py() or 0
        post_null = pc.sum(pc.or_(pc.is_null(post), pc.equal(post, 0))).as_py() or 0
        ip = pc.is_in(pre, value_set=vset)
        iq = pc.is_in(post, value_set=vset)
        both = pc.and_(ip, iq)
        n_both = pc.sum(both).as_py() or 0
        n_pre = pc.sum(ip).as_py() or 0
        n_post = pc.sum(iq).as_py() or 0
        tally["total"] += n
        tally["pre_null"] += int(pre_null)
        tally["post_null"] += int(post_null)
        tally["both"] += int(n_both)
        tally["pre_only"] += int(n_pre - n_both)
        tally["post_only"] += int(n_post - n_both)
        tally["neither"] += int(n - n_pre - n_post + n_both)
        tb = t.filter(both)
        if tb.num_rows:
            tally["self_loop_both"] += int(pc.sum(pc.equal(tb["pre_pt_root_id"],
                                                          tb["post_pt_root_id"])).as_py() or 0)
            g = tb.group_by(["pre_pt_root_id", "post_pt_root_id"]).aggregate([([], "count_all")])
            g = g.rename_columns(["pre", "post", "cnt"]).cast(
                pa.schema([("pre", pa.int64()), ("post", pa.int64()), ("cnt", pa.int64())]))
            con.register("g", g)
            con.execute("INSERT INTO pstage SELECT pre, post, cnt FROM g")
            con.unregister("g")
            gn = tb.group_by(["neuropil"]).aggregate([([], "count_all")])
            gn = gn.rename_columns(["neuropil", "cnt"]).cast(
                pa.schema([("neuropil", pa.string()), ("cnt", pa.int64())]))
            con.register("gn", gn)
            con.execute("INSERT INTO nstage SELECT neuropil, cnt FROM gn")
            con.unregister("gn")
        ur = pa.table({"root": pc.unique(pa.chunked_array(pre.chunks + post.chunks))
                       .cast(pa.int64())})
        con.register("ur", ur)
        con.execute("INSERT INTO rstage SELECT root FROM ur")
        con.unregister("ur")
        nb += 1
    q = con.execute
    q("CREATE TABLE pairs AS SELECT pre, post, sum(cnt)::BIGINT AS count FROM pstage GROUP BY pre, post")
    rep: dict[str, Any] = {
        "file": Path(path).name, "record_batches_scanned": nb,
        "broad_export_synapses": tally["total"],
        "rows_pre_null_or_zero": tally["pre_null"], "rows_post_null_or_zero": tally["post_null"],
        "both_endpoints_proofread": tally["both"], "only_pre_proofread": tally["pre_only"],
        "only_post_proofread": tally["post_only"], "neither_proofread": tally["neither"],
        "contacts_both_proofread": tally["both"],
        "self_loop_contacts_both_proofread": tally["self_loop_both"],
        "directed_pairs_both_proofread": q("SELECT count(*) FROM pairs").fetchone()[0],
        "pair_contacts_check": int(q("SELECT coalesce(sum(count),0) FROM pairs").fetchone()[0]),
        "distinct_root_ids_in_export": q("SELECT count(DISTINCT root) FROM rstage").fetchone()[0],
        "contacts_by_neuropil_both_proofread": {
            r[0]: int(r[1]) for r in q(
                "SELECT neuropil, sum(cnt) FROM nstage GROUP BY neuropil ORDER BY 2 DESC").fetchall()},
        "scope_note": "broad export != proofread contacts != directed pairs; see module docstring",
        "complete_scan": max_batches is None,
    }
    assert rep["pair_contacts_check"] == rep["contacts_both_proofread"], "cross-batch aggregation lost rows"
    if pairs_out is not None:
        pairs_out.parent.mkdir(parents=True, exist_ok=True)
        q(f"COPY (SELECT * FROM pairs ORDER BY pre, post) TO '{pairs_out}' "
          "(FORMAT parquet, COMPRESSION zstd)")
    con.close()
    if not keep_stage:
        shutil.rmtree(tmp)  # project-owned scratch only
    return rep


def compare_with_pairs(scan: dict[str, Any], counts_json: dict[str, Any]) -> dict[str, Any]:
    """Compare the full-export scan with the independently ingested proofread-connections counts."""
    edges = counts_json.get("edges", counts_json)  # ingest counts.json nests these under "edges"
    ref_pairs = edges.get("unique_pairs")
    ref_contacts = edges.get("contacts_in_pairs")
    if ref_pairs is None or ref_contacts is None:
        raise KeyError("counts.json has no unique_pairs/contacts_in_pairs (looked top-level and under 'edges')")
    return {
        "directed_pairs": {"synapse_export": scan["directed_pairs_both_proofread"], "proofread_connections": ref_pairs,
                           "equal": scan["directed_pairs_both_proofread"] == ref_pairs},
        "contacts": {"synapse_export": scan["contacts_both_proofread"], "proofread_connections": ref_contacts,
                     "equal": scan["contacts_both_proofread"] == ref_contacts,
                     "difference": None if ref_contacts is None else scan["contacts_both_proofread"] - ref_contacts},
    }


def diff_pair_tables(exp_pairs: str | Path, prf_pairs: str | Path, *, mem_gib: float = 1.0,
                     threads: int = 2, max_rows: int = 200) -> dict[str, Any]:
    """Pair-level diff of two (pre, post, count) Parquet tables, e.g. the export aggregate vs the proofread file.

    Classifies each differing pair as only_in_export / only_in_proofread_file / count_differs and reports the
    roots involved. Bounded DuckDB (explicit memory cap); root IDs are returned as strings."""
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{mem_gib}GB'; PRAGMA threads={int(threads)}")
    con.execute(f"CREATE VIEW exp AS SELECT pre, post, count AS c_exp FROM '{exp_pairs}'")
    con.execute(f"CREATE VIEW prf AS SELECT pre, post, count AS c_prf FROM '{prf_pairs}'")
    con.execute("""CREATE TABLE d AS
      SELECT coalesce(e.pre,p.pre) pre, coalesce(e.post,p.post) post, e.c_exp, p.c_prf,
        CASE WHEN p.pre IS NULL THEN 'only_in_export' WHEN e.pre IS NULL THEN 'only_in_proofread_file'
             ELSE 'count_differs' END kind
      FROM exp e FULL OUTER JOIN prf p ON e.pre=p.pre AND e.post=p.post
      WHERE e.pre IS NULL OR p.pre IS NULL OR e.c_exp<>p.c_prf""")
    summary = [{"kind": k, "pairs": n, "contacts_export": int(ce or 0), "contacts_proofread": int(cp or 0)}
               for k, n, ce, cp in con.execute(
                   "SELECT kind,count(*),sum(c_exp),sum(c_prf) FROM d GROUP BY 1 ORDER BY 1").fetchall()]
    rows = [{"pre": str(a), "post": str(b), "c_export": c, "c_proofread": e, "kind": k}
            for a, b, c, e, k in con.execute(
                f"SELECT pre,post,c_exp,c_prf,kind FROM d ORDER BY kind,pre,post LIMIT {int(max_rows)}").fetchall()]
    top = [{"root": str(r), "n_pairs": n} for r, n in con.execute(
        "SELECT r,count(*) n FROM (SELECT pre r FROM d UNION ALL SELECT post FROM d) GROUP BY r "
        "ORDER BY n DESC, r LIMIT 10").fetchall()]
    total = con.execute("SELECT count(*) FROM d").fetchone()[0]
    con.close()
    return {"diff_summary": summary, "diff_rows": rows, "diff_rows_total": total, "top_roots_in_diff": top}


def extract_pathway(path: str | Path, pre_ids: np.ndarray, post_ids: np.ndarray, out: Path, *,
                    columns: list[str] | None = None, max_batches: int | None = None) -> dict[str, Any]:
    """Write the synapses with ``pre in pre_ids and post in post_ids`` (coordinates incl.) to Parquet."""
    cols = columns or [c for c in ("id", *REQUIRED, *COORD_COLS, "cleft_score", "connection_score")
                       if c in open_reader(path).schema.names]
    pre_v, post_v = pa.array(np.asarray(pre_ids, np.int64)), pa.array(np.asarray(post_ids, np.int64))
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".parquet.partial")
    writer = None
    n_out = 0
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    nonfinite = 0
    for t in _iter(path, cols, max_batches):
        m = pc.and_(pc.is_in(t["pre_pt_root_id"], value_set=pre_v),
                    pc.is_in(t["post_pt_root_id"], value_set=post_v))
        tb = t.filter(m)
        if not tb.num_rows:
            continue
        if writer is None:
            writer = pq.ParquetWriter(part, tb.schema, compression="zstd")
        writer.write_table(tb)
        n_out += tb.num_rows
        if "pre_pt_position_x" in cols:
            for j, a in enumerate("xyz"):
                x = tb[f"pre_pt_position_{a}"].to_numpy(zero_copy_only=False).astype(float)
                nonfinite += int((~np.isfinite(x)).sum())
                x = x[np.isfinite(x)]
                if len(x):
                    lo[j], hi[j] = min(lo[j], x.min()), max(hi[j], x.max())
    if writer is not None:
        writer.close()
        part.replace(out)
    return {
        "output": str(out) if n_out else None, "synapses_written": n_out, "columns": cols,
        "pre_position_bbox_min": None if not n_out else lo.tolist(),
        "pre_position_bbox_max": None if not n_out else hi.tolist(),
        "nonfinite_coordinates": nonfinite, "coordinates": COORD_METADATA,
        "complete_scan": max_batches is None,
    }
