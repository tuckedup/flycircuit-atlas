"""FlyWire proofread-tier ingestion: Feather V2 -> external-memory aggregation -> Parquet.

Outputs under data/processed/<dataset>/<version>/:
  neurons.parquet, edges_by_neuropil.parquet, edges_pairs.parquet, counts.json
Root IDs are exact int64 (never float). `root_id_str` carries the JS-safe string form.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .feather_io import edge_batches, open_reader, resolve_columns
from .paths import processed_dir, raw_dir, scratch_dir

DEFAULT_MEM_GIB = 2.0  # DuckDB cap; total pipeline budget is 4.5 GiB
COORD_SPACE = {
    "space": "FlyWire FAFB14.1-derived segmentation space (materialization 783)",
    "pos_unit": "voxel, 4x4x40 nm per voxel (x,y,z) per supplemental_files/README.md; "
                "pos_* = backbone anchor, soma_* = soma",
    "note": "Coordinates are nullable; not every neuron has a soma/nucleus position.",
}

ANN_TEXT_COLS = [
    "flow", "super_class", "cell_class", "cell_sub_class", "supertype", "cell_type",
    "hemibrain_type", "ito_lee_hemilineage", "hartenstein_hemilineage", "top_nt",
    "known_nt", "known_nt_source", "side", "nerve", "vfb_id", "fbbt_id", "status",
    "dimorphism", "matching_notes", "fru_dsx", "synonyms"]
ANN_INT_COLS = ["supervoxel_id", "root_id", "nucleus_id"]
ANN_FLOAT_COLS = ["pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z", "top_nt_conf"]


def read_annotations(path: Path) -> pa.Table:
    """Read the annotation TSV with exact int64 IDs; empty strings -> null."""
    types = {c: pa.string() for c in ANN_TEXT_COLS}
    types.update({c: pa.int64() for c in ANN_INT_COLS})
    types.update({c: pa.float64() for c in ANN_FLOAT_COLS})
    t = pacsv.read_csv(
        str(path),
        parse_options=pacsv.ParseOptions(delimiter="\t"),
        convert_options=pacsv.ConvertOptions(
            column_types=types, null_values=[""], strings_can_be_null=True),
    )
    return t


def build_neurons(root_ids: np.ndarray, ann: pa.Table, dataset: str, version: str,
                  ann_source: str) -> tuple[pa.Table, dict]:
    con = duckdb.connect()
    con.register("roots", pa.table({"root_id": pa.array(root_ids.astype(np.int64))}))
    # annotation duplicates: keep first row per root_id deterministically (by supervoxel_id)
    con.register("ann", ann)
    n_ann = ann.num_rows
    n_ann_unique = con.execute("SELECT count(DISTINCT root_id) FROM ann").fetchone()[0]
    cols = [c for c in ann.column_names if c != "root_id"]
    sel = ", ".join(f"a.{c}" for c in cols)
    t = con.execute(
        f"""
        SELECT '{dataset}' AS dataset, '{version}' AS version,
               '{dataset}:{version}:' || CAST(r.root_id AS VARCHAR) AS uid,
               r.root_id, CAST(r.root_id AS VARCHAR) AS root_id_str,
               (a.root_id IS NOT NULL) AS annotated, {sel},
               (a.top_nt IS NULL) AS nt_unknown,
               '{ann_source}' AS annotation_source
        FROM roots r
        LEFT JOIN (SELECT * FROM ann QUALIFY row_number() OVER
                   (PARTITION BY root_id ORDER BY supervoxel_id) = 1) a
          ON r.root_id = a.root_id
        ORDER BY r.root_id
        """
    ).to_arrow_table()
    in_roots = con.execute(
        "SELECT count(DISTINCT root_id) FROM ann WHERE root_id IN (SELECT root_id FROM roots)"
    ).fetchone()[0]
    meta = {
        "annotation_rows": n_ann,
        "annotation_unique_roots": n_ann_unique,
        "annotation_duplicate_rows": n_ann - n_ann_unique,
        "annotation_roots_in_proofread_set": in_roots,
        "annotation_roots_not_in_proofread_set": n_ann_unique - in_roots,
        "proofread_roots_without_annotation": len(root_ids) - in_roots,
    }
    con.close()
    schema_meta = {b"coordinate_space": json.dumps(COORD_SPACE).encode()}
    return t.replace_schema_metadata(schema_meta), meta


def ingest_edges(feather: Path, out: Path, roots: np.ndarray, *, mem_gib: float = DEFAULT_MEM_GIB,
                 threads: int = 4, threshold: int = 5, keep_stage: bool = False) -> dict:
    """Stream Feather batches into an on-disk DuckDB stage, then group globally (spills to disk)."""
    out.mkdir(parents=True, exist_ok=True)
    tmp = scratch_dir() / f"ingest_{out.parent.name}_{out.name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    con = duckdb.connect(str(tmp / "stage.duckdb"))
    con.execute(f"SET memory_limit='{mem_gib}GiB'")
    con.execute(f"SET threads={threads}")
    con.execute(f"SET temp_directory='{tmp / 'spill'}'")
    con.execute("CREATE TABLE stage(pre BIGINT, post BIGINT, neuropil VARCHAR, cnt BIGINT)")
    r = open_reader(feather)
    cols = resolve_columns(r.schema.names)
    nb = 0
    for b in edge_batches(feather):
        con.register("b", pa.Table.from_batches([b]))
        con.execute("INSERT INTO stage SELECT pre, post, neuropil, count FROM b")
        con.unregister("b")
        nb += 1
    con.register("roots", pa.table({"root_id": pa.array(roots.astype(np.int64))}))
    q = con.execute
    src_rows = q("SELECT count(*) FROM stage").fetchone()[0]
    src_contacts = q("SELECT coalesce(sum(cnt),0) FROM stage").fetchone()[0]
    bad = q("SELECT count(*) FROM stage WHERE cnt IS NULL OR cnt<=0 OR pre IS NULL "
            "OR post IS NULL").fetchone()[0]
    q("""CREATE TABLE bynp AS
         SELECT pre, post, neuropil, sum(cnt)::BIGINT AS count, count(*)::BIGINT AS n_source_rows
         FROM stage WHERE cnt>0 GROUP BY pre, post, neuropil""")
    q("""CREATE TABLE pairs AS
         SELECT pre, post, sum(count)::BIGINT AS count, count(*)::INTEGER AS n_neuropils
         FROM bynp GROUP BY pre, post""")
    def f(sql):
        return q(sql).fetchone()[0]
    rep = {
        "feather": feather.name, "record_batches": nb, "physical_columns": cols,
        "source_rows": src_rows, "source_contacts": int(src_contacts),
        "invalid_rows_dropped": bad,
        "duplicate_pair_neuropil_keys_merged": src_rows - bad - f("SELECT count(*) FROM bynp"),
        "pair_neuropil_rows": f("SELECT count(*) FROM bynp"),
        "unique_pairs": f("SELECT count(*) FROM pairs"),
        "contacts_in_pairs": f("SELECT coalesce(sum(count),0) FROM pairs"),
        "self_loop_pairs": f("SELECT count(*) FROM pairs WHERE pre=post"),
        "self_loop_contacts": f("SELECT coalesce(sum(count),0) FROM pairs WHERE pre=post"),
        "unique_roots_in_edges": f(
            "SELECT count(*) FROM (SELECT pre r FROM pairs UNION SELECT post FROM pairs)"),
        "edge_roots_not_in_proofread_set": f(
            "SELECT count(*) FROM (SELECT pre r FROM pairs UNION SELECT post FROM pairs) "
            "WHERE r NOT IN (SELECT root_id FROM roots)"),
        "pairs_with_both_endpoints_internal": f(
            "SELECT count(*) FROM pairs WHERE pre IN (SELECT root_id FROM roots) "
            "AND post IN (SELECT root_id FROM roots)"),
        "pairs_with_unmatched_endpoint": f(
            "SELECT count(*) FROM pairs WHERE pre NOT IN (SELECT root_id FROM roots) "
            "OR post NOT IN (SELECT root_id FROM roots)"),
        "contact_threshold": threshold,
        "pairs_retained_at_threshold": f(f"SELECT count(*) FROM pairs WHERE count>={threshold}"),
        "contacts_retained_at_threshold": f(
            f"SELECT coalesce(sum(count),0) FROM pairs WHERE count>={threshold}"),
        "scope_note": "threshold is an explicit model choice, not applied to saved outputs",
    }
    q(f"COPY (SELECT * FROM bynp ORDER BY pre, post, neuropil) TO "
      f"'{out / 'edges_by_neuropil.parquet'}' (FORMAT parquet, COMPRESSION zstd)")
    q(f"COPY (SELECT * FROM pairs ORDER BY pre, post) TO "
      f"'{out / 'edges_pairs.parquet'}' (FORMAT parquet, COMPRESSION zstd)")
    con.close()
    if not keep_stage:
        shutil.rmtree(tmp)  # project-owned scratch only
    return rep


def run_ingest(dataset: str = "flywire", version: str = "783", *, threshold: int = 5,
               mem_gib: float = DEFAULT_MEM_GIB) -> dict:
    if dataset != "flywire":
        raise NotImplementedError(dataset)
    raw, out = raw_dir(dataset, version), processed_dir(dataset, version)
    out.mkdir(parents=True, exist_ok=True)
    roots = np.load(raw / "proofread_root_ids_783.npy")
    if roots.dtype.kind not in "iu":
        raise ValueError(f"root ids dtype {roots.dtype} is not integer; refusing lossy ids")
    roots = roots.astype(np.int64)
    ann_file = raw / "Supplemental_file1_neuron_annotations.tsv"
    ann = read_annotations(ann_file)
    neurons, nmeta = build_neurons(roots, ann, dataset, version, ann_file.name)
    pq.write_table(neurons, out / "neurons.parquet", compression="zstd")
    erep = ingest_edges(raw / "proofread_connections_783.feather", out, roots,
                        threshold=threshold, mem_gib=mem_gib)
    report = {
        "dataset": dataset, "version": version,
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "proofread_root_id_array_len": len(roots),
        "unique_proofread_roots": len(np.unique(roots)),
        "neurons_rows": neurons.num_rows,
        "excluded_roots": "none at ingestion; root set = proofread_root_ids_783.npy",
        "neurons": nmeta, "edges": erep,
        "provenance": "counts measured locally from the files in SOURCE_MANIFEST.json",
    }
    (out / "counts.json").write_text(json.dumps(report, indent=2))
    return report
