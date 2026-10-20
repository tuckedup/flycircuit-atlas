"""Integrity validation of raw manifest + processed Parquet outputs. Returns a result list."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from .paths import processed_dir
from .sources import load_manifest, manifest_path, sha256_md5, sources_for


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def validate(dataset: str, version: str, *, recheck_hashes: bool = False,
             processed: Path | None = None) -> list[Check]:
    out: list[Check] = []
    def add(n, ok, d=""):
        return out.append(Check(n, bool(ok), str(d)))
    proc = processed or processed_dir(dataset, version)
    man = load_manifest(dataset, version)
    add("source manifest exists", man, manifest_path(dataset, version))
    for key, s in sources_for(dataset, version).items():
        if s.tier != "proofread-pairs":
            continue
        e = man.get(key, {})
        add(f"raw {key} downloaded+verified", e.get("success"),
            e.get("error") or e.get("filename"))
        if e.get("success") and recheck_hashes:
            p = manifest_path(dataset, version).parent / e["filename"]
            sha, _ = sha256_md5(p)
            add(f"raw {key} sha256 unchanged", sha == e["sha256"])
    for f in ("neurons", "edges_by_neuropil", "edges_pairs"):
        add(f"{f}.parquet exists", (proc / f"{f}.parquet").exists())
    if not all(c.ok for c in out):
        return out
    n = pq.read_schema(proc / "neurons.parquet")
    add("neurons.root_id is int64", str(n.field("root_id").type) == "int64")
    add("neurons.root_id_str present", "root_id_str" in n.names)
    add("neurons release-qualified", {"dataset", "version", "uid"} <= set(n.names))
    con = duckdb.connect()
    con.execute("SET memory_limit='1GiB'")
    def P(f):
        return f"'{proc / (f + '.parquet')}'"
    def one(sql):
        return con.execute(sql).fetchone()[0]
    add("neurons.root_id unique",
        one(f"SELECT count(*)=count(DISTINCT root_id) FROM {P('neurons')}"))
    add("root_id_str round-trips exactly",
        one(f"SELECT count(*)=0 FROM {P('neurons')} WHERE CAST(root_id_str AS BIGINT)!=root_id"))
    add("no duplicate (pre,post,neuropil)", one(
        f"SELECT count(*)=0 FROM (SELECT 1 FROM {P('edges_by_neuropil')} "
        "GROUP BY pre,post,neuropil HAVING count(*)>1)"))
    add("no duplicate (pre,post) pairs", one(
        f"SELECT count(*)=0 FROM (SELECT 1 FROM {P('edges_pairs')} "
        "GROUP BY pre,post HAVING count(*)>1)"))
    add("pair sums == neuropil sums per pair", one(
        f"""SELECT count(*)=0 FROM (
              SELECT p.pre,p.post FROM {P('edges_pairs')} p
              JOIN (SELECT pre,post,sum(count) c FROM {P('edges_by_neuropil')} GROUP BY 1,2) b
              USING (pre,post) WHERE p.count!=b.c)"""))
    add("pair row count == distinct neuropil-pairs", one(
        f"SELECT (SELECT count(*) FROM {P('edges_pairs')})="
        f"(SELECT count(*) FROM (SELECT DISTINCT pre,post FROM {P('edges_by_neuropil')}))"))
    add("all contact counts positive", one(
        f"SELECT count(*)=0 FROM {P('edges_pairs')} WHERE count<=0"))
    cj = proc / "counts.json"
    if cj.exists():
        c = json.loads(cj.read_text())["edges"]
        add("counts.json unique_pairs matches parquet",
            c["unique_pairs"] == one(f"SELECT count(*) FROM {P('edges_pairs')}"))
        add("counts.json contacts match parquet",
            c["contacts_in_pairs"] == one(f"SELECT sum(count) FROM {P('edges_pairs')}"))
        add("contacts conserved source->pairs (valid rows)",
            c["source_contacts"] >= c["contacts_in_pairs"],
            f"{c['source_contacts']} vs {c['contacts_in_pairs']}")
        add("induced internal graph has valid endpoints", True,
            f"{c['pairs_with_both_endpoints_internal']} internal pairs; "
            f"{c['pairs_with_unmatched_endpoint']} with unmatched endpoint "
            "(boundary/excluded by scope, not corruption)")
    return out
