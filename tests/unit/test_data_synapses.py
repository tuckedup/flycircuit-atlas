"""Full-synapse-export processing on a tiny SYNTHETIC feather (no download)."""

import numpy as np
import pyarrow as pa
import pytest

from flycircuit.data.feather_io import SchemaError, write_feather_batches
from flycircuit.data.synapses import check_schema, compare_with_pairs, extract_pathway, scan_counts

BIG = 720575940600000000  # > 2^53 (9.007e15 is 2^53; BIG ~ 7.2e17)


def _make(tmp_path, n_batches=4, per=250, seed=3):
    rng = np.random.default_rng(seed)
    roots = BIG + np.arange(1, 21)  # 20 proofread
    unproof = BIG + 1000 + np.arange(5)  # not in proofread array
    pool = np.concatenate([roots, unproof])
    batches, rows = [], []
    for b in range(n_batches):
        pre = rng.choice(pool, per)
        post = rng.choice(pool, per)
        x = rng.uniform(1e5, 9e5, size=(per, 6))
        cols = {"id": np.arange(b * per, (b + 1) * per), "pre_pt_root_id": pre, "post_pt_root_id": post,
                "neuropil": rng.choice(["MB_CA_R", "AL_L", "FB"], per)}
        for j, name in enumerate(["pre_pt_position_x", "pre_pt_position_y", "pre_pt_position_z",
                                  "post_pt_position_x", "post_pt_position_y", "post_pt_position_z"]):
            cols[name] = x[:, j]
        batches.append(pa.RecordBatch.from_pydict({k: pa.array(v) for k, v in cols.items()}))
        rows.append((pre, post))
    p = tmp_path / "syn.feather"
    write_feather_batches(p, batches)
    pre = np.concatenate([r[0] for r in rows])
    post = np.concatenate([r[1] for r in rows])
    return p, roots, pre, post


@pytest.fixture(autouse=True)
def _scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYCIRCUIT_HOME", str(tmp_path / "home"))
    (tmp_path / "home" / "data" / "processed").mkdir(parents=True)


def test_scan_counts_match_oracle_across_batches(tmp_path):
    p, roots, pre, post = _make(tmp_path)
    rep = scan_counts(p, roots, mem_gib=0.5, threads=1)
    inroot = set(roots.tolist())
    ip = np.array([x in inroot for x in pre.tolist()])
    iq = np.array([x in inroot for x in post.tolist()])
    assert rep["record_batches_scanned"] == 4
    assert rep["broad_export_synapses"] == len(pre)
    assert rep["both_endpoints_proofread"] == int((ip & iq).sum())
    assert rep["only_pre_proofread"] == int((ip & ~iq).sum())
    assert rep["only_post_proofread"] == int((~ip & iq).sum())
    assert rep["neither_proofread"] == int((~ip & ~iq).sum())
    assert (rep["both_endpoints_proofread"] + rep["only_pre_proofread"]
            + rep["only_post_proofread"] + rep["neither_proofread"]) == len(pre)
    pairs = {}
    for a, b in zip(pre[ip & iq].tolist(), post[ip & iq].tolist(), strict=True):
        pairs[(a, b)] = pairs.get((a, b), 0) + 1
    assert rep["directed_pairs_both_proofread"] == len(pairs)  # pairs repeat across batches
    assert rep["self_loop_contacts_both_proofread"] == sum(v for (a, b), v in pairs.items() if a == b)
    assert rep["distinct_root_ids_in_export"] == len(set(pre.tolist()) | set(post.tolist()))


def test_pairs_output_exact_big_ids_and_orientation(tmp_path):
    import pyarrow.parquet as pq

    p, roots, pre, post = _make(tmp_path, seed=5)
    out = tmp_path / "pairs.parquet"
    scan_counts(p, roots, mem_gib=0.5, threads=1, pairs_out=out)
    t = pq.read_table(out)
    assert t["pre"].type == pa.int64() and int(t["pre"].to_numpy().min()) > 2**53
    inroot = set(roots.tolist())
    exp = {}
    for a, b in zip(pre.tolist(), post.tolist(), strict=True):
        if a in inroot and b in inroot:
            exp[(a, b)] = exp.get((a, b), 0) + 1
    got = {(int(a), int(b)): int(c) for a, b, c in zip(t["pre"].to_numpy(), t["post"].to_numpy(),
                                                       t["count"].to_numpy(), strict=True)}
    assert got == exp


def test_extract_pathway_filters_and_reports_unit_status(tmp_path):
    import pyarrow.parquet as pq

    p, roots, pre, post = _make(tmp_path, seed=9)
    pre_set, post_set = roots[:5], roots[5:12]
    rep = extract_pathway(p, pre_set, post_set, tmp_path / "pw.parquet")
    exp = sum(1 for a, b in zip(pre.tolist(), post.tolist(), strict=True)
              if a in set(pre_set.tolist()) and b in set(post_set.tolist()))
    assert rep["synapses_written"] == exp
    t = pq.read_table(rep["output"])
    assert t.num_rows == exp
    assert set(t["pre_pt_root_id"].to_pylist()) <= set(pre_set.tolist())
    assert rep["coordinates"]["units"] == "nanometers"
    assert "not independently verified" in rep["coordinates"]["units_status"]
    assert rep["nonfinite_coordinates"] == 0


def test_schema_check_and_missing_essentials(tmp_path):
    p, *_ = _make(tmp_path)
    s = check_schema(p)
    assert s["has_coordinates"] and "gaba" in s["missing_documented"]
    bad = tmp_path / "bad.feather"
    write_feather_batches(bad, [pa.RecordBatch.from_pydict({"x": pa.array([1])})])
    with pytest.raises(SchemaError):
        check_schema(bad)


def test_compare_with_pairs_flags_differences():
    scan = {"directed_pairs_both_proofread": 10, "contacts_both_proofread": 100}
    r = compare_with_pairs(scan, {"unique_pairs": 10, "contacts_in_pairs": 98})
    assert r["directed_pairs"]["equal"] and not r["contacts"]["equal"] and r["contacts"]["difference"] == 2


def test_compare_with_pairs_reads_nested_edges_counts_and_never_returns_null():
    """Regression: the ingest counts.json nests these under 'edges'; a top-level lookup returned None."""
    scan = {"directed_pairs_both_proofread": 15092004, "contacts_both_proofread": 54492970}
    r = compare_with_pairs(scan, {"edges": {"unique_pairs": 15091983, "contacts_in_pairs": 54492922}})
    assert r["directed_pairs"]["proofread_connections"] == 15091983
    assert r["contacts"]["difference"] == 48
    with pytest.raises(KeyError):
        compare_with_pairs(scan, {"edges": {}})


def test_diff_pair_tables_classifies_all_three_kinds_and_keeps_big_ids_exact(tmp_path):
    import pyarrow.parquet as pq

    from flycircuit.data.synapses import diff_pair_tables

    a, b, c, d = (BIG + i for i in (1, 2, 3, 4))
    exp = pa.table({"pre": [a, a, b, c], "post": [b, c, a, d], "count": [5, 2, 7, 1]})
    prf = pa.table({"pre": [a, b, d], "post": [b, a, c], "count": [5, 6, 9]})
    pq.write_table(exp, tmp_path / "e.parquet")
    pq.write_table(prf, tmp_path / "p.parquet")
    r = diff_pair_tables(tmp_path / "e.parquet", tmp_path / "p.parquet")
    kinds = {s["kind"]: s for s in r["diff_summary"]}
    assert kinds["only_in_export"]["pairs"] == 2 and kinds["only_in_export"]["contacts_export"] == 3
    assert kinds["only_in_proofread_file"]["pairs"] == 1 and kinds["only_in_proofread_file"]["contacts_proofread"] == 9
    assert kinds["count_differs"]["pairs"] == 1 and kinds["count_differs"]["contacts_export"] == 7
    assert r["diff_rows_total"] == 4
    assert all(isinstance(x["pre"], str) for x in r["diff_rows"])
    assert str(a) in {x["root"] for x in r["top_roots_in_diff"]}  # exact, >2^53


def test_diff_pair_tables_identical_inputs_report_no_difference(tmp_path):
    import pyarrow.parquet as pq

    from flycircuit.data.synapses import diff_pair_tables

    t = pa.table({"pre": [BIG + 1, BIG + 2], "post": [BIG + 2, BIG + 1], "count": [3, 4]})
    pq.write_table(t, tmp_path / "x.parquet")
    r = diff_pair_tables(tmp_path / "x.parquet", tmp_path / "x.parquet")
    assert r["diff_rows_total"] == 0 and r["diff_summary"] == []
