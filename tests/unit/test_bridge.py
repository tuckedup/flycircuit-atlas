"""Selection -> VersionedGraph bridge on a tiny SYNTHETIC processed release (ids > 2^53)."""

import json

import duckdb
import numpy as np
import pyarrow as pa
import pytest

from flycircuit.graph import bridge as _bridge
from flycircuit.graph.bridge import (
    BridgeError,
    boundary_background_pairs,
    build_shiu_sign_table,
)


def load_circuit_graph(*a, **k):
    """Tests default to the annotation policy; the Shiu policy needs a derived table."""
    k.setdefault("sign_policy", "annotation_top_nt")
    return _bridge.load_circuit_graph(*a, **k)


BASE = 720575940000000000  # > 2**53 (9.007e15)
ID = {k: BASE + 10 + i for i, k in enumerate("abcdef")}  # a,b,c internal; d,e outside; f unused


@pytest.fixture()
def processed(tmp_path):
    proc = tmp_path / "flywire" / "783"
    cdir = proc / "circuits" / "mini"
    cdir.mkdir(parents=True)
    nrn = pa.table({
        "root_id": [ID[k] for k in "abcdef"],
        "top_nt": ["acetylcholine", "gaba", "dopamine", "glutamate", None, "acetylcholine"],
        "top_nt_conf": [0.9, 0.8, 0.7, 0.9, None, 0.2],
    })
    import pyarrow.parquet as pq

    pq.write_table(nrn, proc / "neurons.parquet")
    # a->b 7, b->a 6 (reciprocal), a->c 5, b->c 4 (below thr), c->c 9 (self loop),
    # d->a 8 (incoming boundary), c->e 6 (outgoing boundary), e->a 2 (below thr)
    rows = [("a", "b", 7), ("b", "a", 6), ("a", "c", 5), ("b", "c", 4), ("c", "c", 9),
            ("d", "a", 8), ("c", "e", 6), ("e", "a", 2)]
    pq.write_table(pa.table({"pre": [ID[r[0]] for r in rows], "post": [ID[r[1]] for r in rows],
                             "count": [r[2] for r in rows]}), proc / "edges_pairs.parquet")
    pq.write_table(pa.table({
        "root_id": [ID[k] for k in "abc"], "grp": ["kenyon", "mbon", "dan"],
        "cell_class": ["Kenyon_Cell", "MBON", "DAN"], "cell_type": ["KC", "MBON1", "PAM1"],
        "top_nt": ["acetylcholine", "gaba", "dopamine"], "top_nt_conf": [0.9, 0.8, 0.7]}),
        cdir / "internal_ids.parquet")
    (cdir / "selection.json").write_text(json.dumps(
        {"dataset": "flywire", "version": "783", "threshold": 5, "is_pilot_subset": False}))
    return proc


def test_orientation_threshold_selfloop_ids(processed):
    cg = load_circuit_graph("mini", processed=processed)
    g = cg.graph
    assert g.root_ids.dtype == np.int64 and g.root_ids[0] == ID["a"]
    a, b, c = (g.index_of(ID[k]) for k in "abc")
    W = g.csr  # W[post, pre]
    assert W[b, a] == 7 and W[a, b] == 6 and W[c, a] == 5 and W[c, c] == 9
    assert W[c, b] == 0  # 4 contacts < threshold 5: not an edge at this model choice
    assert g.n_edges == 4 and g.n_self_loops == 1
    assert g.mapping_hash() == load_circuit_graph("mini", processed=processed).graph.mapping_hash()
    assert json.dumps([str(i) for i in g.root_ids.tolist()])  # strings for JS


def test_boundary_masks_and_counts(processed):
    cg = load_circuit_graph("mini", processed=processed)
    g, b = cg.graph, cg.boundary
    a, c = g.index_of(ID["a"]), g.index_of(ID["c"])
    assert b.in_boundary_id.tolist() == [ID["d"]] and b.in_post_idx.tolist() == [a]
    assert b.out_post_id.tolist() == [ID["e"]] and b.out_pre_idx.tolist() == [c]
    assert g.node_masks["boundary_input"].tolist() == [True, False, False]
    assert g.node_masks["boundary_output"].tolist() == [False, False, True]
    s = b.summary()
    assert s["cut_incoming_contacts"] == 8 and s["cut_outgoing_contacts"] == 6  # e->a (2) excluded


def test_annotation_sign_policy_and_unknown_policy(processed):
    cg = load_circuit_graph("mini", processed=processed, sign_policy="annotation_top_nt")
    assert cg.node_sign.tolist() == [1, -1, 1]  # ACh +, GABA -, dopamine + (Shiu-style assumption)
    assert cg.signed_contacts().tolist() == [
        float(s) * c for s, c in zip(cg.edge_sign.tolist(), cg.graph.count.tolist(), strict=True)]
    assert cg.boundary.in_sign.tolist() == [-1]  # outside presyn d is glutamate in the fixture
    # unknown handling: e has no top_nt; with min_conf 0.85 only 'a' (0.9) keeps a sign
    cg2 = load_circuit_graph("mini", processed=processed, sign_policy="annotation_top_nt",
                             min_conf=0.85)
    assert cg2.node_sign.tolist() == [1, 0, 0]
    cg3 = load_circuit_graph("mini", processed=processed, sign_policy="annotation_top_nt",
                             min_conf=0.85, unknown_sign="inhibitory")
    assert cg3.node_sign.tolist() == [1, -1, -1]


def test_shiu_policy_requires_derived_table_and_uses_it(processed, tmp_path):
    with pytest.raises(BridgeError, match="refcheck prepare"):
        load_circuit_graph("mini", processed=processed, sign_policy="shiu_reported")
    ext = tmp_path / "Connectivity_783.parquet"
    import pyarrow.parquet as pq

    pq.write_table(pa.table({
        "Presynaptic_ID": [ID["a"], ID["a"], ID["b"], ID["d"]],
        "Postsynaptic_ID": [ID["b"], ID["c"], ID["a"], ID["a"]],
        "Excitatory": [-1, -1, 1, 1]}), ext)
    info = build_shiu_sign_table(ext, processed / "ref_shiu_presyn_sign.parquet")
    assert info["presynaptic_neurons"] == 3 and info["inconsistent"] == 0
    cg = load_circuit_graph("mini", processed=processed, sign_policy="shiu_reported")
    assert cg.node_sign.tolist() == [-1, 1, 0]  # c absent from the table -> unknown -> 0
    assert cg.boundary.in_sign.tolist() == [1]


def test_inconsistent_sign_in_reference_is_rejected(tmp_path):
    ext = tmp_path / "bad.parquet"
    duckdb.connect().execute(
        f"COPY (SELECT * FROM (VALUES (1,2,1),(1,3,-1)) t(Presynaptic_ID,Postsynaptic_ID,Excitatory)) "
        f"TO '{ext}' (FORMAT parquet)")
    with pytest.raises(BridgeError, match="inconsistent"):
        build_shiu_sign_table(ext, tmp_path / "o.parquet")


def test_threshold_mismatch_with_selection_is_an_error(processed):
    with pytest.raises(BridgeError, match="threshold"):
        load_circuit_graph("mini", processed=processed, threshold=3)


def test_missing_selection_is_actionable(processed):
    with pytest.raises(BridgeError, match="data select"):
        load_circuit_graph("nope", processed=processed)


def test_boundary_modes(processed):
    cg = load_circuit_graph("mini", processed=processed, sign_policy="annotation_top_nt")
    assert boundary_background_pairs(cg, "cut") is None
    post, w = boundary_background_pairs(cg, "background_drive")
    assert post.tolist() == [cg.graph.index_of(ID["a"])] and w.tolist() == [-8.0]  # d glutamate
    with pytest.raises(BridgeError):
        boundary_background_pairs(cg, "poisson")
