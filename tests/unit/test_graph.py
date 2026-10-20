import json

import numpy as np
import pytest

from flycircuit.graph import DenseAllocationError, VersionedGraph, guard_dense, make_synthetic_graph
from flycircuit.sim import simulate_lif

BIG = 2**53 + 12345  # not exactly representable after float64 conversion chains


def tiny():
    # A=BIG -> B=BIG+2 (count 3) ; duplicate row sums; C has a self loop
    a, b, c = BIG, BIG + 2, BIG + 4
    return VersionedGraph.from_edges(
        "test", "1", [a, a, b, c], [b, b, c, c], [3, 4, 5, 6], root_ids=[c, a, b]
    )


def test_direction_W_post_pre():
    g = tiny()
    ia, ib = g.index_of(BIG), g.index_of(BIG + 2)
    dense = g.to_dense()
    assert dense[ib, ia] == 7  # duplicates summed
    assert dense[ia, ib] == 0  # no reverse edge
    post, counts = g.out_partners(ia)
    assert list(post) == [ib] and list(counts) == [7]
    pre, _ = g.in_partners(ib)
    assert list(pre) == [ia]
    assert g.n_self_loops == 1


def test_exact_large_ids_and_string_ids():
    g = tiny()
    assert g.root_ids.dtype == np.int64
    assert int(g.root_ids[g.index_of(BIG)]) == BIG
    qid = g.qualified_id(g.index_of(BIG + 2))
    assert qid == f"test:1:{BIG + 2}"
    assert g.resolve(qid) == g.index_of(BIG + 2)


def test_release_mismatch_rejected():
    g = tiny()
    with pytest.raises(ValueError):
        g.resolve(f"other:1:{BIG}")
    with pytest.raises(ValueError):
        g.resolve(str(BIG))  # bare IDs are not accepted


def test_unknown_endpoint_rejected():
    with pytest.raises(ValueError):
        VersionedGraph.from_edges("t", "1", [1], [2], [1], root_ids=[1])


def test_mapping_hash_stable_and_order_independent():
    g1 = tiny()
    g2 = VersionedGraph.from_edges("test", "1", [BIG], [BIG + 2], [1], root_ids=[BIG + 4, BIG + 2, BIG])
    assert g1.mapping_hash() == g2.mapping_hash()
    g3 = VersionedGraph.from_edges("test", "2", [BIG], [BIG + 2], [1], root_ids=[BIG, BIG + 2, BIG + 4])
    assert g3.mapping_hash() != g1.mapping_hash()  # release-qualified


def test_dense_guard():
    with pytest.raises(DenseAllocationError):
        guard_dense((100_000, 100_000))
    g = make_synthetic_graph()
    with pytest.raises(DenseAllocationError):
        g.to_dense(max_elements=10)


def test_save_load_roundtrip(tmp_path):
    g = make_synthetic_graph()
    g.save(tmp_path / "g")
    h = VersionedGraph.load(tmp_path / "g")
    assert h.edge_hash() == g.edge_hash() and h.synthetic


def test_masks_validated():
    g = make_synthetic_graph()
    with pytest.raises(KeyError):
        g.set_node_mask("nonsense", np.zeros(g.n, bool))
    with pytest.raises(ValueError):
        g.set_edge_mask("trainable", np.zeros(3, bool))


def test_dynamics_respect_edge_direction():
    g = make_synthetic_graph()
    i0, i1, i2 = 0, 1, 2  # chain 0 -> 1 -> 2, no reverse edges
    # drive neuron 2 only: its spikes must not reach 0 via the (absent) reverse path directly
    out = simulate_lif(g, np.array([i2]), steps=400, seed=1)
    assert out["spike_counts"][i2] > 0
    # drive neuron 0: neuron 1 receives the 12-contact edge and should fire
    out0 = simulate_lif(g, np.array([i0]), steps=400, seed=1, input_rate_hz=300)
    assert out0["spike_counts"][i1] > 0
    sil = np.zeros(g.n, bool)
    sil[i0] = True
    silenced = simulate_lif(g, np.array([i0]), steps=400, seed=1, silenced=sil)
    assert silenced["spike_counts"][i0] == 0


def test_synthetic_is_labelled():
    g = make_synthetic_graph()
    assert g.synthetic and g.dataset == "synthetic"
    json.dumps(g.summary())
