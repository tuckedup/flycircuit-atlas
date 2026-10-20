import numpy as np
import pytest

from flycircuit.graph.controls import (
    degrees_equal,
    directed_double_edge_swap,
    is_simple,
    null_ensemble,
    random_sparsity,
    region_constrained_swap,
    weight_sign_permutation,
)
from flycircuit.graph.synthetic import make_synthetic_graph


@pytest.fixture
def g():
    return make_synthetic_graph(60, 0.12, seed=3)


def test_swap_preserves_exact_degrees_and_simplicity(g):
    r = directed_double_edge_swap(g, seed=1, swaps_per_edge=10)
    assert degrees_equal(g, r.graph) and is_simple(r.graph)
    assert r.graph.n_edges == g.n_edges
    assert r.stats["accepted"] > 0 and r.stats["attempted"] == 10 * g.n_edges
    assert r.stats["accepted"] + r.stats["rej_self_loop"] + r.stats["rej_duplicate"] \
        + r.stats["rej_same_endpoint"] == r.stats["attempted"]
    assert (r.graph.pre != r.graph.post).sum() >= g.n_edges - g.n_self_loops
    assert r.graph.n_self_loops == g.n_self_loops


def test_swap_actually_rewires_and_keeps_weight_multiset(g):
    r = directed_double_edge_swap(g, seed=2)
    assert r.graph.edge_hash() != g.edge_hash()
    assert sorted(r.graph.count.tolist()) == sorted(g.count.tolist())
    assert r.stats["fraction_original_edges_retained"] < 0.5


def test_existing_self_loops_are_fixed():
    from flycircuit.graph.versioned import VersionedGraph
    ids = np.arange(10, dtype=np.int64) + 100
    pre = np.array([0, 1, 2, 3, 4, 5, 6, 7, 1, 8])
    post = np.array([1, 2, 3, 4, 5, 6, 7, 0, 1, 8])  # two self loops: (1,1) (8,8)
    gg = VersionedGraph.from_edges("t", "0", ids[pre], ids[post], np.ones(10), root_ids=ids)
    r = directed_double_edge_swap(gg, seed=0)
    assert r.graph.n_self_loops == 2 and degrees_equal(gg, r.graph)
    assert r.stats["fixed_self_loops"] == 2


def test_determinism_and_independent_seeds(g):
    a = directed_double_edge_swap(g, seed=5).graph.edge_hash()
    b = directed_double_edge_swap(g, seed=5).graph.edge_hash()
    c = directed_double_edge_swap(g, seed=6).graph.edge_hash()
    assert a == b and a != c


def test_region_constrained_preserves_region_pair_matrix(g):
    region = np.arange(g.n) % 3
    r = region_constrained_swap(g, region, seed=1)

    def mat(x):
        m = np.zeros((3, 3), int)
        np.add.at(m, (region[x.pre], region[x.post]), 1)
        return m

    assert np.array_equal(mat(g), mat(r.graph))
    assert degrees_equal(g, r.graph) and is_simple(r.graph)
    assert r.stats["accepted"] > 0


def test_weight_sign_permutation_is_not_a_rewire(g):
    sign = np.where(np.arange(g.n) % 2, 1, -1)
    r, new_sign = weight_sign_permutation(g, seed=1, sign=sign)
    assert np.array_equal(r.graph.pre, g.pre) and np.array_equal(r.graph.post, g.post)
    assert not np.array_equal(r.graph.count, g.count)
    assert sorted(new_sign.tolist()) == sorted(sign.tolist())
    assert r.stats["topology_changed"] is False


def test_random_sparsity_size_and_simple(g):
    r = random_sparsity(g, seed=1)
    assert r.graph.n == g.n and r.graph.n_edges == g.n_edges
    assert is_simple(r.graph) and r.graph.n_self_loops == 0
    assert r.stats["degree_sequence_preserved"] is False


def test_ensemble_has_five_independent_nulls_with_mixing_stats(g):
    nulls = null_ensemble(g, k=5)
    hashes = {n.graph.edge_hash() for n in nulls}
    assert len(hashes) == 5
    for n in nulls:
        assert "acceptance_rate" in n.stats and "mixing_ratio" in n.stats
        assert degrees_equal(g, n.graph)
    assert all(n.stats["mixing_adequate"] for n in nulls)


def test_inadequate_mixing_is_flagged_not_hidden(g):
    r = directed_double_edge_swap(g, seed=1, swaps_per_edge=0.01)
    assert r.stats["mixing_adequate"] is False
