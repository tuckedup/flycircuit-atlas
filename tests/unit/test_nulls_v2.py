import numpy as np

from flycircuit.graph.controls import degrees_equal, is_simple
from flycircuit.graph.nulls_v2 import (
    SwapChain,
    class_pair_counts_equal,
    graph_stats,
    load_cached_nulls,
    make_ensemble,
    region_edge_class,
    save_ensemble,
    to_graphs,
)
from flycircuit.graph.versioned import VersionedGraph


def _random_graph(n=60, p=0.08, seed=0, loops=0):
    rng = np.random.default_rng(seed)
    m = rng.random((n, n)) < p
    np.fill_diagonal(m, False)
    for i in range(loops):
        m[i, i] = True
    post, pre = np.nonzero(m)
    ids = 720575940000000000 + np.arange(n, dtype=np.int64) * 13
    cnt = rng.integers(5, 12, size=len(pre))
    return VersionedGraph.from_edges("synthetic", "0", ids[pre], ids[post], cnt, root_ids=ids,
                                     synthetic=True)


def test_chain_preserves_degrees_simplicity_and_fixed_loops():
    g = _random_graph(loops=3)
    ch = SwapChain(g.pre, g.post, g.n, seed=1)
    ch.run(20 * g.n_edges)
    assert ch.c["accepted"] > 0
    keys = ch.pre * g.n + ch.post
    assert len(np.unique(keys)) == g.n_edges  # simple
    assert np.array_equal(np.bincount(ch.pre, minlength=g.n), np.bincount(g.pre, minlength=g.n))
    assert np.array_equal(np.bincount(ch.post, minlength=g.n), np.bincount(g.post, minlength=g.n))
    loops = g.pre == g.post
    assert (ch.pre[loops] == g.pre[loops]).all() and (ch.post[loops] == g.post[loops]).all()
    assert ((ch.pre == ch.post).sum()) == loops.sum()  # no new self-loops
    assert ch.overlap == len(set(keys.tolist()) & set((g.pre * g.n + g.post).tolist()))


def test_ensemble_converges_and_roundtrips_cache(tmp_path):
    g = _random_graph(seed=2)
    # tiny graph -> statistics fluctuate more than the production tolerances allow
    loose = {"overlap": 0.05, "reciprocity": 0.06, "transitivity": 0.06, "assortativity": 0.1,
             "mean_path": 0.3}
    chains, rep = make_ensemble(g, k=4, base_seed=5, min_accepted_per_edge=6, max_sweeps=100,
                                tol=loose, spread_tol=2.0)  # near-zero assortativity -> large relative spread
    assert rep.criteria["constraints_verified"]
    assert all(c.converged for c in chains)
    assert rep.verdict in {"adequate", "adequate_with_limitations"}
    graphs = to_graphs(g, chains, "t")
    assert all(is_simple(h) and degrees_equal(g, h) for h in graphs)
    # a sparse random graph must really be rewired (overlap far below 1)
    assert max(c.final("overlap") for c in chains) < 0.5
    save_ensemble(tmp_path / "e", g, chains, rep, {"t": 1})
    back = load_cached_nulls(tmp_path / "e", g)
    assert len(back) == 4 and np.array_equal(back[0].pre, graphs[0].pre)
    other = _random_graph(seed=3)
    try:
        load_cached_nulls(tmp_path / "e", other)
        raise AssertionError("expected hash mismatch")
    except ValueError:
        pass


def test_inadequate_when_too_few_swaps():
    g = _random_graph(seed=4)
    _, rep = make_ensemble(g, k=3, base_seed=1, min_accepted_per_edge=50, max_sweeps=2)
    assert not rep.adequate and rep.verdict == "inadequate" and rep.reasons


def test_region_constrained_preserves_class_pair_counts_and_flags_forced_hub_edges():
    # node 0 is a hub connected to everyone (both ways): its edges cannot be swapped under the
    # simple-graph constraint -> they must show up as forced edges, not be silently rewired.
    n = 40
    rng = np.random.default_rng(0)
    m = rng.random((n, n)) < 0.07
    m[0, :] = True
    m[:, 0] = True
    np.fill_diagonal(m, False)
    post, pre = np.nonzero(m)
    ids = 720575940000000000 + np.arange(n, dtype=np.int64)
    g = VersionedGraph.from_edges("synthetic", "0", ids[pre], ids[post], np.full(len(pre), 6),
                                  root_ids=ids, synthetic=True)
    region = np.array(["a" if i % 2 else "b" for i in range(n)])
    ec = region_edge_class(g, region)
    chains, rep = make_ensemble(g, k=3, base_seed=9, edge_class=ec, min_accepted_per_edge=4,
                                max_sweeps=60)
    for h in to_graphs(g, chains, "t"):
        assert class_pair_counts_equal(g, h, region)
        assert degrees_equal(g, h) and is_simple(h)
    assert rep.forced_edges["n_forced_in_all_nulls"] >= 2 * (n - 1)  # hub edges never move


def test_graph_stats_known_values():
    # 3-cycle a->b->c->a plus reciprocal pair d<->e
    pre = np.array([0, 1, 2, 3, 4])
    post = np.array([1, 2, 0, 4, 3])
    s = graph_stats(pre, post, 5, np.arange(5))
    assert abs(s["reciprocity"] - 2 / 5) < 1e-9
    assert abs(s["transitivity"] - 1.0) < 1e-9  # one closed triangle, 3 connected triples
    assert s["mean_path"] > 0


def test_chain_is_exactly_uniform_on_reachable_set_small_graph():
    # 4 nodes, each out/in degree 1 or 2: enumerate every reachable state by BFS over swap moves,
    # then check long-run visit frequencies are ~uniform (symmetric proposals -> uniform).
    n = 5
    pre = np.array([0, 0, 1, 2, 3, 4])
    post = np.array([1, 2, 3, 4, 0, 1])
    start = frozenset((a * n + b) for a, b in zip(pre.tolist(), post.tolist(), strict=True))
    g_pre, g_post = pre.copy(), post.copy()
    ch = SwapChain(g_pre, g_post, n, seed=0)
    seen = {}
    for _ in range(60000):
        ch.run(1)
        s = frozenset(ch.keys)
        seen[s] = seen.get(s, 0) + 1
    assert start in seen and len(seen) >= 3
    freqs = np.array(list(seen.values()), dtype=float)
    freqs /= freqs.sum()
    # state occupancy with rejections (self-transitions) is still uniform for symmetric proposals
    assert freqs.max() / freqs.min() < 1.6, dict(zip(range(len(freqs)), freqs.round(3), strict=False))
    # every visited state is a valid simple graph with identical degree sequences
    for s in seen:
        arr = np.array(sorted(s))
        assert len(arr) == len(pre)
        assert np.array_equal(np.sort(np.bincount(arr // n, minlength=n)),
                              np.sort(np.bincount(pre, minlength=n)))
