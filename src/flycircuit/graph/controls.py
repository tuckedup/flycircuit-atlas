"""Null-graph controls. All operate on edge arrays; nothing densifies.

Rules (declared, tested):
* Graphs are SIMPLE directed: no duplicate (pre, post) pairs.
* Self-loops: pre-existing self-loops are held FIXED (excluded from swapping, so in/out degrees
  stay exact); swaps never create new self-loops.
* Edge weights (contact counts) stay attached to their edge *slot* (i.e. the multiset of weights
  is preserved, the slot's endpoints are rewired).
* A swap that would create a duplicate edge or a new self-loop is REJECTED, never repaired.
  Constraints are never silently relaxed; ``stats`` records every rejection reason.

Controls provided:
  directed_double_edge_swap   exact in/out-degree preserving rewire (``edge_class=None``)
  region_constrained_swap     same, within (region[pre], region[post]) classes: also preserves
                              the region-pair edge-count matrix
  weight_sign_permutation     FIXED topology; permutes weights (and optionally node signs).
                              NOT a topology rewire.
  random_sparsity             uniform random simple digraph with the same N and E (degree
                              sequence NOT preserved)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flycircuit.graph.versioned import VersionedGraph


@dataclass
class ControlResult:
    graph: VersionedGraph
    label: str
    stats: dict


def _expected_overlap_fraction(pre, post, n) -> float:
    dout = np.bincount(pre, minlength=n).astype(float)
    din = np.bincount(post, minlength=n).astype(float)
    e = len(pre)
    return float((dout[pre] * din[post]).sum() / max(1, e) / max(1, e))


def _swap_arrays(pre, post, n, n_attempts, seed, edge_class=None, chunks=10):
    rng = np.random.default_rng(seed)
    pre = pre.copy()
    post = post.copy()
    e = len(pre)
    orig_keys = set((pre.astype(np.int64) * n + post).tolist())
    keys = set(orig_keys)
    loops = pre == post
    movable = np.flatnonzero(~loops)
    stats = {"attempted": 0, "accepted": 0, "rej_same_endpoint": 0, "rej_self_loop": 0,
             "rej_duplicate": 0, "n_edges": e, "fixed_self_loops": int(loops.sum()),
             "overlap_curve": []}
    if len(movable) < 2:
        stats["note"] = "fewer than 2 movable edges"
        return pre, post, stats
    if edge_class is None:
        groups = [movable]
    else:
        ec = np.asarray(edge_class)[movable]
        groups = [movable[ec == c] for c in np.unique(ec)]
        groups = [g for g in groups if len(g) >= 2]
        if not groups:
            stats["note"] = "no region class with >=2 edges"
            return pre, post, stats
    sizes = np.array([len(g) for g in groups], dtype=float)
    probs = sizes / sizes.sum()
    overlap = e  # every current edge is an original edge at start
    per_chunk = max(1, n_attempts // chunks)
    exp_frac = _expected_overlap_fraction(pre, post, n)
    for att in range(n_attempts):
        g = groups[0] if len(groups) == 1 else groups[rng.choice(len(groups), p=probs)]
        i, j = rng.integers(0, len(g), size=2)
        i, j = g[i], g[j]
        stats["attempted"] += 1
        a, b, c, d = int(pre[i]), int(post[i]), int(pre[j]), int(post[j])
        if i == j or a == c or b == d:
            stats["rej_same_endpoint"] += 1
        elif a == d or c == b:
            stats["rej_self_loop"] += 1
        else:
            k1, k2 = a * n + d, c * n + b
            if k1 in keys or k2 in keys:
                stats["rej_duplicate"] += 1
            else:
                old1, old2 = a * n + b, c * n + d
                keys.discard(old1)
                keys.discard(old2)
                keys.add(k1)
                keys.add(k2)
                overlap += (k1 in orig_keys) + (k2 in orig_keys)
                overlap -= (old1 in orig_keys) + (old2 in orig_keys)
                post[i], post[j] = d, b
                stats["accepted"] += 1
        if (att + 1) % per_chunk == 0:
            stats["overlap_curve"].append((att + 1, overlap / e))
    stats["acceptance_rate"] = stats["accepted"] / max(1, stats["attempted"])
    stats["final_overlap_with_original"] = overlap / e
    stats["expected_overlap_if_fully_mixed"] = exp_frac
    stats["fraction_original_edges_retained"] = overlap / e
    stats["mixing_ratio"] = (overlap / e) / exp_frac if exp_frac > 0 else float("nan")
    return pre, post, stats


def _rebuild(g: VersionedGraph, pre, post, count, label: str) -> VersionedGraph:
    order = np.lexsort((pre, post))  # canonical (post, pre)
    ng = VersionedGraph(
        g.dataset, g.version, g.root_ids, pre[order], post[order], count[order],
        {k: list(v) for k, v in g.node_attrs.items()}, g.synthetic,
    )
    for name, m in g.node_masks.items():
        ng.node_masks[name] = m.copy()
    ng.edge_masks["trainable"] = np.full(ng.n_edges, bool(g.edge_masks["trainable"].any()))
    ng.control_label = label  # type: ignore[attr-defined]
    return ng


def directed_double_edge_swap(
    g: VersionedGraph, seed: int = 0, swaps_per_edge: float = 10.0, edge_class=None,
    max_mixing_ratio: float = 1.5,
) -> ControlResult:
    n_attempts = int(swaps_per_edge * g.n_edges)
    pre, post, stats = _swap_arrays(g.pre, g.post, g.n, n_attempts, seed, edge_class)
    stats["seed"] = seed
    stats["mixing_adequate"] = bool(
        stats.get("mixing_ratio", float("inf")) <= max_mixing_ratio
    ) if "mixing_ratio" in stats else False
    label = "double_edge_swap" if edge_class is None else "region_constrained_swap"
    return ControlResult(_rebuild(g, pre, post, g.count.copy(), label), label, stats)


def region_constrained_swap(g: VersionedGraph, region: np.ndarray, seed: int = 0,
                            swaps_per_edge: float = 10.0) -> ControlResult:
    """Swap only edges that share the same (region[pre], region[post]) class."""
    region = np.asarray(region)
    if region.shape != (g.n,):
        raise ValueError("region must have one label per node")
    # encode class ids without a dense R x R table
    pair = np.stack([region[g.pre], region[g.post]], axis=1)
    _, cls = np.unique(pair, axis=0, return_inverse=True)
    res = directed_double_edge_swap(g, seed, swaps_per_edge, edge_class=cls.ravel())
    res.stats["constraint"] = "region-pair edge counts preserved"
    return res


def weight_sign_permutation(g: VersionedGraph, seed: int = 0, sign: np.ndarray | None = None
                            ) -> tuple[ControlResult, np.ndarray | None]:
    """FIXED-TOPOLOGY control: permutes edge weights (and node signs if given). NOT a rewire."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(g.n_edges)
    ng = _rebuild(g, g.pre, g.post, g.count[perm], "weight_sign_permutation")
    new_sign = None if sign is None else np.asarray(sign)[rng.permutation(g.n)]
    stats = {"seed": seed, "topology_changed": False,
             "note": "weight/sign permutation on fixed topology; not a topology rewire"}
    return ControlResult(ng, "weight_sign_permutation", stats), new_sign


def random_sparsity(g: VersionedGraph, seed: int = 0, allow_self_loops: bool = False
                    ) -> ControlResult:
    """Uniform random simple digraph with the same N and E; weights resampled from g.count."""
    rng = np.random.default_rng(seed)
    n, e = g.n, g.n_edges
    max_e = n * n - (0 if allow_self_loops else n)
    if e > max_e:
        raise ValueError("requested edge count exceeds simple-digraph capacity")
    keys = np.empty(0, dtype=np.int64)
    while len(keys) < e:
        need = int((e - len(keys)) * 1.3) + 16
        cand = rng.integers(0, n * n, size=need, dtype=np.int64)
        if not allow_self_loops:
            cand = cand[(cand // n) != (cand % n)]
        keys = np.unique(np.concatenate([keys, cand]))
    keys = rng.permutation(keys)[:e]
    pre, post = keys // n, keys % n
    count = rng.permutation(g.count)
    ng = _rebuild(g, pre, post, count, "random_sparsity")
    stats = {"seed": seed, "n_edges": e, "density": e / (n * n),
             "degree_sequence_preserved": False, "self_loops": int((pre == post).sum())}
    return ControlResult(ng, "random_sparsity", stats)


def degrees_equal(a: VersionedGraph, b: VersionedGraph) -> bool:
    return bool(
        np.array_equal(np.bincount(a.pre, minlength=a.n), np.bincount(b.pre, minlength=b.n))
        and np.array_equal(np.bincount(a.post, minlength=a.n), np.bincount(b.post, minlength=b.n))
    )


def is_simple(g: VersionedGraph) -> bool:
    return len(np.unique(g.pre.astype(np.int64) * g.n + g.post)) == g.n_edges


def null_ensemble(g: VersionedGraph, k: int = 5, base_seed: int = 0, **kw) -> list[ControlResult]:
    """k independently seeded degree-preserving nulls (confirmatory: k>=5). Every result keeps
    its own failure/acceptance/mixing stats; inadequate mixing is flagged, not hidden."""
    return [directed_double_edge_swap(g, seed=base_seed + 1000 * i + 1, **kw) for i in range(k)]
