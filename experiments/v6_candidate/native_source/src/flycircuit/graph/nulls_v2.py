"""Null graphs v2: converged, multi-chain, diagnosed degree-preserving rewiring (protocol ``nulls-v2``).

Supersedes the *adequacy test* of ``controls.directed_double_edge_swap`` (``mixing_ratio <= 1.5``
against a configuration-model overlap that is not attainable for simple graphs with hub blocks).
``controls.py`` is untouched so historical runs stay reproducible.

Constraints are identical and exact: simple digraph, pre-existing self-loops held fixed, no new
self-loops/duplicates, in/out-degrees (and with ``edge_class``, the class-pair edge counts)
preserved; weights stay attached to the edge slot. Nothing is relaxed to obtain mixing.

Diagnostics per chain (one chain = one null graph, independently seeded, started from the original):
* swaps are applied in sweeps of E attempts; after every sweep we record accepted-swaps-so-far,
  overlap with the original, reciprocity, undirected transitivity, out->in degree assortativity and
  a sampled directed mean shortest path;
* a chain is *converged* when it has >= ``min_accepted_per_edge`` * E ACCEPTED swaps and every tracked
  statistic has moved less than its tolerance over the last ``window`` sweeps;
* ensemble adequacy additionally needs (a) between-chain relative spread of the final statistics
  below tolerance, (b) a restart test (continuing a converged null shows no drift beyond the
  between-null spread) and (c) forced-edge accounting (edges present in every null). A forced edge
  fraction is a limitation of the simple-graph constraint, and is reported, not hidden.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

from flycircuit.graph.controls import _rebuild, degrees_equal, is_simple
from flycircuit.graph.versioned import VersionedGraph

PROTOCOL = "nulls-v2"
TRACKED = ("overlap", "reciprocity", "transitivity", "assortativity", "mean_path")


def graph_stats(pre: np.ndarray, post: np.ndarray, n: int, sources: np.ndarray) -> dict[str, float]:
    e = len(pre)
    keys = pre.astype(np.int64) * n + post
    rev = post.astype(np.int64) * n + pre
    nonloop = pre != post
    reciprocity = float(np.isin(rev[nonloop], keys).mean()) if nonloop.any() else float("nan")
    a = sp.csr_matrix((np.ones(e, dtype=np.float32), (pre, post)), shape=(n, n))
    au = ((a + a.T) > 0).astype(np.float32).tolil()
    au.setdiag(0)
    au = au.tocsr()
    au.eliminate_zeros()
    deg = np.asarray(au.sum(1)).ravel()
    w = float((deg * (deg - 1) / 2).sum())
    closed = float(((au @ au).multiply(au)).sum())
    transitivity = closed / (2 * w) if w > 0 else float("nan")
    dout = np.bincount(pre, minlength=n).astype(float)
    din = np.bincount(post, minlength=n).astype(float)
    x, y = dout[pre], din[post]
    assort = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    d = shortest_path(a, method="D", directed=True, unweighted=True, indices=sources)
    fin = np.isfinite(d) & (d > 0)
    mean_path = float(d[fin].mean()) if fin.any() else float("nan")
    return {"reciprocity": reciprocity, "transitivity": transitivity,
            "assortativity": assort, "mean_path": mean_path,
            "reach_fraction": float(fin.mean())}


class SwapChain:
    """Resumable exact double-edge-swap Markov chain (symmetric proposals; rejection on violation)."""

    def __init__(self, pre, post, n, seed, edge_class=None):
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.pre = np.asarray(pre).astype(np.int64).copy()
        self.post = np.asarray(post).astype(np.int64).copy()
        self.e = len(self.pre)
        self.orig_keys = frozenset((self.pre * n + self.post).tolist())
        self.keys = set(self.orig_keys)
        loops = self.pre == self.post
        movable = np.flatnonzero(~loops)
        self.fixed_loops = int(loops.sum())
        if edge_class is None:
            groups = [movable]
        else:
            ec = np.asarray(edge_class)[movable]
            groups = [movable[ec == c] for c in np.unique(ec)]
        self.groups = [g for g in groups if len(g) >= 2]
        sizes = np.array([len(g) for g in self.groups], dtype=float)
        self.probs = sizes / sizes.sum() if len(sizes) else sizes
        self.overlap = self.e
        self.c = {"attempted": 0, "accepted": 0, "rej_same_endpoint": 0, "rej_self_loop": 0,
                  "rej_duplicate": 0}
        self.swappable_edges = int(sum(len(g) for g in self.groups))

    def run(self, k: int) -> None:
        if not self.groups:
            return
        pre, post, keys, orig, n = self.pre, self.post, self.keys, self.orig_keys, self.n
        c = self.c
        gi = (np.zeros(k, dtype=np.int64) if len(self.groups) == 1
              else self.rng.choice(len(self.groups), size=k, p=self.probs))
        u = self.rng.random((k, 2))
        ov = self.overlap
        for t in range(k):
            g = self.groups[gi[t]]
            L = len(g)
            i = g[int(u[t, 0] * L)]
            j = g[int(u[t, 1] * L)]
            c["attempted"] += 1
            a, b, cc, d = int(pre[i]), int(post[i]), int(pre[j]), int(post[j])
            if i == j or a == cc or b == d:
                c["rej_same_endpoint"] += 1
            elif a == d or cc == b:
                c["rej_self_loop"] += 1
            else:
                k1, k2 = a * n + d, cc * n + b
                if k1 in keys or k2 in keys:
                    c["rej_duplicate"] += 1
                else:
                    o1, o2 = a * n + b, cc * n + d
                    keys.discard(o1)
                    keys.discard(o2)
                    keys.add(k1)
                    keys.add(k2)
                    ov += (k1 in orig) + (k2 in orig) - (o1 in orig) - (o2 in orig)
                    post[i], post[j] = d, b
                    c["accepted"] += 1
        self.overlap = ov


@dataclass
class ChainResult:
    seed: int
    pre: np.ndarray
    post: np.ndarray
    trajectory: list[dict]
    counters: dict
    converged: bool
    reason: str
    swappable_edges: int

    def final(self, name: str) -> float:
        return self.trajectory[-1][name]


def run_chain(g: VersionedGraph, seed: int, edge_class=None, *, min_accepted_per_edge: float = 10.0,
              max_sweeps: int = 120, window: int = 4, tol: dict | None = None,
              n_path_sources: int = 64, start: tuple | None = None) -> ChainResult:
    tol = {"overlap": 0.01, "reciprocity": 0.01, "transitivity": 0.01, "assortativity": 0.02,
           "mean_path": 0.05, **(tol or {})}
    pre0, post0 = (g.pre, g.post) if start is None else start
    ch = SwapChain(pre0, post0, g.n, seed, edge_class)
    srng = np.random.default_rng(10_000 + seed)
    sources = np.sort(srng.choice(g.n, size=min(n_path_sources, g.n), replace=False))
    traj: list[dict] = []

    def record():
        s = graph_stats(ch.pre, ch.post, g.n, sources)
        s.update({"sweep": len(traj), "accepted": ch.c["accepted"], "attempted": ch.c["attempted"],
                  "overlap": ch.overlap / ch.e})
        traj.append(s)

    record()
    converged, reason = False, "max_sweeps_reached"
    if not ch.groups:
        return ChainResult(seed, ch.pre, ch.post, traj, ch.c, False, "fewer than 2 swappable edges", 0)
    for _ in range(max_sweeps):
        ch.run(ch.e)
        record()
        if ch.c["accepted"] < min_accepted_per_edge * ch.e or len(traj) <= window:
            continue
        ok = all(np.nanmax([t[k] for t in traj[-window:]]) - np.nanmin([t[k] for t in traj[-window:]])
                 <= tol[k] for k in TRACKED)
        if ok:
            converged, reason = True, "plateau"
            break
    return ChainResult(seed, ch.pre, ch.post, traj, dict(ch.c), converged, reason,
                       ch.swappable_edges)


@dataclass
class EnsembleReport:
    protocol: str = PROTOCOL
    label: str = ""
    n_nodes: int = 0
    n_edges: int = 0
    n_nulls: int = 0
    adequate: bool = False  # chains mixed/converged and constraints verified (see ``verdict``)
    verdict: str = "unassessed"  # adequate | adequate_with_limitations (>20% forced edges) | inadequate
    reasons: list = field(default_factory=list)
    criteria: dict = field(default_factory=dict)
    per_null: list = field(default_factory=list)
    graph_stats_original: dict = field(default_factory=dict)
    graph_stats_nulls: dict = field(default_factory=dict)
    forced_edges: dict = field(default_factory=dict)
    limitations: list = field(default_factory=list)


def make_ensemble(g: VersionedGraph, k: int = 10, base_seed: int = 2000, edge_class=None, label="",
                  spread_tol: float = 0.05, **kw) -> tuple[list[ChainResult], EnsembleReport]:
    chains = [run_chain(g, base_seed + 7919 * i, edge_class, **kw) for i in range(k)]
    sources = np.sort(np.random.default_rng(1).choice(g.n, size=min(64, g.n), replace=False))
    orig = graph_stats(g.pre, g.post, g.n, sources)
    rep = EnsembleReport(label=label, n_nodes=g.n, n_edges=g.n_edges, n_nulls=k,
                         graph_stats_original=orig)
    finals = {name: np.array([c.final(name) for c in chains], dtype=float) for name in TRACKED}
    rep.graph_stats_nulls = {n: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)),
                                 "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}
                             for n, v in finals.items() if np.isfinite(v).any()}
    rep.per_null = [{"seed": c.seed, "converged": c.converged, "reason": c.reason,
                     "sweeps": len(c.trajectory) - 1, **c.counters,
                     "acceptance_rate": c.counters["accepted"] / max(1, c.counters["attempted"]),
                     "accepted_per_edge": c.counters["accepted"] / g.n_edges,
                     **{f"final_{n}": float(finals[n][i]) for n in TRACKED}}
                    for i, c in enumerate(chains)]
    reasons = []
    c1 = all(c.counters["accepted"] >= kw.get("min_accepted_per_edge", 10.0) * g.n_edges
             for c in chains)
    c2 = all(c.converged for c in chains)
    # (a) between-chain spread: std / max(|mean|, 0.05) for each tracked statistic
    spread = {}
    for n, v in finals.items():
        if np.isfinite(v).all():
            spread[n] = float(np.std(v) / max(abs(np.mean(v)), 0.05))
    c3 = all(s <= spread_tol for s in spread.values())
    # (b) restart test on first null: continue 3 sweeps; drift must be <= 3x between-null std (+tiny)
    drift = {}
    if chains[0].converged:
        rs = run_chain(g, chains[0].seed + 1, edge_class, min_accepted_per_edge=0, max_sweeps=3,
                       window=99, start=(chains[0].pre, chains[0].post))
        for n in ("reciprocity", "transitivity", "assortativity", "mean_path"):
            a, b = chains[0].final(n), rs.final(n)
            if np.isfinite(a) and np.isfinite(b):
                drift[n] = {"drift": abs(b - a), "allowed": 3 * float(np.std(finals[n])) + 1e-3}
    c4 = bool(drift) and all(v["drift"] <= v["allowed"] for v in drift.values())
    # (c) forced edges (present in every null) = constraint-limited structure
    ek = np.stack([np.sort(c.pre.astype(np.int64) * g.n + c.post) for c in chains])
    from functools import reduce

    common = reduce(np.intersect1d, list(ek))
    rep.forced_edges = {
        "n_forced_in_all_nulls": int(len(common)),
        "fraction_of_edges": float(len(common) / g.n_edges),
        "note": "edges retained in every null; high fractions mean the exact simple-graph degree "
                "constraint (hub / near-complete blocks) leaves little to rewire",
    }
    rep.criteria = {
        "C1_accepted_swaps_ge_min_per_edge": c1,
        "C2_every_chain_plateaued": c2,
        "C3_between_chain_spread_ok": {"ok": c3, "spread": spread, "tol": spread_tol},
        "C4_restart_no_drift": {"ok": c4, "drift": drift},
        "constraints_verified": all(degrees_equal(g, _rebuild(g, c.pre, c.post, g.count, "x"))
                                    for c in chains)
        and all(len(np.unique(c.pre * g.n + c.post)) == g.n_edges for c in chains),
    }
    if not c1:
        reasons.append("fewer accepted swaps per edge than required in at least one chain")
    if not c2:
        reasons.append("at least one chain did not plateau within max_sweeps")
    if not c3:
        reasons.append("between-chain spread of final statistics exceeds tolerance")
    if not c4:
        reasons.append("restart test showed drift (or could not run)")
    if not rep.criteria["constraints_verified"]:
        reasons.append("DEGREE/SIMPLE-GRAPH CONSTRAINT VIOLATED")
    rep.reasons = reasons
    rep.adequate = not reasons
    if rep.forced_edges["fraction_of_edges"] > 0.2:
        rep.limitations.append(
            f"{rep.forced_edges['fraction_of_edges']:.0%} of edges are identical in all nulls: the "
            "degree-preserving ensemble cannot separate topology from degree structure there")
    rep.verdict = ("inadequate" if reasons else
                   "adequate_with_limitations" if rep.forced_edges["fraction_of_edges"] > 0.2
                   else "adequate")
    if chains[0].swappable_edges < g.n_edges:
        rep.limitations.append(
            f"{g.n_edges - chains[0].swappable_edges} edges are not swappable "
            "(fixed self-loops / class-singleton edges)")
    return chains, rep


def to_graphs(g: VersionedGraph, chains: list[ChainResult], label: str) -> list[VersionedGraph]:
    return [_rebuild(g, c.pre.copy(), c.post.copy(), g.count.copy(), label) for c in chains]


def class_pair_counts_equal(g: VersionedGraph, h: VersionedGraph, region: np.ndarray) -> bool:
    def tab(x):
        pair = np.stack([region[x.pre], region[x.post]], 1)
        u, c = np.unique(pair.astype(str), axis=0, return_counts=True)
        return {tuple(a): int(b) for a, b in zip(u.tolist(), c.tolist(), strict=True)}

    return tab(g) == tab(h)


def region_edge_class(g: VersionedGraph, region: np.ndarray) -> np.ndarray:
    pair = np.stack([region[g.pre].astype(str), region[g.post].astype(str)], 1)
    _, cls = np.unique(pair, axis=0, return_inverse=True)
    return cls.ravel()


def save_ensemble(outdir: Path, g: VersionedGraph, chains, rep: EnsembleReport, meta: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(chains):
        np.savez_compressed(outdir / f"null_{i:02d}.npz", pre=c.pre.astype(np.int32),
                            post=c.post.astype(np.int32), seed=c.seed,
                            trajectory=json.dumps(c.trajectory))
    doc = {"protocol": PROTOCOL, "meta": meta, "mapping_hash": g.mapping_hash(),
           "edge_hash": g.edge_hash(), "report": rep.__dict__}
    (outdir / "report.json").write_text(json.dumps(doc, indent=1, default=float))


def load_cached_nulls(outdir: Path, g: VersionedGraph, label: str = "nulls_v2") -> list[VersionedGraph]:
    """Reload cached nulls for ``g`` (verifies the base graph hashes first)."""
    doc = json.loads((Path(outdir) / "report.json").read_text())
    if doc["mapping_hash"] != g.mapping_hash() or doc["edge_hash"] != g.edge_hash():
        raise ValueError(f"cached nulls in {outdir} were made for a different graph (hash mismatch)")
    out = []
    for f in sorted(Path(outdir).glob("null_*.npz")):
        z = np.load(f)
        h = _rebuild(g, z["pre"].astype(np.int64), z["post"].astype(np.int64), g.count.copy(), label)
        if not (is_simple(h) and degrees_equal(g, h)):
            raise ValueError(f"{f} violates the null constraints")
        out.append(h)
    return out
