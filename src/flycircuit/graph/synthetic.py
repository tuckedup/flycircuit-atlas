"""Tiny SYNTHETIC graph fixture. Not biological data; engineering validation only."""

from __future__ import annotations

import numpy as np

from flycircuit.graph.versioned import VersionedGraph

SYNTHETIC_BANNER = "SYNTHETIC DEMO DATA - not a connectome, not a biological result"
# Above 2**53 so any JS-number conversion would silently corrupt IDs.
ID_BASE = 720575940000000000


def make_synthetic_graph(n: int = 40, density: float = 0.08, seed: int = 0) -> VersionedGraph:
    rng = np.random.default_rng(seed)
    root_ids = ID_BASE + np.arange(n, dtype=np.int64) * 7919 + 13
    mask = rng.random((n, n)) < density
    post, pre = np.nonzero(mask)
    counts = rng.integers(1, 11, size=len(pre))
    # Guaranteed directed chain 0 -> 1 -> 2 (reverse edges absent) for direction tests.
    chain_pre = np.array([0, 1])
    chain_post = np.array([1, 2])
    keep = ~(((pre == 1) & (post == 0)) | ((pre == 2) & (post == 1)))
    pre = np.concatenate([pre[keep], chain_pre])
    post = np.concatenate([post[keep], chain_post])
    counts = np.concatenate([counts[keep], [12, 12]])
    kinds = {int(r): ("synthetic_input" if i < 4 else "synthetic_core") for i, r in enumerate(root_ids)}
    g = VersionedGraph.from_edges(
        "synthetic", "0", root_ids[pre], root_ids[post], counts, root_ids=root_ids,
        node_attrs={"cell_type": kinds}, synthetic=True,
    )
    g.set_node_mask("sensory_input", np.arange(n) < 4)
    g.set_node_mask("action_readout", np.arange(n) >= n - 4)
    g.set_edge_mask("trainable", np.ones(g.n_edges, dtype=bool))
    return g
