"""Graph/model builders for screens: a labelled SYNTHETIC graph and real circuit selections.

The real loader reads agent-B outputs (circuits/<name>/internal_ids.parquet + edges_pairs.parquet)
with a pyarrow filter (bounded columns) and induces the internal graph at ``min_contacts``.
That threshold is a MODEL CHOICE. Engineered interface (not biology): sensory sites = a seeded
subset of Kenyon cells; readout = MBONs (or a fallback), both recorded in provenance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flycircuit.config import project_root
from flycircuit.graph.versioned import VersionedGraph

NT_SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}  # assumption; others -> unknown (0)
SYNTH_LABEL = "SYNTHETIC screen graph: random sparse graph, NOT a connectome, not a real-neuron screen"


def synthetic_screen_graph(n: int = 400, deg: int = 12, seed: int = 0) -> VersionedGraph:
    rng = np.random.default_rng(seed)
    keys = np.unique(rng.integers(0, n * n, size=int(n * deg * 1.2), dtype=np.int64))
    keys = keys[(keys // n) != (keys % n)]
    keys = rng.permutation(keys)[: n * deg]  # unique() sorts by post; shuffle before truncating
    pre, post = keys % n, keys // n
    ids = 720575940000000000 + np.arange(n, dtype=np.int64) * 7919 + 13
    ctype = {int(r): f"synthetic_type_{i % 40}" for i, r in enumerate(ids)}
    cclass = {int(r): f"synthetic_class_{i % 8}" for i, r in enumerate(ids)}
    g = VersionedGraph.from_edges(
        "synthetic", "0", ids[pre], ids[post], rng.integers(1, 12, size=len(pre)), root_ids=ids,
        node_attrs={"cell_type": ctype, "cell_class": cclass}, synthetic=True)
    g.set_node_mask("sensory_input", np.arange(g.n) < 16)
    g.set_node_mask("action_readout", np.arange(g.n) >= g.n - 8)
    g.set_edge_mask("trainable", np.ones(g.n_edges, dtype=bool))
    return g


def circuit_dir(circuit: str, dataset="flywire", version="783", root: Path | None = None) -> Path:
    return (root or project_root()) / "data" / "processed" / dataset / version / "circuits" / circuit


LEGACY_SIGN_POLICY = "legacy_top_nt"  # v1 screens: top_nt only (ach/gaba/glu) -> ~1.8% usable signs


def load_circuit_graph(
    circuit: str, dataset: str = "flywire", version: str = "783", min_contacts: int = 5,
    n_sensory: int = 64, seed: int = 0, root: Path | None = None,
    sign_policy="annotation_v2",
) -> tuple[VersionedGraph, dict, np.ndarray]:
    """Induced internal graph for a saved selection. Returns (graph, info, node_sign).

    ``sign_policy``: ``"annotation_v2"`` (default; versioned layer in ``graph/signs.py``: measured
    ``known_nt`` > classifier ``top_nt`` > explicit priors; monoamines = modulatory, fast sign 0), a
    ``SignPolicy`` instance (sensitivity variants), or ``"legacy_top_nt"`` (the v1 behaviour, kept
    only to reproduce/contrast the v1 screens). Signs of 0 mean "silent fast output": pair them with
    ``unknown_sign_policy='zero'`` when building the model (the unknown fill is already inside the
    v2 policy)."""
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    cdir = circuit_dir(circuit, dataset, version, root)
    if not (cdir / "internal_ids.parquet").exists():
        raise FileNotFoundError(
            f"selection '{circuit}' not found at {cdir}; run `flycircuit data select` first")
    nodes = pq.read_table(cdir / "internal_ids.parquet").to_pandas()
    ids = nodes["root_id"].to_numpy(dtype=np.int64)
    pairs = cdir.parent.parent / "edges_pairs.parquet"
    flt = (ds.field("pre").isin(ids.tolist()) & ds.field("post").isin(ids.tolist())
           & (ds.field("count") >= min_contacts))
    tab = ds.dataset(pairs).to_table(columns=["pre", "post", "count"], filter=flt)
    attrs = {c: dict(zip(nodes["root_id"].tolist(), nodes[c].tolist(), strict=True))
             for c in ("cell_type", "cell_class", "side", "top_nt", "grp") if c in nodes}
    g = VersionedGraph.from_edges(
        dataset, version, tab["pre"].to_numpy(), tab["post"].to_numpy(),
        tab["count"].to_numpy(), root_ids=ids, node_attrs=attrs, synthetic=False)
    grp = np.array([str(x) for x in g.node_attrs.get("grp", [""] * g.n)])
    rng = np.random.default_rng(seed)
    kc = np.flatnonzero(grp == "kenyon")
    sens = np.sort(rng.choice(kc, size=min(n_sensory, len(kc)), replace=False)) if len(kc) else np.arange(0)
    read = np.flatnonzero(grp == "mbon")
    g.set_node_mask("sensory_input", np.isin(np.arange(g.n), sens))
    g.set_node_mask("action_readout", np.isin(np.arange(g.n), read))
    g.set_edge_mask("trainable", np.ones(g.n_edges, dtype=bool))
    sign_extra: dict = {}
    if sign_policy == LEGACY_SIGN_POLICY:
        nt = [str(x).lower() for x in g.node_attrs.get("top_nt", [""] * g.n)]
        sign = np.array([NT_SIGN.get(x, 0) for x in nt], dtype=np.float32)
        sign_extra = {"sign_policy": LEGACY_SIGN_POLICY,
                      "sign_note": "v1 legacy: top_nt ach=+1,gaba=-1,glu=-1 only; ignores known_nt"}
    else:
        from flycircuit.graph.signs import SignPolicy, load_assignment_for_ids

        spec = SignPolicy() if sign_policy == "annotation_v2" else sign_policy
        if not isinstance(spec, SignPolicy):
            raise ValueError("sign_policy must be 'annotation_v2', 'legacy_top_nt' or a SignPolicy")
        asg = load_assignment_for_ids(g.root_ids, spec)
        sign = asg.sign.astype(np.float32)
        g.node_attrs["modulatory"] = [bool(v) for v in asg.modulatory]
        sign_extra = {"sign_policy": spec.tag(), "sign_assignment_hash": asg.hash(),
                      "sign_summary": asg.summary,
                      "sign_note": "annotation_v2: monoamine neurons = modulatory (fast sign 0); "
                                   "pair with unknown_sign_policy='zero'"}
    info = {
        "circuit": circuit, "is_pilot_subset": "pilot" in circuit, "min_contacts": min_contacts,
        "n_nodes": g.n, "n_edges": g.n_edges, "n_sensory_sites": int(len(sens)),
        "n_readout_nodes": int(len(read)), "unknown_sign_nodes": int((sign == 0).sum()),
        "mapping_hash": g.mapping_hash(), "edge_hash": g.edge_hash(),
        "interface_note": "sensory sites = seeded KC subset; readout = MBONs (engineered interface)",
    }
    info.update(sign_extra)
    return g, info, sign
