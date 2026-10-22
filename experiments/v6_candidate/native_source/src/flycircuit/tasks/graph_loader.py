"""Learning-circuit loaders: a SYNTHETIC KC/MBON-like graph and the real FlyWire v783 pilot circuit.

Sensory interface = Kenyon cells (engineered: task cues are injected into KCs through a fixed random
sparse projection; in the animal KCs receive PN input, which this circuit does not include).
Readout = MBONs (decoder to 2 actions is an engineered interface).
Plastic mask = every edge whose postsynaptic neuron is a readout node (includes KC->MBON, the
documented plastic synapse class; the same rule on a degree-preserving null keeps the edge count
EXACTLY because in-degrees are preserved).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flycircuit.config import project_root
from flycircuit.graph.versioned import VersionedGraph

NEUROMOD = {"dopamine", "serotonin", "octopamine"}


@dataclass
class CircuitBundle:
    graph: VersionedGraph
    sign: np.ndarray  # (n,) in {-1,0,+1}; 0 = unknown
    sensory_idx: np.ndarray
    readout_idx: np.ndarray
    label: str
    notes: dict
    modulatory: np.ndarray | None = None  # (n,) bool monoamine neurons (sign policy v2 only)

    @property
    def plastic_edge_mask(self) -> np.ndarray:
        return np.isin(self.graph.post, self.readout_idx)

    def with_graph(self, g: VersionedGraph, label: str) -> CircuitBundle:
        return CircuitBundle(g, self.sign, self.sensory_idx, self.readout_idx, label, self.notes,
                             self.modulatory)


def make_learning_synthetic(n_kc: int = 200, n_mbon: int = 12, n_aux: int = 40, seed: int = 0,
                            p_kc_mbon: float = 0.4, p_other: float = 0.03) -> CircuitBundle:
    """SYNTHETIC KC/MBON-like circuit for engineering tests. Not a biological result."""
    rng = np.random.default_rng(seed)
    n = n_kc + n_mbon + n_aux
    kc = np.arange(n_kc)
    mb = np.arange(n_kc, n_kc + n_mbon)
    pre, post = [], []
    a = rng.random((n_mbon, n_kc)) < p_kc_mbon
    pm, pk = np.nonzero(a)
    pre += list(kc[pk])
    post += list(mb[pm])
    m = rng.random((n, n)) < p_other
    np.fill_diagonal(m, False)
    m[:n_kc, :n_kc] = rng.random((n_kc, n_kc)) < p_other / 2
    np.fill_diagonal(m[:n_kc, :n_kc], False)
    q, p = np.nonzero(m)
    pre += list(p)
    post += list(q)
    key = np.unique(np.array(post, dtype=np.int64) * n + np.array(pre, dtype=np.int64))
    post_a, pre_a = key // n, key % n
    ids = 720575940000000000 + np.arange(n, dtype=np.int64) * 31 + 7
    cnt = rng.integers(5, 16, size=len(key))
    g = VersionedGraph.from_edges("synthetic", "0", ids[pre_a], ids[post_a], cnt, root_ids=ids,
                                  synthetic=True)
    sign = np.ones(n, dtype=np.float32)
    sign[n_kc + n_mbon:] = -1.0  # inhibitory auxiliary pool
    g.set_node_mask("sensory_input", np.isin(np.arange(n), kc))
    g.set_node_mask("action_readout", np.isin(np.arange(n), mb))
    g.set_edge_mask("trainable", np.ones(g.n_edges, dtype=bool))
    return CircuitBundle(g, sign, kc, mb, "synthetic_kc_mbon",
                         {"synthetic": True, "n_kc": n_kc, "n_mbon": n_mbon})


def _sign_from_annotations(known_nt, top_nt, top_conf) -> tuple[np.ndarray, dict]:
    """Per-neuron sign. Prefer literature ``known_nt`` (first token); fall back to predicted
    ``top_nt`` only when confidence >= 0.5 and it is not a neuromodulator. ASSUMPTION (declared in
    docs/learning.md): acetylcholine -> +1; GABA and glutamate -> -1; everything else 0 (unknown).
    NOTE: FlyWire v783 ``top_nt`` labels ~all Kenyon cells 'dopamine' (prediction artifact; the
    literature value is acetylcholine), which is why ``known_nt`` takes precedence."""
    def one(k, t, c):
        k = (k or "").split(";")[0].strip().lower() if isinstance(k, str) else ""
        if k.startswith("acetylcholine"):
            return 1, "known"
        if k in ("gaba", "glutamate"):
            return -1, "known"
        if k:
            return 0, "known_unmapped"
        t = t.lower() if isinstance(t, str) else ""
        if t in NEUROMOD or not t or not (c is not None and c >= 0.5):
            return 0, "unknown"
        return (1 if t == "acetylcholine" else -1 if t in ("gaba", "glutamate") else 0), "predicted"

    out = [one(k, t, c) for k, t, c in zip(known_nt, top_nt, top_conf, strict=True)]
    s = np.array([o[0] for o in out], dtype=np.float32)
    src: dict[str, int] = {}
    for _, how in out:
        src[how] = src.get(how, 0) + 1
    return s, {"sign_source_counts": src}


def load_flywire_circuit(circuit: str = "mushroom_body_right_pilot", release: str = "783",
                         min_contacts: int = 5, root: Path | None = None,
                         sign_policy=None) -> CircuitBundle:
    """Real circuit from data/processed (ingested by ``flycircuit data``). Raises FileNotFoundError
    with the command to run if artifacts are missing. Pairs below ``min_contacts`` are dropped (an
    explicit MODEL CHOICE, recorded in notes); cut boundary edges are not modelled."""
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    root = root or project_root()
    base = root / "data" / "processed" / "flywire" / release
    cdir = base / "circuits" / circuit
    for p in (base / "edges_pairs.parquet", base / "neurons.parquet", cdir / "internal_ids.parquet"):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing; run `flycircuit data fetch/ingest/select` for FlyWire v{release}")
    ids = pq.read_table(cdir / "internal_ids.parquet").to_pandas()
    nrn = pq.read_table(base / "neurons.parquet",
                        columns=["root_id", "known_nt", "top_nt", "top_nt_conf"]).to_pandas()
    nrn = nrn.drop_duplicates("root_id").set_index("root_id")
    root_ids = np.sort(ids["root_id"].to_numpy(dtype=np.int64))
    ids = ids.drop_duplicates("root_id").set_index("root_id").loc[root_ids]
    nn = nrn.reindex(root_ids)
    modulatory = None
    if sign_policy is None:  # v1 behaviour (kept for reproducibility of the v1 study)
        sign, sign_notes = _sign_from_annotations(nn["known_nt"].to_numpy(),
                                                  nn["top_nt"].to_numpy(),
                                                  nn["top_nt_conf"].to_numpy())
    else:  # sign-policy layer v2 (graph/signs.py): known / predicted / prior / unknown separated
        from flycircuit.graph.signs import load_assignment_for_ids

        asg = load_assignment_for_ids(root_ids, sign_policy, base)
        sign = asg.sign.astype(np.float32)
        modulatory = asg.modulatory
        sign_notes = {"sign_policy_v2": asg.policy.tag(), "sign_assignment_hash": asg.hash(),
                      "sign_summary": asg.summary}
    # Filter before materializing: a pilot must not load all 15M whole-brain pairs.
    flt = (ds.field("pre").isin(root_ids.tolist()) & ds.field("post").isin(root_ids.tolist())
           & (ds.field("count") >= min_contacts))
    e = ds.dataset(base / "edges_pairs.parquet").to_table(columns=["pre", "post", "count"], filter=flt)
    # Keep selection annotations aligned by exact root ID. Trained type/class screens need these;
    # annotations do not change the graph mapping, edges, signs or model parameters.
    attrs = {col: ids[col].to_dict() for col in
             ("cell_type", "cell_class", "side", "top_nt", "grp", "region") if col in ids}
    g = VersionedGraph.from_edges("flywire", release, e["pre"].to_numpy(), e["post"].to_numpy(),
                                  e["count"].to_numpy(), root_ids=root_ids, node_attrs=attrs)
    grp = ids["grp"].to_numpy()
    kc = np.flatnonzero(grp == "kenyon")
    mb = np.flatnonzero(grp == "mbon")
    if len(kc) == 0 or len(mb) < 2:
        raise ValueError(f"circuit {circuit}: need Kenyon cells and >=2 MBONs; got {len(kc)}/{len(mb)}")
    g.set_node_mask("sensory_input", np.isin(np.arange(g.n), kc))
    g.set_node_mask("action_readout", np.isin(np.arange(g.n), mb))
    g.set_edge_mask("trainable", np.ones(g.n_edges, dtype=bool))
    sel = json.loads((cdir / "selection.json").read_text())
    notes = {"synthetic": False, "circuit": circuit, "release": release,
             "min_contacts_model_choice": min_contacts, "n_kc": int(len(kc)), "n_mbon": int(len(mb)),
             "is_pilot_subset": bool(sel.get("is_pilot_subset")), "selection": sel.get("pilot"),
             "boundary": "cut edges removed (no boundary drive); see circuit BOUNDARY.md",
             "mapping_hash": g.mapping_hash(), "edge_hash": g.edge_hash(), **sign_notes}
    return CircuitBundle(g, sign, kc, mb, circuit, notes, modulatory)
