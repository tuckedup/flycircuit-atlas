"""Bridge: a circuit selection (``data.selectors``) -> ``VersionedGraph`` with signs and boundary.

Convention ``W[post, pre]``. Root IDs stay exact int64 (strings on the wire). Everything that is a
*modelling choice* (contact threshold, sign policy, boundary treatment) is an explicit argument and
is recorded in ``CircuitGraph.provenance``; nothing is silently defaulted into the science.

Sign policies (assumed physiological sign of a presynaptic neuron, NOT a measured effect):

``shiu_reported``
    The per-presynaptic-neuron sign shipped in Shiu et al.'s ``Connectivity_783.parquet``
    (``Excitatory`` column, constant per neuron: measured 0 inconsistent neurons of 138,005).
    Needed to replicate that model; it is *not* a simple function of ``top_nt`` (measured
    agreement with the ACh/DA/5HT/OA=+1, GABA/Glu=-1 rule: see ``docs/refcheck.md``).
``annotation_top_nt``
    Rule applied to ``neurons.parquet:top_nt`` (``DEFAULT_NT_SIGN``), with an explicit
    ``unknown`` policy (``zero`` | ``excitatory`` | ``inhibitory``) and optional ``min_conf``.
    Dopamine/serotonin/octopamine = +1 is the Shiu-style simplification, not a physiological claim.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa

from flycircuit.graph.versioned import VersionedGraph

DEFAULT_NT_SIGN: dict[str, int] = {
    "acetylcholine": 1,
    "dopamine": 1,
    "serotonin": 1,
    "octopamine": 1,
    "gaba": -1,
    "glutamate": -1,
}
SIGN_POLICIES = ("shiu_reported", "annotation_top_nt", "annotation_v2")
UNKNOWN_POLICIES = {"zero": 0, "excitatory": 1, "inhibitory": -1}
SHIU_SIGN_TABLE = "ref_shiu_presyn_sign.parquet"


class BridgeError(ValueError):
    """Actionable failure while building a graph from a selection."""


def processed_release_dir(dataset: str = "flywire", version: str = "783") -> Path:
    from flycircuit.data.paths import processed_dir

    return Path(processed_dir(dataset, version))


@dataclass
class BoundaryInfo:
    """One-hop boundary of a selection at the chosen contact threshold (pairs, not neurons)."""

    in_boundary_id: np.ndarray  # presynaptic outside-neuron root ids (int64)
    in_post_idx: np.ndarray  # internal node index receiving the cut edge
    in_contacts: np.ndarray  # contacts of the cut pair
    in_sign: np.ndarray  # assumed sign of the outside presynaptic neuron (0 unknown)
    out_post_id: np.ndarray  # outside postsynaptic root ids (int64)
    out_pre_idx: np.ndarray  # internal node index sending the cut edge
    out_contacts: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "cut_incoming_pairs": int(len(self.in_contacts)),
            "cut_incoming_contacts": int(self.in_contacts.sum()),
            "cut_incoming_excitatory_contacts": int(self.in_contacts[self.in_sign > 0].sum()),
            "cut_incoming_inhibitory_contacts": int(self.in_contacts[self.in_sign < 0].sum()),
            "cut_incoming_unknown_sign_contacts": int(self.in_contacts[self.in_sign == 0].sum()),
            "cut_outgoing_pairs": int(len(self.out_contacts)),
            "cut_outgoing_contacts": int(self.out_contacts.sum()),
        }


@dataclass
class CircuitGraph:
    graph: VersionedGraph
    node_sign: np.ndarray  # int8 per node, 0 = unknown/no sign assumed
    group: np.ndarray  # selector group label per node (object array)
    cell_type: np.ndarray
    cell_class: np.ndarray
    boundary: BoundaryInfo
    selection: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)
    modulatory: np.ndarray | None = None  # bool per node (annotation_v2 only): monoamine neurons
    sign_evidence: np.ndarray | None = None  # object per node: known|predicted|prior|unknown

    @property
    def edge_sign(self) -> np.ndarray:
        """Sign of every graph edge (aligned with ``graph.pre``/``post``) = sign of its presyn node."""
        return self.node_sign[self.graph.pre]

    def signed_contacts(self) -> np.ndarray:
        """Edge-aligned signed contact counts (float64): sign[pre] * contacts."""
        return self.edge_sign.astype(np.float64) * self.graph.count.astype(np.float64)

    def indices_of_group(self, name: str) -> np.ndarray:
        return np.flatnonzero(self.group == name)

    def indices_of_type(self, cell_type: str) -> np.ndarray:
        return np.flatnonzero(self.cell_type == cell_type)


def ids_hash(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def build_shiu_sign_table(
    shiu_connectivity: str | Path, out: str | Path, memory_limit: str = "2GiB"
) -> dict[str, Any]:
    """Derive one sign per presynaptic neuron from Shiu's Connectivity parquet (bounded memory)."""
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    src = str(shiu_connectivity)
    if not Path(src).exists():
        raise BridgeError(f"missing {src}; clone external/Drosophila_brain_model first")
    stats = con.execute(
        f"""WITH per AS (SELECT Presynaptic_ID id, min(Excitatory) mn, max(Excitatory) mx
                         FROM '{src}' GROUP BY 1)
            SELECT count(*), count(*) FILTER (WHERE mn != mx) FROM per"""
    ).fetchone()
    if stats[1]:
        raise BridgeError(f"{stats[1]} presynaptic neurons have inconsistent signs in {src}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""COPY (SELECT Presynaptic_ID AS root_id, max(Excitatory)::TINYINT AS sign
                   FROM '{src}' GROUP BY 1 ORDER BY 1) TO '{out}' (FORMAT parquet)"""
    )
    return {"presynaptic_neurons": int(stats[0]), "inconsistent": int(stats[1]), "path": str(out)}


def _node_signs(
    ids: np.ndarray,
    top_nt: list,
    conf: list,
    policy: str,
    unknown: str,
    min_conf: float | None,
    proc: Path,
) -> np.ndarray:
    if unknown not in UNKNOWN_POLICIES:
        raise BridgeError(f"unknown policy {unknown!r}; choose from {sorted(UNKNOWN_POLICIES)}")
    if policy == "shiu_reported":
        p = proc / SHIU_SIGN_TABLE
        if not p.exists():
            raise BridgeError(
                f"{p} missing; run `flycircuit refcheck prepare` (derives it from "
                "external/Drosophila_brain_model/Connectivity_783.parquet)"
            )
        con = duckdb.connect()
        con.register("q", pa.table({"root_id": ids}))
        rows = con.execute(
            f"SELECT q.root_id, s.sign FROM q LEFT JOIN '{p}' s USING (root_id)"
        ).fetchall()
        sg = np.array([0 if r[1] is None else int(r[1]) for r in rows], dtype=np.int8)
        missing = np.array([r[1] is None for r in rows])
        sg[missing] = UNKNOWN_POLICIES[unknown]
        return sg
    if policy == "annotation_top_nt":
        out = np.zeros(len(ids), dtype=np.int8)
        for i, (nt, c) in enumerate(zip(top_nt, conf, strict=True)):
            ok = nt in DEFAULT_NT_SIGN and (
                min_conf is None or (c is not None and not np.isnan(c) and c >= min_conf)
            )
            out[i] = DEFAULT_NT_SIGN[nt] if ok else UNKNOWN_POLICIES[unknown]
        return out
    raise BridgeError(f"unknown sign policy {policy!r}; choose from {SIGN_POLICIES}")


def load_circuit_graph(
    circuit: str,
    *,
    dataset: str = "flywire",
    version: str = "783",
    threshold: int = 5,
    sign_policy: str = "shiu_reported",
    unknown_sign: str = "zero",
    min_conf: float | None = None,
    processed: Path | None = None,
    sign_spec: Any = None,
) -> CircuitGraph:
    """Build the induced internal graph of a saved selection (``flycircuit data select``)."""
    proc = Path(processed) if processed else processed_release_dir(dataset, version)
    cdir = proc / "circuits" / circuit
    for f in ("selection.json", "internal_ids.parquet"):
        if not (cdir / f).exists():
            raise BridgeError(f"missing {cdir / f}; run `flycircuit data select` for {circuit!r}")
    sel = json.loads((cdir / "selection.json").read_text())
    if sel.get("dataset") != dataset or str(sel.get("version")) != str(version):
        raise BridgeError(
            f"selection is {sel.get('dataset')}:{sel.get('version')}, requested {dataset}:{version}"
        )
    if int(sel.get("threshold", threshold)) != int(threshold):
        raise BridgeError(
            f"selection {circuit} was written at threshold {sel.get('threshold')}, bridge asked for "
            f"{threshold}; boundary/size statistics would disagree - re-run selection"
        )
    con = duckdb.connect()
    con.execute("SET memory_limit='1GiB'")
    node_tbl = con.execute(
        f"SELECT root_id, grp, cell_class, cell_type, top_nt, top_nt_conf "
        f"FROM '{cdir / 'internal_ids.parquet'}' ORDER BY root_id"
    ).fetchall()
    ids = np.array([r[0] for r in node_tbl], dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise BridgeError("duplicate root ids in internal_ids.parquet")
    con.register("ids", pa.table({"root_id": ids}))
    E = f"'{proc / 'edges_pairs.parquet'}'"
    ed = con.execute(
        f"""SELECT pre, post, count FROM {E}
            WHERE count >= {int(threshold)} AND pre IN (SELECT root_id FROM ids)
              AND post IN (SELECT root_id FROM ids) ORDER BY pre, post"""
    ).fetchnumpy()
    node_attrs = {
        "group": {r[0]: r[1] for r in node_tbl},
        "cell_class": {r[0]: r[2] for r in node_tbl},
        "cell_type": {r[0]: r[3] for r in node_tbl},
        "top_nt": {r[0]: r[4] for r in node_tbl},
    }
    g = VersionedGraph.from_edges(
        dataset, version, ed["pre"], ed["post"], ed["count"], root_ids=ids, node_attrs=node_attrs
    )
    top_nt = [r[4] for r in node_tbl]
    conf = [r[5] for r in node_tbl]
    v2 = None
    if sign_policy == "annotation_v2":
        from flycircuit.graph.signs import SignPolicy, load_assignment_for_ids

        sign_spec = sign_spec or SignPolicy()
        v2 = load_assignment_for_ids(g.root_ids, sign_spec, proc, dataset, version)
        node_sign = v2.sign
    else:
        node_sign = _node_signs(g.root_ids, top_nt, conf, sign_policy, unknown_sign, min_conf, proc)

    # one-hop boundary at the same threshold
    bin_ = con.execute(
        f"""SELECT pre, post, count FROM {E} WHERE count >= {int(threshold)}
            AND post IN (SELECT root_id FROM ids) AND pre NOT IN (SELECT root_id FROM ids)
            ORDER BY pre, post"""
    ).fetchnumpy()
    bout = con.execute(
        f"""SELECT pre, post, count FROM {E} WHERE count >= {int(threshold)}
            AND pre IN (SELECT root_id FROM ids) AND post NOT IN (SELECT root_id FROM ids)
            ORDER BY pre, post"""
    ).fetchnumpy()
    in_pre = np.asarray(bin_["pre"], dtype=np.int64)
    out_nt = {}
    if len(in_pre):
        uq = np.unique(in_pre)
        con.register("bq", pa.table({"root_id": uq}))
        rows = con.execute(
            f"SELECT n.root_id, n.top_nt, n.top_nt_conf FROM '{proc / 'neurons.parquet'}' n "
            "WHERE n.root_id IN (SELECT root_id FROM bq) ORDER BY n.root_id"
        ).fetchall()
        rid = np.array([r[0] for r in rows], dtype=np.int64)
        if v2 is not None:
            bs = load_assignment_for_ids(rid, sign_spec, proc, dataset, version).sign
        else:
            bs = _node_signs(rid, [r[1] for r in rows], [r[2] for r in rows], sign_policy,
                             unknown_sign, min_conf, proc)
        out_nt = dict(zip(rid.tolist(), bs.tolist(), strict=True))
    in_sign = np.array([out_nt.get(int(b), UNKNOWN_POLICIES[unknown_sign]) for b in in_pre],
                       dtype=np.int8)
    boundary = BoundaryInfo(
        in_boundary_id=in_pre,
        in_post_idx=np.searchsorted(g.root_ids, np.asarray(bin_["post"], dtype=np.int64)),
        in_contacts=np.asarray(bin_["count"], dtype=np.int64),
        in_sign=in_sign,
        out_post_id=np.asarray(bout["post"], dtype=np.int64),
        out_pre_idx=np.searchsorted(g.root_ids, np.asarray(bout["pre"], dtype=np.int64)),
        out_contacts=np.asarray(bout["count"], dtype=np.int64),
    )
    bi = np.zeros(g.n, dtype=bool)
    bi[boundary.in_post_idx] = True
    bo = np.zeros(g.n, dtype=bool)
    bo[boundary.out_pre_idx] = True
    g.set_node_mask("boundary_input", bi)
    g.set_node_mask("boundary_output", bo)
    prov = {
        "circuit": circuit,
        "release": f"{dataset}:{version}",
        "threshold_contacts": int(threshold),
        "sign_policy": sign_policy,
        "unknown_sign_policy": unknown_sign,
        "sign_v2": None if v2 is None else {**v2.summary, "hash": v2.hash()},
        "min_conf": min_conf,
        "n_nodes": g.n,
        "n_edges": g.n_edges,
        "mapping_hash": g.mapping_hash(),
        "edge_hash": g.edge_hash(),
        "nodes_by_sign": {str(s): int((node_sign == s).sum()) for s in (-1, 0, 1)},
        "is_pilot_subset": bool(sel.get("is_pilot_subset")),
        "boundary": boundary.summary(),
    }
    return CircuitGraph(
        graph=g,
        node_sign=node_sign,
        group=np.array([r[1] for r in node_tbl], dtype=object),
        cell_type=np.array([r[3] for r in node_tbl], dtype=object),
        cell_class=np.array([r[2] for r in node_tbl], dtype=object),
        boundary=boundary,
        selection=sel,
        provenance=prov,
        modulatory=None if v2 is None else v2.modulatory,
        sign_evidence=None if v2 is None else v2.evidence,
    )


def boundary_background_pairs(
    cg: CircuitGraph, mode: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Boundary treatment. ``cut`` (default): cut incoming edges are removed -> returns None.

    ``background_drive``: every cut incoming pair becomes an independent background Poisson source
    (rate chosen by the caller) delivering ``sign * contacts`` synaptic units to its internal
    target; returns (post_idx, signed_contacts), unknown-sign pairs excluded. A fixed Poisson
    background is NOT equivalent to the missing upstream dynamics (sensitivity comparison only).
    """
    if mode == "cut":
        return None
    if mode != "background_drive":
        raise BridgeError(f"unknown boundary mode {mode!r}; use 'cut' or 'background_drive'")
    b = cg.boundary
    keep = b.in_sign != 0
    return b.in_post_idx[keep], b.in_contacts[keep].astype(float) * b.in_sign[keep]


def induced_subgraph_check(cg: CircuitGraph) -> dict[str, Any]:
    """Cheap structural invariants of the induced internal graph."""
    g = cg.graph
    return {
        "endpoints_valid": bool(((g.pre >= 0) & (g.pre < g.n) & (g.post >= 0) & (g.post < g.n)).all()),
        "self_loops": g.n_self_loops,
        "sorted_unique_ids": bool(np.all(np.diff(g.root_ids) > 0)),
        "min_contacts": float(g.count.min()) if g.n_edges else None,
    }
