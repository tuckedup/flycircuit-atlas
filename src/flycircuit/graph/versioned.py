"""Release-qualified directed graph with deterministic node ordering.

Convention: ``W[post, pre]`` (row = postsynaptic, column = presynaptic).
Root IDs are exact int64 on the Python side and always strings in JSON/JS.
Edge arrays (``pre``, ``post``, ``count``) are the canonical store, sorted by
(post, pre); the CSR/CSC matrices are derived. Edge masks align with that order.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from flycircuit.graph.guard import guard_dense

NODE_MASKS = (
    "retained",
    "stimulation",
    "sensory_input",
    "action_readout",
    "boundary_input",
    "boundary_output",
)
EDGE_MASKS = ("retained", "trainable")


class VersionedGraph:
    def __init__(
        self,
        dataset: str,
        version: str,
        root_ids: np.ndarray,
        pre: np.ndarray,
        post: np.ndarray,
        count: np.ndarray,
        node_attrs: dict[str, list[Any]] | None = None,
        synthetic: bool = False,
    ):
        self.dataset = dataset
        self.version = str(version)
        self.root_ids = np.asarray(root_ids, dtype=np.int64)
        self.pre = np.asarray(pre, dtype=np.int64)
        self.post = np.asarray(post, dtype=np.int64)
        self.count = np.asarray(count, dtype=np.float32)
        self.node_attrs = node_attrs or {}
        self.synthetic = synthetic
        self.node_masks: dict[str, np.ndarray] = {
            "retained": np.ones(self.n, dtype=bool),
            **{k: np.zeros(self.n, dtype=bool) for k in NODE_MASKS if k != "retained"},
        }
        self.edge_masks: dict[str, np.ndarray] = {
            "retained": np.ones(self.n_edges, dtype=bool),
            "trainable": np.zeros(self.n_edges, dtype=bool),
        }
        self._csr: sp.csr_matrix | None = None
        self._csc: sp.csc_matrix | None = None

    # ---- construction -------------------------------------------------
    @classmethod
    def from_edges(
        cls,
        dataset: str,
        version: str,
        pre_ids,
        post_ids,
        counts,
        root_ids=None,
        node_attrs: dict[str, dict[int, Any]] | None = None,
        synthetic: bool = False,
    ) -> VersionedGraph:
        """Build from root-ID edge lists. Duplicate (pre, post) rows are summed.

        Unknown endpoints raise ValueError (classify boundary IDs before calling).
        """
        pre_ids = np.asarray(pre_ids, dtype=np.int64)
        post_ids = np.asarray(post_ids, dtype=np.int64)
        counts = np.asarray(counts, dtype=np.float64)
        if not (len(pre_ids) == len(post_ids) == len(counts)):
            raise ValueError("pre/post/count length mismatch")
        if root_ids is None:
            root_ids = np.union1d(pre_ids, post_ids)
        root_ids = np.unique(np.asarray(root_ids, dtype=np.int64))  # sorted, deterministic
        pi = np.searchsorted(root_ids, pre_ids)
        qi = np.searchsorted(root_ids, post_ids)
        bad = (
            (pi >= len(root_ids))
            | (qi >= len(root_ids))
            | (root_ids[np.minimum(pi, len(root_ids) - 1)] != pre_ids)
            | (root_ids[np.minimum(qi, len(root_ids) - 1)] != post_ids)
        )
        if len(root_ids) == 0 and len(pre_ids) > 0 or bad.any():
            raise ValueError(f"{int(bad.sum())} edge endpoint(s) not in root_ids")
        n = len(root_ids)
        key = qi.astype(np.int64) * n + pi
        uniq, inv = np.unique(key, return_inverse=True)
        summed = np.zeros(len(uniq), dtype=np.float64)
        np.add.at(summed, inv, counts)
        post_idx = uniq // n
        pre_idx = uniq % n
        attrs: dict[str, list[Any]] = {}
        if node_attrs:
            for name, mapping in node_attrs.items():
                attrs[name] = [mapping.get(int(r)) for r in root_ids]
        return cls(dataset, version, root_ids, pre_idx, post_idx, summed, attrs, synthetic)

    # ---- basic properties ---------------------------------------------
    @property
    def n(self) -> int:
        return len(self.root_ids)

    @property
    def n_edges(self) -> int:
        return len(self.pre)

    @property
    def n_self_loops(self) -> int:
        return int((self.pre == self.post).sum())

    @property
    def csr(self) -> sp.csr_matrix:
        """W[post, pre] as CSR."""
        if self._csr is None:
            self._csr = sp.csr_matrix((self.count, (self.post, self.pre)), shape=(self.n, self.n))
        return self._csr

    @property
    def csc(self) -> sp.csc_matrix:
        if self._csc is None:
            self._csc = self.csr.tocsc()
        return self._csc

    # ---- identity -----------------------------------------------------
    def mapping_hash(self) -> str:
        h = hashlib.sha256()
        h.update(f"{self.dataset}\n{self.version}\n".encode())
        h.update(self.root_ids.astype("<i8").tobytes())
        return h.hexdigest()

    def edge_hash(self) -> str:
        h = hashlib.sha256()
        for a in (self.pre, self.post):
            h.update(a.astype("<i8").tobytes())
        h.update(self.count.astype("<f4").tobytes())
        return h.hexdigest()

    def qualified_id(self, idx: int) -> str:
        return f"{self.dataset}:{self.version}:{int(self.root_ids[idx])}"

    def index_of(self, root_id: int) -> int:
        i = int(np.searchsorted(self.root_ids, root_id))
        if i >= self.n or int(self.root_ids[i]) != int(root_id):
            raise KeyError(f"root id {root_id} not in {self.dataset}:{self.version}")
        return i

    def resolve(self, qid: str) -> int:
        """Resolve 'dataset:version:rootid' (or bare string id only when release matches)."""
        parts = qid.split(":")
        if len(parts) != 3:
            raise ValueError(f"id must be 'dataset:version:root_id', got {qid!r}")
        ds, ver, rid = parts
        if ds != self.dataset or ver != self.version:
            raise ValueError(f"id release {ds}:{ver} != graph release {self.dataset}:{self.version}")
        return self.index_of(int(rid))

    # ---- queries ------------------------------------------------------
    def in_partners(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Presynaptic partners of ``idx`` (row of W). Returns (pre_idx, counts)."""
        m = self.csr
        s, e = m.indptr[idx], m.indptr[idx + 1]
        return m.indices[s:e].copy(), m.data[s:e].copy()

    def out_partners(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Postsynaptic targets of ``idx`` (column of W). Returns (post_idx, counts)."""
        m = self.csc
        s, e = m.indptr[idx], m.indptr[idx + 1]
        return m.indices[s:e].copy(), m.data[s:e].copy()

    # ---- masks --------------------------------------------------------
    def set_node_mask(self, name: str, mask: np.ndarray) -> None:
        if name not in NODE_MASKS:
            raise KeyError(f"unknown node mask {name!r}; allowed {NODE_MASKS}")
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n,):
            raise ValueError("node mask shape mismatch")
        self.node_masks[name] = mask

    def set_edge_mask(self, name: str, mask: np.ndarray) -> None:
        if name not in EDGE_MASKS:
            raise KeyError(f"unknown edge mask {name!r}; allowed {EDGE_MASKS}")
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_edges,):
            raise ValueError("edge mask shape mismatch")
        self.edge_masks[name] = mask

    # ---- dense (guarded) ----------------------------------------------
    def to_dense(self, max_elements: int | None = None) -> np.ndarray:
        guard_dense((self.n, self.n), max_elements, what="adjacency")
        return self.csr.toarray()

    # ---- reporting / persistence -------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "version": self.version,
            "synthetic": self.synthetic,
            "n_neurons": self.n,
            "n_edges": self.n_edges,
            "n_self_loops": self.n_self_loops,
            "total_contacts": float(self.count.sum()),
            "mapping_hash": self.mapping_hash(),
            "edge_hash": self.edge_hash(),
            "convention": "W[post, pre]",
        }

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            d / "graph.npz", root_ids=self.root_ids, pre=self.pre, post=self.post, count=self.count
        )
        meta = {**self.summary(), "node_attrs": self.node_attrs}
        (d / "meta.json").write_text(json.dumps(meta, default=str), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> VersionedGraph:
        d = Path(directory)
        z = np.load(d / "graph.npz")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        g = cls(
            meta["dataset"], meta["version"], z["root_ids"], z["pre"], z["post"], z["count"],
            meta.get("node_attrs") or {}, meta.get("synthetic", False),
        )
        if g.mapping_hash() != meta["mapping_hash"] or g.edge_hash() != meta["edge_hash"]:
            raise ValueError(f"{d}: stored hashes do not match graph content")
        return g
