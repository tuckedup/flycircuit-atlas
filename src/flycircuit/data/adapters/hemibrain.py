"""Hemibrain adapter via the neuPrint HTTP API (dataset ``hemibrain:v1.2.1``).

Verified 2026-10-15:
 - ``GET https://neuprint.janelia.org/api/dbmeta/datasets`` is public and lists ``hemibrain:v1.2.1``
   (among others) with last-mod/uuid/ROIs.
 - Queries are ``POST {server}/api/custom/custom`` with JSON ``{"cypher": ..., "dataset": ...}`` and header
   ``Authorization: Bearer <token>``; response ``{"columns": [...], "data": [[...]]}``. Request format read
   from neuprint-python ``client.py``. The Cypher patterns (``(a:Neuron)-[e:ConnectsTo]->(b:Neuron)``,
   ``e.weight``, ``n.bodyId/type/instance/status``) follow neuprint-python ``queries/connectivity.py``.
 - A token is REQUIRED for queries: env ``NEUPRINT_APPLICATION_CREDENTIALS`` (same variable as
   neuprint-python). Without it every query raises ``MissingPrerequisite`` with the exact steps; the
   authenticated path was exercised only against a mocked transport (no token on this machine).
Bounded by design: whole-graph pulls are refused; pass explicit ``body_ids`` or a ``limit``.
External identity ``hemibrain:v1.2.1:<bodyId>`` (hemibrain = partial adult central brain, one hemisphere).
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import httpx
import pyarrow as pa

from .base import DatasetAdapter, MissingPrerequisite, UnsupportedOperation

SERVER = "https://neuprint.janelia.org"
TOKEN_ENV = "NEUPRINT_APPLICATION_CREDENTIALS"
DEFAULT_PROPS = ("type", "instance", "status")
_IDENT = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


class HemibrainAdapter(DatasetAdapter):
    dataset = "hemibrain"
    default_version = "v1.2.1"
    access = f"neuPrint API; token required in ${TOKEN_ENV} (dataset listing is public)"
    capabilities = {
        "datasets": "public, verified live",
        "neurons": "token required; bounded by limit",
        "edge_batches": "token required; explicit body_ids only",
        "not_supported": "synapse points, skeletons, meshes",
    }

    def __init__(self, version: str | None = None, *, server: str = SERVER, token: str | None = None,
                 transport: httpx.BaseTransport | None = None):
        super().__init__(version)
        self.server = server.rstrip("/")
        self._token = token
        self._client = httpx.Client(timeout=120, transport=transport)

    # -- plumbing ---------------------------------------------------------
    def _tok(self) -> str:
        tok = self._token or os.environ.get(TOKEN_ENV)
        if not tok:
            raise MissingPrerequisite(
                f"hemibrain/neuPrint queries need an API token: log in at {self.server} with a Google "
                f"account, copy 'Auth Token' from your account page, then `export {TOKEN_ENV}=<token>`. "
                "Public dataset listing works without a token (`datasets()`).")
        if ":" in tok:  # neuprint-python also accepts a JSON blob containing a 'token' key
            import json
            tok = json.loads(tok)["token"]
        return tok.strip().strip("'\"")

    @property
    def dataset_name(self) -> str:
        return f"{self.dataset}:{self.version}"

    def datasets(self) -> dict:
        r = self._client.get(f"{self.server}/api/dbmeta/datasets")
        r.raise_for_status()
        return r.json()

    def verify_release(self) -> dict:
        d = self.datasets()
        if self.dataset_name not in d:
            raise UnsupportedOperation(f"{self.dataset_name} not listed by {self.server}; available: {sorted(d)}")
        return {k: d[self.dataset_name].get(k) for k in ("last-mod", "uuid")}

    def cypher(self, query: str) -> tuple[list[str], list[list]]:
        if any(ch in query for ch in "‘’“”"):
            raise ValueError("curly quotes in Cypher")
        r = self._client.post(
            f"{self.server}/api/custom/custom",
            headers={"Authorization": f"Bearer {self._tok()}", "Content-Type": "application/json"},
            json={"cypher": query, "dataset": self.dataset_name})
        if r.status_code in (401, 403):
            raise MissingPrerequisite(f"neuPrint rejected the token (HTTP {r.status_code}); renew it")
        r.raise_for_status()
        j = r.json()
        return j["columns"], j["data"]

    # -- data ---------------------------------------------------------------
    def neurons(self, *, limit: int = 1000, skip: int = 0, status: str | None = "Traced",
                properties: tuple[str, ...] = DEFAULT_PROPS) -> pa.Table:
        if not (1 <= limit <= 50_000):
            raise ValueError("limit must be in 1..50000 (bounded queries only)")
        for p in properties:
            if not set(p) <= _IDENT:
                raise ValueError(f"bad property name {p!r}")
        where = f'WHERE n.status = "{status}" ' if status else ""
        props = "".join(f", n.{p} AS {p}" for p in properties)
        cols, rows = self.cypher(
            f"MATCH (n:Neuron) {where}RETURN n.bodyId AS bodyId{props} ORDER BY n.bodyId "
            f"SKIP {int(skip)} LIMIT {int(limit)}")
        t = pa.table({c: [r[i] for r in rows] for i, c in enumerate(cols)})
        if "bodyId" not in t.column_names:
            raise UnsupportedOperation("neuPrint response lacks bodyId")
        return t.append_column("uid", pa.array([self.uid(i) for i in t["bodyId"].to_pylist()]))

    def edge_batches(self, body_ids: list[int] | None = None, *, min_weight: int = 1,
                     chunk: int = 200) -> Iterator[pa.RecordBatch]:
        if not body_ids:
            raise UnsupportedOperation(
                "hemibrain edge_batches needs explicit body_ids (no whole-graph pull over the API); "
                "use the downloadable neuPrint exports for full-graph work")
        ids = [int(b) for b in body_ids]
        for i in range(0, len(ids), chunk):
            c = ids[i:i + chunk]
            lst = "[" + ",".join(map(str, c)) + "]"
            cols, rows = self.cypher(
                f"MATCH (a:Neuron)-[e:ConnectsTo]->(b:Neuron) WHERE a.bodyId IN {lst} AND "
                f"e.weight >= {int(min_weight)} RETURN a.bodyId AS body_pre, b.bodyId AS body_post, "
                "e.weight AS weight")
            yield pa.RecordBatch.from_arrays(
                [pa.array([r[j] for r in rows], type=pa.int64()) for j in range(3)], names=cols)
