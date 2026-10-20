"""First-instar larva adapter: Winding, Pedigo et al. 2023, Science (DOI 10.1126/science.add9330).

Verified 2026-10-15: a public CATMAID instance (https://l1em.catmaid.virtualflybrain.org, server
2021.12.21.dev295, project 1 "L1 CNS") answers anonymous reads. GET works directly (``/projects/``,
``/1/skeletons/``, ``/1/annotations/``, ``/1/skeletons/{id}/compact-detail``); POST endpoints
(``/1/skeletons/connectivity``, ``/1/annotations/query-targets``) require Django's CSRF cookie+header,
obtained by a prior GET to ``/`` (cookie named ``csrftoken_<hash>``) -- no login or token.
This is a LIVE, MUTABLE service, not an immutable release: every result records the server version and
access time; pin by exporting responses. Identity: ``larva:l1em-p1:<skeletonId>`` (skeleton IDs, not neuron
IDs). Connectivity counts come as per-confidence synapse counts ([c1..c5]); ``count`` below is their sum
(confidence filtering is the caller's choice). Larva != adult: never merge with adult IDs.
"""

from __future__ import annotations

import datetime as dt

import httpx
import pyarrow as pa

from .base import DatasetAdapter, UnsupportedOperation

SERVER = "https://l1em.catmaid.virtualflybrain.org"


class LarvaAdapter(DatasetAdapter):
    dataset = "larva"
    default_version = "l1em-p1"
    access = "public CATMAID reads (anonymous; CSRF cookie handshake), live service"
    capabilities = {
        "neurons": "skeleton list (+ optional neuron names via annotations query)",
        "edge_batches": "skeletons/connectivity for explicit skeleton ids",
        "not_supported": "morphology/meshes beyond compact-detail; whole-graph pull",
    }

    def __init__(self, version: str | None = None, *, server: str = SERVER, project: int = 1,
                 transport: httpx.BaseTransport | None = None):
        super().__init__(version)
        self.server, self.project = server.rstrip("/"), project
        self._c = httpx.Client(timeout=120, transport=transport, follow_redirects=True,
                               headers={"Referer": self.server + "/"})
        self._csrf: str | None = None

    def _post_headers(self) -> dict:
        if self._csrf is None:
            self._c.get(self.server + "/")
            self._csrf = next((v for k, v in self._c.cookies.items() if k.startswith("csrftoken")), None)
            if self._csrf is None:
                raise UnsupportedOperation("CATMAID did not set a CSRF cookie; POST endpoints unavailable")
        return {"X-CSRFToken": self._csrf, "Referer": self.server + "/"}

    def _get(self, path: str, **params):
        r = self._c.get(f"{self.server}{path}", params=params or None)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, data: dict):
        r = self._c.post(f"{self.server}{path}", data=data, headers=self._post_headers())
        r.raise_for_status()
        return r.json()

    def provenance(self) -> dict:
        v = self._get("/version")
        return {"server": self.server, "server_version": v.get("SERVER_VERSION"), "project": self.project,
                "accessed_utc": dt.datetime.now(dt.UTC).isoformat(), "mutable_service": True}

    def projects(self) -> list[dict]:
        return self._get("/projects/")

    def neurons(self, *, min_nodes: int = 0, limit: int | None = None) -> pa.Table:
        ids = self._get(f"/{self.project}/skeletons/", nodecount_gt=min_nodes) if min_nodes else \
            self._get(f"/{self.project}/skeletons/")
        ids = sorted(int(i) for i in ids)
        if limit is not None:
            ids = ids[:limit]
        return pa.table({"skeleton_id": pa.array(ids, pa.int64()),
                         "uid": pa.array([self.uid(i) for i in ids])})

    def edge_batches(self, skeleton_ids: list[int] | None = None, *, chunk: int = 50):
        """Directed (pre -> post, count) batches for the outgoing connections of ``skeleton_ids``."""
        if not skeleton_ids:
            raise UnsupportedOperation("larva edge_batches needs explicit skeleton_ids (bounded queries only)")
        ids = [int(i) for i in skeleton_ids]
        for k in range(0, len(ids), chunk):
            sub = ids[k:k + chunk]
            data = {f"source_skeleton_ids[{i}]": s for i, s in enumerate(sub)}
            data.update({"boolean_op": "OR", "with_nodes": "false"})
            j = self._post(f"/{self.project}/skeletons/connectivity", data)
            pre, post, cnt = [], [], []
            for tgt, d in j.get("outgoing", {}).items():
                for src, conf in d["skids"].items():
                    pre.append(int(src))
                    post.append(int(tgt))
                    cnt.append(int(sum(conf)))
            yield pa.RecordBatch.from_arrays(
                [pa.array(pre, pa.int64()), pa.array(post, pa.int64()), pa.array(cnt, pa.int64())],
                names=["skeleton_pre", "skeleton_post", "count"])
