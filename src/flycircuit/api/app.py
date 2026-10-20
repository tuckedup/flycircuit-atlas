"""Local FastAPI backend. IDs are release-qualified strings end to end."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from flycircuit.api.extra import COORD_SPACE, mode_from_env, register_extra
from flycircuit.config import project_root
from flycircuit.graph import SYNTHETIC_BANNER, VersionedGraph, make_synthetic_graph

KNOWN_FIELDS = ("cell_type", "hemisphere", "neurotransmitter", "soma_position", "morphology")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
MAX_LIMIT = 500
# FlyWire v783 annotation README (flyconnectome/flywire_annotations, supplemental_files/README.md):
# soma_x/y/z are "soma coordinates in 4x4x40nm voxel space".
VOXEL_NM = (4.0, 4.0, 40.0)


def load_default_graph(root: Path) -> VersionedGraph:
    return load_default_graph_and_info(root)[0]


def load_default_graph_and_info(root: Path, mode: str | None = None, circuit: str | None = None):
    """mode: 'real' (error if not cached), 'synthetic', or 'auto' (real when cached).

    Programmatic default is 'synthetic' so tests/CI never depend on cached data; the start
    scripts set FLYCIRCUIT_API_MODE=auto.
    """
    from flycircuit.api.extra import DEFAULT_CIRCUIT, load_real_graph, real_available

    env_mode, env_circuit = mode_from_env()
    mode = (mode or env_mode if mode or "FLYCIRCUIT_API_MODE" in os.environ else "synthetic")
    circuit = circuit or env_circuit or DEFAULT_CIRCUIT
    if mode in ("real", "auto") and real_available(root, circuit):
        return load_real_graph(root, circuit)
    if mode == "real":
        raise RuntimeError(f"real circuit {circuit!r} not ingested; run `flycircuit data select`")
    legacy = root / "data" / "processed" / "graph"
    g = VersionedGraph.load(legacy) if (legacy / "meta.json").exists() else make_synthetic_graph()
    return g, {"mode": "synthetic" if g.synthetic else "real-legacy",
               "note": SYNTHETIC_BANNER if g.synthetic else "legacy graph"}


class State:
    """Mutable holder so a circuit can be switched at runtime (real mode only)."""

    def __init__(self, g: VersionedGraph, info: dict[str, Any], switchable: bool):
        self.g, self.info, self.switchable = g, info, switchable
        self.cache: dict[str, tuple[VersionedGraph, dict[str, Any]]] = {}
        if info.get("circuit"):
            self.cache[info["circuit"]] = (g, info)


def create_app(graph: VersionedGraph | None = None, root: Path | None = None) -> FastAPI:
    root = root or project_root()
    if graph is not None:
        g0, info0 = graph, {"mode": "synthetic" if graph.synthetic else "real-supplied",
                            "note": SYNTHETIC_BANNER if graph.synthetic else "supplied graph"}
    else:
        g0, info0 = load_default_graph_and_info(root)
    S = State(g0, info0, switchable=info0.get("mode") == "real" and graph is None)
    app = FastAPI(title="FlyCircuit Atlas API", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["GET"], allow_headers=["*"],
    )

    def wrap(payload: dict[str, Any]) -> dict[str, Any]:
        g = S.g
        return {"synthetic": g.synthetic, "banner": SYNTHETIC_BANNER if g.synthetic else None,
                "data_mode": S.info.get("mode"),
                "data_label": None if g.synthetic else S.info.get("note"), **payload}

    def soma(idx: int) -> list[float] | None:
        sp = S.g.node_attrs.get("soma_position")
        return sp[idx] if sp else None

    def neuron(idx: int, detail: bool = False) -> dict[str, Any]:
        g = S.g
        attrs = {k: v[idx] for k, v in g.node_attrs.items() if v[idx] is not None}
        for alias, src in (("hemisphere", "side"), ("neurotransmitter", "top_nt")):
            if src in attrs:
                attrs.setdefault(alias, attrs[src])
        d: dict[str, Any] = {
            "id": g.qualified_id(idx), "root_id": str(int(g.root_ids[idx])),
            "dataset": g.dataset, "version": g.version,
            "cell_type": attrs.get("cell_type"),
            "has_soma": bool(soma(idx)),
        }
        if detail:
            pre, c_in = g.in_partners(idx)
            post, c_out = g.out_partners(idx)
            sp = soma(idx)
            if sp:
                attrs["soma_position_um"] = [round(sp[i] * VOXEL_NM[i] / 1000.0, 3) for i in range(3)]
            d.update({
                "in_degree": len(pre), "out_degree": len(post),
                "in_contacts": int(c_in.sum()), "out_contacts": int(c_out.sum()),
                "attributes": attrs,
                "unknown_fields": [f for f in KNOWN_FIELDS if f not in attrs],
                "coordinate_space": COORD_SPACE if sp else None,
            })
        return d

    def resolve(qid: str) -> int:
        try:
            return S.g.resolve(qid)
        except (ValueError, KeyError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/health")
    def health():
        return wrap({"status": "ok", "project": "flycircuit", "owner": "Atul Pandey"})

    @app.get("/api/graph")
    def graph_info():
        return wrap({"graph": S.g.summary()})

    @app.get("/api/neurons")
    def neurons(q: str = "", offset: int = Query(0, ge=0),
                limit: int = Query(50, ge=1, le=MAX_LIMIT),
                cell_type: str = "", group: str = ""):
        g = S.g
        ct = g.node_attrs.get("cell_type")
        gp = g.node_attrs.get("group")
        ql = q.lower()
        sel = [
            i for i in range(g.n)
            if (not q or q in str(int(g.root_ids[i]))
                or (ct is not None and ct[i] and ql in str(ct[i]).lower()))
            and (not cell_type or (ct is not None and ct[i] == cell_type))
            and (not group or (gp is not None and gp[i] == group))
        ]
        total = len(sel)
        page = sel[offset: offset + limit]
        return wrap({"total": total, "offset": offset, "limit": limit,
                     "items": [neuron(i) for i in page]})

    @app.get("/api/neurons/{qid}")
    def neuron_detail(qid: str):
        return wrap({"neuron": neuron(resolve(qid), detail=True)})

    def partner_edges(qid: str, direction: str, hops: int, limit: int):
        g = S.g
        start = resolve(qid)
        edges: dict[tuple[int, int], float] = {}
        truncated = False
        frontier = {start}
        for _ in range(hops):
            nxt: set[int] = set()
            for u in sorted(frontier):
                if direction in ("in", "both"):
                    for p, c in zip(*g.in_partners(u), strict=True):
                        edges[(int(p), u)] = float(c)
                        nxt.add(int(p))
                if direction in ("out", "both"):
                    for p, c in zip(*g.out_partners(u), strict=True):
                        edges[(u, int(p))] = float(c)
                        nxt.add(int(p))
                if len(edges) > limit * 20:
                    truncated = True
                    break
            frontier = nxt
        items = sorted(edges.items(), key=lambda kv: -kv[1])
        if len(items) > limit:
            truncated = True
            items = items[:limit]
        return start, items, truncated

    @app.get("/api/neurons/{qid}/partners")
    def partners(qid: str, direction: str = Query("both", pattern="^(in|out|both)$"),
                 hops: int = Query(1, ge=1, le=2), limit: int = Query(100, ge=1, le=MAX_LIMIT)):
        g = S.g
        start, items, truncated = partner_edges(qid, direction, hops, limit)
        nodes = {start} | {a for (a, _), _c in items} | {b for (_, b), _c in items}
        info = {g.qualified_id(i): {"cell_type": (g.node_attrs.get("cell_type") or [None] * g.n)[i],
                                    "group": (g.node_attrs.get("group") or [None] * g.n)[i]}
                for i in nodes}
        return wrap({
            "center": g.qualified_id(start), "direction": direction, "hops": hops,
            "truncated": truncated, "node_info": info,
            "edges": [{"pre": g.qualified_id(a), "post": g.qualified_id(b), "contacts": int(c)}
                      for (a, b), c in items],
        })

    @app.get("/api/runs")
    def runs(limit: int = Query(50, ge=1, le=MAX_LIMIT)):
        out = []
        for m in sorted((root / "runs").glob("*/manifest.json"), reverse=True)[:limit]:
            d = json.loads(m.read_text(encoding="utf-8"))
            out.append({"has_replay": (m.parent / "replay.json").exists(),
                        **{k: d.get(k) for k in ("run_id", "timestamp_utc", "exit_status",
                                                  "data_release", "label", "metrics")}})
        return wrap({"runs": out})

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str):
        if not RUN_ID_RE.match(run_id):
            raise HTTPException(400, "invalid run id")
        p = root / "runs" / run_id / "manifest.json"
        if not p.exists():
            raise HTTPException(404, "run not found")
        return wrap({"manifest": json.loads(p.read_text(encoding="utf-8"))})

    register_extra(app, S, root, wrap, partner_edges)
    return app


def app_factory() -> FastAPI:
    return create_app()
