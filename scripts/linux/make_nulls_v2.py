"""Generate + cache protocol nulls-v2 ensembles. Usage: make_nulls_v2.py <circuit> [k] [kinds]"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from flycircuit.graph.bridge import load_circuit_graph
from flycircuit.graph.nulls_v2 import (class_pair_counts_equal, make_ensemble, region_edge_class,
                                       save_ensemble, to_graphs)

circuit = sys.argv[1] if len(sys.argv) > 1 else "mushroom_body_right_pilot"
k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
kinds = (sys.argv[3] if len(sys.argv) > 3 else "plain,group_constrained").split(",")
cg = load_circuit_graph(circuit)
g = cg.graph
root = Path("results/nulls_v2") / circuit
summary = {"circuit": circuit, "n": g.n, "edges": g.n_edges, "k": k, "kinds": {}}
for kind in kinds:
    t0 = time.time()
    ec = None
    if kind == "group_constrained":
        ec = region_edge_class(g, cg.group)
    chains, rep = make_ensemble(g, k=k, base_seed=2000 if kind == "plain" else 3000, edge_class=ec,
                                label=f"{circuit}/{kind}")
    graphs = to_graphs(g, chains, "nulls_v2_" + kind)
    if kind == "group_constrained":
        rep.criteria["class_pair_counts_preserved"] = all(
            class_pair_counts_equal(g, h, cg.group) for h in graphs)
        if not rep.criteria["class_pair_counts_preserved"]:
            rep.adequate = False
            rep.verdict = "inadequate"
            rep.reasons.append("CLASS-PAIR COUNTS NOT PRESERVED")
    # edge persistence: in how many of the k nulls does each ORIGINAL edge reappear?
    okeys = g.pre.astype(np.int64) * g.n + g.post
    pers = np.zeros(g.n_edges, dtype=int)
    for c in chains:
        pers += np.isin(okeys, c.pre.astype(np.int64) * g.n + c.post)
    forced = pers == k
    pair = np.char.add(np.char.add(cg.group[g.pre].astype(str), "->"), cg.group[g.post].astype(str))
    by_pair = {p: {"edges": int((pair == p).sum()), "forced": int(((pair == p) & forced).sum())}
               for p in np.unique(pair)}
    rep.forced_edges["persistence_histogram"] = np.bincount(pers, minlength=k + 1).tolist()
    rep.forced_edges["by_group_pair"] = by_pair
    meta = {"kind": kind, "wall_s": round(time.time() - t0, 1), "circuit": circuit,
            "n_classes": None if ec is None else int(len(np.unique(ec)))}
    save_ensemble(root / kind, g, chains, rep, meta)
    summary["kinds"][kind] = {"adequate": rep.adequate, "verdict": rep.verdict,
                              "reasons": rep.reasons,
                              "forced_edge_fraction": rep.forced_edges["fraction_of_edges"],
                              "forced_by_group_pair": by_pair,
                              "persistence_histogram": rep.forced_edges["persistence_histogram"],
                              "wall_s": meta["wall_s"], "limitations": rep.limitations,
                              "nulls_stats": rep.graph_stats_nulls,
                              "original_stats": rep.graph_stats_original,
                              "accepted_per_edge": [p["accepted_per_edge"] for p in rep.per_null],
                              "sweeps": [p["sweeps"] for p in rep.per_null]}
    print(kind, "adequate=", rep.adequate, rep.reasons, f"forced={rep.forced_edges['fraction_of_edges']:.3f}",
          f"{meta['wall_s']}s")
(root / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
