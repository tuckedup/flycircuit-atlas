"""``flycircuit models ...``: benchmark / controls / match (SYNTHETIC graphs only here)."""

from __future__ import annotations

import json
import time

import numpy as np


def _synth(n: int, deg: int, seed: int = 0):
    from flycircuit.graph.versioned import VersionedGraph

    rng = np.random.default_rng(seed)
    e = n * deg
    keys = np.unique(rng.integers(0, n * n, size=int(e * 1.2), dtype=np.int64))
    keys = keys[(keys // n) != (keys % n)][:e]
    pre, post = keys % n, keys // n
    ids = 720575940000000000 + np.arange(n, dtype=np.int64)
    g = VersionedGraph.from_edges("synthetic", "0", ids[pre], ids[post],
                                  rng.integers(1, 12, size=len(pre)), root_ids=ids, synthetic=True)
    g.set_node_mask("sensory_input", np.arange(n) < 8)
    g.set_node_mask("action_readout", np.arange(n) >= n - 4)
    g.set_edge_mask("trainable", np.ones(g.n_edges, dtype=bool))
    return g


def _cmd_benchmark(args) -> int:
    import torch

    from flycircuit.models.sparse_rnn import ModelSpec, SparseConnectomeRNN
    from flycircuit.models.train import estimate_memory_bytes

    torch.set_num_threads(4)
    rows = []
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for n in args.sizes:
        g = _synth(n, args.degree)
        for dev in devs:
            for backend in ("scatter", "coo"):
                spec = ModelSpec(n_in=8, n_out=4, backend=backend)
                m = SparseConnectomeRNN.sparse_from_graph(g, spec).to(dev)
                x = torch.randn(args.steps, 1, 8, device=dev)
                est = estimate_memory_bytes(g.n, g.n_edges, 1, args.steps, sum(p.numel() for p in m.parameters()))
                try:
                    def run(m=m, x=x):
                        y, _ = m(x)
                        y.square().mean().backward()
                    run()  # warm-up
                    if dev == "cuda":
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    t = time.perf_counter()
                    for _ in range(args.reps):
                        m.zero_grad()
                        run()
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    dt = (time.perf_counter() - t) / args.reps
                    row = {"n": g.n, "edges": g.n_edges, "device": dev, "backend": backend,
                           "steps": args.steps, "sec_per_fwd_bwd": dt,
                           "est_gib_theoretical": est["total_gib"]}
                    if dev == "cuda":
                        row["peak_cuda_alloc_gib_measured"] = torch.cuda.max_memory_allocated() / 2**30
                except RuntimeError as exc:  # e.g. OOM: record, don't continue silently
                    row = {"n": g.n, "device": dev, "backend": backend, "error": str(exc)[:200]}
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                rows.append(row)
                print(json.dumps(row))
    return 0


def _cmd_controls(args) -> int:
    from flycircuit.graph.controls import degrees_equal, null_ensemble

    g = _synth(args.n, args.degree)
    for r in null_ensemble(g, k=args.k):
        s = {k: v for k, v in r.stats.items() if k != "overlap_curve"}
        s["degrees_preserved"] = degrees_equal(g, r.graph)
        print(json.dumps(s))
    return 0


def _cmd_match(args) -> int:
    from flycircuit.models.baselines import matching_report
    from flycircuit.models.sparse_rnn import ModelSpec, SparseConnectomeRNN

    g = _synth(args.n, args.degree)
    m = SparseConnectomeRNN.sparse_from_graph(g, ModelSpec(n_in=8, n_out=4))
    print(json.dumps(matching_report(m, 8, 4), indent=2))
    return 0


def register(sub) -> None:
    p = sub.add_parser("models", help="sparse connectome RNN, baselines, null controls (synthetic)")
    ms = p.add_subparsers(dest="models_cmd", required=True)
    b = ms.add_parser("benchmark", help="sparse CPU vs GPU fwd+bwd timing")
    b.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000])
    b.add_argument("--degree", type=int, default=30)
    b.add_argument("--steps", type=int, default=32)
    b.add_argument("--reps", type=int, default=3)
    b.set_defaults(func=_cmd_benchmark)
    c = ms.add_parser("controls", help="null-graph ensemble stats on a synthetic graph")
    c.add_argument("--n", type=int, default=300)
    c.add_argument("--degree", type=int, default=10)
    c.add_argument("--k", type=int, default=5)
    c.set_defaults(func=_cmd_controls)
    m = ms.add_parser("match", help="parameter/compute-matched baseline report")
    m.add_argument("--n", type=int, default=1000)
    m.add_argument("--degree", type=int, default=30)
    m.set_defaults(func=_cmd_match)
