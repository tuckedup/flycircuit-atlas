"""Measured benchmarks behind the adaptive profile.

Workers run in a fresh subprocess so thread settings (set via env BEFORE numpy/torch import) and
peak-RSS are clean per configuration. Orchestrator functions only parse worker JSON.

Worker protocol::

    python -m flycircuit.resources.bench worker <mode> '<json list of configs>'
    -> prints one line  ``RESULT <json list of result dicts>``

Everything here is measurement code; nothing is cached between runs and no number is assumed.
"""

from __future__ import annotations

import contextlib
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from typing import Any

GIB = 1024**3
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


# --------------------------------------------------------------------------- workers
def _rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def _build(n: int, e: int, batch: int, tbptt: int, device: str, seed: int):
    import numpy as np
    import torch

    from flycircuit.models.sparse_rnn import ModelSpec, SparseConnectomeRNN

    rng = np.random.default_rng(seed)
    pre, post = rng.integers(0, n, e), rng.integers(0, n, e)
    cnt = rng.integers(5, 40, e).astype(np.float32)
    m = SparseConnectomeRNN(n, pre, post, cnt, ModelSpec(n_in=8, n_out=2, seed=seed),
                            unknown_sign_policy="random_fixed").to(device)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(tbptt, batch, 8, generator=g).to(device)
    return m, x


def _window(m, x, precision: str, fwd=None):
    import torch

    m.zero_grad(set_to_none=True)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else contextlib.nullcontext()
    with ctx:
        y, _ = (fwd if fwd is not None else m)(x)
        loss = y.float().pow(2).mean()
    loss.backward()
    return y.detach().float(), m.raw_gain.grad.detach().clone()


def _sync(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()


def _model_one(c: dict[str, Any]) -> dict[str, Any]:
    import torch

    from flycircuit.resources.estimate import estimate_sparse_rnn

    n, e, b, t = c["n"], c["e"], c.get("batch", 1), c.get("tbptt", 32)
    dev, prec = c.get("device", "cpu"), c.get("precision", "fp32")
    res: dict[str, Any] = {k: c.get(k) for k in ("n", "e", "batch", "tbptt", "device", "precision",
                                                 "compile", "tag")}
    est = estimate_sparse_rnn(n, e, b, t)
    res["theoretical_cuda_gib"] = est["cuda_gib"]
    if dev == "cpu":  # autograd-saved activations live in host RAM: cap them so Windows is safe
        import psutil

        need = est["cuda_gib"] * 1.3 + 0.6
        avail = psutil.virtual_memory().available / GIB - 0.75
        if need > c.get("cpu_cap_gib", 2.5) or need > avail:
            return {**res, "skipped": f"CPU activation estimate {need:.2f} GiB exceeds benchmark cap "
                                      f"{c.get('cpu_cap_gib', 2.5)} GiB or WSL available {avail:.2f} GiB"}
    if dev == "cuda":
        free, _tot = torch.cuda.mem_get_info()
        limit = free / GIB - c.get("reserve_gib", 1.0)
        if est["cuda_gib"] * c.get("safety", 1.5) > limit:
            return {**res, "skipped": f"theoretical {est['cuda_gib']:.2f} GiB x safety exceeds "
                                      f"free-minus-reserve {limit:.2f} GiB"}
    try:
        torch.backends.cuda.matmul.allow_tf32 = prec == "tf32"
        torch.set_float32_matmul_precision("high" if prec == "tf32" else "highest")
        m, x = _build(n, e, b, t, dev, c.get("seed", 0))
        # fp32 oracle (same weights/inputs) for numerical-correctness comparison
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        y0, g0 = _window(m, x, "fp32")
        torch.backends.cuda.matmul.allow_tf32 = prec == "tf32"
        torch.set_float32_matmul_precision("high" if prec == "tf32" else "highest")
        fwd, t_compile = None, None
        if c.get("compile"):
            t0 = time.perf_counter()
            fwd = torch.compile(m)
            _window(m, x, prec, fwd)
            _sync(dev)
            t_compile = time.perf_counter() - t0
        for _ in range(2):  # warmup
            y, g = _window(m, x, prec, fwd)
        _sync(dev)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        times, t_end = [], time.perf_counter() + c.get("duration", 0.0)
        while len(times) < c.get("reps", 5) or time.perf_counter() < t_end:
            t0 = time.perf_counter()
            y, g = _window(m, x, prec, fwd)
            _sync(dev)
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        denom = max(float(y0.abs().max()), 1e-12)
        res.update({
            "ms_per_window_median": med * 1e3, "ms_per_window_min": min(times) * 1e3,
            "windows": len(times), "windows_per_s": len(times) / sum(times),
            "steps_per_s": b * t / med,  # batch*time-steps per second (median window)
            "y_rel_err_vs_fp32": float((y - y0).abs().max()) / denom,
            "grad_rel_l2_err_vs_fp32": float((g - g0).norm() / g0.norm().clamp_min(1e-30)),
            "compile_seconds": t_compile,
            "peak_cuda_alloc_gib": torch.cuda.max_memory_allocated() / GIB if dev == "cuda" else None,
            "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / GIB if dev == "cuda" else None,
            "torch_threads": torch.get_num_threads(), "peak_rss_gib": _rss_gib(),
        })
        del m, x
    except torch.cuda.OutOfMemoryError as exc:
        res["error"] = f"cuda_oom: {str(exc)[:120]}"
    except Exception as exc:  # compile/triton failures are data, not crashes
        res["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        if dev == "cuda":
            torch.cuda.empty_cache()
    return res


def _prep_one(c: dict[str, Any]) -> dict[str, Any]:
    import duckdb
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    path, k = c["path"], c["threads"]
    out: dict[str, Any] = {"threads": k, "path": path}
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={k}")
    con.execute(f"PRAGMA memory_limit='{c.get('memory_limit', '1000MB')}'")
    if c.get("spill_dir"):  # project-owned spill directory; caller removes it
        con.execute(f"PRAGMA temp_directory='{c['spill_dir']}'")
    cols = [r[0] for r in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [path]).fetchall()]
    pre, post = cols[0], cols[1]
    val = next((x for x in cols if x in ("count", "syn_count", "n_syn", "contacts")), cols[-1])
    out["columns"] = cols
    t0 = time.perf_counter()
    n = con.execute(f'SELECT count(*) FROM (SELECT "{pre}","{post}",sum("{val}") s '
                    f'FROM read_parquet(?) GROUP BY 1,2)', [path]).fetchone()[0]
    out["duckdb_groupby_s"] = time.perf_counter() - t0
    out["duckdb_unique_pairs"] = n
    pa.set_cpu_count(k)
    pa.set_io_thread_count(max(1, k))
    # Arrow part is bounded: stream record batches until ``arrow_max_rows`` (never read_all).
    t0 = time.perf_counter()
    batches, got = [], 0
    for rb in pq.ParquetFile(path).iter_batches(batch_size=500_000, columns=[pre, post, val]):
        batches.append(rb)
        got += rb.num_rows
        if got >= c.get("arrow_max_rows", 4_000_000):
            break
    tbl = pa.Table.from_batches(batches)
    out["arrow_read_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    agg = tbl.group_by([pre, post]).aggregate([(val, "sum")])
    out["arrow_groupby_s"] = time.perf_counter() - t0
    out["arrow_unique_pairs"] = agg.num_rows
    out["total_contacts"] = int(pc.sum(tbl[val]).as_py())
    out["rows"] = tbl.num_rows
    out["peak_rss_gib"] = _rss_gib()
    return out


def _transfer_one(c: dict[str, Any]) -> dict[str, Any]:
    import torch

    mib = c["mib"]
    n = mib * 1024 * 1024 // 4
    res: dict[str, Any] = {"mib": mib}
    for pinned in (False, True):
        a = torch.empty(n)
        a = a.pin_memory() if pinned else a
        d = torch.empty(n, device="cuda")
        for direction in ("h2d", "d2h"):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                (d.copy_(a) if direction == "h2d" else a.copy_(d))
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 10
            res[f"{'pinned' if pinned else 'pageable'}_{direction}_GBps"] = (n * 4 / 1e9) / dt
            res[f"{'pinned' if pinned else 'pageable'}_{direction}_ms"] = dt * 1e3
    return res


def _ctx_one(_c: dict[str, Any]) -> dict[str, Any]:
    import torch

    from flycircuit.resources.live import gpu_state

    r0 = _rss_gib()
    g0 = gpu_state().get("used_gib")
    torch.zeros(1, device="cuda")
    (torch.randn(8, 8, device="cuda", requires_grad=True) @ torch.randn(8, 8, device="cuda")).sum().backward()
    torch.cuda.synchronize()
    g1 = gpu_state().get("used_gib")
    return {"cuda_context_gib": (g1 - g0) if g0 is not None and g1 is not None else None,
            "rss_gib_before_cuda": r0, "rss_gib_after_cuda": _rss_gib()}


_MODES = {"model": _model_one, "prep": _prep_one, "transfer": _transfer_one, "ctx": _ctx_one}


def worker_main(mode: str, configs_json: str) -> int:
    results = [_MODES[mode](c) for c in json.loads(configs_json)]
    print("RESULT " + json.dumps(results, default=str), flush=True)
    return 0


# --------------------------------------------------------------------------- orchestration
def _env(threads: int | None) -> dict[str, str]:
    env = dict(os.environ)
    if threads is not None:
        for v in THREAD_VARS:
            env[v] = str(threads)
    return env


def spawn(mode: str, configs: list[dict[str, Any]], threads: int | None = None) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "flycircuit.resources.bench", "worker", mode, json.dumps(configs)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_env(threads))


def collect(p: subprocess.Popen, timeout: float = 900) -> list[dict[str, Any]]:
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        return [{"error": f"worker timeout after {timeout}s"}]
    for line in reversed(out.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return [{"error": f"worker rc={p.returncode} no RESULT; stderr tail: {err[-400:]}"}]


def run_worker(mode: str, configs: list[dict[str, Any]], threads: int | None = None,
               timeout: float = 900) -> list[dict[str, Any]]:
    return collect(spawn(mode, configs, threads), timeout)


def pick_threads(rows: list[dict[str, Any]], key: str, higher_is_better: bool,
                 tolerance: float = 0.05) -> int:
    """Smallest thread count within ``tolerance`` of the best (avoids needless oversubscription)."""
    ok = [r for r in rows if key in r and r.get("error") is None]
    if not ok:
        raise ValueError("no successful benchmark rows")
    best = max(r[key] for r in ok) if higher_is_better else min(r[key] for r in ok)
    good = [r for r in ok if (r[key] >= best * (1 - tolerance) if higher_is_better
                              else r[key] <= best * (1 + tolerance))]
    return int(min(r["threads"] for r in good))


def plateau_batch(rows: list[dict[str, Any]], frac: float = 0.95) -> int:
    """Smallest batch reaching ``frac`` of the best measured throughput among batches that fit.

    (A first-gain<10% stopping rule is fooled by noise/non-monotone curves; this is not.)"""
    ok = [r for r in rows if "steps_per_s" in r and r.get("error") is None]
    if not ok:
        raise ValueError("no successful rows")
    top = max(r["steps_per_s"] for r in ok)
    return int(min(r["batch"] for r in ok if r["steps_per_s"] >= frac * top))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "worker":
        sys.exit(worker_main(sys.argv[2], sys.argv[3]))
    sys.exit("usage: python -m flycircuit.resources.bench worker <mode> '<json configs>'")
