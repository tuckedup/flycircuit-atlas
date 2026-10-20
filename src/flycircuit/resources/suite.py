"""Run the benchmark suite and derive ``configs/hardware/adaptive_profile.yaml``.

Must be run while holding the project heavy-job lock (scripts/linux/bench_profile.sh does this and
sets FLYCIRCUIT_HEAVY_HELD=1). All numbers in the YAML come from this run; none are assumed.
"""

from __future__ import annotations

import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import yaml

from flycircuit.config import project_root
from flycircuit.resources import bench, live

SIZES = {"pilot_1500": (1500, 7399), "mb_5608": (5608, 44024), "n20k": (20000, 600000),
         "n50k": (50000, 1500000)}
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
WIN_MIN_FREE_GIB = 1.0  # abort a phase if Windows free RAM falls below this
PREC_TOL = {"fp32": 1e-6, "tf32": 1e-3, "bf16": 5e-2}  # max rel err vs fp32 oracle (policy)


class Suite:
    def __init__(self, root: Path, quick: bool = False, log=print):
        self.root, self.quick, self.log = root, quick, log
        self.data: dict[str, Any] = {"phases": {}, "skipped": []}

    def guard(self, phase: str, need_gib: float = 0.0) -> bool:
        w = live.windows_memory()
        # same heuristic as adaptive.check: memory the VM already holds as reclaimable cache
        # can be reused without taking more from Windows
        from_windows = live.windows_growth_gib(need_gib, w, live.wsl_memory())
        if w is not None and from_windows > 0.05 and w["free_gib"] - from_windows < WIN_MIN_FREE_GIB:
            msg = (f"{phase}: Windows free RAM {w['free_gib']:.2f} GiB - up to {from_windows:.1f} "
                   f"GiB growth < {WIN_MIN_FREE_GIB} GiB floor")
            self.log("SKIP " + msg)
            self.data["skipped"].append(msg)
            return False
        return True

    # ---- phases -------------------------------------------------------------
    def cpu_threads(self) -> int:
        self.log("phase cpu_threads (model fwd/bwd, CPU)")
        sizes = ["mb_5608", "n20k"]
        cfgs = [{"n": SIZES[s][0], "e": SIZES[s][1], "batch": b, "tbptt": 32, "device": "cpu",
                 "reps": 4 if self.quick else 6, "tag": s} for s in sizes for b in (1, 8)]
        table: dict[int, list[dict]] = {}
        tl = [1, 2, 4, 8]
        for k in tl:
            table[k] = bench.run_worker("model", cfgs, threads=k, timeout=900)
            for r in table[k]:
                r["threads"] = k
        def score(k: int) -> float:  # geometric mean of per-config speed-up vs 1 thread
            ratios = [max(r.get("steps_per_s", 0), 1e-9) / max(table[1][i].get("steps_per_s", 1e-9), 1e-9)
                      for i, r in enumerate(table[k])]
            return statistics.geometric_mean(ratios)

        s4, s8 = score(4), score(8)
        if s8 > 1.10 * s4:  # test 16 only when 8 still helped materially
            table[16] = bench.run_worker("model", cfgs, threads=16, timeout=900)
            for r in table[16]:
                r["threads"] = 16
            tl.append(16)
        rel = {k: score(k) for k in tl}
        best_val = max(rel.values())
        best = min(k for k in tl if rel[k] >= 0.95 * best_val)
        by_cfg = {}  # per-workload best (smallest within 5% of that workload's best)
        for i, c in enumerate(cfgs):
            vals = {k: table[k][i].get("steps_per_s", 0) for k in tl}
            top = max(vals.values())
            by_cfg[f"{c['tag']}_b{c['batch']}"] = min(k for k in tl if vals[k] >= 0.95 * top)
        self.data["phases"]["cpu_threads"] = {"relative_throughput_vs_1_thread": rel,
                                              "tested": tl, "best_threads_model": best,
                                              "best_threads_by_workload": by_cfg,
                                              "tested_16": 16 in tl, "rows": table}
        self.log(f"  rel throughput {rel} -> best_threads_model={best}")
        return best

    def cpu_prep(self) -> int | None:
        """Best DuckDB/Arrow thread count, or None when NOT measured (never a silent default)."""
        self.log("phase cpu_prep (DuckDB/Arrow aggregation of edges_by_neuropil.parquet)")
        path = self.root / "data/processed/flywire/783/edges_by_neuropil.parquet"
        if not path.exists():
            self.data["skipped"].append("cpu_prep: edges_by_neuropil.parquet not present")
            return None
        if not self.guard("cpu_prep", need_gib=1.5):
            return None
        spill = self.root / "runs" / "tmp_bench_duckdb"
        spill.mkdir(parents=True, exist_ok=True)
        rows = []
        try:
            for k in [1, 2, 4, 8, 16]:
                r = bench.run_worker("prep", [{"path": str(path), "threads": k,
                                               "spill_dir": str(spill)}], threads=k, timeout=900)[0]
                r["threads"] = k
                rows.append(r)
                self.log(f"  threads={k}: duckdb {r.get('duckdb_groupby_s')} s, "
                         f"arrow {r.get('arrow_groupby_s')} s, rss {r.get('peak_rss_gib')}")
        finally:
            shutil.rmtree(spill, ignore_errors=True)  # project-owned temp only
        ok = [r for r in rows if "duckdb_groupby_s" in r]
        best = bench.pick_threads(ok, "duckdb_groupby_s", higher_is_better=False)
        self.data["phases"]["cpu_prep"] = {"best_threads_prep": best, "rows": rows}
        return best

    def device_compare(self, cpu_threads: int) -> dict[str, Any]:
        self.log("phase device_compare (CPU vs CUDA by size)")
        plan = {"pilot_1500": (1, 4, 16, 64), "mb_5608": (1, 4, 16, 64), "n20k": (1, 16)}
        if not self.quick:
            plan["n50k"] = (1,)
        combos = [(s, b) for s, bs in plan.items() for b in bs]
        # CPU rows run in ONE worker; the worker itself skips configs whose activation memory would
        # exceed the CPU benchmark cap (so a huge CPU batch can never squeeze Windows).
        base = {"tbptt": 32, "reps": 3}
        ccfg = [{**base, "n": SIZES[s][0], "e": SIZES[s][1], "batch": b, "device": "cpu", "tag": s}
                for s, b in combos]
        gcfg = [{**base, "reps": 5, "n": SIZES[s][0], "e": SIZES[s][1], "batch": b, "device": "cuda",
                 "tag": s, "reserve_gib": 1.2, "safety": 1.1} for s, b in combos]
        out = bench.run_worker("model", ccfg, threads=cpu_threads, timeout=1500)
        out += bench.run_worker("model", gcfg, threads=cpu_threads, timeout=900)
        cross = {}
        for s, b in combos:
            c = next((r for r in out if r.get("tag") == s and r.get("batch") == b
                      and r.get("device") == "cpu" and "steps_per_s" in r), None)
            g = next((r for r in out if r.get("tag") == s and r.get("batch") == b
                      and r.get("device") == "cuda" and "steps_per_s" in r), None)
            if c and g:
                cross[f"{s}_b{b}"] = {"cpu_steps_per_s": c["steps_per_s"],
                                      "cuda_steps_per_s": g["steps_per_s"],
                                      "cuda_speedup": g["steps_per_s"] / c["steps_per_s"],
                                      "edges": SIZES[s][1], "batch": b,
                                      "edges_x_batch": SIZES[s][1] * b}
            else:
                why = next((r.get("skipped") or r.get("error") for r in out
                            if r.get("tag") == s and r.get("batch") == b
                            and (r.get("skipped") or r.get("error"))), "no result")
                self.data["skipped"].append(f"device_compare {s} b{b}: {why}")
        self.data["phases"]["device_compare"] = {"summary": cross, "rows": out}
        for k, v in cross.items():
            self.log(f"  {k}: cuda/cpu = {v['cuda_speedup']:.2f}x")
        return cross

    def batch_sweep(self, cpu_threads: int) -> dict[str, int]:
        self.log("phase batch_sweep (CUDA)")
        plateau, rows_all = {}, {}
        for s in ("mb_5608", "n20k"):
            n, e = SIZES[s]
            cfgs = [{"n": n, "e": e, "batch": b, "tbptt": 32, "device": "cuda", "reps": 5,
                     "tag": s, "reserve_gib": 1.2, "safety": 1.1} for b in BATCHES]
            rows = bench.run_worker("model", cfgs, threads=cpu_threads, timeout=1500)
            rows_all[s] = rows
            plateau[s] = bench.plateau_batch(rows)
            self.log(f"  {s}: plateau batch {plateau[s]}; "
                     + ", ".join(f"b{r.get('batch')}={r.get('steps_per_s', 0):.0f}/s"
                                 + ("(skip)" if "skipped" in r else "") for r in rows))
        self.data["phases"]["batch_sweep"] = {"plateau_batch": plateau, "rows": rows_all}
        return plateau

    def seq_sweep(self, cpu_threads: int, batch: int) -> None:
        self.log("phase seq_sweep (tbptt length)")
        n, e = SIZES["mb_5608"]
        cfgs = [{"n": n, "e": e, "batch": batch, "tbptt": t, "device": "cuda", "reps": 4,
                 "tag": "mb_5608", "reserve_gib": 1.2} for t in (16, 32, 64, 128)]
        self.data["phases"]["seq_sweep"] = {
            "batch": batch, "rows": bench.run_worker("model", cfgs, threads=cpu_threads, timeout=900)}

    def precision(self, cpu_threads: int, plateau: dict[str, int]) -> dict[str, Any]:
        self.log("phase precision / compile (numerical check vs fp32 oracle)")
        rows = []
        for s in ("mb_5608", "n20k"):
            n, e = SIZES[s]
            b = min(plateau[s], 16)
            variants = [{"precision": "fp32"}, {"precision": "tf32"}, {"precision": "bf16"},
                        {"precision": "fp32", "compile": True}]
            cfgs = [{"n": n, "e": e, "batch": b, "tbptt": 32, "device": "cuda", "reps": 5,
                     "tag": s, "reserve_gib": 1.2, **v} for v in variants]
            if not self.guard(f"precision/{s}", need_gib=1.2):
                continue
            rows += bench.run_worker("model", cfgs, threads=cpu_threads, timeout=1200)
        rec: dict[str, Any] = {}
        for s in ("mb_5608", "n20k"):
            base = next((r for r in rows if r.get("tag") == s and r.get("precision") == "fp32"
                         and not r.get("compile") and "steps_per_s" in r), None)
            for r in rows:
                if r.get("tag") != s:
                    continue
                name = ("compile_fp32" if r.get("compile") else r.get("precision"))
                key = f"{s}/{name}"
                if "steps_per_s" not in r:
                    rec[key] = {"usable": False, "reason": r.get("error") or r.get("skipped")}
                    continue
                tol = PREC_TOL.get(r["precision"], 1e-6)
                ok = r["y_rel_err_vs_fp32"] <= tol and r["grad_rel_l2_err_vs_fp32"] <= max(tol, 1e-5)
                sp = r["steps_per_s"] / base["steps_per_s"] if base else None
                rec[key] = {"usable": bool(ok), "speedup_vs_fp32": sp,
                            "y_rel_err": r["y_rel_err_vs_fp32"],
                            "grad_rel_l2_err": r["grad_rel_l2_err_vs_fp32"],
                            "compile_seconds": r.get("compile_seconds"),
                            "peak_cuda_alloc_gib": r.get("peak_cuda_alloc_gib")}
        self.data["phases"]["precision"] = {"summary": rec, "tolerances": PREC_TOL, "rows": rows}
        for k, v in rec.items():
            self.log(f"  {k}: {v}")
        return rec

    def transfer(self) -> None:
        self.log("phase transfer")
        self.data["phases"]["transfer"] = {"rows": bench.run_worker(
            "transfer", [{"mib": m} for m in (1, 16, 128)], timeout=300)}

    def ctx(self) -> dict[str, Any]:
        r = bench.run_worker("ctx", [{}], timeout=300)[0]
        self.data["phases"]["cuda_context"] = r
        self.log(f"  cuda context {r}")
        return r

    def concurrency(self, cpu_threads: int, batch: int) -> dict[str, Any]:
        self.log("phase gpu_concurrency (1 vs 2 processes)")
        n, e = SIZES["mb_5608"]
        cfg = {"n": n, "e": e, "batch": batch, "tbptt": 32, "device": "cuda", "reps": 5,
               "duration": 8.0, "tag": "mb_5608", "reserve_gib": 1.2}
        if not self.guard("gpu_concurrency", need_gib=2.5):
            return {}
        solo = bench.run_worker("model", [cfg], threads=cpu_threads, timeout=600)[0]
        ps = [bench.spawn("model", [cfg], threads=max(1, cpu_threads // 2)) for _ in range(2)]
        duo = [bench.collect(p, 600)[0] for p in ps]
        res: dict[str, Any] = {"cfg": cfg, "solo": solo, "duo": duo}
        if "steps_per_s" in solo and all("steps_per_s" in d for d in duo):
            agg = sum(d["steps_per_s"] for d in duo)
            res["solo_steps_per_s"] = solo["steps_per_s"]
            res["two_process_aggregate_steps_per_s"] = agg
            res["aggregate_ratio"] = agg / solo["steps_per_s"]
            res["per_process_slowdown"] = solo["steps_per_s"] / statistics.fmean(
                d["steps_per_s"] for d in duo)
            res["decision_gpu_processes"] = 2 if res["aggregate_ratio"] >= 1.15 else 1
        else:
            res["decision_gpu_processes"] = 1
            res["note"] = "concurrency measurement incomplete; defaulting to 1 process"
        self.data["phases"]["gpu_concurrency"] = res
        self.log(f"  concurrency: {({k: v for k, v in res.items() if k not in ('solo', 'duo', 'cfg')})}")
        return res

    # ---- derivation -----------------------------------------------------------
    def _all_rows(self):
        for ph in self.data["phases"].values():
            rows = ph.get("rows") if isinstance(ph, dict) else None
            if isinstance(rows, dict):
                for rr in rows.values():
                    yield from (r for r in rr if isinstance(r, dict))
            elif isinstance(rows, list):
                yield from (r for r in rows if isinstance(r, dict))

    def calibration(self) -> dict[str, Any]:
        """peak_alloc ~= fixed + ratio * theoretical. ``ratio`` comes only from runs whose theoretical
        footprint is >= 0.2 GiB (tiny runs are dominated by fixed overhead and would inflate it);
        ``fixed`` is the largest observed positive residual."""
        pts, rss = [], []
        for r in self._all_rows():
            if r.get("peak_cuda_alloc_gib") and r.get("theoretical_cuda_gib"):
                pts.append((r["theoretical_cuda_gib"], r["peak_cuda_alloc_gib"]))
            if r.get("peak_rss_gib") and r.get("device") == "cuda":
                rss.append(r["peak_rss_gib"])
        big = [p / t for t, p in pts if t >= 0.2]
        ratio = max(big) if big else None
        fixed = max((p - ratio * t for t, p in pts), default=0.0) if ratio else None
        cal = {
            "n_points": len(pts), "n_points_ratio_fit": len(big),
            "cuda_ratio_max_theo_ge_0p2gib": ratio,
            "cuda_ratio_median_all": statistics.median(p / t for t, p in pts) if pts else None,
            "cuda_fixed_gib": max(0.0, fixed) if fixed is not None else None,
            "cuda_process_rss_gib_max": max(rss) if rss else None,
        }
        self.data["phases"]["calibration"] = cal
        return cal


def run_suite(quick: bool = False, out: Path | None = None, log=print) -> dict[str, Any]:
    root = project_root()
    s = Suite(root, quick, log)
    before = live.snapshot()
    s.data["snapshot_before"] = before
    t0 = time.time()
    load: dict[str, float] = {"start": before["cpu"]["load_1min"]}

    def mark(name: str) -> None:  # contention evidence: 1-min load average at each phase end
        load[name] = live.cpu_state()["load_1min"]

    threads_model = s.cpu_threads()
    mark("cpu_threads")
    threads_prep = s.cpu_prep()
    mark("cpu_prep")
    s.ctx()
    s.device_compare(threads_model)
    mark("device_compare")
    plateau = s.batch_sweep(threads_model)
    mark("batch_sweep")
    s.seq_sweep(threads_model, min(plateau["mb_5608"], 16))
    prec = s.precision(threads_model, plateau)
    mark("precision")
    s.transfer()
    conc = s.concurrency(threads_model, min(plateau["mb_5608"], 16))
    mark("concurrency")
    s.data["load_avg_1min_by_phase"] = load
    cal = s.calibration()
    after = live.snapshot()
    s.data["snapshot_after"] = after
    s.data["elapsed_s"] = time.time() - t0
    profile = derive_profile(s.data, before, threads_model, threads_prep, plateau, prec, conc, cal)
    outp = out or (root / "configs/hardware/adaptive_profile.yaml")
    raw = root / "results" / "resources" / f"profile_raw_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(s.data, indent=1, default=str))
    profile["provenance"]["raw_results"] = str(raw.relative_to(root))
    header = ("# MEASURED adaptive resource profile (generated by `flycircuit resources profile`).\n"
              "# Supersedes configs/hardware/atul_laptop.yaml (kept as the historical conservative\n"
              "# starting limits). `policy:` and `hardware_profile:` are decisions; `measured:` is from the\n"
              "# benchmark run referenced in provenance.raw_results. Re-run before heavy campaigns.\n")
    outp.write_text(header + yaml.safe_dump(profile, sort_keys=False))
    log(f"wrote {outp} and {raw}")
    return profile


def rederive(raw_new: Path, carry_from: Path | None, out: Path, log=print) -> dict[str, Any]:
    """Rebuild the profile YAML from stored raw JSON WITHOUT running any benchmark.

    Phases that the new run skipped (e.g. under Windows memory pressure) can be carried forward from
    an older raw file; each carried phase is labelled ``carried_from`` so nothing is passed off as
    freshly measured. A phase absent from both stays absent (value null / not measured)."""
    root = project_root()
    data = json.loads(Path(raw_new).read_text())
    carried: dict[str, str] = {}
    if carry_from:
        old = json.loads(Path(carry_from).read_text())
        for ph in ("precision", "gpu_concurrency", "cpu_prep"):
            cur = data["phases"].get(ph) or {}
            empty = not cur or (not cur.get("summary") and not cur.get("rows")
                                and cur.get("best_threads_prep") is None
                                and "decision_gpu_processes" not in cur)
            if empty and old["phases"].get(ph):
                data["phases"][ph] = {**old["phases"][ph], "carried_from": str(carry_from)}
                carried[ph] = str(carry_from)
    s = Suite(root)
    s.data = data
    prec = data["phases"].get("precision", {}).get("summary", {})
    conc = data["phases"].get("gpu_concurrency", {})
    cal = s.calibration()  # recomputed over all rows incl. carried ones
    plateau = data["phases"]["batch_sweep"]["plateau_batch"]
    tp = data["phases"].get("cpu_prep", {}).get("best_threads_prep")
    profile = derive_profile(data, data["snapshot_before"], data["phases"]["cpu_threads"]["best_threads_model"],
                             tp, plateau, prec, conc, cal)
    profile["provenance"]["raw_results"] = str(Path(raw_new))
    profile["provenance"]["carried_forward_phases"] = carried
    if carry_from:  # run-to-run spread of the CPU thread table (machine shared with other agents)
        profile["measured"]["cpu"]["relative_throughput_other_run"] = (
            old["phases"]["cpu_threads"]["relative_throughput_vs_1_thread"])
        profile["measured"]["cpu"]["other_run_best_threads_model"] = (
            old["phases"]["cpu_threads"]["best_threads_model"])
    profile["provenance"]["rederived_without_benchmarking"] = True
    header = ("# MEASURED adaptive resource profile (`flycircuit resources profile --rederive ...`).\n"
              "# Supersedes configs/hardware/atul_laptop.yaml (kept as the historical conservative\n"
              "# starting limits). `policy:` and `hardware_profile:` are decisions; `measured:` comes from\n"
              "# the raw benchmark JSON(s) in provenance. Phases in provenance.carried_forward_phases come\n"
              "# from an earlier run on the same machine; null means NOT measured.\n")
    out.write_text(header + yaml.safe_dump(profile, sort_keys=False))
    data_path = Path(raw_new).with_name(Path(raw_new).stem + "_merged.json")
    data_path.write_text(json.dumps(data, indent=1, default=str))
    log(f"wrote {out}; merged raw {data_path}; carried: {carried}")
    return profile


def derive_crossover(dev: dict[str, Any]) -> dict[str, Any]:
    """CUDA-vs-CPU rule in terms of edges x batch (work per step), from measured rows only."""
    pts = sorted((v["edges_x_batch"], v["cuda_speedup"]) for v in dev.values() if "edges_x_batch" in v)
    wins = [p for p, sp in pts if sp > 1.0]
    losses = [p for p, sp in pts if sp <= 1.0]
    return {
        "points_edges_x_batch_vs_cuda_speedup": pts,
        "smallest_tested_edges_x_batch_where_cuda_wins": min(wins) if wins else None,
        "largest_tested_edges_x_batch_where_cpu_wins": max(losses) if losses else None,
        "consistent_threshold": (not wins or not losses or max(losses) < min(wins)),
        "note": "crossover lies between the two values; not resolved more finely than the tested grid",
    }


def compile_summary(prec_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for tag in {r.get("tag") for r in prec_rows}:
        e = next((r for r in prec_rows if r.get("tag") == tag and not r.get("compile")
                  and r.get("precision") == "fp32" and "ms_per_window_median" in r), None)
        c = next((r for r in prec_rows if r.get("tag") == tag and r.get("compile")
                  and "ms_per_window_median" in r), None)
        if e and c and e["ms_per_window_median"] > c["ms_per_window_median"]:
            saving_s = (e["ms_per_window_median"] - c["ms_per_window_median"]) / 1e3
            out[tag] = {"eager_ms": e["ms_per_window_median"], "compiled_ms": c["ms_per_window_median"],
                        "compile_seconds": c.get("compile_seconds"),
                        "break_even_windows": (c["compile_seconds"] or 0) / saving_s,
                        "batch": c["batch"], "tbptt": c["tbptt"]}
    return out


def derive_profile(data, before, threads_model, threads_prep, plateau, prec, conc, cal):
    win = before.get("windows") or {}
    wsl, gpu = before["wsl"], before["gpu"]
    gpu_total = gpu.get("total_gib", 8.0)
    cuda_reserve = round(max(1.0, 0.15 * gpu_total), 2)
    wsl_headroom = 0.75
    win_reserve = 1.0
    dev = data["phases"].get("device_compare", {}).get("summary", {})
    cpu_phase = data["phases"]["cpu_threads"]
    # tf32 / bf16 / fp32 numeric formats (compile handled separately)
    fmt = {k: v for k, v in prec.items() if not k.endswith("/compile_fp32")}
    fmt_ok = {k: v for k, v in fmt.items() if v.get("usable") and (v.get("speedup_vs_fp32") or 0) >= 1.10
              and not k.endswith("/fp32")}
    comp = compile_summary(data["phases"].get("precision", {}).get("rows", []))
    rss_job = max(cal.get("cuda_process_rss_gib_max") or 1.0, 1.0)
    mem_budget = max(1.0, wsl["total_gib"] - wsl_headroom - 0.75)
    cpu_conc = max(1, min(before["cpu"]["physical"] // max(1, threads_model), int(mem_budget // rss_job)))
    gpu_procs = conc.get("decision_gpu_processes", 1) if conc else 1
    hw = {
        "profile": "adaptive_v1", "cpu_threads": threads_model, "dataloader_workers": 0,
        "experiment_concurrency": gpu_procs,  # GPU jobs; CPU-only concurrency is under measured.cpu
        "max_pipeline_ram_gib": round(wsl["total_gib"] - 1.5, 2),
        "max_cuda_allocated_gib": round(gpu_total - cuda_reserve, 2),
        "min_host_free_disk_gib": 40, "max_new_project_storage_gib": 60,
        "initial_batch_size": 1, "initial_tbptt_steps": 32, "initial_training_seeds": 2,
        "confirmatory_training_seeds": 5, "initial_trial_walltime_minutes": 30,
        "max_default_job_walltime_minutes": 480, "max_default_sweep_walltime_minutes": 1440,
        "whole_brain_gradient_training": False, "full_synapse_download_default": False,
        "cuda_headroom_gib": cuda_reserve, "host_ram_headroom_gib": wsl_headroom,
    }
    return {
        "profile": "adaptive_v1",
        "generated_utc": before["time"],
        "kind": "measured_with_policy_fields",
        "provenance": {"suite_elapsed_s": data["elapsed_s"], "skipped_phases": data["skipped"],
                       "load_avg_1min_by_phase": data.get("load_avg_1min_by_phase"),
                       "caveat": "other project agents may share the CPU; load averages include "
                                 "this benchmark's own threads. Re-run on a quiet machine to refine."},
        "host_at_measurement": {"windows": win, "wsl": wsl, "gpu": gpu, "cpu": before["cpu"]},
        "measured": {
            "cpu": {"best_threads_model": threads_model,
                    "best_threads_by_workload": cpu_phase.get("best_threads_by_workload"),
                    "best_threads_prep": threads_prep,  # null => NOT measured (see skipped_phases)
                    "tested_16_threads": cpu_phase["tested_16"],
                    "relative_throughput_vs_1_thread": cpu_phase["relative_throughput_vs_1_thread"],
                    "cpu_only_job_concurrency": cpu_conc},
            "gpu": {"device_rule": derive_crossover(dev),
                    "plateau_batch_tbptt32": plateau,
                    "precision": prec,
                    "recommended_numeric_format": ("fp32" if not fmt_ok else max(
                        fmt_ok, key=lambda k: fmt_ok[k]["speedup_vs_fp32"])),
                    "torch_compile": {"per_workload": comp,
                                      "caveat": "verified on model fwd/bwd only, not inside the "
                                                "plasticity/training harnesses; shape changes recompile"},
                    "gpu_processes": gpu_procs, "concurrency": {
                        k: v for k, v in (conc or {}).items() if k not in ("solo", "duo", "cfg")},
                    "cuda_context_gib": data["phases"].get("cuda_context", {}).get("cuda_context_gib")},
            "calibration": cal,
        },
        "policy": {  # decisions, not measurements
            "cuda_reserve_gib": cuda_reserve, "windows_free_reserve_gib": win_reserve,
            "wsl_ram_headroom_gib": wsl_headroom, "downsize_before_refusing": True,
            "note": "Live free memory is re-read before each job; these reserves are floors.",
        },
        "hardware_profile": hw,
    }
