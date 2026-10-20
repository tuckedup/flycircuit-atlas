import json
import sys
import time

import pytest

from flycircuit.resources import adaptive, bench, registry

CAL = {"cuda_reserve_gib": 1.2, "windows_free_reserve_gib": 1.0, "wsl_ram_headroom_gib": 0.75,
       "cuda_ratio": 1.5, "cuda_context_gib": 0.4, "process_rss_gib": 1.5, "gpu_processes": 1}


def snap(win_free=8.0, wsl_avail=6.0, reclaim=3.0, gpu_free=6.5, gpu_util=0.0, windows=True):
    return {
        "windows": ({"free_gib": win_free, "total_gib": 15.0, "commit_free_gib": 10.0,
                     "commit_total_gib": 40.0} if windows else None),
        "wsl": {"total_gib": 7.4, "available_gib": wsl_avail, "memfree_gib": 1.0,
                "reclaimable_gib": reclaim, "swap_used_gib": 0.0, "swap_total_gib": 2.0},
        "gpu": {"available": True, "used_gib": 1.3, "free_gib": gpu_free, "total_gib": 8.0,
                "util_pct": gpu_util, "temp_c": 50.0, "compute_apps": []},
        "cpu": {"logical": 16, "physical": 8, "load_1min": 0.5, "util_pct": 3.0},
    }


REQ = {"neurons": 5608, "edges": 44024, "batch": 1, "tbptt": 32}


def test_small_job_passes_with_roomy_host():
    r = adaptive.check(REQ, snap(), CAL)
    assert r["ok"] and not r["failures"]


def test_refuses_when_windows_would_be_squeezed():
    # needs ~2 GiB RAM, VM cache can supply only 0.2 GiB -> ~1.8 GiB from Windows, only 1.5 free
    r = adaptive.check(REQ, snap(win_free=1.5, reclaim=0.2), CAL)
    assert not r["ok"] and any("Windows RAM" in f for f in r["failures"])


def test_vm_reclaimable_cache_does_not_count_against_windows():
    r = adaptive.check(REQ, snap(win_free=1.2, reclaim=4.4), CAL)
    assert r["ok"], r["failures"]


def test_windows_growth_uses_vmmem_footprint_when_known():
    from flycircuit.resources.live import windows_growth_gib

    s = snap(wsl_avail=5.5)  # Linux used = 7.4 - 5.5 = 1.9 GiB
    s["windows"]["vmmem_gib"] = 3.75
    assert windows_growth_gib(1.5, s["windows"], s["wsl"]) == 0.0  # fits in current VM footprint
    assert windows_growth_gib(3.0, s["windows"], s["wsl"]) == pytest.approx(1.9 + 3.0 - 3.75)
    s["windows"]["vmmem_gib"] = None  # fallback: need minus reclaimable cache
    assert windows_growth_gib(5.0, s["windows"], s["wsl"]) == pytest.approx(5.0 - 3.0)


def test_unreadable_windows_is_a_warning_not_a_guess():
    r = adaptive.check(REQ, snap(windows=False), CAL)
    assert r["ok"] and any("NOT checked" in w for w in r["warnings"])


def test_refuses_when_wsl_ram_short():
    r = adaptive.check(REQ, snap(wsl_avail=1.0), CAL)
    assert not r["ok"] and any("WSL RAM" in f for f in r["failures"])


def test_refuses_on_cuda_headroom_with_numbers():
    big = {"neurons": 20000, "edges": 600000, "batch": 128, "tbptt": 64}
    r = adaptive.check(big, snap(gpu_free=2.0), CAL)
    assert not r["ok"] and any("CUDA" in f and "reserve" in f for f in r["failures"])


def test_cpu_only_ignores_gpu():
    r = adaptive.check(REQ, snap(gpu_free=0.1), CAL, use_cuda=False)
    assert r["ok"]


def test_downsize_halves_batch_then_tbptt_and_keeps_graph_size():
    big = {"neurons": 20000, "edges": 600000, "batch": 256, "tbptt": 32}
    r = adaptive.downsize(big, snap(gpu_free=3.0), CAL)
    assert r["ok"] and r["downsized"]
    assert r["request"]["neurons"] == 20000 and r["request"]["edges"] == 600000
    assert r["request"]["batch"] < 256
    assert "SAME reduction" in r["advice"]
    assert r["original_failures"]


def test_downsize_gives_up_with_advice():
    huge = {"neurons": 140000, "edges": 50_000_000, "batch": 1, "tbptt": 32}
    r = adaptive.downsize(huge, snap(), CAL)
    assert not r["ok"] and not r["downsized"] and "does not fit" in r["advice"]


def test_calibration_uses_measured_values_with_margin():
    prof = {"policy": {"cuda_reserve_gib": 1.5},
            "measured": {"calibration": {"cuda_ratio_max_theo_ge_0p2gib": 1.2, "cuda_fixed_gib": 0.1,
                                         "cuda_process_rss_gib_max": 2.2},
                         "gpu": {"cuda_context_gib": 0.35, "gpu_processes": 2}}}
    c = adaptive.calibration(prof)
    assert c["cuda_reserve_gib"] == 1.5 and c["cuda_ratio"] == pytest.approx(1.38)
    assert c["cuda_fixed_gib"] == pytest.approx(0.125)
    assert c["process_rss_gib"] == 2.2 and c["gpu_processes"] == 2
    assert adaptive.calibration(None)["cuda_ratio"] == adaptive.DEFAULTS["cuda_ratio"]


def test_suite_calibration_ignores_tiny_runs_for_ratio(tmp_path):
    from flycircuit.resources.suite import Suite

    s = Suite(tmp_path)
    s.data["phases"]["x"] = {"rows": [
        {"theoretical_cuda_gib": 0.01, "peak_cuda_alloc_gib": 0.06, "device": "cuda", "peak_rss_gib": 1.1},
        {"theoretical_cuda_gib": 1.0, "peak_cuda_alloc_gib": 1.05, "device": "cuda", "peak_rss_gib": 1.2},
        {"theoretical_cuda_gib": 4.0, "peak_cuda_alloc_gib": 4.0, "device": "cuda", "peak_rss_gib": 1.3}]}
    cal = s.calibration()
    assert cal["cuda_ratio_max_theo_ge_0p2gib"] == pytest.approx(1.05)  # not 6.0 from the tiny run
    assert cal["cuda_fixed_gib"] == pytest.approx(0.06 - 1.05 * 0.01)
    assert cal["cuda_process_rss_gib_max"] == 1.3


def test_crossover_and_compile_break_even():
    from flycircuit.resources.suite import compile_summary, derive_crossover

    dev = {"a": {"edges_x_batch": 7399, "cuda_speedup": 0.3}, "b": {"edges_x_batch": 704_000, "cuda_speedup": 5.0},
           "c": {"edges_x_batch": 118_000, "cuda_speedup": 0.9}}
    r = derive_crossover(dev)
    assert r["smallest_tested_edges_x_batch_where_cuda_wins"] == 704_000
    assert r["largest_tested_edges_x_batch_where_cpu_wins"] == 118_000 and r["consistent_threshold"]
    rows = [{"tag": "t", "precision": "fp32", "ms_per_window_median": 50.0, "batch": 16, "tbptt": 32},
            {"tag": "t", "precision": "fp32", "compile": True, "ms_per_window_median": 10.0,
             "compile_seconds": 30.0, "batch": 16, "tbptt": 32}]
    assert compile_summary(rows)["t"]["break_even_windows"] == pytest.approx(30.0 / 0.04)


def test_cpu_benchmark_skips_configs_that_would_squeeze_host():
    out = bench.run_worker("model", [{"n": 20000, "e": 600000, "batch": 64, "tbptt": 32, "device": "cpu",
                                      "reps": 1}], threads=1, timeout=300)[0]
    assert "skipped" in out and "CPU activation estimate" in out["skipped"]


def test_rederive_carries_skipped_phases_with_provenance(tmp_path):
    import yaml

    from flycircuit.config import project_root
    from flycircuit.resources.suite import rederive

    r = project_root() / "results" / "resources"
    new, old = r / "profile_raw_20261015T231721.json", r / "profile_raw_20261015T220305.json"
    if not (new.exists() and old.exists()):
        pytest.skip("measured raw profile files not present")
    out = tmp_path / "p.yaml"
    prof = rederive(new, old, out, log=lambda *_: None)
    assert yaml.safe_load(out.read_text())["profile"] == "adaptive_v1"
    carried = prof["provenance"]["carried_forward_phases"]
    assert set(carried) == {"precision", "gpu_concurrency"}  # cpu_prep was never measured -> stays null
    assert prof["measured"]["cpu"]["best_threads_prep"] is None
    assert prof["measured"]["gpu"]["precision"]  # carried numbers are present, and labelled


# ---- registry --------------------------------------------------------------------
def test_registry_blocks_duplicate_and_output_conflict(tmp_path):
    j = registry.register_job("exp", {"seed": 1}, outputs=["results/x"], root=tmp_path)
    with pytest.raises(registry.DuplicateJobError, match="duplicate"):
        registry.register_job("exp", {"seed": 1}, root=tmp_path)
    with pytest.raises(registry.DuplicateJobError, match="output path"):
        registry.register_job("other", {"seed": 2}, outputs=["results/x"], root=tmp_path)
    registry.register_job("exp", {"seed": 2}, root=tmp_path)  # different config is fine
    registry.finish_job(j, 0, {"x": 1}, root=tmp_path)
    registry.register_job("exp", {"seed": 1}, root=tmp_path)  # finished -> allowed again
    assert "REG" in (tmp_path / "runs" / "jobs.log").read_text()


def test_registry_reaps_dead_jobs(tmp_path):
    import subprocess

    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    registry.register_job("sleeper", {}, pid=p.pid, root=tmp_path)
    assert len(registry.list_jobs(active_only=True, root=tmp_path)) == 1
    p.kill()
    p.wait()
    time.sleep(0.2)
    assert registry.list_jobs(active_only=True, root=tmp_path) == []
    assert registry.list_jobs(root=tmp_path)[0]["status"] == "lost"


def test_run_tracked_records_exit_status_and_peak_rss(tmp_path):
    rc = registry.run_tracked("alloc", {"n": 1}, [sys.executable, "-c",
                              "import time; a=bytearray(60*1024*1024); time.sleep(1.2)"], root=tmp_path)
    assert rc == 0
    rec = registry.list_jobs(root=tmp_path)[0]
    assert rec["status"] == "ok" and rec["measurements"]["tree_peak_mem_gib"] > 0.04
    rc = registry.run_tracked("bad", {"n": 2}, [sys.executable, "-c", "raise SystemExit(7)"], root=tmp_path)
    assert rc == 7 and registry.list_jobs(root=tmp_path)[-1]["status"] == "failed"


# ---- benchmark helpers + real (tiny) worker ---------------------------------------
def test_pick_threads_prefers_fewest_within_tolerance():
    rows = [{"threads": 1, "t": 10.0}, {"threads": 4, "t": 3.0}, {"threads": 8, "t": 2.95},
            {"threads": 16, "t": 3.5}]
    assert bench.pick_threads(rows, "t", higher_is_better=False) == 4
    rows = [{"threads": 1, "s": 1.0}, {"threads": 8, "s": 5.0}, {"threads": 4, "s": 4.9}]
    assert bench.pick_threads(rows, "s", higher_is_better=True) == 4


def test_plateau_batch_is_smallest_within_5pct_of_best_and_robust_to_noise():
    rows = [{"batch": 1, "steps_per_s": 100}, {"batch": 2, "steps_per_s": 190},
            {"batch": 4, "steps_per_s": 340}, {"batch": 8, "steps_per_s": 360},
            {"batch": 16, "steps_per_s": 500}]
    assert bench.plateau_batch(rows) == 16  # later real gain after a flat step is NOT ignored
    rows = [{"batch": 1, "steps_per_s": 100}, {"batch": 8, "steps_per_s": 480},
            {"batch": 16, "steps_per_s": 500}, {"batch": 32, "steps_per_s": 498}]
    assert bench.plateau_batch(rows) == 8  # 480 >= 0.95*500
    skipped = [{"batch": 64, "skipped": "no headroom"}] + rows
    assert bench.plateau_batch(skipped) == 8


def test_worker_cpu_reports_fp32_oracle_agreement():
    out = bench.run_worker("model", [{"n": 200, "e": 2000, "batch": 2, "tbptt": 8, "device": "cpu",
                                      "reps": 2}], threads=1, timeout=300)[0]
    assert "error" not in out, out
    assert out["torch_threads"] == 1 and out["steps_per_s"] > 0
    assert out["y_rel_err_vs_fp32"] == 0.0 and out["grad_rel_l2_err_vs_fp32"] == 0.0
    json.dumps(out)
