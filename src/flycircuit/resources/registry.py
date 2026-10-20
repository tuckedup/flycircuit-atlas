"""Job registry: who is running what, with resource measurements, layered over runs/jobs.log.

``runs/jobs/registry.json`` holds one record per job. Writes are serialized with an flock on
``runs/jobs/.registry.lock``. A job is a *duplicate* when an entry with the same ``dedupe_key``
(default: label + config hash) is still ``running`` and its process (pid + start time) is alive.
Lines are also appended to ``runs/jobs.log`` (the shell-level heavyjob.sh log) as ``REG``/``REGEND``.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import psutil

from flycircuit.config import project_root


class DuplicateJobError(RuntimeError):
    pass


def _dir(root: Path | None) -> Path:
    d = (root or project_root()) / "runs" / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


@contextlib.contextmanager
def _locked(root: Path | None):
    d = _dir(root)
    with open(d / ".registry.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            yield d / "registry.json"
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _load(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    return json.loads(p.read_text() or "[]")


def _save(p: Path, rows: list[dict[str, Any]]) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=1, default=str))
    os.replace(tmp, p)


def _alive(rec: dict[str, Any]) -> bool:
    try:
        pr = psutil.Process(rec["pid"])
        return pr.is_running() and pr.status() != psutil.STATUS_ZOMBIE and \
            abs(pr.create_time() - rec["pid_create_time"]) < 2.0
    except (psutil.Error, KeyError):
        return False


def _log(root: Path | None, line: str) -> None:
    log = (root or project_root()) / "runs" / "jobs.log"
    with open(log, "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {line}\n")


def config_key(label: str, config: Any) -> str:
    blob = json.dumps({"label": label, "config": config}, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def reap(root: Path | None = None) -> int:
    """Mark ``running`` entries whose process is gone as ``lost`` (crash/kill). Returns count."""
    n = 0
    with _locked(root) as p:
        rows = _load(p)
        for r in rows:
            if r["status"] == "running" and not _alive(r):
                r["status"], r["ended"] = "lost", time.time()
                n += 1
        if n:
            _save(p, rows)
    return n


def register_job(label: str, config: Any, cmd: list[str] | None = None,
                 outputs: list[str] | None = None, estimate: dict[str, Any] | None = None,
                 pid: int | None = None, root: Path | None = None) -> str:
    pid = pid or os.getpid()
    key = config_key(label, config)
    with _locked(root) as p:
        rows = _load(p)
        for r in rows:
            if r["status"] == "running" and not _alive(r):
                r["status"], r["ended"] = "lost", time.time()
        dup = [r for r in rows if r["status"] == "running" and r["dedupe_key"] == key]
        if dup:
            _save(p, rows)
            raise DuplicateJobError(
                f"duplicate job: '{label}' with identical config is already running as "
                f"{dup[0]['job_id']} (pid {dup[0]['pid']}). Wait for it, or change the label/config.")
        out_clash = [r for r in rows if r["status"] == "running" and outputs
                     and set(outputs) & set(r.get("outputs") or [])]
        if out_clash:
            _save(p, rows)
            raise DuplicateJobError(
                f"output path conflict with running job {out_clash[0]['job_id']}: "
                f"{sorted(set(outputs) & set(out_clash[0]['outputs']))}")
        job_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:6]}"
        rows.append({
            "job_id": job_id, "label": label, "dedupe_key": key, "config": config,
            "cmd": cmd, "outputs": outputs or [], "estimate": estimate, "pid": pid,
            "pid_create_time": psutil.Process(pid).create_time(), "status": "running",
            "started": time.time(), "ended": None, "measurements": {},
        })
        _save(p, rows)
    _log(root, f"REG    job={job_id} pid={pid} label={label} key={key}")
    return job_id


def finish_job(job_id: str, exit_status: int, measurements: dict[str, Any] | None = None,
               root: Path | None = None) -> None:
    with _locked(root) as p:
        rows = _load(p)
        for r in rows:
            if r["job_id"] == job_id:
                r["status"] = "ok" if exit_status == 0 else "failed"
                r["exit_status"], r["ended"] = exit_status, time.time()
                r["elapsed_s"] = r["ended"] - r["started"]
                r["measurements"] = measurements or {}
                break
        else:
            raise KeyError(job_id)
        _save(p, rows)
    _log(root, f"REGEND job={job_id} rc={exit_status} {json.dumps(measurements or {})}")


def list_jobs(active_only: bool = False, root: Path | None = None) -> list[dict[str, Any]]:
    reap(root)
    with _locked(root) as p:
        rows = _load(p)
    return [r for r in rows if r["status"] == "running"] if active_only else rows


def _mem(p: psutil.Process) -> int:
    """PSS (shared pages split fairly, so a tree of torch workers is not double counted); RSS fallback."""
    try:
        return p.memory_full_info().pss
    except (psutil.Error, AttributeError):
        try:
            return p.memory_info().rss
        except psutil.Error:
            return 0


class _Sampler(threading.Thread):
    """Samples PSS of a process tree (the job + all children) to get a tree peak."""

    def __init__(self, pid: int, period: float = 0.5):
        super().__init__(daemon=True)
        self.pid, self.period, self.peak_rss, self.stop = pid, period, 0, threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                pr = psutil.Process(self.pid)
                tot = sum(_mem(p) for p in [pr, *pr.children(recursive=True)])
                self.peak_rss = max(self.peak_rss, tot)
            except psutil.Error:
                pass
            self.stop.wait(self.period)


def _gpu_peak_used_gib() -> float | None:
    from flycircuit.resources.live import gpu_state
    g = gpu_state()
    return g.get("used_gib") if g.get("available") else None


def run_tracked(label: str, config: Any, cmd: list[str], outputs: list[str] | None = None,
                estimate: dict[str, Any] | None = None, root: Path | None = None) -> int:
    """Register, run ``cmd`` as a child, record tree-peak RSS and elapsed, then finish."""
    job_id = register_job(label, config, cmd, outputs, estimate, root=root)  # runner pid; dup -> raises
    child = subprocess.Popen(cmd, cwd=str(root or project_root()))
    s = _Sampler(child.pid)
    s.start()
    g0 = _gpu_peak_used_gib()
    rc = child.wait()
    s.stop.set()
    s.join(1)
    g1 = _gpu_peak_used_gib()
    finish_job(job_id, rc, {"tree_peak_mem_gib": s.peak_rss / 1024**3,
                            "gpu_used_gib_before": g0, "gpu_used_gib_after": g1}, root=root)
    return rc


def main_run(argv: list[str]) -> int:  # pragma: no cover - thin wrapper
    return run_tracked(argv[0], {"argv": argv[1:]}, argv[1:])


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main_run(sys.argv[1:]))
