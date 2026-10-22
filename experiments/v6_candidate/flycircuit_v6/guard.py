"""No-wait guard for future v6 execution. Never stops unrelated processes."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .plan import acceptance, digest, verify_plan


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def windows_available_gib():
    try:
        response = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        value = int(response.stdout.strip()) / 1048576
        return value if value >= 0 else None
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


def pdelab_jobs(proc=Path("/proc")):
    """Conservative detection of Python processes tied to the separate project."""
    found = []
    for folder in proc.iterdir():
        if not folder.name.isdigit():
            continue
        try:
            command = (
                (folder / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
            if "python" not in command.lower():
                continue
            cwd = str((folder / "cwd").resolve())
            if "/projects/pdelab" in cwd.lower() or "pdelab" in command.lower():
                found.append(int(folder.name))
        except (OSError, RuntimeError):
            continue
    return found


def status():
    return {
        "available_windows_gib": windows_available_gib(),
        "pdelab_python_pids": pdelab_jobs(),
        "starts_training": False,
        "minimum_runtime_available_gib": 1.5,
        "minimum_launch_available_gib": 2.6,
    }


def run(plan_path, receipt_path):
    """One deliberate guarded attempt. Resume requires the same plan and budget."""
    import fcntl

    plan_path, receipt_path = Path(plan_path).resolve(), Path(receipt_path).resolve()
    initial = status()
    if (
        initial["pdelab_python_pids"]
        or initial["available_windows_gib"] is None
        or initial["available_windows_gib"] < 2.6
    ):
        raise ValueError(
            f"no launch: competing PDELab or insufficient/unknown headroom: {initial}"
        )
    plan, receipt_sha = acceptance(plan_path, receipt_path, with_binding=True)
    budget_path = plan_path.parent / "resource_ledger.json"
    canonical = Path(plan["source_root"]).parent / "flycircuit"
    lock_path = canonical / "runs/.heavyjob.lock"
    # Open EXISTING shared serialization lock without modifying canonical files.
    with lock_path.open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # All ledger reads and updates belong INSIDE the serialization lock.
        # A pre-lock reader could otherwise overwrite a newer attempt's charge.
        budget = (
            json.loads(budget_path.read_text())
            if budget_path.exists()
            else {
                "charged_wall_s": 0.0,
                "attempts": 0,
                "inflight": None,
                "plan_sha256": digest(plan_path),
                "acceptance_sha256": receipt_sha,
            }
        )
        charge, attempts = budget.get("charged_wall_s"), budget.get("attempts")
        if (
            budget.get("plan_sha256") != digest(plan_path)
            or budget.get("acceptance_sha256") != receipt_sha
            or budget.get("inflight") is not None
            or type(charge) not in (int, float)
            or not math.isfinite(charge)
            or charge < 0
            or type(attempts) is not int
            or attempts < 0
        ):
            raise ValueError(
                "invalid budget identity/charge or unresolved attempt; audit before resuming"
            )
        remaining = plan["protocol"]["wall_limit_s"] - charge
        if remaining <= 0:
            raise ValueError("prospective 30-minute cumulative budget exhausted")
        state = status()
        free = state["available_windows_gib"]
        if state["pdelab_python_pids"] or free is None or free < 2.6:
            raise ValueError(
                f"no launch: competing PDELab or insufficient/unknown headroom: {state}"
            )
        verify_plan(plan)
        budget["attempts"] += 1
        started = time.monotonic()
        original_charge = budget["charged_wall_s"]
        budget["inflight"] = {
            "supervisor_pid": os.getpid(),
            "started_unix": time.time(),
        }
        atomic(budget_path, budget)
        env = dict(
            os.environ,
            FLYCIRCUIT_HOME=str(canonical),
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            CUDA_VISIBLE_DEVICES="",
            FC_V6_PLAN_SHA=digest(plan_path),
            FC_V6_ACCEPTANCE_SHA=receipt_sha,
            FC_V6_LOCK_FD=str(lock.fileno()),
        )
        command = [
            sys.executable,
            "-m",
            "flycircuit_v6",
            "_child",
            "--plan",
            str(plan_path),
            "--acceptance",
            str(receipt_path),
        ]
        child, reason, low_since = None, None, None
        interrupted = False

        def on_signal(_signum, _frame):
            nonlocal interrupted
            interrupted = True

        old_handlers = {
            s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            # Charge this deliberate headroom wait to the same cumulative cap.
            time.sleep(min(30.0, remaining))
            second = status()
            if (
                interrupted
                or second["pdelab_python_pids"]
                or second["available_windows_gib"] is None
                or second["available_windows_gib"] < 2.6
            ):
                reason = "headroom_not_sustained"
                return 4
            if time.monotonic() - started >= remaining:
                reason = "cumulative_wall_limit"
                return 4
            with (plan_path.parent / "training.log").open("a") as output:
                child = subprocess.Popen(
                    command,
                    cwd=plan["bundle_root"],
                    env=env,
                    start_new_session=True,
                    pass_fds=(lock.fileno(),),
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
                budget["inflight"]["child_pid"] = child.pid
                atomic(budget_path, budget)
                print(
                    json.dumps(
                        {
                            "supervisor_pid": os.getpid(),
                            "child_pid": child.pid,
                            "output": str(plan_path.parent),
                            "command": command,
                        }
                    ),
                    flush=True,
                )
                while child.poll() is None:
                    elapsed = time.monotonic() - started
                    budget["charged_wall_s"] = original_charge + elapsed
                    atomic(budget_path, budget)
                    if interrupted:
                        reason = "interrupted"
                        break
                    if elapsed >= remaining:
                        reason = "cumulative_wall_limit"
                        break
                    verify_plan(plan)
                    free = windows_available_gib()
                    if free is None or free < 1.5:
                        low_since = (
                            low_since if low_since is not None else time.monotonic()
                        )
                        if time.monotonic() - low_since >= 90:
                            reason = "memory_guard"
                            break
                    else:
                        low_since = None
                    if pdelab_jobs():
                        reason = "pdelab_started"
                        break
                    time.sleep(min(10.0, max(0.1, remaining - elapsed)))
        except BaseException as exc:
            reason = reason or f"supervisor_error:{type(exc).__name__}"
            raise
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            budget["charged_wall_s"] = original_charge + time.monotonic() - started
            budget["inflight"] = None
            budget["last_result"] = {
                "reason": reason,
                "child_rc": child.returncode if child else None,
            }
            atomic(budget_path, budget)
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
        return 4 if reason else (child.returncode if child is not None else 1)
