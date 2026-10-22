"""Per-run JSON manifest + optional local MLflow logging. Writes are serialized by a file lock."""

from __future__ import annotations

import fcntl
import json
import platform
import subprocess
import sys
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flycircuit import __version__
from flycircuit.config import config_hash, project_root
from flycircuit.resources.probe import peak_cuda_mib, peak_ram_mib

MANIFEST_FIELDS = (
    "run_id", "timestamp_utc", "git_revision", "git_dirty", "config_hash", "resolved_config",
    "data_release", "data_hashes", "circuit_mapping_hash", "seeds", "env_versions", "algorithm",
    "stopping_rule", "elapsed_seconds", "exit_status", "peak_ram_mib", "peak_cuda_mib",
    "artifact_paths", "metrics", "label", "mlflow",
)


def _git(root: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def git_state(root: Path) -> tuple[str | None, bool | None]:
    rev = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    return rev, (None if status is None else bool(status))


def env_versions() -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": sys.version.split()[0], "executable": sys.executable,
        "platform": platform.platform(), "flycircuit": __version__,
    }
    for mod in ("numpy", "scipy", "pyarrow", "torch", "mlflow", "fastapi", "duckdb"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = None
    try:
        import torch

        out["torch_cuda"] = torch.version.cuda
    except Exception:
        pass
    return out


class RunRecorder:
    """Context manager that records a run and always writes a manifest, even on failure."""

    def __init__(
        self, name: str, config: dict[str, Any], seeds: list[int], algorithm: str,
        stopping_rule: str, data_release: str | None = None,
        data_hashes: dict[str, str] | None = None, mapping_hash: str | None = None,
        root: Path | None = None, use_mlflow: bool = True, label: str = "",
    ):
        self.root = root or project_root()
        ts = datetime.now(UTC)
        self.run_id = f"{ts:%Y%m%dT%H%M%S}-{name}-{uuid.uuid4().hex[:8]}"
        self.dir = self.root / "runs" / self.run_id
        self.timestamp = ts.isoformat()
        self.config = config
        self.seeds = seeds
        self.algorithm = algorithm
        self.stopping_rule = stopping_rule
        self.data_release = data_release
        self.data_hashes = data_hashes or {}
        self.mapping_hash = mapping_hash
        self.use_mlflow = use_mlflow
        self.label = label
        self.artifacts: list[str] = []
        self.metrics: dict[str, float] = {}
        self.manifest: dict[str, Any] = {}

    def add_artifact(self, path: Path | str) -> None:
        self.artifacts.append(str(path))

    def __enter__(self) -> RunRecorder:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        rev, dirty = git_state(self.root)
        status = "ok" if exc_type is None else "failed"
        m: dict[str, Any] = {
            "run_id": self.run_id, "timestamp_utc": self.timestamp, "git_revision": rev,
            "git_dirty": dirty, "config_hash": config_hash(self.config),
            "resolved_config": self.config, "data_release": self.data_release,
            "data_hashes": self.data_hashes, "circuit_mapping_hash": self.mapping_hash,
            "seeds": self.seeds, "env_versions": env_versions(), "algorithm": self.algorithm,
            "stopping_rule": self.stopping_rule,
            "elapsed_seconds": time.perf_counter() - self._t0, "exit_status": status,
            "peak_ram_mib": peak_ram_mib(), "peak_cuda_mib": peak_cuda_mib(),
            "artifact_paths": self.artifacts, "metrics": self.metrics, "label": self.label,
            "mlflow": {"status": "disabled"},
        }
        if exc_type is not None:
            m["error"] = "".join(traceback.format_exception(exc_type, exc, tb))[-4000:]
        path = self.dir / "manifest.json"
        lock = self.root / "runs" / ".write.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with open(lock, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            path.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
            if self.use_mlflow:
                try:
                    from flycircuit.tracking.mlflow_backend import log_manifest

                    m["mlflow"] = log_manifest(m, path, self.root)
                except Exception as e:  # tracking must never mask the scientific result
                    m["mlflow"] = {"status": "error", "error": repr(e)}
                path.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
        self.manifest = m
        return False  # never swallow exceptions
