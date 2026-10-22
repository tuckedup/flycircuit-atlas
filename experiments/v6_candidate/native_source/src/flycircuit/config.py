"""Validated YAML configuration and project-root discovery."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

DEFAULT_ROOT = Path("/home/atulp/projects/flycircuit")


def project_root() -> Path:
    """Locate the repository root independent of the current directory."""
    env = os.environ.get("FLYCIRCUIT_HOME")
    if env:
        return Path(env)
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").exists():
        return candidate
    return DEFAULT_ROOT


class HardwareProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str
    cpu_threads: int
    dataloader_workers: int
    experiment_concurrency: int
    max_pipeline_ram_gib: float
    max_cuda_allocated_gib: float
    min_host_free_disk_gib: float
    max_new_project_storage_gib: float
    initial_batch_size: int
    initial_tbptt_steps: int
    initial_training_seeds: int
    confirmatory_training_seeds: int
    initial_trial_walltime_minutes: float
    max_default_job_walltime_minutes: float
    max_default_sweep_walltime_minutes: float
    whole_brain_gradient_training: bool
    full_synapse_download_default: bool
    cuda_headroom_gib: float = 1.0
    host_ram_headroom_gib: float = 1.0


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at top level")
    return data


def load_hardware(name: str = "atul_laptop", root: Path | None = None) -> HardwareProfile:
    root = root or project_root()
    path = root / "configs" / "hardware" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"hardware profile not found: {path}")
    return HardwareProfile(**load_yaml(path))


def config_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
