"""Project-root and data-directory resolution (works from any cwd)."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("FLYCIRCUIT_HOME")
    if env:
        return Path(env)
    for start in (Path.cwd(), Path(__file__).resolve()):
        for p in [start, *start.parents]:
            pp = p / "pyproject.toml"
            if pp.is_file() and 'name = "flycircuit"' in pp.read_text(encoding="utf-8"):
                return p
    return Path.home() / "projects" / "flycircuit"


def raw_dir(dataset: str, version: str) -> Path:
    return project_root() / "data" / "raw" / dataset / version


def processed_dir(dataset: str, version: str) -> Path:
    return project_root() / "data" / "processed" / dataset / version


def scratch_dir() -> Path:
    return project_root() / "data" / "processed" / "_tmp"


def host_free_gib(path: Path | None = None) -> float:
    """Physical free space: min(WSL fs, Windows C: drive). WSL's own capacity is thin-provisioned."""
    import shutil

    vals = [shutil.disk_usage(path or project_root()).free / 2**30]
    if Path("/mnt/c").exists():
        vals.append(shutil.disk_usage("/mnt/c").free / 2**30)
    return min(vals)
