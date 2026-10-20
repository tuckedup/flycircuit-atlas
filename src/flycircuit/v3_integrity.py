"""Study-v3-only provenance primitives. Historical study artifacts are inputs.

Prepared as a candidate outside the shared repository. Callers must explicitly
identify their protocol and input hashes; this module does not choose models.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
import zipfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from flycircuit.config import project_root

ROOT = project_root().resolve()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def runtime_identity() -> dict:
    """Read installed distribution metadata without importing Torch or CUDA."""
    packages = sorted((d.metadata.get("Name", ""), d.version) for d in importlib.metadata.distributions())
    return {"python": sys.version, "executable": str(Path(sys.executable).resolve()),
            "installed_distributions_sha256": canonical_hash(packages),
            "environment": {k: os.environ.get(k) for k in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
                "PYTHONHASHSEED", "MUJOCO_GL", "PYOPENGL_PLATFORM")}}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_v3_output(path: Path) -> None:
    allowed = (ROOT / "results/learning_v3", ROOT / "results/flygym_v3",
               ROOT / "results/provenance/study_v3")
    # Resolve the requested path but not the allowed literal namespace. Resolving
    # both would let a symlink learning_v3 -> learning_v2 authorize old-study writes.
    if not any(path.resolve().is_relative_to(p) for p in allowed):
        raise ValueError(f"v3 output must stay in a new study namespace: {path}")


def atomic_json(path: Path, value) -> None:
    """Replace a directory entry, never truncate a possibly shared inode."""
    _require_v3_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".v3_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _sync_directory(path: Path) -> None:
    """The target is Ubuntu WSL; fsync the directory entry as well as its file."""
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _manifest_lock(path: Path):
    # Advisory coordination between helper users; does not prevent human edits.
    lock = path.with_name(path.name + ".lock")
    _require_v3_output(lock)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def source_paths() -> list[Path]:
    paths = list((ROOT / "src/flycircuit").rglob("*.py"))
    required = [ROOT / rel for rel in (
        "pyproject.toml", "uv.lock", "scripts/linux/sequential_driver_v3.sh", "scripts/linux/heavyjob.sh",
        "scripts/linux/make_results_v3.py",
        "external/flygym_pilot/fly_rollout.py", "external/flygym_pilot/nav_env.py")]
    for path in [*paths, *required]:
        if not path.resolve().is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"required source/dependency file missing or escapes repository: {path}")
    return sorted(set(paths + required))


@lru_cache(maxsize=1)
def source_identity() -> tuple[str, dict[str, str]]:
    """Identity observed once per process, while its implementation is loaded.

    The driver must not permit source edits during a running stage. Every newly
    launched stage recomputes this identity before accepting an existing cache.
    Configuration hashes are separate because body evaluation N is predeclared
    to be derived later from validation data.
    """
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in source_paths()}
    return canonical_hash(hashes), hashes


def _unchanged_source() -> tuple[str, dict[str, str]]:
    identity, original = source_identity()
    current = {str(p.relative_to(ROOT)): file_hash(p) for p in source_paths()}
    if current != original:
        raise ValueError("source changed during the current process; refusing cached state or freeze")
    return identity, original


def input_inventory(paths: list[Path]) -> dict[str, str]:
    result = {}
    for path in sorted(set(paths)):
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f"study input escapes repository: {path}")
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def circuit_inputs(circuit: str = "mushroom_body_right_pilot", include_group: bool = False) -> dict[str, str]:
    """Exact immutable processed graph/sign inputs and cached null files consumed.

    Does not hash the unused multi-gigabyte raw synapse export. Callers run this
    after the resource guard, before loading models, and bind its result in their
    stage manifest. No data is created or modified here.
    """
    base = ROOT / "data/processed/flywire/783"
    files = [base / "neurons.parquet", base / "edges_pairs.parquet",
             base / "circuits" / circuit / "internal_ids.parquet"]
    for kind in (["plain", "group_constrained"] if include_group else ["plain"]):
        folder = ROOT / "results/nulls_v2" / circuit / kind
        nulls = sorted(folder.glob("null_*.npz"))
        if len(nulls) < 5:
            raise ValueError(f"five cached nulls required: {folder}")
        files += [folder / "report.json", *nulls]
    return input_inventory(files)


def require_manifest(path: Path, identity: dict) -> dict:
    _require_v3_output(path)
    if not path.is_file():
        raise ValueError(f"required v3 manifest missing: {path}")
    saved = json.loads(path.read_text())
    source_id, hashes = _unchanged_source()
    expected = {**identity, "source_id": source_id}
    if saved.get("identity") != expected:
        raise ValueError(f"v3 source/protocol/input identity changed: {path}; preserve records and record a prospective amendment")
    if saved.get("source_files") != hashes:
        raise ValueError("manifest source inventory disagrees with the running implementation")
    archive = ROOT / saved["source_archive"]
    if not archive.resolve().is_relative_to((ROOT / "results/provenance/study_v3").resolve()):
        raise ValueError("v3 manifest source archive escapes its provenance namespace")
    if not archive.is_file() or file_hash(archive) != saved.get("source_archive_sha256"):
        raise ValueError(f"v3 source archive missing or changed: {archive}")
    with zipfile.ZipFile(archive) as bundle:
        inventories = {**hashes, **saved.get("configuration_files", {})}
        if set(bundle.namelist()) != set(inventories):
            raise ValueError("archive file inventory differs from manifest")
        if any(hashlib.sha256(bundle.read(name)).hexdigest() != digest for name, digest in inventories.items()):
            raise ValueError("archive bytes differ from declared source/configuration inventories")
    return saved


def freeze_manifest(path: Path, identity: dict, configuration_paths: list[Path]) -> dict:
    """Idempotent freeze with timestamp/source/config archive and serialized writers.

    Callers must include effective protocol, inputs and selections in identity.
    Configuration files archive original bytes; later mutable derived fields (such
    as validation-derived body N) are governed by the caller's phase-specific
    effective identity, rather than comparing the whole mutable YAML on resume.
    """
    _require_v3_output(path)
    with _manifest_lock(path):
        return _freeze_locked(path, identity, configuration_paths)


def _freeze_locked(path: Path, identity: dict, configuration_paths: list[Path]) -> dict:
    if path.exists():
        return require_manifest(path, identity)
    source_id, hashes = _unchanged_source()
    configurations = input_inventory(configuration_paths)
    now = dt.datetime.now(dt.UTC)
    archive_dir = ROOT / "results/provenance/study_v3"
    _require_v3_output(archive_dir / "source_archive.zip")
    archive_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".sources_", suffix=".zip", dir=archive_dir)
    os.close(fd)
    archive = Path(temporary_name).with_name(
        path.stem + "_" + now.strftime("%Y%m%dT%H%M%S%fZ") + "_" + Path(temporary_name).name[1:])
    try:
        with zipfile.ZipFile(temporary_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            paths = set(source_paths()) | set(configuration_paths)
            for item in sorted(paths):
                if not item.resolve().is_relative_to(ROOT.resolve()):
                    raise ValueError(f"archive input escapes repository: {item}")
                if item.is_file():
                    zf.writestr(str(item.relative_to(ROOT)), item.read_bytes())
        with zipfile.ZipFile(temporary_name) as zf:
            inventories = {**hashes, **configurations}
            if set(zf.namelist()) != set(inventories):
                raise ValueError("source/configuration files changed while building their freeze archive")
            if any(hashlib.sha256(zf.read(name)).hexdigest() != digest for name, digest in inventories.items()):
                raise ValueError("source/configuration bytes changed while building their freeze archive")
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, archive)
        _sync_directory(archive_dir)
        payload = {"frozen_utc": now.isoformat(), "identity": {**identity, "source_id": source_id},
                   "source_files": hashes, "source_archive": str(archive.relative_to(ROOT)),
                   "configuration_files": configurations,
                   "source_archive_sha256": file_hash(archive),
                   "scope": "local pre-evaluation provenance; not an external timestamp or biological validation"}
        atomic_json(path, payload)
        return payload
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def validate_record_identity(records: list[dict], protocol_hash: str,
                             allowed_legacy_records: set[str] | None = None) -> None:
    source_id, _ = _unchanged_source()
    legacy = allowed_legacy_records or set()
    for record in records:
        if record.get("protocol_hash") != protocol_hash:
            raise ValueError("v3 records mix different protocols")
        if record.get("source_id") == source_id:
            continue
        # Legacy eligibility is an explicit, immutable list of the exact saved
        # pre-correction validation records. It is never a blanket hash bypass.
        if (record.get("source_id") is None and record.get("stage") == "search"
                and record.get("splits") == ["val"] and canonical_hash(record) in legacy):
            continue
        raise ValueError("v3 record source is unknown or differs from this implementation")
