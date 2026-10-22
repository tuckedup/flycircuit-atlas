"""Verified source registry, resumable atomic download, and source manifest.

Sizes/MD5s below were read from the Zenodo API (record 10676866) on 2026-10-15 and
are re-checked against the live API by `flycircuit data sources --verify`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .paths import host_free_gib, raw_dir

ZENODO_API = "https://zenodo.org/api/records/10676866"
ANN_COMMIT = "a83b2776d60d5764cef36b927f5f9679c16c47a2"
ANN_BASE = f"https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{ANN_COMMIT}"


@dataclass(frozen=True)
class Source:
    key: str
    filename: str
    url: str
    tier: str
    size: int | None = None  # expected bytes (None = not published by source)
    md5: str | None = None
    record: str = ""
    license: str = ""
    citation: str = ""


_ZEN = "zenodo:10676866 (FlyWire connectivity v783.0; concept 10676865)"
_ZLIC = "CC-BY-4.0"
_ZCITE = "FlyWire Consortium, doi:10.5281/zenodo.10676866; Dorkenwald et al., Nature 2024"


def _z(key: str, name: str, size: int, md5: str, tier: str) -> Source:
    return Source(
        key, name, f"{ZENODO_API}/files/{name}/content", tier, size, md5, _ZEN, _ZLIC, _ZCITE
    )


SOURCES: dict[str, dict[str, Source]] = {
    "flywire": {
        "root_ids": _z(
            "root_ids", "proofread_root_ids_783.npy", 1114168,
            "e0e6c19732fd8c7a4e39a2d170105421", "proofread-pairs"),
        "proofread_connections": _z(
            "proofread_connections", "proofread_connections_783.feather", 852022274,
            "f48f972d262323a102aed49af1396b8a", "proofread-pairs"),
        "annotations": Source(
            "annotations", "Supplemental_file1_neuron_annotations.tsv",
            f"{ANN_BASE}/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
            "proofread-pairs", 31720298, None,
            f"github:flyconnectome/flywire_annotations@{ANN_COMMIT}",
            "not declared by GitHub license API (inspect repo before redistributing)",
            "Schlegel et al., Nature 2024; Dorkenwald et al., Nature 2024"),
        "per_neuron_neuropil_post": _z(
            "per_neuron_neuropil_post", "per_neuron_neuropil_count_post_783.feather",
            233843050, "bb5999f10920ade803d9f37097a43a56", "neuropil-counts"),
        "per_neuron_neuropil_pre": _z(
            "per_neuron_neuropil_pre", "per_neuron_neuropil_count_pre_783.feather",
            16853770, "90fcdb42c1ba05ed92820840fa1e6ba0", "neuropil-counts"),
        "synapses": _z(
            "synapses", "flywire_synapses_783.feather", 9492998242,
            "f8f1b97c9d4b0ea9b4c8b287f6b99091", "synapses"),
    }
}


def sources_for(dataset: str, version: str) -> dict[str, Source]:
    if dataset != "flywire" or version != "783":
        raise NotImplementedError(f"No verified source registry for {dataset} v{version}")
    return SOURCES[dataset]


def tier_sources(dataset: str, version: str, tier: str) -> list[Source]:
    out = [s for s in sources_for(dataset, version).values() if s.tier == tier]
    if not out:
        raise ValueError(f"unknown tier {tier!r}")
    return out


def sha256_md5(path: Path, chunk: int = 8 << 20) -> tuple[str, str]:
    h, m = hashlib.sha256(), hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
            m.update(b)
    return h.hexdigest(), m.hexdigest()


@dataclass
class ManifestEntry:
    key: str
    filename: str
    url: str
    record: str
    license: str
    citation: str
    expected_size: int | None
    actual_size: int | None = None
    source_md5: str | None = None
    local_md5: str | None = None
    sha256: str | None = None
    download_date_utc: str | None = None
    success: bool = False
    error: str | None = None
    extra: dict = field(default_factory=dict)


def manifest_path(dataset: str, version: str) -> Path:
    return raw_dir(dataset, version) / "SOURCE_MANIFEST.json"


def load_manifest(dataset: str, version: str) -> dict[str, dict]:
    p = manifest_path(dataset, version)
    return json.loads(p.read_text()) if p.exists() else {}


def _save_manifest(dataset: str, version: str, m: dict) -> None:
    p = manifest_path(dataset, version)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=2, sort_keys=True))
    tmp.replace(p)


def verify_live_metadata() -> list[str]:
    """Compare registry against the live Zenodo API; return list of discrepancies."""
    with urllib.request.urlopen(ZENODO_API, timeout=60) as r:
        d = json.load(r)
    live = {f["key"]: f for f in d["files"]}
    problems = []
    for s in SOURCES["flywire"].values():
        if not s.url.startswith(ZENODO_API):
            continue
        f = live.get(s.filename)
        if f is None:
            problems.append(f"{s.filename}: missing from live record")
        elif f["size"] != s.size or f["checksum"] != f"md5:{s.md5}":
            problems.append(f"{s.filename}: size/checksum differs from registry")
    return problems


def _open(url: str, start: int):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    return urllib.request.urlopen(req, timeout=120)


def download(dataset: str, version: str, src: Source, *, min_free_gib: float = 40.0,
             progress: bool = True, max_resume_attempts: int = 30) -> ManifestEntry:
    """Resumable download to <name>.partial; finalize atomically only after validation."""
    d = raw_dir(dataset, version)
    d.mkdir(parents=True, exist_ok=True)
    final, part = d / src.filename, d / (src.filename + ".partial")
    man = load_manifest(dataset, version)
    entry = ManifestEntry(src.key, src.filename, src.url, src.record, src.license,
                          src.citation, src.size, source_md5=src.md5)
    if final.exists() and man.get(src.key, {}).get("success"):
        return ManifestEntry(**man[src.key])  # immutable raw: never overwrite
    free = host_free_gib(d)
    need = ((src.size or 0) - (part.stat().st_size if part.exists() else 0)) / 2**30
    if free - need < min_free_gib:
        entry.error = f"disk budget: free {free:.1f} GiB - need {need:.1f} GiB < {min_free_gib}"
        return entry
    try:
        have = part.stat().st_size if part.exists() else 0
        if src.size and have > src.size:
            part.unlink()
            have = 0
        attempts = 0
        # a server may close the stream early (seen: 3.48 of 9.49 GB); resume from the partial file
        while not (src.size and have == src.size) and attempts < max_resume_attempts:
            attempts += 1
            try:
                resp = _open(src.url, have)
                if have and getattr(resp, "status", 200) != 206:
                    have = 0  # server ignored Range: restart
                with open(part, "ab" if have else "wb") as f:
                    done = have
                    while b := resp.read(4 << 20):
                        f.write(b)
                        done += len(b)
                        if progress and src.size and (done // (4 << 20)) % 25 == 0:
                            print(f"  {src.filename}: {done / 2**20:.0f}/{src.size / 2**20:.0f} MiB",
                                  flush=True)
            except OSError as e:  # connection reset / timeout: keep partial, resume
                print(f"  stream interrupted ({type(e).__name__}: {e}); resuming", flush=True)
            have = part.stat().st_size if part.exists() else 0
            if not src.size:
                break
            if have != src.size:
                time.sleep(min(30, 2 * attempts))
        entry.actual_size = part.stat().st_size
        entry.sha256, entry.local_md5 = sha256_md5(part)
        if src.size is not None and entry.actual_size != src.size:
            raise ValueError(f"size mismatch {entry.actual_size} != {src.size}")
        if src.md5 and entry.local_md5 != src.md5:
            part.unlink()
            raise ValueError(f"md5 mismatch {entry.local_md5} != {src.md5}; partial deleted")
        part.replace(final)
        entry.success = True
        entry.download_date_utc = dt.datetime.now(dt.UTC).isoformat()
    except Exception as e:  # noqa: BLE001 - recorded in manifest, partial kept for resume
        entry.error = f"{type(e).__name__}: {e}"
    man[src.key] = asdict(entry)
    _save_manifest(dataset, version, man)
    return entry


def dry_run(dataset: str, version: str, tier: str) -> dict:
    srcs = tier_sources(dataset, version, tier)
    total = sum(s.size or 0 for s in srcs)
    d = raw_dir(dataset, version)
    d.mkdir(parents=True, exist_ok=True)
    return {
        "tier": tier,
        "files": [{"file": s.filename, "bytes": s.size, "url": s.url} for s in srcs],
        "total_gib": round(total / 2**30, 3),
        "host_free_gib": round(host_free_gib(d), 1),
        "note": "dry run: nothing downloaded",
    }
