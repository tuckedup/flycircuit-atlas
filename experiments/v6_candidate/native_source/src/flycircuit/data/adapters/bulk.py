"""Bulk public-file adapters (GCS HTTPS Feather/CSV): MaleCNS and BANC.

Downloads reuse ``flycircuit.data.sources.download`` (resumable .partial, md5 check, atomic finalize,
SOURCE_MANIFEST.json, host-disk budget). Raw files are immutable and live under data/raw/<dataset>/<version>/.
Large files are read only through record batches with column projection (``feather_io``).
"""

from __future__ import annotations

import base64
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from flycircuit.data import feather_io
from flycircuit.data.paths import host_free_gib, raw_dir
from flycircuit.data.sources import ManifestEntry, Source, download

from .base import DatasetAdapter, MissingPrerequisite


@dataclass(frozen=True)
class BulkFile:
    key: str
    filename: str
    url: str
    size: int
    md5: str | None
    description: str


class BulkFeatherAdapter(DatasetAdapter):
    files: dict[str, BulkFile] = {}
    record = ""
    license = ""
    citation = ""

    def path(self, key: str) -> Path:
        return raw_dir(self.dataset, str(self.version)) / self.files[key].filename

    def dry_run(self, *keys: str) -> dict:
        ks = keys or tuple(self.files)
        tot = sum(self.files[k].size for k in ks)
        d = raw_dir(self.dataset, str(self.version))
        d.mkdir(parents=True, exist_ok=True)
        return {"dataset": self.dataset, "version": self.version, "total_gib": round(tot / 2**30, 3),
                "host_free_gib": round(host_free_gib(d), 1),
                "files": [{"key": k, "file": self.files[k].filename, "bytes": self.files[k].size,
                           "url": self.files[k].url} for k in ks], "note": "dry run: nothing downloaded"}

    def fetch(self, key: str, *, min_free_gib: float = 40.0) -> ManifestEntry:
        if key not in self.files:
            raise KeyError(f"unknown file key {key!r}; available: {sorted(self.files)}")
        f = self.files[key]
        src = Source(key, f.filename, f.url, key, f.size, f.md5, self.record, self.license, self.citation)
        return download(self.dataset, str(self.version), src, min_free_gib=min_free_gib)

    def require(self, key: str) -> Path:
        p = self.path(key)
        if not p.exists():
            raise MissingPrerequisite(
                f"{self.dataset} {self.version}: {self.files[key].filename} not downloaded; run "
                f"`python scripts/linux/fetch_adapter.py {self.dataset} {key}` "
                f"({self.files[key].size / 2**30:.2f} GiB)")
        return p

    def verify_live(self) -> list[str]:
        """HEAD each file (GCS) and compare size / md5 to the registry. Returns discrepancies."""
        out = []
        for f in self.files.values():
            req = urllib.request.Request(f.url, method="HEAD")
            with urllib.request.urlopen(req, timeout=60) as r:
                size = int(r.headers.get("Content-Length", -1))
                md5 = None
                for h in r.headers.get_all("x-goog-hash") or []:
                    if h.startswith("md5="):
                        md5 = base64.b64decode(h[4:]).hex()
                if size != f.size or (f.md5 and md5 and md5 != f.md5):
                    out.append(f"{f.filename}: live size/md5 ({size}, {md5}) differs from registry")
        return out

    # helpers -----------------------------------------------------------
    def schema(self, key: str) -> dict:
        return feather_io.describe(self.require(key))

    def iter_batches(self, key: str, columns: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        return feather_io.iter_batches(self.require(key), columns)
