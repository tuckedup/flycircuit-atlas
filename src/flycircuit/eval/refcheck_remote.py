"""Fetch individual members of a huge remote ZIP via HTTP Range requests (bounded download).

Used to obtain a few small published model-output files from the 4.5 GB Shiu et al. archive
(Edmond doi:10.17617/3.CZODIW, results.zip) without downloading the whole archive. ``zipfile``
verifies each member's CRC32 on read; the whole-archive md5 is NOT checked (never downloaded).
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zlib
from pathlib import Path, PurePosixPath, PureWindowsPath


class RangeFile(io.RawIOBase):
    """Read-only, seekable file object over an HTTP URL that supports ``Range``."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url = url
        self.block = block
        req = urllib.request.Request(url, method="HEAD")
        r = urllib.request.urlopen(req, timeout=60)
        self.final_url = r.geturl()
        self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self._cache: dict[int, bytes] = {}
        self.bytes_fetched = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _block(self, idx: int) -> bytes:
        if idx not in self._cache:
            a = idx * self.block
            b = min(self.size, a + self.block) - 1
            req = urllib.request.Request(self.final_url, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=180) as x:
                if x.status != 206:
                    raise OSError(f"server ignored Range (status {x.status})")
                data = x.read()
            self.bytes_fetched += len(data)
            if len(self._cache) > 64:  # bound memory: 64 MiB
                self._cache.pop(next(iter(self._cache)))
            self._cache[idx] = data
        return self._cache[idx]

    def readinto(self, buf) -> int:
        n = min(len(buf), self.size - self.pos)
        out = bytearray()
        while len(out) < n:
            idx, off = divmod(self.pos + len(out), self.block)
            out += self._block(idx)[off : off + (n - len(out))]
        buf[:n] = out
        self.pos += n
        return n


def open_remote_zip(url: str):
    import zipfile

    rf = RangeFile(url)
    return zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 16)), rf


def extract_members(url: str, names: list[str], out_dir: Path, *, preserve_paths: bool = False) -> dict:
    """CRC/SHA-verified extraction; reject collisions and preserve existing raw files."""
    root = out_dir.resolve()
    destinations = {}
    for name in names:
        relative = PurePosixPath(name.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or PureWindowsPath(name).drive:
            raise ValueError(f"unsafe archive member path: {name!r}")
        dest = root / (str(relative) if preserve_paths else relative.name)
        dest = dest.resolve()
        if not dest.is_relative_to(root):
            raise ValueError(f"archive member escapes output directory: {name!r}")
        if str(dest).casefold() in {str(p).casefold() for p in destinations.values()}:
            raise ValueError("archive member destination collision; use preserve_paths=True")
        destinations[name] = dest
    out_dir.mkdir(parents=True, exist_ok=True)
    zf, rf = open_remote_zip(url)
    res = {}
    with zf:
        for name, dest in destinations.items():
            info = zf.getinfo(name)
            if info.is_dir():
                raise ValueError(f"requested archive member is a directory: {name!r}")
            digest, crc = hashlib.sha256(), 0
            reused = dest.exists()
            if reused:
                with dest.open("rb") as source:
                    for chunk in iter(lambda: source.read(1 << 20), b""):
                        digest.update(chunk)
                        crc = zlib.crc32(chunk, crc)
                if dest.stat().st_size != info.file_size or crc != info.CRC:
                    raise ValueError(f"existing raw file differs from archive; refusing overwrite: {dest}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                part = dest.with_suffix(dest.suffix + ".partial")
                with zf.open(info) as source, part.open("wb") as target:
                    while chunk := source.read(1 << 20):
                        target.write(chunk)
                        digest.update(chunk)
                        crc = zlib.crc32(chunk, crc)
                if part.stat().st_size != info.file_size or crc != info.CRC:
                    raise ValueError(f"archive member failed size/CRC verification: {name}")
                part.replace(dest)
            res[name] = {"path": str(dest), "size": info.file_size, "crc32": f"{crc:08x}",
                         "sha256": digest.hexdigest(), "reused": reused}
    res["_bytes_fetched"] = rf.bytes_fetched
    return res
