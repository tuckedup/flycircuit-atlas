"""Bounded-memory Feather V2 (Arrow IPC file) reading: record batches + column projection.

Never calls read_all()/to_pandas() on a whole file. Memory-mapping is used only as an I/O
convenience; compressed buffers are decompressed per record batch.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
from pyarrow import ipc

# Candidate names for the logical columns of proofread_connections (verified at ingest).
PRE_NAMES = ("pre_pt_root_id", "pre_root_id", "pre")
POST_NAMES = ("post_pt_root_id", "post_root_id", "post")
NEUROPIL_NAMES = ("neuropil",)
COUNT_NAMES = ("syn_count", "count", "synapse_count")


class SchemaError(ValueError):
    pass


def open_reader(path: str | Path) -> ipc.RecordBatchFileReader:
    return ipc.open_file(pa.memory_map(str(path), "r"))


def describe(path: str | Path) -> dict:
    r = open_reader(path)
    sch = r.schema
    return {
        "num_record_batches": r.num_record_batches,
        "columns": [(f.name, str(f.type)) for f in sch],
    }


def resolve_columns(names: list[str]) -> dict[str, str]:
    """Map logical -> physical column names; raise SchemaError with a clear message."""
    out = {}
    for logical, cands in (("pre", PRE_NAMES), ("post", POST_NAMES),
                           ("neuropil", NEUROPIL_NAMES), ("count", COUNT_NAMES)):
        hit = next((c for c in cands if c in names), None)
        if hit is None:
            raise SchemaError(f"no column for {logical!r}; expected one of {cands}; have {names}")
        out[logical] = hit
    return out


def iter_batches(path: str | Path, columns: list[str] | None = None) -> Iterator[pa.RecordBatch]:
    """Yield record batches one at a time (projected). Peak memory ~ one batch."""
    r = open_reader(path)
    for i in range(r.num_record_batches):
        b = r.get_batch(i)
        if columns is not None:
            b = pa.RecordBatch.from_arrays(
                [b.column(c) for c in columns], names=columns)
        yield b


def edge_batches(path: str | Path) -> Iterator[pa.RecordBatch]:
    """Projected batches with canonical names pre, post, neuropil, count (int64/int64/str/int64)."""
    r = open_reader(path)
    cols = resolve_columns(r.schema.names)
    for b in iter_batches(path, list(cols.values())):
        pre = b.column(cols["pre"]).cast(pa.int64())
        post = b.column(cols["post"]).cast(pa.int64())
        npil = b.column(cols["neuropil"]).cast(pa.string())
        cnt = b.column(cols["count"]).cast(pa.int64())
        yield pa.RecordBatch.from_arrays([pre, post, npil, cnt],
                                         names=["pre", "post", "neuropil", "count"])


def write_feather_batches(path: str | Path, batches: list[pa.RecordBatch]) -> None:
    """Test/fixture helper: write several record batches as Feather V2 (uncompressed IPC)."""
    with pa.OSFile(str(path), "wb") as sink, ipc.new_file(sink, batches[0].schema) as w:
        for b in batches:
            w.write_batch(b)
