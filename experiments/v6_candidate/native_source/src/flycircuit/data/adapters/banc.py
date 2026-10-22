"""BANC (Brain And Nerve Cord, female adult Drosophila) adapter, compiled release banc_888.

Verified 2026-10-15: public GCS bucket ``lee-lab_brain-and-nerve-cord-fly-connectome`` (JSON listing
works anonymously) with compiled tables for CAVE materialization 888 (snapshot 2026-04-16, per
https://github.com/sjcabs/fly_connectome_data_tutorial docs/dataset_documentation/banc_data.md);
cited as Bates, Phelps, Kim, Yang et al. (2026) Nature. Recommended primary key ``banc_888_id``;
edgelist ``pre``/``post`` use v888 root IDs. v3 synapse table (size >= 10) recommended for new
analyses, v2 (size >= 5) is used in the paper. The live CAVE datastack (caveclient + token) is NOT used.
No explicit data license was found on the pages read: check before redistributing derived assets.
External identity: ``banc:888:<banc_888_id>``; not interchangeable with FlyWire IDs.
Column names beyond the documented ``banc_888_id``/``pre``/``post`` are discovered from the downloaded
file and validated at read time (SchemaError lists what is present).
"""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

from flycircuit.data import feather_io
from flycircuit.data.feather_io import SchemaError

from .base import TypeMapping
from .bulk import BulkFeatherAdapter, BulkFile

_B = "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888"


class BANCAdapter(BulkFeatherAdapter):
    dataset = "banc"
    default_version = "888"
    access = "public bulk files (GCS, anonymous); live CAVE datastack needs a token (not used)"
    record = "lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888 (CAVE mat. 888)"
    license = "not stated on pages read (UNVERIFIED); cite Bates et al. 2026 Nature"
    citation = "Bates, Phelps, Kim, Yang et al. (2026), Nature (BANC); see docs/DATA_SECONDARY.md"
    files = {
        "meta": BulkFile("meta", "banc_888_meta.feather", f"{_B}/banc_888_meta.feather", 57503026,
                         "8c2b93a608c7163ec9d94d68e0756ff4", "per-neuron metadata/annotations"),
        "edgelist_simple_v3": BulkFile(
            "edgelist_simple_v3", "banc_888_edgelist_simple_v3.feather",
            f"{_B}/banc_888_edgelist_simple_v3.feather", 359161658, "08542b0771db7418ed474be60dc9886c",
            "neuron-to-neuron edges, synapse size >= 10 (recommended)"),
        "edgelist_simple_v2": BulkFile(
            "edgelist_simple_v2", "banc_888_edgelist_simple_v2.feather",
            f"{_B}/banc_888_edgelist_simple_v2.feather", 305250378, "394406f8a9bdf093c895f95aff4f6c49",
            "neuron-to-neuron edges, synapse size >= 5 (used in the paper)"),
        "neurotransmitters_v2": BulkFile(
            "neurotransmitters_v2", "banc_888_neurotransmitter_prediction_v2.csv",
            f"{_B}/banc_888_neurotransmitter_prediction_v2.csv", 21107592,
            "4ebbd1d6e05d4192ad0c6db27739a8e3", "CSV transmitter predictions"),
    }
    capabilities = {
        "neurons": "banc_888_meta.feather",
        "edge_batches": "edgelist_simple v2/v3 via record batches",
        "not_supported": "per-synapse parquet (17-20 GB), meshes, skeletons, influence tables: not wired",
    }
    id_column = "banc_888_id"

    def neurons(self) -> pa.Table:
        """All columns of banc_888_meta (verified schema: 77 columns, ids stored as STRINGS to keep
        exact 18-digit integers; coordinates are strings in ``position``/``root_position_nm`` etc. and
        are passed through untouched -- their space/unit is not reinterpreted here)."""
        t = ipc.open_file(self.require("meta")).read_all()
        if self.id_column not in t.column_names:
            raise SchemaError(f"BANC meta: no {self.id_column!r}; have {sorted(t.column_names)}")
        if t[self.id_column].null_count:
            raise SchemaError("BANC meta: null banc_888_id")
        ids = pc.cast(t[self.id_column], pa.int64())          # exact: ids < 2^63
        t = t.take(pc.sort_indices(ids))
        uid = pa.array([self.uid(i) for i in t[self.id_column].to_pylist()])
        return t.append_column("uid", uid)

    def type_mappings(self) -> list[TypeMapping]:
        """Type-name links from BANC's own meta columns (fafb = the FAFB/FlyWire EM volume)."""
        cols = {"fafb_cell_type": "flywire", "manc_cell_type": "manc", "malecns_cell_type": "malecns",
                "hemibrain_cell_type": "hemibrain", "fanc_cell_type": "fanc"}
        t = ipc.open_file(self.require("meta")).read_all()
        miss = [c for c in ("cell_type", *cols) if c not in t.column_names]
        if miss:
            raise SchemaError(f"BANC meta missing columns {miss}")
        out, seen = [], set()
        for k, tgt in cols.items():
            sub = t.filter(pc.and_(pc.is_valid(t["cell_type"]), pc.is_valid(t[k])))
            for s, o in zip(sub["cell_type"].to_pylist(), sub[k].to_pylist(), strict=True):
                if (k, s, o) not in seen:
                    seen.add((k, s, o))
                    out.append(TypeMapping(self.dataset, str(self.version), s, tgt,
                                           "unspecified in source annotation", o,
                                           f"BANC banc_888_meta column {k} (not independently validated)", 0.5))
        return out

    def edge_batches(self, version: str = "v3") -> Iterator[pa.RecordBatch]:
        """Verified edgelist schema: pre, post (STRING ids), count int32, norm, post_count, pre_count.
        Convert ids with int() only when needed; strings are kept to guarantee exactness."""
        key = f"edgelist_simple_{version}"
        if key not in self.files:
            raise KeyError(f"edgelist versions: v2, v3 (got {version!r})")
        path = self.require(key)
        cols = [c[0] for c in feather_io.describe(path)["columns"]]
        for need in ("pre", "post"):
            if need not in cols:
                raise SchemaError(f"BANC {key}: no {need!r} column; have {cols}")
        yield from feather_io.iter_batches(path)
