"""MaleCNS (Janelia/Cambridge male central nervous system) adapter, release v1.0.

Verified 2026-10-15 at https://male-cns.janelia.org/release/ and /download/ :
 - releases: v0.9 (2025-10-05), v1.0 (2026-06-08); data license CC-BY 4.0
 - public bulk Feather (no token) at https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/
 - neuPrint dataset `male-cns:v1.0` at https://neuprint.janelia.org (API token required; not used here)
Sizes/md5 below were read from the live GCS object headers on 2026-10-15 (`verify_live()` rechecks) and the
downloaded files' Arrow schemas were inspected (annotations 211,577 rows; transmitters 1,835,518 rows;
weights columns body_pre/body_post/weight, 2,318 record batches).
Scope: male CNS = central brain + optic lobes + VNC. IDs are neuPrint ``bodyId``; the external
identity is ``malecns:v1.0:<bodyId>``. Not FlyWire IDs: cross-dataset links are TYPE-level only
(`type_mappings`), taken from the release's own annotation columns, never ID equality.
"""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa
import pyarrow.compute as pc

from flycircuit.data.feather_io import SchemaError

from .base import TypeMapping
from .bulk import BulkFeatherAdapter, BulkFile

_B = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

ANN_COLS = ["bodyId", "type", "instance", "superclass", "class", "subclass", "somaSide", "statusLabel",
            "status", "flywireType", "hemibrainType", "mancType", "somaLocation"]
NT_COLS = ["body", "predicted_nt", "predicted_nt_confidence", "ground_truth", "consensus_nt",
           "celltype_predicted_nt", "celltype_predicted_nt_confidence"]


def _need(names: list[str], have: list[str], what: str) -> None:
    miss = [c for c in names if c not in have]
    if miss:
        raise SchemaError(f"MaleCNS {what}: missing columns {miss}; have {sorted(have)}")


class MaleCNSAdapter(BulkFeatherAdapter):
    dataset = "malecns"
    default_version = "v1.0"
    access = "public bulk Feather (GCS, no token); neuPrint needs a token (not used)"
    record = "flyem-male-cns v1.0 (https://male-cns.janelia.org/release/)"
    license = "CC-BY-4.0"
    citation = ("Male CNS connectome project (FlyEM/HHMI Janelia, Cambridge, MRC LMB, Google Research); "
                "cite per male-cns.janelia.org")
    files = {
        "annotations": BulkFile(
            "annotations", "body-annotations-male-cns-v1.0-minconf-0.5.feather",
            f"{_B}/body-annotations-male-cns-v1.0-minconf-0.5.feather", 14483314,
            "50a7718770c57220f160ba4f431ab89e", "curated classes/types/sides; no transmitters"),
        "neurotransmitters": BulkFile(
            "neurotransmitters", "body-neurotransmitters-male-cns-v1.0.feather",
            f"{_B}/body-neurotransmitters-male-cns-v1.0.feather", 43282834,
            "3d842b12fe5c49eefade528d7dd24a1f", "aggregate predicted transmitter per neuron"),
        "weights": BulkFile(
            "weights", "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
            f"{_B}/connectome-weights-male-cns-v1.0-minconf-0.5.feather", 1051241946,
            "f30e9dcca25cfd021bf1e7b3d975599e", "segment-to-segment connection strengths (full graph)"),
    }
    capabilities = {
        "neurons": "annotations + transmitter predictions (small files)",
        "edge_batches": "connectome-weights.feather via record batches (1.1 GB)",
        "not_supported": "synapse points (12.7 GB), t-bar transmitters (2.7 GB), skeletons: not wired",
    }

    def neurons(self) -> pa.Table:
        """Annotated bodies (211,577) left-joined to transmitter predictions. Coordinates are kept as the
        raw ``somaLocation`` list (nullable); the voxel size is NOT assumed here (the download page states
        8 nm for synapse points; the annotation table's unit is not documented on the pages read)."""
        ann_p, nt_p = self.require("annotations"), self.require("neurotransmitters")
        import pyarrow.ipc as ipc
        a = ipc.open_file(ann_p).read_all()
        _need(ANN_COLS, a.column_names, "annotations")
        a = a.select(ANN_COLS)
        if a["bodyId"].null_count:
            raise SchemaError("MaleCNS annotations: null bodyId")
        n = ipc.open_file(nt_p).read_all()
        _need(NT_COLS, n.column_names, "neurotransmitters")
        n = n.select(NT_COLS)
        if pc.count_distinct(n["body"]).as_py() != n.num_rows:
            raise SchemaError("MaleCNS: transmitter table has duplicate body ids (join would fan out)")
        # index-based left join (Arrow's hash join rejects list/null-typed non-key columns)
        idx = pc.index_in(a["bodyId"], value_set=n["body"])
        t = a
        for c in NT_COLS[1:]:
            t = t.append_column(f"nt_{c}", n[c].take(idx))
        t = t.sort_by("bodyId")
        uid = pa.array([self.uid(i) for i in t["bodyId"].to_pylist()])
        return t.append_column("uid", uid).append_column(
            "provenance", pa.array([f"malecns:{self.version}:annotations+neurotransmitters"] * t.num_rows))

    def edge_batches(self) -> Iterator[pa.RecordBatch]:
        """Directed (body_pre -> body_post, weight) batches. Aggregate externally; never read_all()."""
        path = self.require("weights")
        from flycircuit.data import feather_io
        d = feather_io.describe(path)
        _need(["body_pre", "body_post", "weight"], [c[0] for c in d["columns"]], "weights")
        yield from feather_io.iter_batches(path, ["body_pre", "body_post", "weight"])

    def type_mappings(self, kinds=("flywireType", "hemibrainType", "mancType")) -> list[TypeMapping]:
        """Distinct type-name correspondences published in the release's own annotation columns."""
        import pyarrow.ipc as ipc
        a = ipc.open_file(self.require("annotations")).read_all().select(["type", *kinds])
        tgt = {"flywireType": "flywire", "hemibrainType": "hemibrain", "mancType": "manc"}
        out, seen = [], set()
        for k in kinds:
            sub = a.filter(pc.and_(pc.is_valid(a["type"]), pc.is_valid(a[k])))
            for s, t in zip(sub["type"].to_pylist(), sub[k].to_pylist(), strict=True):
                if (k, s, t) in seen:
                    continue
                seen.add((k, s, t))
                out.append(TypeMapping(self.dataset, str(self.version), s, tgt[k],
                                       "unspecified in source annotation", t,
                                       f"male-cns v1.0 annotation column {k} (not independently validated)",
                                       0.5))
        return out
