"""Offline tests with tiny SYNTHETIC Feather fixtures (not biological data)."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flycircuit.data import adapters, feather_io, ingest, selectors, sources, validate
from flycircuit.data.paths import processed_dir, raw_dir

BIG = 720575940628857210  # > 2**53: float64 would corrupt it
A, B, C, D = BIG, BIG + 1, BIG + 2, BIG + 3  # consecutive ids differ below float64 resolution
EXT = BIG + 99  # not in the proofread set


def _batch(rows):
    pre, post, npil, cnt = zip(*rows, strict=False)
    return pa.RecordBatch.from_arrays(
        [pa.array(pre, pa.uint64()), pa.array(post, pa.uint64()),
         pa.array(npil), pa.array(cnt, pa.int32())],
        names=["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count"])


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYCIRCUIT_HOME", str(tmp_path))
    return tmp_path


def _make_raw(home):
    raw = raw_dir("flywire", "783")
    raw.mkdir(parents=True)
    # the same (A->B, MB_CA) key appears in 3 different batches; A->B also in another neuropil
    b1 = _batch([(A, B, "MB_CA", 3), (B, A, "MB_CA", 7), (A, A, "LH", 2)])
    b2 = _batch([(A, B, "MB_CA", 4), (A, B, "LH", 1), (C, D, "MB_CA", 9)])
    b3 = _batch([(A, B, "MB_CA", 1), (EXT, A, "MB_CA", 6), (D, C, "MB_CA", 2)])
    feather_io.write_feather_batches(raw / "proofread_connections_783.feather", [b1, b2, b3])
    np.save(raw / "proofread_root_ids_783.npy", np.array([A, B, C, D], dtype=np.int64))
    cls = {A: ("Kenyon_Cell", "KCab"), B: ("MBON", "MBON01"), C: ("DAN", "PAM01"),
           D: ("CX", "EPG")}
    lines = ["supervoxel_id\troot_id\tsuper_class\tcell_class\tcell_type\ttop_nt\ttop_nt_conf\tside\tpos_x"]
    for i, (rid, (c, t)) in enumerate(cls.items()):
        lines.append(f"{i + 1}\t{rid}\tcentral\t{c}\t{t}\tacetylcholine\t0.9\tleft\t")
    lines.append(f"99\t{A}\tdup\tdup\tdup\t\t\tleft\t")  # duplicate annotation row for root A
    (raw / "Supplemental_file1_neuron_annotations.tsv").write_text("\n".join(lines) + "\n")


def test_cross_batch_aggregation_and_orientation(home):
    _make_raw(home)
    rep = ingest.run_ingest(threshold=5)
    out = processed_dir("flywire", "783")
    pairs = {(r["pre"], r["post"]): r for r in pq.read_table(out / "edges_pairs.parquet").to_pylist()}
    assert pairs[(A, B)]["count"] == 3 + 4 + 1 + 1  # summed across 3 batches and 2 neuropils
    assert pairs[(A, B)]["n_neuropils"] == 2
    assert pairs[(B, A)]["count"] == 7  # directed: reverse is a separate edge
    assert pairs[(A, A)]["count"] == 2 and rep["edges"]["self_loop_pairs"] == 1
    bn = pq.read_table(out / "edges_by_neuropil.parquet").to_pylist()
    mb = [r for r in bn if (r["pre"], r["post"], r["neuropil"]) == (A, B, "MB_CA")][0]
    assert mb["count"] == 8 and mb["n_source_rows"] == 3
    assert rep["edges"]["record_batches"] == 3
    assert rep["edges"]["duplicate_pair_neuropil_keys_merged"] == 2


def test_counts_threshold_and_exclusions(home):
    _make_raw(home)
    e = ingest.run_ingest(threshold=5)["edges"]
    assert e["source_rows"] == 9 and e["source_contacts"] == 35
    assert e["unique_pairs"] == 6
    assert e["edge_roots_not_in_proofread_set"] == 1
    assert e["pairs_with_unmatched_endpoint"] == 1 and e["pairs_with_both_endpoints_internal"] == 5
    assert e["pairs_retained_at_threshold"] == 4  # A->B 9, B->A 7, C->D 9, EXT->A 6
    assert e["contacts_in_pairs"] == e["source_contacts"]


def test_exact_ids_and_neuron_table(home):
    _make_raw(home)
    ingest.run_ingest()
    n = pq.read_table(processed_dir("flywire", "783") / "neurons.parquet").to_pylist()
    ids = {r["root_id"] for r in n}
    assert ids == {A, B, C, D}  # distinct despite float64 collisions
    assert all(r["root_id_str"] == str(r["root_id"]) for r in n)
    assert all(r["uid"] == f"flywire:783:{r['root_id']}" for r in n)
    a = [r for r in n if r["root_id"] == A][0]
    assert a["cell_class"] == "Kenyon_Cell"  # duplicate annotation row dropped deterministically
    assert a["pos_x"] is None  # coordinates stay nullable


def test_validate_roundtrip(home):
    _make_raw(home)
    ingest.run_ingest()
    # fake a verified manifest for the three proofread files
    man = {k: {"success": True, "filename": s.filename}
           for k, s in sources.sources_for("flywire", "783").items()}
    p = sources.manifest_path("flywire", "783")
    p.write_text(json.dumps(man))
    res = validate.validate("flywire", "783")
    assert all(c.ok for c in res), [c for c in res if not c.ok]


def test_schema_errors():
    with pytest.raises(feather_io.SchemaError):
        feather_io.resolve_columns(["a", "b"])
    assert feather_io.resolve_columns(
        ["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count", "gaba_avg"])["count"] == "syn_count"


def test_batches_are_projected(home):
    _make_raw(home)
    f = raw_dir("flywire", "783") / "proofread_connections_783.feather"
    b = next(feather_io.iter_batches(f, ["syn_count"]))
    assert b.schema.names == ["syn_count"]
    assert sum(x.num_rows for x in feather_io.edge_batches(f)) == 9


def test_selector_missing_label_is_actionable(home):
    _make_raw(home)
    ingest.run_ingest()
    with pytest.raises(selectors.SelectionError, match="matched 0 neurons"):
        selectors.select_circuit("mushroom_body")


def test_selector_cx_boundary(home):
    _make_raw(home)
    ingest.run_ingest()
    s = selectors.select_circuit("central_complex", threshold=5, out_name="cx")
    assert s["n_neurons"] == 1 and s["internal_pairs_retained"] == 0
    d = processed_dir("flywire", "783") / "circuits" / "cx"
    assert (d / "BOUNDARY.md").exists() and (d / "boundary_ids.parquet").exists()
    assert selectors.select_circuit("central_complex", threshold=5, out_name="cx2")[
        "ids_hash_sha256"] == s["ids_hash_sha256"]  # deterministic


def test_adapters_never_fake_data(home):
    # real adapters (tests/unit/test_adapters.py) fail with a precise prerequisite, never fabricate
    for ds in ("malecns", "banc", "hemibrain", "larva"):
        a = adapters.get_adapter(ds)
        assert a.describe()["dataset"] == ds
    with pytest.raises(NotImplementedError):
        adapters.get_adapter("not-a-dataset")
    assert adapters.qualified_id("flywire", "783", BIG) == f"flywire:783:{BIG}"


def test_download_checksum_mismatch_deletes_partial(home, tmp_path):
    payload = tmp_path / "x.bin"
    payload.write_bytes(b"hello")
    src = sources.Source("x", "x.bin", payload.as_uri(), "t", 5, "0" * 32)
    e = sources.download("flywire", "783", src, min_free_gib=0)
    assert not e.success and "md5" in e.error
    d = raw_dir("flywire", "783")
    assert not (d / "x.bin").exists() and not (d / "x.bin.partial").exists()


def test_download_success_atomic_and_immutable(home, tmp_path):
    import hashlib

    payload = tmp_path / "y.bin"
    payload.write_bytes(b"abcdef")
    md5 = hashlib.md5(b"abcdef").hexdigest()  # noqa: S324
    src = sources.Source("y", "y.bin", payload.as_uri(), "t", 6, md5)
    e = sources.download("flywire", "783", src, min_free_gib=0)
    assert e.success and e.sha256 == hashlib.sha256(b"abcdef").hexdigest()
    assert not (raw_dir("flywire", "783") / "y.bin.partial").exists()
    assert sources.download("flywire", "783", src).success  # no re-download/overwrite


def test_download_resumes_partial(home, tmp_path):
    import hashlib

    payload = tmp_path / "z.bin"
    payload.write_bytes(b"0123456789")
    d = raw_dir("flywire", "783")
    d.mkdir(parents=True)
    (d / "z.bin.partial").write_bytes(b"01234")  # file:// ignores Range -> restart path
    src = sources.Source("z", "z.bin", payload.as_uri(), "t", 10,
                         hashlib.md5(b"0123456789").hexdigest())  # noqa: S324
    assert sources.download("flywire", "783", src, min_free_gib=0).success
    assert (d / "z.bin").read_bytes() == b"0123456789"


def test_dry_run_synapses_does_not_download(home):
    r = sources.dry_run("flywire", "783", "synapses")
    assert r["total_gib"] > 8 and not (raw_dir("flywire", "783") / "flywire_synapses_783.feather").exists()


def test_unsupported_version():
    with pytest.raises(NotImplementedError):
        sources.sources_for("flywire", "630")
