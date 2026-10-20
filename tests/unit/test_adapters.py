"""Secondary-dataset adapters: offline tests with mocked transports and tiny Feather fixtures."""
import json

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from flycircuit.data.adapters import (
    CrossDatasetMapping,
    MappingTable,
    MissingPrerequisite,
    TypeMapping,
    UnsupportedOperation,
    get_adapter,
    qualified_id,
    split_qualified_id,
)
from flycircuit.data.adapters.banc import BANCAdapter
from flycircuit.data.adapters.hemibrain import HemibrainAdapter
from flycircuit.data.adapters.larva import LarvaAdapter
from flycircuit.data.adapters.malecns import MaleCNSAdapter
from flycircuit.data.feather_io import SchemaError

BIG = 2**53 + 12345


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYCIRCUIT_HOME", str(tmp_path))
    return tmp_path


def _write_feather(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as f, ipc.new_file(f, table.schema) as w:
        for b in table.to_batches(max_chunksize=2):
            w.write_batch(b)


# ---- identity contracts -------------------------------------------------
def test_qualified_ids_exact_and_validated():
    assert qualified_id("malecns", "v1.0", BIG) == f"malecns:v1.0:{BIG}"
    assert split_qualified_id(f"banc:888:{BIG}") == ("banc", "888", BIG)
    with pytest.raises(ValueError):
        split_qualified_id(str(BIG))  # bare id rejected


def test_cross_dataset_mapping_requires_justification_and_two_releases():
    a, b = f"malecns:v1.0:{BIG}", "flywire:783:123"
    m = CrossDatasetMapping(a, b, "published type match", 0.7, "cell type")
    t = MappingTable([m])
    assert t.targets(a, 0.5) == [m] and t.targets(a, 0.9) == []
    with pytest.raises(ValueError):
        t.targets(str(BIG))  # bare ID lookup refused
    with pytest.raises(ValueError):
        CrossDatasetMapping(a, "malecns:v1.0:5", "x", 0.5, "y")  # same release
    with pytest.raises(ValueError):
        CrossDatasetMapping(a, b, "", 0.5, "y")                  # no method
    with pytest.raises(ValueError):
        CrossDatasetMapping(a, b, "m", 1.5, "y")                 # bad confidence
    with pytest.raises(ValueError):
        TypeMapping("malecns", "v1.0", "KC", "malecns", "v1.0", "KC", "m", 0.5)


def test_registry_and_unknown_dataset():
    assert {type(get_adapter(d)).__name__ for d in ("malecns", "banc", "hemibrain", "larva", "larval")} == {
        "MaleCNSAdapter", "BANCAdapter", "HemibrainAdapter", "LarvaAdapter"}
    with pytest.raises(UnsupportedOperation):
        get_adapter("nonexistent")
    with pytest.raises(NotImplementedError):  # UnsupportedOperation is a NotImplementedError
        get_adapter("nonexistent")


# ---- bulk adapters ------------------------------------------------------
def test_malecns_missing_download_reports_exact_step(home):
    a = MaleCNSAdapter()
    with pytest.raises(MissingPrerequisite, match="not downloaded"):
        a.neurons()
    d = a.dry_run("annotations")
    assert d["total_gib"] < 0.02 and "download" in d["note"] or "nothing downloaded" in d["note"]


def test_malecns_neurons_join_edges_and_type_mappings(home):
    a = MaleCNSAdapter()
    ann = pa.table({
        "bodyId": pa.array([BIG, 7, 9], pa.int64()), "type": ["KCg-m", "MBON01", None],
        "instance": ["a", "b", "c"], "superclass": ["cb_intrinsic"] * 3, "class": [None] * 3,
        "subclass": [None] * 3, "somaSide": ["R", "L", None], "statusLabel": ["t"] * 3,
        "status": ["Traced"] * 3, "flywireType": ["KCg-m", None, None], "hemibrainType": [None] * 3,
        "mancType": [None] * 3, "somaLocation": pa.array([[1, 2, 3], None, None], pa.list_(pa.int64()))})
    nt = pa.table({
        "body": pa.array([BIG, 7], pa.int64()), "predicted_nt": ["acetylcholine", "gaba"],
        "predicted_nt_confidence": [0.9, 0.8], "ground_truth": [None, "gaba"],
        "consensus_nt": ["acetylcholine", "gaba"], "celltype_predicted_nt": ["acetylcholine", "gaba"],
        "celltype_predicted_nt_confidence": [0.9, 0.7]})
    _write_feather(a.path("annotations"), ann)
    _write_feather(a.path("neurotransmitters"), nt)
    t = a.neurons()
    assert t.num_rows == 3                      # left join: no row lost, none duplicated
    d = {r["bodyId"]: r for r in t.to_pylist()}
    assert d[BIG]["uid"] == f"malecns:v1.0:{BIG}"           # exact >2^53 integer preserved
    assert d[BIG]["nt_consensus_nt"] == "acetylcholine" and d[9]["nt_consensus_nt"] is None  # unknown stays unknown
    maps = a.type_mappings()
    assert [(m.source_type, m.target_dataset, m.target_type) for m in maps] == [("KCg-m", "flywire", "KCg-m")]
    assert "not single-neuron identity" in maps[0].scope and maps[0].confidence <= 0.5

    _write_feather(a.path("weights"), pa.table({
        "body_pre": pa.array([BIG, 7, BIG], pa.int64()), "body_post": pa.array([7, BIG, 9], pa.int64()),
        "weight": pa.array([5, 3, 1], pa.int64())}))
    rows = [(p, q, w) for b in a.edge_batches() for p, q, w in zip(*[c.to_pylist() for c in b.columns], strict=True)]
    assert rows == [(BIG, 7, 5), (7, BIG, 3), (BIG, 9, 1)]   # direction preserved (pre -> post)


def test_malecns_schema_errors_are_actionable(home):
    a = MaleCNSAdapter()
    _write_feather(a.path("annotations"), pa.table({"bodyId": pa.array([1], pa.int64())}))
    _write_feather(a.path("neurotransmitters"), pa.table({"body": pa.array([1], pa.int64())}))
    with pytest.raises(SchemaError, match="missing columns"):
        a.neurons()


def test_banc_neurons_and_edges(home):
    a = BANCAdapter()
    _write_feather(a.path("meta"), pa.table({
        "banc_888_id": [str(BIG), "3"], "cell_type": ["a", None], "fafb_cell_type": ["a2", None],
        "manc_cell_type": [None, None], "malecns_cell_type": [None, None],
        "hemibrain_cell_type": [None, None], "fanc_cell_type": [None, None]}))
    t = a.neurons()
    assert t["uid"].to_pylist() == ["banc:888:3", f"banc:888:{BIG}"]   # numeric (not lexicographic) order
    assert t["banc_888_id"].to_pylist() == ["3", str(BIG)]             # source string ids untouched
    tm = a.type_mappings()
    assert [(m.source_type, m.target_dataset, m.target_type) for m in tm] == [("a", "flywire", "a2")]
    _write_feather(a.path("edgelist_simple_v3"), pa.table({
        "pre": [str(BIG)], "post": ["3"], "count": pa.array([12], pa.int32())}))
    b = next(iter(a.edge_batches("v3")))
    assert b.to_pylist() == [{"pre": str(BIG), "post": "3", "count": 12}]
    with pytest.raises(KeyError):
        list(a.edge_batches("v9"))
    _write_feather(a.path("meta"), pa.table({"x": [1]}))
    with pytest.raises(SchemaError):
        a.neurons()


# ---- hemibrain (token-gated REST, mocked) ---------------------------------
def _neuprint_transport(seen):
    def h(req: httpx.Request):
        seen.append(req)
        if req.url.path == "/api/dbmeta/datasets":
            return httpx.Response(200, json={"hemibrain:v1.2.1": {"last-mod": "2020-12-05", "uuid": "u"}})
        body = json.loads(req.content)
        if req.headers.get("authorization") != "Bearer tok123":
            return httpx.Response(401)
        if "ConnectsTo" in body["cypher"]:
            return httpx.Response(200, json={"columns": ["body_pre", "body_post", "weight"],
                                             "data": [[BIG, 5, 20], [BIG, 6, 4]]})
        return httpx.Response(200, json={"columns": ["bodyId", "type", "instance", "status"],
                                         "data": [[BIG, "KCab", "KCab(R)", "Traced"]]})
    return httpx.MockTransport(h)


def test_hemibrain_public_listing_without_token(monkeypatch):
    monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
    seen = []
    a = HemibrainAdapter(transport=_neuprint_transport(seen))
    assert a.verify_release()["uuid"] == "u"
    assert all("authorization" not in r.headers for r in seen)
    with pytest.raises(MissingPrerequisite, match="NEUPRINT_APPLICATION_CREDENTIALS"):
        a.neurons()


def test_hemibrain_authenticated_queries_and_bounds(monkeypatch):
    monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
    seen = []
    a = HemibrainAdapter(token="tok123", transport=_neuprint_transport(seen))
    t = a.neurons(limit=10)
    assert t["uid"].to_pylist() == [f"hemibrain:v1.2.1:{BIG}"]
    q = json.loads(seen[-1].content)
    assert q["dataset"] == "hemibrain:v1.2.1" and "LIMIT 10" in q["cypher"]
    rows = [r for b in a.edge_batches([BIG]) for r in b.to_pylist()]
    assert rows[0] == {"body_pre": BIG, "body_post": 5, "weight": 20}
    with pytest.raises(UnsupportedOperation):
        list(a.edge_batches())                      # no whole-graph pull
    with pytest.raises(ValueError):
        a.neurons(limit=10**9)
    with pytest.raises(ValueError):
        a.neurons(properties=("type; DROP",))       # no Cypher injection through property names
    bad = HemibrainAdapter(token="wrong", transport=_neuprint_transport([]))
    with pytest.raises(MissingPrerequisite, match="rejected"):
        bad.neurons()


def test_hemibrain_unknown_release_reports_available(monkeypatch):
    a = HemibrainAdapter("v9.9", transport=_neuprint_transport([]))
    with pytest.raises(UnsupportedOperation, match="hemibrain:v1.2.1"):
        a.verify_release()


# ---- larva (CATMAID, mocked) ------------------------------------------------
def _catmaid_transport():
    def h(req: httpx.Request):
        p = req.url.path
        if p == "/":
            return httpx.Response(200, text="ok", headers={"set-cookie": "csrftoken_abc=T0K; Path=/"})
        if p == "/version":
            return httpx.Response(200, json={"SERVER_VERSION": "2021.12.21"})
        if p == "/1/skeletons/":
            return httpx.Response(200, json=[BIG, 29, 11995])
        if p == "/1/skeletons/connectivity":
            if req.headers.get("x-csrftoken") != "T0K":
                return httpx.Response(403, json={"detail": "CSRF"})
            return httpx.Response(200, json={"outgoing": {"16": {"num_nodes": 3, "skids": {"29": [0, 0, 1, 0, 2]}}},
                                             "incoming": {}})
        return httpx.Response(404)
    return httpx.MockTransport(h)


def test_larva_neurons_connectivity_and_provenance():
    a = LarvaAdapter(transport=_catmaid_transport())
    t = a.neurons()
    assert t["skeleton_id"].to_pylist() == [29, 11995, BIG]
    assert t["uid"][0].as_py() == "larva:l1em-p1:29"
    assert a.provenance()["mutable_service"] is True
    rows = [r for b in a.edge_batches([29]) for r in b.to_pylist()]
    assert rows == [{"skeleton_pre": 29, "skeleton_post": 16, "count": 3}]   # direction + summed confidences
    with pytest.raises(UnsupportedOperation):
        list(a.edge_batches())


def test_adapters_never_fabricate_without_prerequisites(home, monkeypatch):
    monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
    for ds in ("malecns", "banc"):
        with pytest.raises(MissingPrerequisite):
            get_adapter(ds).neurons()
        with pytest.raises(MissingPrerequisite):
            next(iter(get_adapter(ds).edge_batches()))
    with pytest.raises(MissingPrerequisite):
        get_adapter("hemibrain").neurons()
