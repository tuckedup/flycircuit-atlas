"""Regression: type screens retain real labels; pilot filtering preserves exact graph identity."""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from flycircuit.graph.versioned import VersionedGraph
from flycircuit.interventions import trained as T
from flycircuit.interventions.targets import eligible_units
from flycircuit.tasks.graph_loader import load_flywire_circuit


def test_flywire_loader_retains_annotations_and_filters_pairs(tmp_path):
    base = tmp_path / "data/processed/flywire/783"
    circuit = base / "circuits/test_pilot"
    circuit.mkdir(parents=True)
    ids = np.array([720575940000000003, 720575940000000001, 720575940000000004,
                    720575940000000002], dtype=np.int64)
    pq.write_table(pa.table({"root_id": ids, "grp": ["mbon", "kenyon", "mbon", "kenyon"],
                             "cell_type": ["out1", "KC-a", "out2", "KC-a"],
                             "cell_class": ["MBON", "KC", "MBON", "KC"]}),
                   circuit / "internal_ids.parquet")
    pq.write_table(pa.table({"root_id": ids, "known_nt": ["gaba", "acetylcholine"] * 2,
                             "top_nt": [None] * 4, "top_nt_conf": [0.0] * 4}),
                   base / "neurons.parquet")
    # Duplicate pair, a subthreshold pair and an external endpoint exercise legacy equivalence.
    pre = ids[[1, 1, 3, 0]].tolist() + [int(ids[1]) + 100]
    post = ids[[0, 0, 2, 2]].tolist() + [int(ids[2])]
    counts = [5, 6, 8, 4, 99]
    pq.write_table(pa.table({"pre": pre, "post": post, "count": counts}), base / "edges_pairs.parquet")
    (circuit / "selection.json").write_text(json.dumps({"is_pilot_subset": True}))
    bundle = load_flywire_circuit("test_pilot", root=tmp_path)
    legacy = VersionedGraph.from_edges("flywire", "783", pre[:3], post[:3], counts[:3], root_ids=ids)
    assert bundle.graph.edge_hash() == legacy.edge_hash()
    assert bundle.graph.mapping_hash() == legacy.mapping_hash()
    assert bundle.graph.node_attrs["cell_type"] == ["KC-a", "KC-a", "out1", "out2"]
    units, report = eligible_units(bundle.graph, "type", exclude_sensory=False)
    assert report["n_units"] == 1
    assert units[0].label == "KC-a" and units[0].nodes == (0, 1)


def test_empty_aggregate_has_no_invented_results():
    assert T.aggregate([], [], None, seed=0, n_controls=0) == []


def test_missing_type_annotations_are_reported_not_crashed(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from flycircuit.tasks.graph_loader import make_learning_synthetic

    graph = make_learning_synthetic(n_kc=8, n_mbon=2, n_aux=2).graph
    directory = tmp_path / "instances/v2_default"
    directory.mkdir(parents=True)
    for seed in [1, 2]:
        (directory / f"bptt_s{seed}.pt").touch()
    monkeypatch.setitem(T._BUNDLES, ("fixture", 5, "v2_default"), SimpleNamespace(graph=graph))

    def load_instance(path, device):
        return SimpleNamespace(seed=int(path.stem.split("_s")[1]), baseline_val_acc=0.8,
                               path=str(path), meta={"train_summary": {"val_acc": 0.8, "test_acc": 0.8}})

    monkeypatch.setattr(T, "load_instance", load_instance)
    cfg = {"torch_threads": 1, "device": "cpu", "min_baseline_acc": 0.6,
           "sign_variants": ["v2_default"], "learners": ["bptt"], "circuit": "fixture",
           "min_contacts": 5, "level": "type", "n_interventions": 120,
           "min_distinct_targets": 100, "t_start": 0, "seed": 0, "n_controls": 2,
           "sweep": {"n_targets": 30, "doses": {}, "windows": [[0, None]]}}
    result = T.screen_stage(cfg, tmp_path, tmp_path / "out")
    screen = result["screens"]["v2_default|bptt"]
    assert screen["status"] == "no_eligible_targets"
    assert "cell_type" in screen["eligibility"]["error"]
    assert result["plan"]["planned_distinct_targets"] == 0
    assert not list((tmp_path / "out").glob("screen__*.parquet"))
