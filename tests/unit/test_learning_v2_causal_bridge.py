"""Causal export must preserve the learned policy and reject mismatched provenance."""
import copy
from pathlib import Path

import pytest

from flycircuit.eval import learning_v2 as V
from flycircuit.interventions import trained as T
from flycircuit.interventions.learning_v2_bridge import export_instance
from flycircuit.tasks.graph_loader import make_learning_synthetic


@pytest.fixture
def source(tmp_path, monkeypatch):
    proto = copy.deepcopy(V.load_protocol())
    proto["task"].update(n_train=40, n_val=16, n_test=16)
    proto["training"]["trials_per_phase"] = 40
    proto["circuit"]["name"] = "bridge_fixture"
    base = make_learning_synthetic(n_kc=24, n_mbon=4, n_aux=4)
    task = V.L.make_task(proto, 7)
    agent, _ = V.make_agent_v2(proto, base, {}, "bptt", "connectome", 0.01, 7, task)
    V.L.train_phase(agent, task, 0, 40, V.L._rngs(7, 0))
    monkeypatch.setattr(V, "ckpt_dir", lambda stage: tmp_path)
    monkeypatch.setattr(T, "build_bundle", lambda *args: base)
    monkeypatch.setattr(T, "_BUNDLES", {})
    meta = {"arm": "bptt", "family": "connectome", "mode": "std", "seed": 7, "task_seed": 7,
            "lr": 0.01, "decoder_lr": None, "protocol_hash": V.protocol_hash(proto), "tag": "post_acq"}
    path = Path(V.save_checkpoint(agent, meta, "fixture", "post_acq"))
    record = {"arm": "bptt", "family": "connectome", "mode": "std", "seed": 7, "lr": 0.01,
              "status": "ok", "protocol_hash": V.protocol_hash(proto), "elapsed_s": 1.0,
              "n_params_trainable": sum(p.numel() for p in agent.params),
              "edge_hash": base.graph.edge_hash(),
              "association": {"ckpt_post_acquisition": str(path),
                              "val_acc_acq": agent.evaluate("val", 0),
                              "test_acc_acq": agent.evaluate("test", 0)}}
    return record, proto, base, tmp_path


def test_export_roundtrip_preserves_logits_and_reuses_identical_source(source):
    record, proto, base, root = source
    destination = root / "export/instances/v2_default/bptt_s7.pt"
    result = export_instance(record, proto, base, destination, root)
    assert result["forward_max_abs_error"] < 1e-5
    assert not result["reused"]
    assert export_instance(record, proto, base, destination, root)["reused"]
    instance = T.load_instance(destination)
    assert "learning-v2" in instance.meta["origin"]


@pytest.mark.parametrize("field,value", [("protocol_hash", "wrong"), ("seed", 8), ("family", "gru"), ("edge_hash", "wrong")])
def test_export_rejects_wrong_protocol_identity_or_model(source, field, value):
    record, proto, base, root = source
    record[field] = value
    with pytest.raises(ValueError):
        export_instance(record, proto, base, root / "bad.pt", root)
    assert not (root / "bad.pt").exists()


@pytest.mark.parametrize("source", [None, "learning-v2 acquisition follow-up"])
def test_screen_artifact_records_its_source_study(tmp_path, source):
    import json

    import pyarrow.parquet as pq

    cfg = {} if source is None else {"instance_study": source, "instance_protocol_hash": "fixture"}
    path = tmp_path / "screen.parquet"
    T._write(path, [{"example": 1}], cfg, {})
    metadata = json.loads(pq.read_schema(path).metadata[b"flycircuit_provenance"])
    if source:
        assert metadata["instance_study"] == source
        assert metadata["instance_protocol_hash"] == "fixture"
        assert "v1 acquisition" not in metadata["label"]
    else:
        assert "v1 acquisition" in metadata["instance_study"]
