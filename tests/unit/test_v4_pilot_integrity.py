"""Focused integrity tests for the validation-only v4 learning pilot."""
from __future__ import annotations

import json

import pytest

from flycircuit.eval import learning as L
from flycircuit.eval import learning_v3 as V3
from flycircuit.eval import learning_v4 as P
from flycircuit.eval import learning_v4_select as S
from flycircuit.tasks.graph_loader import make_learning_synthetic


def _record(coord, manifest="manifest", *, status="ok", acq=0.7, cont=0.6, forget=0.1):
    out = {**coord, "pilot": "learning_v4_validation_only", "splits": ["val"],
           "pilot_manifest_sha256": manifest, "status": status, "mode": "std", "stage": None,
           "lr": coord["core_lr"]}
    if status == "ok":
        out.update(association={"val_acc_acq": acq},
                   continual={"final_mean_acc_val": cont,
                              "avg_forgetting_keep_val": forget})
    return out


def test_prepare_manifest_archives_unmanifested_rows_byte_exactly(tmp_path, monkeypatch):
    out = tmp_path / "pilot"
    out.mkdir()
    old = b'{"legacy":true}\n{"legacy":false}\n'
    (out / "search_records.jsonl").write_bytes(old)
    monkeypatch.setattr(P, "OUT", out)
    monkeypatch.setattr(P, "MANIFEST", out / "pilot_manifest.json")
    monkeypatch.setattr(P, "LEGACY", out / "legacy_unmanifested")
    monkeypatch.setattr(P, "_manifest_identity", lambda: {"fixed": "identity"})

    doc = P.prepare_manifest()
    assert doc["identity"] == {"fixed": "identity"}
    assert not (out / "search_records.jsonl").exists()
    archived = list((out / "legacy_unmanifested").glob("search_records.*.jsonl"))
    assert len(archived) == 1 and archived[0].read_bytes() == old
    journal = json.loads((out / "legacy_unmanifested/migration.json").read_text())
    assert journal["files"][0]["sha256"] == P._sha_bytes(old)
    assert P.prepare_manifest() == doc


def test_jsonl_reader_fails_closed_on_truncation_and_malformed_rows(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"a": 1}')
    with pytest.raises(ValueError, match="truncated"):
        P.read(path)
    path.write_bytes(b'{bad}\n')
    with pytest.raises(ValueError, match="malformed"):
        P.read(path)


def test_record_validator_reports_partial_and_rejects_failed_duplicate_unknown_nonfinite():
    a = P._coord("connectome", 1e-3, 2e-3, 501)
    b = P._coord("connectome", 1e-3, 2e-3, 502)
    one = _record(a)
    status = P.validate_records([one], [a, b], "search", "manifest")
    assert status == {"phase": "search", "expected": 2, "records": 1, "ok": 1,
                      "failed": 0, "missing": 1, "complete": False,
                      "missing_keys": [P._key(b)]}

    failed = _record(b, status="failed: RuntimeError('boom')")
    status = P.validate_records([one, failed], [a, b], "search", "manifest")
    assert status["failed"] == 1 and not status["complete"]
    with pytest.raises(ValueError, match="duplicate"):
        P.validate_records([one, one], [a, b], "search", "manifest")
    with pytest.raises(ValueError, match="unknown"):
        P.validate_records([_record(P._coord("gru", 1e-3, 2e-3, 501))], [a],
                           "search", "manifest")
    with pytest.raises(ValueError, match="nonfinite"):
        P.validate_records([_record(a, acq=float("nan"))], [a], "search", "manifest")
    wrong_split = _record(a)
    wrong_split["splits"] = ["val", "test"]
    with pytest.raises(ValueError, match="split identity"):
        P.validate_records([wrong_split], [a], "search", "manifest")


def test_two_group_one_se_uses_joint_simplicity_not_core_first():
    # Lexicographic core-first would choose (1/9, 9); the prospective rule minimizes
    # the largest normalized rate first and therefore chooses the balanced setting.
    table = {
        (1 / 9, 9.0): [0.70, 0.72, 0.68, 0.70],
        (1.0, 1.0): [0.70, 0.72, 0.68, 0.70],
        (3.0, 1 / 3): [0.70, 0.72, 0.68, 0.70],
    }
    out = S.one_se_choice_2d(table, base_lr=1.0)
    assert out["chosen"] == (1.0, 1.0)
    assert S.two_group_simplicity((1.0, 1.0), 1.0) < S.two_group_simplicity((1 / 9, 9.0), 1.0)


def test_forgetting_target_requires_positive_gru_minus_connectome_improvement():
    assert P.forgetting_improvement_status(0.07) is True
    assert P.forgetting_improvement_status(0.069) is False
    assert P.forgetting_improvement_status(-0.08) is False
    assert P.forgetting_improvement_status(None) is None


@pytest.mark.parametrize("family", P.ARMS)
def test_v4_optimizer_has_exact_core_and_decoder_groups_for_each_family(family):
    proto = V3.load_protocol()
    base = make_learning_synthetic(n_kc=24, n_mbon=4, n_aux=4, seed=0)
    task = L.make_task(proto, 501)
    core_lr, decoder_lr = 1e-4, 3e-4
    P.CUR["core"], P.CUR["dec"] = core_lr, decoder_lr
    agent, meta = P.make_agent_v4(proto, base, {}, "bptt", family, core_lr, 501, task,
                                  decoder_lr)
    core, dec = P._split(agent)
    assert core and dec
    assert {id(x) for x in core}.isdisjoint({id(x) for x in dec})
    assert {id(x) for x in core + dec} == {id(x) for x in agent.params}
    assert [g["lr"] for g in agent.opt.param_groups] == [core_lr, decoder_lr]
    assert meta["param_groups"] == {
        "core_n": sum(x.numel() for x in core),
        "decoder_n": sum(x.numel() for x in dec),
        "core_lr": core_lr,
        "decoder_lr": decoder_lr,
    }
