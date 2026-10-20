"""Protocol freezing, split leakage, matched-training smoke and summary statistics (offline)."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from flycircuit.eval import navigation as nv

ROOT = Path(__file__).resolve().parents[2]


def _proto():
    return yaml.safe_load((ROOT / nv.PROTOCOL_REL).read_text())


def test_protocol_declares_primary_endpoint_and_targets():
    p = _proto()
    assert p["primary_endpoint"]["target_threshold"] == 0.68
    assert p["primary_endpoint"]["condition"] == "dropout_channel_episode_25"
    assert p["dropout_protocol_25"] == {**p["dropout_protocol_25"], "kind": "channel",
                                        "per": "episode", "level": 0.25}
    assert p["primary_endpoint"]["comparator_primary"] == "gru_param_matched"
    assert p["training"]["hyperparameter_search_budget"].startswith("0")
    names = {c["name"] for c in p["evaluation_conditions"]}
    assert p["primary_endpoint"]["condition"] in names
    assert {0.0, 0.1, 0.25, 0.5} <= {c.get("dropout", 0.0) for c in p["evaluation_conditions"]}
    assert p["seeds"]["pilot_exploratory"] == [0, 1] and len(p["seeds"]["confirmatory"]) == 5
    assert not set(p["seeds"]["pilot_exploratory"]) & set(p["seeds"]["confirmatory"])
    # targets.yaml agrees with the protocol threshold
    t = yaml.safe_load((ROOT / "configs/targets.yaml").read_text())
    nav = next(x for x in t["targets"] if x["id"] == "navigation_success_25pct_dropout")
    assert nav["threshold"] == p["primary_endpoint"]["target_threshold"]


def _sandbox(tmp_path):
    (tmp_path / "configs/experiments").mkdir(parents=True)
    shutil.copy(ROOT / nv.PROTOCOL_REL, tmp_path / nv.PROTOCOL_REL)
    return tmp_path


def test_freeze_verify_detects_edit_and_accepts_amendment(tmp_path):
    root = _sandbox(tmp_path)
    assert nv.verify_protocol(root)["frozen"] is False
    h = nv.freeze_protocol(root)
    v = nv.verify_protocol(root)
    assert v["frozen"] and v["match"] and v["hash"] == h
    p = nv.load_protocol(root)
    p["episodes_per_condition"] = 5
    (root / nv.PROTOCOL_REL).write_text(yaml.safe_dump(p))
    assert nv.verify_protocol(root)["match"] is False  # silent edit is detected
    p2 = copy.deepcopy(p)
    p2["amendments"] = [{"date": "2026-10-16", "reason": "test", "old_hash": h,
                         "new_hash": nv.protocol_hash(p2)}]
    (root / nv.PROTOCOL_REL).write_text(yaml.safe_dump(p2))
    assert nv.verify_protocol(root)["match"] is True  # recorded amendment accepted


def test_run_benchmark_refuses_unfrozen_protocol(tmp_path):
    root = _sandbox(tmp_path)
    with pytest.raises(RuntimeError, match="not frozen"):
        nv.run_benchmark([0], root=root)


def test_no_layout_leakage_between_splits():
    r = nv.assert_no_layout_leakage(_proto(), n_train=500)
    assert r["shared"] == 0 and r["test"] == 200


def _tiny_proto():
    p = _proto()
    p["training"].update(dagger_betas=[1.0, 0.0], episodes_per_round=8, epochs_per_round=2,
                         dropout_levels=[0.0, 0.25])
    return p


def test_matched_training_is_deterministic_and_finite():
    from flycircuit.models.baselines import GRUBaseline

    p = _tiny_proto()
    losses = []
    for _ in range(2):
        m = GRUBaseline(10, 2, 8, seed=3)
        s = nv.train_student(m, p, seed=3, log=lambda *_: None)
        assert not s["diverged"] and s["dataset_sequences"] == 16
        assert s["train_env_episodes"] == 16
        losses.append(s["loss_by_round"])
    assert losses[0] == losses[1]  # same seed + same schedule => identical training
    assert all(np.isfinite(losses[0]))


def test_training_reduces_imitation_loss():
    from flycircuit.models.baselines import GRUBaseline

    p = _tiny_proto()
    p["training"]["epochs_per_round"] = 12
    m = GRUBaseline(10, 2, 16, seed=0)
    s = nv.train_student(m, p, seed=0, log=lambda *_: None)
    assert s["loss_by_round"][0] < 1.2  # starts near label variance and moves down
    assert s["loss_by_round"][-1] < 0.9


def test_summary_uses_trained_models_as_units_and_labels_exploratory():
    p = _proto()
    names = [c["name"] for c in p["evaluation_conditions"]]

    def mk(cond, seed, v):
        return {"condition": cond, "seed": seed, "training": {"diverged": False},
                "eval": {n: {"success_rate": v} for n in names}}

    models = [mk("connectome_pilot_mb", s, 0.7 + 0.01 * s) for s in range(3)]
    models += [mk("gru_param_matched", s, 0.6) for s in range(3)]
    models += [mk("gru_compute_matched", 1, 0.5)]
    out = nv.summarize(models, p, exploratory=True)
    assert out["primary"]["status"] == "exploratory"
    assert out["primary"]["n_trained_models"] == 3
    assert out["primary"]["measured_mean"] == pytest.approx(0.71)
    c = out["contrasts_exploratory"]["connectome_pilot_mb_minus_gru_param_matched"]
    assert c["n_units"] == 3 and c["estimate"] == pytest.approx(0.11)
    assert out["per_condition"]["connectome_pilot_mb"]["n_models"] == 3
    # confirmatory status is decided against the frozen threshold
    assert nv.summarize(models, p, exploratory=False)["primary"]["status"] == "evaluated/met"
    low = [mk("connectome_pilot_mb", s, 0.5) for s in range(3)]
    assert nv.summarize(low, p, exploratory=False)["primary"]["status"] == "evaluated/missed"
    assert nv.summarize([], p, exploratory=True)["primary"]["status"] == "not run"


def test_diverged_models_are_reported_not_silently_dropped():
    p = _proto()
    names = [c["name"] for c in p["evaluation_conditions"]]
    good = {"condition": "connectome_pilot_mb", "seed": 0, "training": {"diverged": False},
            "eval": {n: {"success_rate": 0.8} for n in names}}
    bad = {"condition": "connectome_pilot_mb", "seed": 1, "training": {"diverged": True},
           "eval": {n: {"success_rate": 0.0} for n in names}}
    out = nv.summarize([good, bad], p, exploratory=True)
    row = out["per_condition"]["connectome_pilot_mb"]
    assert row["n_models"] == 2 and row["n_diverged"] == 1
