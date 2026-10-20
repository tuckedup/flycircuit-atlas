import json

import pytest

from flycircuit.tracking.replay import MAX_VALUES, REPLAY_FORMAT, validate_replay, write_replay


def _doc(**kw):
    d = {
        "format": REPLAY_FORMAT, "label": "t", "kind": "structural_forward", "recorded": True,
        "synthetic": False, "structural_untrained": True,
        "provenance": {"run_id": "r", "release": "flywire:783", "circuit": "c", "seed": 1,
                       "sign_policy": "excitatory", "code_note": "x"},
        "dt": 1.0, "time_unit": "step", "n_steps": 2,
        "neurons": ["flywire:783:720575940621084771", "flywire:783:5"],
        "activity": [[0.1, 0.2], [0.3, float("nan")]], "stimulus": [1.0, 0.0], "trials": [],
    }
    d.update(kw)
    return d


def test_roundtrip_keeps_exact_string_ids_and_nulls_nan(tmp_path):
    p = write_replay(tmp_path, _doc())
    d = json.loads(p.read_text())
    assert d["neurons"][0] == "flywire:783:720575940621084771"  # exact, string
    assert d["activity"][1][1] is None  # NaN never written as a number
    assert d["recorded"] is True


def test_rejects_numeric_ids_and_shape_mismatch():
    with pytest.raises(ValueError, match="strings"):
        validate_replay(_doc(neurons=[720575940621084771, "x"]))
    with pytest.raises(ValueError, match="one value per tracked neuron"):
        validate_replay(_doc(activity=[[0.1], [0.2]]))
    with pytest.raises(ValueError, match="n_steps"):
        validate_replay(_doc(n_steps=3))


def test_requires_provenance_and_bounded_size():
    bad = _doc()
    del bad["provenance"]["sign_policy"]
    with pytest.raises(ValueError, match="sign_policy"):
        validate_replay(bad)
    t = MAX_VALUES // 2 + 1
    with pytest.raises(ValueError, match="too large"):
        validate_replay(_doc(n_steps=t, activity=[[0.0, 0.0]] * t, stimulus=[0.0] * t))


def test_gain_trajectory_must_match_edges():
    g = {"edges": [{"pre": "a", "post": "b"}], "trajectory": [[1.0, 2.0]]}
    with pytest.raises(ValueError, match="gain trajectory"):
        validate_replay(_doc(gains=g))
