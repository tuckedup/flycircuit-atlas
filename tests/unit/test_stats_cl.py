import numpy as np
import pytest

from flycircuit.eval import stats

R = np.array([[0.9, 0.5, 0.5],
              [0.8, 0.9, 0.5],
              [0.7, 0.8, 0.9]])


def test_forgetting_excludes_final_task():
    f = stats.forgetting_per_task(R)
    # task0: max(R[0,0],R[1,0]) - R[2,0] = 0.9-0.7 ; task1: R[1,1]-R[2,1] = 0.9-0.8
    assert np.allclose(f, [0.2, 0.1]) and len(f) == 2
    assert stats.average_forgetting(R) == pytest.approx(0.15)


def test_negative_forgetting_policy():
    Rn = np.array([[0.5, 0.5], [0.8, 0.9]])
    assert stats.average_forgetting(Rn, "keep") == pytest.approx(-0.3)
    assert stats.average_forgetting(Rn, "clip") == 0.0
    with pytest.raises(ValueError):
        stats.average_forgetting(Rn, "x")


def test_bwt_and_fwt():
    assert stats.backward_transfer(R) == pytest.approx(np.mean([0.7 - 0.9, 0.8 - 0.9]))
    with pytest.raises(ValueError):
        stats.forward_transfer(R, None)
    assert stats.forward_transfer(R, [0.5, 0.4, 0.4]) == pytest.approx(np.mean([0.5 - 0.4, 0.5 - 0.4]))


def test_forgetting_improvement_definitions():
    r = stats.forgetting_improvement(0.20, 0.10)
    assert r["absolute"] == pytest.approx(0.10) and r["percentage_points"] == pytest.approx(10)
    assert r["relative"] == pytest.approx(0.5) and r["meets_target_abs"]
    z = stats.forgetting_improvement(0.0, 0.05)
    assert z["relative"] is None and not z["meets_target_abs"]


def test_bootstrap_small_n_flagged_and_single_unit_has_no_interval():
    one = stats.bootstrap_ci([0.8])
    assert np.isnan(one["lo"])
    b = stats.bootstrap_ci([0.7, 0.8, 0.75, 0.9, 0.85], n_boot=2000)
    assert b["lo"] <= b["estimate"] <= b["hi"] and "exploratory" in b["note"]


def test_paired_diff_uses_pairs():
    a, b = [0.9, 0.8, 0.85], [0.8, 0.7, 0.75]
    d = stats.paired_bootstrap_diff(a, b, n_boot=500)
    assert d["estimate"] == pytest.approx(0.1) and d["lo"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        stats.paired_bootstrap_diff([1, 2], [1])


def test_nested_bootstrap_respects_structure():
    groups = {"g1": [0.5, 0.52], "g2": [0.9, 0.88], "g3": [0.7, 0.71]}
    r = stats.nested_bootstrap(groups, n_boot=1000)
    assert r["lo"] < r["estimate"] < r["hi"] and r["n_graphs"] == 3
    # between-graph variance must widen the interval relative to pooling seeds as independent
    pooled = stats.bootstrap_ci(np.concatenate(list(groups.values())), n_boot=1000)
    assert (r["hi"] - r["lo"]) > (pooled["hi"] - pooled["lo"])


def test_target_status_vocabulary():
    assert stats.target_status(None, 0.8) == "not run"
    assert stats.target_status(0.9, 0.8) == "exploratory"
    assert stats.target_status(0.9, 0.8, exploratory=False) == "evaluated/met"
    assert stats.target_status(0.7, 0.8, exploratory=False) == "evaluated/missed"
    assert stats.target_status(None, 0.8, blocked=True) == "blocked"
