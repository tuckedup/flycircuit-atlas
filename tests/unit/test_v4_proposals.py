"""Fixture checks for the PROPOSED v4 modules. Pure Python/NumPy, seconds, no training or physics.
Written but NOT yet executed (awaiting the owner's approval to run any new computation)."""
import json

import numpy as np
import pytest

from flycircuit.bodies import diag_v4
from flycircuit.eval import learning_v4_select as S
from flycircuit.plasticity.diagnostics_v4 import TrialDiag


def test_one_se_prefers_smallest_lr_among_indistinguishable():
    # best = 2e-3 (mean .74, SE .018 -> threshold .722); 1e-3 and 3e-3 (mean .73) are within 1 SE
    table = {1e-3: [0.69, 0.77, 0.71, 0.75], 2e-3: [0.70, 0.78, 0.72, 0.76],
             3e-3: [0.69, 0.77, 0.70, 0.76], 3e-2: [0.40, 0.42, 0.41, 0.43]}
    out = S.one_se_choice(table)
    assert out["best_lr"] == 2e-3
    assert out["chosen_lr"] == 1e-3          # smallest within 1 SE of the best
    assert out["n_indistinguishable"] >= 2
    assert 3e-2 not in out["indistinguishable_lrs"]


def test_one_se_rejects_too_few_seeds():
    with pytest.raises(ValueError):
        S.one_se_choice({1e-3: [0.7, 0.7]}, min_seeds=4)


def test_next_lr_stops_when_neighbours_are_too_close():
    scores = {1e-3: 0.70, 1.1e-3: 0.72, 2e-3: 0.60}
    assert S.next_lr(scores, min_ratio=1.25) is None      # best 1.1e-3 vs neighbour 1e-3: ratio 1.1
    edge = {1e-3: 0.60, 3e-3: 0.70}
    assert S.next_lr(edge) == pytest.approx(9e-3)         # best at upper edge: step outward


def test_composite_score_weights_and_nonfinite():
    assert S.composite_score(0.8, 0.6) == pytest.approx(0.7)
    assert S.composite_score(float("nan"), 0.6) == float("-inf")


def test_trial_diag_sidecar_roundtrip(tmp_path):
    d = TrialDiag(tmp_path / "diag.jsonl")
    for i, gn in enumerate([0.5, 2.0, 0.1, float("inf")]):
        d.log(trial=i, delta=0.1, logp=-0.7, grad_norm=gn, clip_max=1.0, loss=0.07)
    s = TrialDiag.summarize(tmp_path / "diag.jsonl")
    assert s["n"] == 4 and s["nonfinite"] == 1
    assert s["clip_fraction"] == pytest.approx(0.5)       # 2.0 and inf exceed the clip
    json.dumps(s)


def _encdec_bundle(w_in_channel=None):
    """Tiny 'encdec' policy: output depends only on input channel ``w_in_channel`` (or nothing)."""
    w1 = np.zeros((2, 10))
    if w_in_channel is not None:
        w1[0, w_in_channel] = 1.0
        w1[1, w_in_channel] = -1.0
    return {"kind": np.asarray("encdec"), "w1": w1, "b1": np.zeros(2),
            "w2": np.eye(2), "b2": np.zeros(2)}


def test_channel_report_flags_the_channel_the_policy_depends_on(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(3):
        x = rng.normal(size=(20, 10)).astype(np.float32)
        # NumpyPolicy.act = tanh(raw) and this bundle's raw = tanh(+-x5), so the matching teacher is tanh(tanh(x5))
        y = np.tanh(np.tanh(np.stack([x[:, 5], -x[:, 5]], 1))).astype(np.float32)
        np.savez(tmp_path / f"rec_val{i}.npz", x=x, y=y)
    pol = tmp_path / "pol.npz"
    np.savez(pol, **_encdec_bundle(w_in_channel=5))
    # np_policy.NumpyPolicy.load needs the numpy-only module; run the real report
    rep = diag_v4.channel_report(pol, str(tmp_path / "rec_val*.npz"))
    assert rep["mse_no_dropout"] < 1e-6                 # policy reproduces the teacher exactly
    ex = np.array(rep["excess_over_no_dropout"])
    assert int(ex.argmax()) == 5                        # dropping channel 5 hurts; others do not
    assert np.delete(ex, 5).max() < 1e-6
