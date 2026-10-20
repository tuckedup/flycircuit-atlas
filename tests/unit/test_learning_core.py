"""Contracts for the learning benchmark: task leakage, plasticity maths, resets, interfaces."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from flycircuit.eval import learning as L
from flycircuit.eval import stats as S
from flycircuit.graph.controls import directed_double_edge_swap
from flycircuit.plasticity.agent import LEARNERS
from flycircuit.plasticity.three_factor import PlasticityConfig, PlasticSynapses, ValueFunction
from flycircuit.tasks.association import ContingencyTask, TaskSpec
from flycircuit.tasks.graph_loader import _sign_from_annotations, make_learning_synthetic


@pytest.fixture(scope="module")
def proto():
    p = L.load_protocol()
    p = copy.deepcopy(p)
    p["training"].update(trials_per_phase=25, reversal_trials=25, perseveration_window=10,
                         acquisition_window=5)
    return p


@pytest.fixture(scope="module")
def bundle():
    return make_learning_synthetic(n_kc=40, n_mbon=6, n_aux=8, seed=0)


# ---- task ----------------------------------------------------------------------------------
def test_splits_disjoint_balanced_and_reversal_flips():
    t = ContingencyTask(TaskSpec(), seed=3)
    keys = {}
    for (split, ctx), x in t.cues.items():
        for row in x:
            k = row.astype(np.uint8).tobytes()
            assert k not in keys, f"cue shared by {keys[k]} and {split}/{ctx}"
            keys[k] = f"{split}/{ctx}"
        assert t.labels[(split, ctx)].sum() == len(x) // 2  # class balanced
        # labels follow the context teacher
        assert (t._label(x, ctx) == t.labels[(split, ctx)]).all()
    y = t.labels_for("test", 0)
    assert (t.labels_for("test", 0, reversed_=True) == 1 - y).all()


def test_same_seed_same_task_different_seed_different():
    a, b, c = (ContingencyTask(TaskSpec(), seed=s) for s in (1, 1, 2))
    assert np.array_equal(a.teachers, b.teachers)
    assert np.array_equal(a.cues[("test", 1)], b.cues[("test", 1)])
    assert not np.array_equal(a.teachers, c.teachers)


def test_trial_input_context_observable_and_cue_timing():
    t = ContingencyTask(TaskSpec(), seed=0)
    cue = t.cues[("train", 0)][:2]
    x = t.make_input(cue, ctx=2, noise_rng=None)
    s = t.spec
    assert x.shape == (s.T, 2, s.n_in)
    assert (x[: s.cue_steps, :, : s.n_cue] == cue).all() and (x[s.cue_steps :, :, : s.n_cue] == 0).all()
    assert (x[:, :, s.n_cue + 2] == 1).all() and (x[:, :, s.n_cue : s.n_cue + 2] == 0).all()


# ---- plasticity maths ----------------------------------------------------------------------
def _tiny_synapses(cfg):
    pre = torch.tensor([0, 1, 2, 0])
    post = torch.tensor([3, 3, 4, 4])
    readout = torch.tensor([3, 4])
    mask = np.array([True, True, False, True])  # edge 2 is NOT plastic
    return PlasticSynapses(mask, pre, post, readout, init_gain=0.1, cfg=cfg)


def test_eligibility_and_gain_update_match_hand_calculation():
    cfg = PlasticityConfig(lr=0.5, decay=0.9, gain_lo=0.0, gain_hi=0.3)
    ps = _tiny_synapses(cfg)
    assert ps.n_plastic == 3 and ps.idx.tolist() == [0, 1, 3]
    h = torch.tensor([[2.0, 3.0, 5.0, 0.0, 0.0]])
    eps = torch.tensor([[1.0, -1.0]])  # readout order: node 3, node 4
    ps.accumulate(h, eps)
    # edges (pre,post): (0->3), (1->3), (0->4): elig = h[pre]*eps[post]
    assert np.allclose(ps.elig.numpy(), [2.0 * 1.0, 3.0 * 1.0, 2.0 * -1.0])
    ps.accumulate(h, eps)
    assert np.allclose(ps.elig.numpy(), [0.9 * 2 + 2, 0.9 * 3 + 3, 0.9 * -2 - 2])
    ps.apply(0.1)  # gain + 0.5*0.1*elig, clipped to [0, 0.3]
    expect = np.clip(0.1 + 0.05 * np.array([3.8, 5.7, -3.8]), 0.0, 0.3)
    assert np.allclose(ps.gain.numpy(), expect)
    assert ps.gain[1] <= 0.3 and ps.gain[2] >= 0.0


def test_modulation_disabled_leaves_gains_and_override_touches_only_plastic_edges():
    ps = _tiny_synapses(PlasticityConfig(lr=1.0, modulation=False))
    ps.accumulate(torch.ones(1, 5), torch.ones(1, 2))
    ps.apply(5.0)
    assert (ps.gain == 0.1).all()
    w = torch.tensor([9.0, 9.0, 9.0, 9.0])
    sign = torch.ones(5)
    prior = torch.tensor([2.0, 2.0, 2.0, 2.0])
    out = ps.override(w, sign, prior)
    assert out[2] == 9.0 and torch.allclose(out[[0, 1, 3]], torch.tensor([0.2, 0.2, 0.2]))
    assert not ps.gain.requires_grad  # updates are outside autograd


def test_td_error_terminal_and_bootstrapped():
    v = ValueFunction(3)
    v.v[:] = [1.0, 0.0, 0.0, 0.5]
    phi = np.array([1.0, 0.0, 0.0, 1.0])
    assert v.value(phi) == pytest.approx(1.5)
    assert v.td_error(1.0, phi, gamma=0.9) == pytest.approx(1.0 - 1.5)  # terminal: V(next)=0
    assert v.td_error(1.0, phi, gamma=0.9, next_phi=phi) == pytest.approx(1.0 + 0.9 * 1.5 - 1.5)
    v.update(0.5, phi)
    assert v.v[0] == pytest.approx(1.0 + 0.05 * 0.5 * 1.0)


# ---- agent interfaces and resets -----------------------------------------------------------
def _agent(proto, bundle, learner, lr=1e-4, carry=False, seed=0):
    task = L.make_task(proto, seed)
    return L.make_agent(proto, bundle, "connectome", None, learner, lr, seed, task)[0], task


def test_trainable_sets_per_learner(proto, bundle):
    for learner in LEARNERS:
        ag, _ = _agent(proto, bundle, learner)
        m = ag.model
        trainable = {n for n, p in m.named_parameters() if p.requires_grad}
        if learner == "bptt":
            assert trainable == {"raw_gain", "decoder"}
        elif learner in ("decoder_only", "plastic_plus_dec"):
            assert trainable == {"decoder"}
        else:
            assert trainable == set()
        if learner.startswith("plastic"):
            assert ag.plastic is not None and ag.plastic.n_plastic == int(bundle.plastic_edge_mask.sum())
            assert not ag.plastic.gain.requires_grad


def test_modulation_disabled_control_never_changes_gains(proto, bundle):
    ag, task = _agent(proto, bundle, "plastic_mod_off", lr=1e-2)
    g0 = ag.plastic.gain.clone()
    rngs = L._rngs(0, 0)
    L.train_phase(ag, task, 0, 15, rngs)
    assert torch.equal(ag.plastic.gain, g0)


def test_plasticity_learner_changes_gains_but_not_decoder_when_fixed(proto, bundle):
    ag, task = _agent(proto, bundle, "plastic_fixed_dec", lr=1e-2)
    g0, d0 = ag.plastic.gain.clone(), ag.model.decoder.detach().clone()
    L.train_phase(ag, task, 0, 25, L._rngs(0, 0))
    assert not torch.equal(ag.plastic.gain, g0)
    assert torch.equal(ag.model.decoder.detach(), d0)
    assert (ag.plastic.gain >= 0).all() and (ag.plastic.gain <= ag.plastic.cfg.gain_hi + 1e-9).all()


def test_reset_kinds_are_separable(proto, bundle):
    ag, task = _agent(proto, bundle, "plastic_plus_dec", lr=1e-2)
    ag.carry_state = True
    L.train_phase(ag, task, 0, 25, L._rngs(0, 0))
    dec_init = ag._init_snapshot["decoder"].clone()
    gain_init = torch.full_like(ag.plastic.gain, ag.model.spec.init_gain)
    s = ag.snapshot()
    assert s["state"] is not None and (s["state"]["h"] != 0).any()
    assert not torch.equal(s["decoder"][0], dec_init) and not torch.equal(s["plastic_gain"], gain_init)

    ag.reset("state")
    a = ag.snapshot()
    assert a["state"] is None
    assert torch.equal(a["decoder"][0], s["decoder"][0]) and torch.equal(a["plastic_gain"], s["plastic_gain"])

    ag.reset("gain")
    b = ag.snapshot()
    assert torch.equal(b["plastic_gain"], gain_init) and (b["elig"] == 0).all()
    assert torch.equal(b["decoder"][0], s["decoder"][0])  # decoder untouched by gain reset

    ag.reset("decoder")
    c = ag.snapshot()
    assert torch.equal(c["decoder"][0], dec_init)

    # full: restore everything incl. critic
    ag.carry_state = True
    L.train_phase(ag, task, 0, 10, L._rngs(0, 1))
    assert np.abs(ag.value.v).sum() > 0
    ag.reset("full")
    d = ag.snapshot()
    assert d["state"] is None and torch.equal(d["decoder"][0], dec_init)
    assert torch.equal(d["plastic_gain"], gain_init) and np.abs(ag.value.v).sum() == 0
    with pytest.raises(ValueError):
        ag.reset("everything")


def test_evaluation_does_not_learn_or_use_test_for_training(proto, bundle):
    ag, task = _agent(proto, bundle, "bptt", lr=1e-3)
    before = [p.detach().clone() for p in ag.params]
    a = ag.evaluate("test", 0)
    b = ag.evaluate("test", 0)
    assert a == b  # deterministic, no noise
    assert all(torch.equal(x, p.detach()) for x, p in zip(before, ag.params, strict=True))


def test_paired_streams_same_stimuli_across_learners(proto, bundle):
    """Cue/noise streams depend only on (seed, stream), not on the learner's actions."""
    seen = []
    for learner in ("bptt", "plastic_fixed_dec"):
        ag, task = _agent(proto, bundle, learner, lr=1e-4, seed=5)
        order, noise, act = L._rngs(5, 0)
        cues = []
        for _ in range(12):
            cue, lab = task.sample_train(order, 0)
            ag.trial(cue, 0, int(lab[0]), noise, act)
            cues.append(cue.copy())
        seen.append(np.concatenate(cues))
    assert np.array_equal(seen[0], seen[1])


# ---- protocol outputs ----------------------------------------------------------------------
def test_run_one_records_matrix_and_forgetting_definition(proto, bundle):
    r = L.run_one(proto, bundle, None, "connectome", "decoder_only", 3e-3, 0)
    assert r["status"] == "ok"
    R = np.array(r["continual"]["R_test"])
    assert R.shape == (3, 3)
    assert r["continual"]["avg_forgetting_keep_test"] == pytest.approx(S.average_forgetting(R, "keep"))
    assert r["continual"]["bwt_test"] == pytest.approx(S.backward_transfer(R))
    assert "not computed" in r["continual"]["forward_transfer"]
    a = r["association"]
    assert 0 <= a["test_acc_acq"] <= 1 and a["acquisition_censored"] in (True, False)
    # search-style call must not evaluate the test split
    rv = L.run_one(proto, bundle, None, "connectome", "decoder_only", 3e-3, 0, splits=("val",),
                   protocols=("continual",))
    assert "R_test" not in rv["continual"] and "R_val" in rv["continual"]


def test_dense_baselines_use_same_interface(proto, bundle):
    for fam, learner in (("gru", "bptt"), ("encdec", "decoder_only")):
        r = L.run_one(proto, bundle, None, fam, learner, 3e-3, 0, protocols=("continual",))
        assert r["status"] == "ok", r["status"]
    with pytest.raises(ValueError):
        L.make_agent(proto, bundle, "gru", None, "plastic_fixed_dec", 1e-4, 0, L.make_task(proto, 0))


# ---- circuit interface ---------------------------------------------------------------------
def test_plastic_edge_count_is_preserved_by_degree_preserving_null(bundle):
    null = directed_double_edge_swap(bundle.graph, seed=1)
    nb = bundle.with_graph(null.graph, "null")
    assert int(nb.plastic_edge_mask.sum()) == int(bundle.plastic_edge_mask.sum())


def test_sign_policy_prefers_known_nt_over_artifact_prediction():
    known = ["acetylcholine; sNPF", "gaba", None, "gaba-negative", None, None]
    top = ["dopamine", "dopamine", "dopamine", "acetylcholine", "gaba", "acetylcholine"]
    conf = [0.9, 0.9, 0.9, 0.9, 0.9, 0.2]
    s, notes = _sign_from_annotations(known, top, conf)
    assert s.tolist() == [1, -1, 0, 0, -1, 0]  # KC-like dopamine prediction ignored when known_nt
    assert notes["sign_source_counts"]["known"] == 2
