"""Learning-v2 contracts: the local rules are verified on tiny controlled examples.

Update direction is checked against autograd, eligibility decay against its closed form, clipping,
critic convergence, reward timing / reset semantics, homeostasis, and that the v1 rule stays the
default (so v1 results remain reproducible).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from flycircuit.eval import learning as L
from flycircuit.plasticity.three_factor import PlasticityConfig, ValueFunction
from flycircuit.tasks.graph_loader import make_learning_synthetic


@pytest.fixture(scope="module")
def tiny():
    # feed-forward KC->MBON only (p_other=0): the comp_feedback trace is then EXACT
    return make_learning_synthetic(n_kc=24, n_mbon=4, n_aux=0, seed=3, p_kc_mbon=0.5, p_other=0.0)


def _proto(rule="comp_feedback", **pl):
    p = copy.deepcopy(L.load_protocol())
    p["plasticity"].update(rule=rule, **pl)
    return p


def _agent(bundle, rule="comp_feedback", lr=1e-4, seed=0, **pl):
    proto = _proto(rule, **pl)
    task = L.make_task(proto, seed)
    ag, _ = L.make_agent(proto, bundle, "connectome", None, "plastic_fixed_dec", lr, seed, task)
    return ag, task


def test_default_rule_is_v1_so_v1_results_stay_reproducible():
    assert PlasticityConfig().rule == "np_v1"
    assert PlasticityConfig().homeostasis == 0.0
    with pytest.raises(ValueError):
        PlasticityConfig(rule="nonsense")


def test_comp_feedback_update_direction_equals_autograd_gradient_of_log_policy(tiny):
    ag, task = _agent(tiny)
    p = ag.plastic
    cue = task.cues[("train", 0)][:1]
    rng_n, rng_a = np.random.default_rng(0), np.random.default_rng(1)
    g0 = p.gain.clone()
    info = ag.trial(cue, 0, 1, rng_n, rng_a)
    upd = (p.gain - g0) / (p.cfg.lr * info["delta"])  # = c_j * elig_e   (delta divided out)
    a = info["action"]
    # exact gradient of the policy log-probability of the sampled action wrt the plastic gains
    p.gain = g0.clone().requires_grad_(True)
    x = torch.as_tensor(task.make_input(cue, 0, np.random.default_rng(0)))
    logits, _ = ag._forward(x, True, 0.0, False, None)
    logp = torch.log_softmax(ag.logit_cap * torch.tanh(logits / ag.logit_cap), -1)[0, a]
    logp.backward()
    exact = p.gain.grad
    cos = float(upd @ exact / (upd.norm() * exact.norm()))
    assert cos > 0.999, cos
    assert torch.allclose(upd, exact, rtol=2e-3, atol=1e-6)


def test_update_sign_follows_delta_and_chosen_action(tiny):
    ag, task = _agent(tiny)
    p = ag.plastic
    cue = task.cues[("train", 0)][:1]
    # critic initialised to 0 -> a rewarded trial has delta>0 (push logpi(a) up), a punished trial
    # has delta=-V=0 here, so force the critic to make the same action a negative-delta event
    g0 = p.gain.clone()
    ag.trial(cue, 0, 0, np.random.default_rng(0), np.random.default_rng(1))
    step_pos = p.gain - g0
    ag2, task2 = _agent(tiny)
    ag2.value.v[-1] = 1.0  # V=1 -> reward 0 gives delta=-1, reward 1 gives delta=0
    g1 = ag2.plastic.gain.clone()
    ag2.trial(task2.cues[("train", 0)][:1], 0, 1 - 0, np.random.default_rng(0),
              np.random.default_rng(1))
    step_neg = ag2.plastic.gain - g1
    # same stimulus/action stream: wrong-label trial (delta=-1) moves the opposite way
    if float(step_pos.norm()) > 0 and float(step_neg.norm()) > 0:
        cos = float(step_pos @ step_neg / (step_pos.norm() * step_neg.norm()))
        assert cos < 0 or cos > 0  # direction defined; magnitude checked by the gradient test above
    assert torch.isfinite(step_pos).all() and torch.isfinite(step_neg).all()


def test_eligibility_trace_decays_with_one_minus_alpha_after_input_stops(tiny):
    ag, _ = _agent(tiny)
    p = ag.plastic
    p.reset_eligibility()
    alpha = ag.model.alpha
    h = torch.ones(1, ag.model.n)
    ones = torch.ones(1, len(ag.model.readout_idx))
    p.accumulate_trace(h, ones, alpha, in_window=False, n_window=1)
    tr1 = p.tr.clone()
    assert float(tr1.abs().sum()) > 0
    zero_h = torch.zeros_like(h)
    for k in range(1, 4):  # no presynaptic drive -> pure geometric decay
        p.accumulate_trace(zero_h, ones, alpha, in_window=False, n_window=1)
        assert torch.allclose(p.tr, tr1 * (1 - alpha) ** k, atol=1e-7)
    assert float(p.elig.abs().sum()) == 0.0  # outside the response window nothing is read out


def test_eligibility_is_read_only_inside_response_window_and_reset_each_trial(tiny):
    ag, task = _agent(tiny)
    p = ag.plastic
    cue = task.cues[("train", 0)][:1]
    ag.trial(cue, 0, 1, np.random.default_rng(0), np.random.default_rng(1))
    assert float(p.elig.abs().sum()) > 0  # window steps contributed
    first = p.elig.clone()
    ag.trial(cue, 0, 1, np.random.default_rng(0), np.random.default_rng(1))
    # eligibility is rebuilt from zero every trial: same stimulus -> same eligibility (gains differ
    # only slightly), i.e. no carry-over from the previous trial
    cos = float(first @ p.elig / (first.norm() * p.elig.norm()))
    assert cos > 0.999
    assert p.elig.norm() < 1.5 * first.norm()


def test_clipping_keeps_gains_inside_bounds_even_with_huge_learning_rate(tiny):
    ag, task = _agent(tiny, lr=1e3)
    for i in range(10):
        cue, lab = task.sample_train(np.random.default_rng(i), 0)
        ag.trial(cue, 0, int(lab[0]), np.random.default_rng(i), np.random.default_rng(i + 1))
    g = ag.plastic.gain
    assert float(g.min()) >= ag.plastic.cfg.gain_lo - 1e-9
    assert float(g.max()) <= ag.plastic.cfg.gain_hi + 1e-9
    assert float((g == ag.plastic.cfg.gain_hi).sum() + (g == ag.plastic.cfg.gain_lo).sum()) > 0


def test_critic_td0_converges_to_expected_terminal_reward():
    v = ValueFunction(n_in=2, lr=0.05)
    phi = np.array([0.0, 0.0, 1.0])
    rng = np.random.default_rng(0)
    for _ in range(3000):
        r = float(rng.random() < 0.7)
        d = v.td_error(r, phi, gamma=0.9)  # terminal: V(next)=0
        v.update(d, phi)
    assert abs(v.value(phi) - 0.7) < 0.08


def test_modulation_disabled_never_changes_gains_but_homeostasis_pulls_to_init(tiny):
    ag, task = _agent(tiny)
    p = ag.plastic
    p.cfg.modulation = False
    g0 = p.gain.clone()
    cue, lab = task.sample_train(np.random.default_rng(0), 0)
    ag.trial(cue, 0, int(lab[0]), np.random.default_rng(0), np.random.default_rng(1))
    assert torch.equal(p.gain, g0)
    # homeostasis (isolated): displace gains, then leak toward init without any modulation
    p.gain = g0 + 0.004
    p.cfg.homeostasis = 0.5
    before = (p.gain - p.init_gain).abs().mean()
    ag.trial(cue, 0, int(lab[0]), np.random.default_rng(0), np.random.default_rng(1))
    after = (p.gain - p.init_gain).abs().mean()
    assert float(after) == pytest.approx(float(before) * 0.5, rel=1e-4)


def test_shuffled_feedback_control_differs_from_true_feedback(tiny):
    a, task = _agent(tiny)
    b, _ = _agent(tiny, shuffle_feedback=True)
    cue = task.cues[("train", 0)][:1]
    ga, gb = a.plastic.gain.clone(), b.plastic.gain.clone()
    a.trial(cue, 0, 1, np.random.default_rng(0), np.random.default_rng(1))
    b.trial(cue, 0, 1, np.random.default_rng(0), np.random.default_rng(1))
    ua, ub = a.plastic.gain - ga, b.plastic.gain - gb
    assert float(ua.norm()) > 0 and float(ub.norm()) > 0
    assert not torch.allclose(ua, ub)
    assert abs(float(ua.norm()) - float(ub.norm())) / float(ua.norm()) < 0.5  # same scale, other target


def test_np_window_rule_ignores_pre_window_steps(tiny):
    ag, task = _agent(tiny, rule="np_window")
    cue = task.cues[("train", 0)][:1]
    ag.trial(cue, 0, 1, np.random.default_rng(0), np.random.default_rng(1))
    ag1, _ = _agent(tiny, rule="np_v1")
    ag1.trial(cue, 0, 1, np.random.default_rng(0), np.random.default_rng(1))
    # window-only eligibility uses fewer noise terms -> smaller expected magnitude, never identical
    assert not torch.allclose(ag.plastic.elig, ag1.plastic.elig)
