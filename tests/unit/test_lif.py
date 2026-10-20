"""Independent LIF engine: exact integration, delay, refractory conditional-write, lesion, events."""

import numpy as np
import pytest
import scipy.sparse as sp

from flycircuit.models.lif import (
    LIFParams,
    poisson_events,
    silence_outgoing,
    simulate,
    weights_mV,
)

P = LIFParams()


def _W(edges, n):
    pre = [e[0] for e in edges]
    post = [e[1] for e in edges]
    c = [e[2] for e in edges]
    return weights_mV(np.array(c, float), np.array(pre), np.array(post), n, P)


def test_params_parse_from_published_source(tmp_path):
    src = tmp_path / "model.py"
    src.write_text(
        "default_params = {\n 'v_0': -52 * mV, 'v_rst': -52 * mV, 'v_th': -45 * mV,\n"
        " 't_mbr': 20 * ms, 'tau': 5 * ms, 't_rfc': 2.2 * ms, 't_dly': 1.8*ms,\n"
        " 'w_syn': .275 * mV, 'r_poi': 150*Hz, 'f_poi': 250,\n 'eqs': 'x',\n}\n"
    )
    p = LIFParams.from_shiu_source(src)
    assert p == LIFParams()  # package defaults == constants parsed from source text
    assert p.w_ext == pytest.approx(68.75)
    assert (p.n_delay, p.n_refractory) == (18, 22)


def test_matches_checked_out_reference_constants():
    from flycircuit.config import project_root

    src = project_root() / "external" / "Drosophila_brain_model" / "model.py"
    if not src.exists():
        pytest.skip("reference not cloned")
    assert LIFParams.from_shiu_source(src) == LIFParams()


def test_exact_exponential_integration_matches_closed_form():
    # 0 -> 1 with a sub-threshold weight; neuron 1 sees g0 = w at record index s+D+1.
    W = _W([(0, 1, 40)], 2)
    r = simulate(W, 400, P, np.array([5]), np.array([0]), no_refractory=np.array([0]),
                 record=np.array([1]))
    spike_step = int(r.spike_steps[0])
    assert spike_step == 6  # event at 5 -> threshold crossing detected at the next step
    arrive = spike_step + P.n_delay
    g0 = 40 * P.w_syn
    assert r.g_trace[arrive, 0] == 0 and r.g_trace[arrive + 1, 0] == pytest.approx(g0)
    for m in (1, 5, 40, 120):
        t = m * P.dt
        A = g0 * P.tau / (P.tau - P.t_mbr)
        u = (0 - A) * np.exp(-t / P.t_mbr) + A * np.exp(-t / P.tau)
        assert r.v_trace[arrive + 1 + m, 0] - P.v_0 == pytest.approx(u, abs=1e-9)
        assert r.g_trace[arrive + 1 + m, 0] == pytest.approx(g0 * np.exp(-t / P.tau), abs=1e-9)


def test_inhibition_hyperpolarises_below_rest():
    W = _W([(0, 1, -40)], 2)
    r = simulate(W, 300, P, np.array([5]), np.array([0]), no_refractory=np.array([0]),
                 record=np.array([1]))
    assert r.v_trace[:, 0].min() < P.v_0 - 0.1 and len(r.spike_steps) == 1


def test_synaptic_delay_is_round_t_dly_over_dt_and_follows_dt():
    W = _W([(0, 1, 40)], 2)
    for dt in (0.1, 0.05):
        p = P.with_(dt=dt)
        r = simulate(W, 600, p, np.array([5]), np.array([0]), no_refractory=np.array([0]),
                     record=np.array([1]))
        s = int(r.spike_steps[0])
        first_g = int(np.flatnonzero(r.g_trace[:, 0] > 0)[0])
        assert first_g == s + p.n_delay + 1


def test_refractory_neuron_discards_synaptic_and_external_writes():
    # pacemaker 1 (no refractory) -> 0 strongly; 0 must not accumulate g while refractory.
    W = _W([(1, 0, 150)], 2)
    ev = np.arange(100, 400, 4)
    r = simulate(W, 600, P, ev, np.full(len(ev), 1), no_refractory=np.array([1]),
                 record=np.array([0]))
    s0 = r.spike_steps[r.spike_neurons == 0]
    assert len(s0) >= 2
    first = int(s0[0])
    window = slice(first + 1, first + P.n_refractory + 1)
    assert np.all(r.g_trace[window, 0] == 0.0)  # inputs arriving while refractory are dropped
    assert np.all(r.v_trace[window, 0] == P.v_rst)
    assert s0[1] - s0[0] >= P.n_refractory


def test_deterministic_given_events():
    W = _W([(0, 1, 300)], 2)
    ev = np.array([5, 5 + 18 + 1])
    a = simulate(W, 100, P, ev, np.array([0, 1]), no_refractory=np.array([0, 1]))
    b = simulate(W, 100, P, ev, np.array([0, 1]), no_refractory=np.array([0, 1]))
    assert len(a.spike_steps) > 0 and np.array_equal(a.spike_steps, b.spike_steps)
    assert np.array_equal(a.spike_neurons, b.spike_neurons)


def test_silence_zeroes_all_outgoing_synapses_only():
    W = _W([(0, 1, 5), (0, 2, 5), (1, 2, 5), (2, 0, 5)], 3)
    S = silence_outgoing(W, [0]).toarray()
    assert not S[:, 0].any() and S[2, 1] > 0 and S[0, 2] > 0
    r = simulate(silence_outgoing(_W([(0, 1, 250)], 2), [0]), 200, P, np.array([5]),
                 np.array([0]), no_refractory=np.array([0]))
    assert set(r.spike_neurons.tolist()) == {0}


def test_direction_pre_to_post_orientation():
    W = _W([(0, 1, 250)], 2)  # W[post=1, pre=0]
    assert W[1, 0] > 0 and W[0, 1] == 0
    r = simulate(W, 300, P, np.array([5]), np.array([1]), no_refractory=np.array([1]))
    assert set(r.spike_neurons.tolist()) == {1}  # driving the POST neuron never drives the pre


def test_poisson_event_streams_are_per_target_and_reproducible():
    s1, t1 = poisson_events(np.array([3, 7]), 5000, 150.0, 0.1, 11)
    s2, t2 = poisson_events(np.array([7]), 5000, 150.0, 0.1, 11)
    assert np.array_equal(s1[t1 == 7], s2)  # stream of target 7 independent of target 3
    s3, _ = poisson_events(np.array([7]), 5000, 150.0, 0.1, 12)
    assert not np.array_equal(s2, s3)
    assert abs(len(s2) / (5000 * 0.1e-3) - 150) < 40  # rate ~150 Hz


def test_background_drive_changes_activity_and_respects_refractory():
    W = sp.csc_matrix((3, 3))
    base = simulate(W, 2000, P)
    bg = simulate(W, 2000, P, background=(np.array([1, 1, 2]), np.array([400.0, 400.0, -400.0]), 200.0),
                  background_seed=1)
    assert len(base.spike_steps) == 0 and len(bg.spike_steps) > 0
    assert 2 not in set(bg.spike_neurons.tolist())  # inhibitory-only target never spikes
