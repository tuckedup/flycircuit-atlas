"""Streaming LIF must reproduce the validated reference engine's spikes (synthetic network)."""

import numpy as np
import pytest
import scipy.sparse as sp

from flycircuit.models.lif import LIFParams, poisson_events, simulate
from flycircuit.models.lif_stream import (
    estimate_whole_brain_inference,
    estimate_whole_brain_training,
    simulate_stream,
)


def _net(n=300, deg=12, seed=0):
    rng = np.random.default_rng(seed)
    pre = np.repeat(np.arange(n), deg)
    post = rng.integers(0, n, size=n * deg)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    sign = np.where(rng.random(n) < 0.7, 1.0, -1.0)[pre]
    contacts = rng.integers(1, 20, size=len(pre)).astype(float)
    p = LIFParams()
    W = sp.coo_matrix((sign * contacts * p.w_syn, (post, pre)), shape=(n, n)).tocsc()
    W.sum_duplicates()
    return W, p


def _events(targets, steps, p):
    return poisson_events(np.asarray(targets), steps, 200.0, p.dt, seed=7)


@pytest.mark.parametrize("backend,device", [("numpy", "cpu"), ("torch", "cpu")])
def test_stream_matches_reference_counts_and_trains(backend, device):
    W, p = _net()
    steps = 4000
    targets = np.arange(0, 40)
    es, et = _events(targets, steps, p)
    ref = simulate(W, steps, p, es, et, no_refractory=targets)
    watch = np.array([0, 5, 77, 123, 299])
    out = simulate_stream(W, steps, p, es, et, no_refractory=targets, watch=watch,
                          backend=backend, device=device)
    assert ref.counts().sum() > 200, "test network must actually spike"
    np.testing.assert_array_equal(out.counts, ref.counts())
    trains = ref.trains()
    for w, tr in out.watch_trains().items():
        np.testing.assert_array_equal(tr, trains.get(w, np.empty(0, np.int64)))
    assert out.pop_per_step.sum() == out.counts.sum()
    # first spike consistent with the full train
    for nid, tr in trains.items():
        assert out.first_step[nid] == tr[0]
    assert out.psth.sum() == sum(len(trains.get(int(w), [])) for w in watch)


def test_stream_keeps_no_spike_list():
    W, p = _net()
    out = simulate_stream(W, 500, p, watch=np.array([1, 2]))
    assert out.watch_raster.shape == (500, 2)
    assert out.counts.shape == (300,)


def test_watch_limit_enforced():
    W, p = _net(n=50)
    with pytest.raises(ValueError):
        simulate_stream(W, 10, p, watch=np.arange(5000) % 50)


def test_estimates_are_labelled_theoretical_and_scale():
    a = estimate_whole_brain_inference(139_000, 15_000_000, 10_000)
    assert a["kind"] == "theoretical" and 0.1 < a["total_gib"] < 4
    t1 = estimate_whole_brain_training(139_000, 15_000_000, tbptt=32)
    t2 = estimate_whole_brain_training(139_000, 15_000_000, tbptt=64)
    assert t2["edge_saved_gib"] == pytest.approx(2 * t1["edge_saved_gib"])
    assert t1["kind"] == "theoretical"
