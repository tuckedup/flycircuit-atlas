import numpy as np
import pytest
import torch

from flycircuit.graph.synthetic import make_synthetic_graph
from flycircuit.models import ModelSpec, SparseConnectomeRNN
from flycircuit.models.baselines import (
    EncoderDecoderOnly,
    GRUBaseline,
    match_gru_hidden,
    matching_report,
    random_sparse_rnn,
)
from flycircuit.models.sparse_rnn import coupling_prior, resolve_signs
from flycircuit.models.train import estimate_memory_bytes, train_tbptt


def _tiny(backend="scatter", prior="log1p", **kw):
    n = 6
    pre = np.array([0, 0, 1, 2, 3, 4, 5, 2])
    post = np.array([1, 2, 2, 3, 4, 5, 0, 0])
    cnt = np.array([3, 1, 5, 2, 8, 4, 6, 1], dtype=float)
    sign = np.array([1, -1, 1, 1, -1, 1], dtype=float)
    spec = ModelSpec(n_in=2, n_out=2, backend=backend, prior=prior, tau=4.0, dt=1.0, **kw)
    return SparseConnectomeRNN(n, pre, post, cnt, spec, sign=sign), (n, pre, post, cnt, sign)


def _dense_forward(m, n, pre, post, cnt, sign, x_seq):
    """Tiny dense oracle (explicitly bounded size) built independently in numpy/torch."""
    prior = {"identity": cnt, "log1p": np.log1p(cnt), "sqrt": np.sqrt(cnt)}[m.spec.prior]
    gain = torch.nn.functional.softplus(m.raw_gain).detach().numpy()
    W = np.zeros((n, n), dtype=np.float32)
    for e in range(len(pre)):
        W[post[e], pre[e]] += sign[pre[e]] * gain[e] * prior[e]
    return torch.as_tensor(W)


@pytest.mark.parametrize("backend", ["scatter", "coo"])
@pytest.mark.parametrize("prior", ["identity", "log1p", "sqrt"])
def test_forward_matches_dense_oracle(backend, prior):
    m, (n, pre, post, cnt, sign) = _tiny(backend, prior)
    W = _dense_forward(m, n, pre, post, cnt, sign, None)
    h = torch.randn(3, n)
    out = m.recurrent(h, m.edge_values())
    assert torch.allclose(out, h @ W.T, atol=1e-5)


@pytest.mark.parametrize("backend", ["scatter", "coo"])
def test_gradients_match_dense_oracle(backend):
    m, (n, pre, post, cnt, sign) = _tiny(backend)
    h = torch.randn(2, n)
    tgt = torch.randn(2, n)
    # sparse path
    loss = ((m.recurrent(h, m.edge_values()) - tgt) ** 2).sum()
    loss.backward()
    g_sparse = m.raw_gain.grad.clone()
    # dense oracle: W built from the same raw params with autograd
    raw = m.raw_gain.detach().clone().requires_grad_(True)
    gain = torch.nn.functional.softplus(raw)
    prior = coupling_prior(torch.as_tensor(cnt, dtype=torch.float32), "log1p")
    vals = torch.as_tensor(sign, dtype=torch.float32)[torch.as_tensor(pre)] * gain * prior
    W = torch.zeros(n, n).index_put((torch.as_tensor(post), torch.as_tensor(pre)), vals, accumulate=True)
    ((h @ W.T - tgt) ** 2).sum().backward()
    assert torch.allclose(g_sparse, raw.grad, atol=1e-4)


def test_end_to_end_backward_through_time_nonzero_and_finite():
    m, _ = _tiny(z_dim=2)
    x = torch.randn(10, 2, 2)
    y, st = m(x)
    y.square().mean().backward()
    assert m.raw_gain.grad is not None and torch.isfinite(m.raw_gain.grad).all()
    assert m.raw_gain.grad.abs().sum() > 0
    assert set(st) == {"h", "z"}


def test_no_dense_allocation_at_scale():
    # 20k nodes: dense W would be 1.6 GB; model must hold only O(E) parameters
    n, e = 20000, 60000
    rng = np.random.default_rng(0)
    pre, post = rng.integers(0, n, e), rng.integers(0, n, e)
    m = SparseConnectomeRNN(n, pre, post, np.ones(e), ModelSpec(n_in=4, n_out=2))
    assert m.raw_gain.numel() == e
    assert max(p.numel() for p in m.parameters()) <= max(e, n * 4)


def test_alpha_validation():
    with pytest.raises(ValueError):
        ModelSpec(n_in=1, n_out=1, dt=10.0, tau=5.0).resolved_alpha()
    assert ModelSpec(n_in=1, n_out=1, dt=1.0, tau=4.0).resolved_alpha() == 0.25


def test_unknown_sign_policies():
    s = np.array([1, -1, 0, 0], dtype=float)
    assert resolve_signs(s, 4, "excitatory").tolist() == [1, -1, 1, 1]
    assert resolve_signs(s, 4, "inhibitory").tolist() == [1, -1, -1, -1]
    assert resolve_signs(s, 4, "zero").tolist() == [1, -1, 0, 0]
    r1, r2 = resolve_signs(s, 4, "random_fixed", 3), resolve_signs(s, 4, "random_fixed", 3)
    assert r1.tolist() == r2.tolist() and set(r1) <= {-1, 1}
    with pytest.raises(ValueError):
        resolve_signs(s, 4, "bogus")


def test_nonneg_gain_keeps_sign_and_free_gain_can_flip():
    m, _ = _tiny()
    assert (m.gain() >= 0).all()
    m2, _ = _tiny(nonneg_gain=False)
    with torch.no_grad():
        m2.raw_gain[0] = -1.0
    assert m2.gain()[0] < 0


def test_class_gain_shares_parameters():
    n = 5
    pre, post = np.array([0, 1, 2, 3]), np.array([1, 2, 3, 4])
    spec = ModelSpec(n_in=1, n_out=1, gain_mode="class")
    m = SparseConnectomeRNN(n, pre, post, np.ones(4), spec, edge_class=np.array([7, 7, 9, 9]))
    assert m.n_gain_params == 2 and m.gain().shape == (4,)


def test_frozen_edges_get_no_gradient():
    n = 4
    pre, post = np.array([0, 1, 2]), np.array([1, 2, 3])
    mask = np.array([True, False, True])
    m = SparseConnectomeRNN(n, pre, post, np.ones(3), ModelSpec(n_in=1, n_out=1),
                            trainable_edge_mask=mask)
    y, _ = m(torch.randn(6, 1, 1))
    y.sum().backward()
    assert m.raw_gain.grad[1] == 0


def test_graph_builder_and_tbptt_runs_and_detaches():
    g = make_synthetic_graph(30, 0.15, seed=1)
    m = SparseConnectomeRNN.sparse_from_graph(g, ModelSpec(n_in=4, n_out=4))
    x = torch.randn(64, 2, 4)
    tgt = torch.randn(64, 2, 4)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    l1 = train_tbptt(m, x, tgt, torch.nn.functional.mse_loss, opt, tbptt_steps=16)
    l2 = train_tbptt(m, x, tgt, torch.nn.functional.mse_loss, opt, tbptt_steps=16)
    assert np.isfinite(l1) and np.isfinite(l2)


def test_baselines_and_matching_report():
    g = make_synthetic_graph(40, 0.1, seed=0)
    m = SparseConnectomeRNN.sparse_from_graph(g, ModelSpec(n_in=4, n_out=4))
    rep = matching_report(m, 4, 4)
    ref = rep["reference"]["trainable_params"]
    assert rep["gru_param_matched"]["trainable_params"] >= ref
    h = match_gru_hidden(ref, 4, 4)
    assert GRUBaseline(4, 4, h - 1).param_report()["trainable_params"] < ref if h > 1 else True
    assert "encoder_params" in rep["gru_param_matched"] and "discrepancies" in rep
    y, _ = GRUBaseline(4, 4, 8)(torch.randn(5, 2, 4))
    assert y.shape == (5, 2, 4)
    y2, _ = EncoderDecoderOnly(4, 4, 8)(torch.randn(5, 2, 4))
    assert y2.shape == (5, 2, 4)
    rs = random_sparse_rnn(g, ModelSpec(n_in=4, n_out=4))
    assert rs.n_edges == m.n_edges and rs.kind == "random_sparse_rnn"


def test_memory_estimate_scales_with_batch_and_steps():
    a = estimate_memory_bytes(1000, 30000, 1, 32, 30000)
    b = estimate_memory_bytes(1000, 30000, 2, 64, 30000)
    assert b["activations_bytes"] == 4 * a["activations_bytes"]
    assert a["kind"] == "theoretical"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_forward_backward_matches_cpu():
    m, _ = _tiny()
    x = torch.randn(8, 2, 2)
    y, _ = m(x)
    y.sum().backward()
    gc = m.raw_gain.grad.clone()
    m.zero_grad()
    mg = m.to("cuda")
    yg, _ = mg(x.cuda())
    yg.sum().backward()
    assert torch.allclose(gc, mg.raw_gain.grad.cpu(), atol=1e-4)
