"""Numpy policy runtime must reproduce the torch models exactly (engineering contract)."""
import numpy as np
import pytest
import torch

from flycircuit.bodies.export import export_policy
from flycircuit.bodies.np_policy import NumpyPolicy
from flycircuit.models.baselines import EncoderDecoderOnly, GRUBaseline
from flycircuit.models.sparse_rnn import ModelSpec, SparseConnectomeRNN


def _rollout_parity(model, tmp_path, seq):
    p = export_policy(model, tmp_path / "p.npz")
    pol = NumpyPolicy.load(p)
    x = torch.as_tensor(seq, dtype=torch.float32)[:, None, :]
    with torch.no_grad():
        y, _ = model(x)
    got = np.stack([pol.act(s) for s in seq])
    np.testing.assert_allclose(got, np.tanh(y[:, 0].numpy()), atol=2e-5)


@pytest.fixture
def seq():
    return np.random.default_rng(0).normal(size=(40, 10)).astype(np.float32)


def test_sparse_rnn_parity(tmp_path, seq):
    rng = np.random.default_rng(1)
    n, e = 60, 300
    pre, post = rng.integers(0, n, e), rng.integers(0, n, e)
    cnt = rng.integers(5, 40, e)
    spec = ModelSpec(n_in=10, n_out=2, dt=1.0, tau=5.0, trainable_encoder=True, init_gain=0.3, seed=3)
    sign = rng.choice([-1, 1], n).astype(np.int8)
    m = SparseConnectomeRNN(n, pre, post, cnt, spec, sign=sign,
                            sensory_idx=np.arange(10), readout_idx=np.arange(n - 5, n))
    with torch.no_grad():  # make the model non-trivial
        m.raw_gain.add_(torch.randn_like(m.raw_gain) * 0.3)
        m.bias.add_(torch.randn_like(m.bias) * 0.1)
        m.encoder.add_(torch.randn_like(m.encoder) * 0.2)
    _rollout_parity(m, tmp_path, seq)


def test_gru_parity(tmp_path, seq):
    _rollout_parity(GRUBaseline(10, 2, 17, seed=2), tmp_path, seq)


def test_encdec_parity(tmp_path, seq):
    _rollout_parity(EncoderDecoderOnly(10, 2, 13, seed=2), tmp_path, seq)


def test_np_policy_is_numpy_only():
    import ast
    import inspect

    import flycircuit.bodies.np_policy as mod
    tree = ast.parse(inspect.getsource(mod))
    roots = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    roots |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert roots <= {"__future__", "numpy"}, roots   # no network/LLM/torch deps in the policy runtime
