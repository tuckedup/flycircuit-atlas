"""Fixture checks from the plan (c)5. Run against the ORIGINAL package (PYTHONPATH=<orig>/src), read-only.

 1. gradient presence / finiteness per parameter group, and frozen groups really stay frozen
 2. checkpoint payload owns its tensors (no aliasing with the live learner) and restore is bit-exact
 3. agents are fresh: same seed -> identical init, no shared storage, training one never moves the other
 4. Torch vs NumPy export parity on a REAL-graph model + determinism vs the saved untrained export
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from flycircuit.eval import learning as L
from flycircuit.eval import learning_v2 as V2
from flycircuit.tasks.graph_loader import make_learning_synthetic

ORIG = Path(os.environ.get("FLYCIRCUIT_HOME") or Path(__file__).resolve().parents[2])


@pytest.fixture(scope="module")
def proto():
    p = copy.deepcopy(L.load_protocol())
    p["training"].update(trials_per_phase=25, reversal_trials=25, perseveration_window=10, acquisition_window=5)
    return p


@pytest.fixture(scope="module")
def bundle():
    return make_learning_synthetic(n_kc=40, n_mbon=6, n_aux=8, seed=0)


def _agent(proto, bundle, learner="bptt", lr=1e-3, seed=0):
    task = L.make_task(proto, seed)
    return L.make_agent(proto, bundle, "connectome", None, learner, lr, seed, task)[0], task


# ---- 1. gradients per parameter group -------------------------------------------------------------
def test_gradients_present_finite_for_trainable_groups_and_frozen_stay_frozen(proto, bundle):
    ag, task = _agent(proto, bundle, "bptt", lr=1e-2)
    before = {n: p.detach().clone() for n, p in ag.model.named_parameters()}
    L.train_phase(ag, task, 0, 12, L._rngs(0, 0))
    trainable = {n for n, p in ag.model.named_parameters() if p.requires_grad}
    assert trainable == {"raw_gain", "decoder"}
    for n, p in ag.model.named_parameters():
        if n in trainable:
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
            assert not torch.equal(p.detach(), before[n]), f"{n} did not update"
        else:
            assert p.grad is None or float(p.grad.abs().sum()) == 0.0, n
            assert torch.equal(p.detach(), before[n]), f"frozen {n} changed"


def test_clip_returns_preclip_norm_so_diagnostics_hook_is_possible():
    p = torch.nn.Parameter(torch.ones(4))
    (p * 10).sum().backward()
    n = torch.nn.utils.clip_grad_norm_([p], 1.0)
    assert float(n) == pytest.approx(20.0)                 # PRE-clip norm is returned
    assert float(p.grad.norm()) == pytest.approx(1.0)


# ---- 2. checkpoint ownership and restore identity -------------------------------------------------
def test_checkpoint_payload_owns_tensors_and_restore_is_exact(proto, bundle):
    ag, task = _agent(proto, bundle, "bptt", lr=1e-2)
    L.train_phase(ag, task, 0, 20, L._rngs(0, 0))          # nonconstant, trained learner
    payload = ag.checkpoint_payload()
    snap = {k: v.clone() for k, v in payload["model"].items()}
    acc = ag.evaluate("val", 0)
    # payload must not alias live params: mutate live params in place, payload must not move
    with torch.no_grad():
        for p in ag.model.parameters():
            p.add_(1.0)
    for k, v in payload["model"].items():
        assert torch.equal(v, snap[k]), f"payload[{k}] aliases the live model"
    V2._restore(ag, payload)
    for k, v in ag.model.state_dict().items():
        assert torch.equal(v.cpu(), snap[k]), f"restore not exact for {k}"
    assert ag.evaluate("val", 0) == acc


# ---- 3. fresh agents -------------------------------------------------------------------------------
def test_agents_are_fresh_identical_init_and_isolated(proto, bundle):
    a1, t1 = _agent(proto, bundle, "bptt", lr=1e-2, seed=3)
    a2, t2 = _agent(proto, bundle, "bptt", lr=1e-2, seed=3)   # as run_one_v2 does for continual after association
    s1 = {k: v.clone() for k, v in a1.model.state_dict().items()}
    for k, v in a2.model.state_dict().items():
        assert torch.equal(v, s1[k]), f"init differs for {k}"
    # Measured: only the immutable graph-index buffers (pre/post/count/sensory_idx/readout_idx) are shared
    # through the circuit bundle; every nn.Parameter is private to its agent.
    p1, p2 = dict(a1.model.named_parameters()), dict(a2.model.named_parameters())
    for n in p1:
        assert p1[n].data_ptr() != p2[n].data_ptr(), f"parameter {n} is shared between agents"
    sd1, sd2 = a1.model.state_dict(), a2.model.state_dict()
    shared = [k for k in sd1 if sd1[k].numel() and sd1[k].data_ptr() == sd2[k].data_ptr()]
    assert not (set(shared) & set(p1)), "a trainable parameter is shared"
    L.train_phase(a1, t1, 0, 15, L._rngs(3, 0))
    for k, v in a2.model.state_dict().items():
        assert torch.equal(v, s1[k]), f"training agent 1 moved agent 2 ({k})"   # incl. shared buffers unchanged


# ---- 4. Torch vs NumPy parity on real-graph models ---------------------------------------------------
@pytest.mark.skipif(not (ORIG / "results/flygym_v3/bc_dart").exists(), reason="needs saved v3 recordings")
@pytest.mark.parametrize("cond", ["connectome_pilot_mb", "gru_param_matched", "encdec_param_matched"])
def test_torch_numpy_parity_and_untrained_export_determinism(cond, tmp_path):
    import importlib.util

    from flycircuit.bodies.export import export_policy

    # Use the SAME builder the v3 body study used (audited signs, v2 nulls). The v1 builder
    # (eval.navigation.build_model) differs from it in 266/7399 connectome edge signs (measured).
    from flycircuit.bodies.stages_v3 import build_model_v3

    model, _ = build_model_v3(cond, 411, 0)
    model.eval()
    path = tmp_path / f"{cond}.npz"
    export_policy(model, path)

    spec = importlib.util.spec_from_file_location("npp", ORIG / "src/flycircuit/bodies/np_policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pol = mod.NumpyPolicy.load(path)

    rec = sorted((ORIG / "results/flygym_v3/bc_dart").glob("rec_val*.npz"))[0]
    x = np.load(rec)["x"][:60].astype(np.float32)
    with torch.no_grad():
        out, _ = model(torch.as_tensor(x)[:, None, :], None)
    ref = torch.tanh(out)[:, 0, :].numpy()
    pol.reset()
    got = np.stack([pol.act(xt.astype(np.float64)) for xt in x])
    assert float(np.abs(ref - got).max()) < 1e-4, f"{cond} torch-vs-numpy parity {np.abs(ref - got).max()}"

    saved = ORIG / f"results/flygym_v3/models/untrained_{cond}_s411.npz"
    if saved.exists():                                       # build + export is deterministic given the seed
        with np.load(saved, allow_pickle=False) as a, np.load(path, allow_pickle=False) as b:
            assert set(a.files) == set(b.files)
            for k in a.files:
                if a[k].dtype.kind in "fiu":
                    assert np.array_equal(a[k], b[k]), f"untrained export differs for {k}"
