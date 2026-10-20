"""Tier-1 navigation arena: transitions, success criterion, dropout, splits, metrics, no-LLM policy."""

from __future__ import annotations

import math
import socket
import sys
from pathlib import Path

import numpy as np
import pytest

from flycircuit.envs import arena as A
from flycircuit.envs.arena import DropoutSpec, EvalCondition, Layout, NavEnv, make_layout
from flycircuit.envs.controllers import (
    BatchController,
    NeuralController,
    PrivilegedTeacher,
    RandomWalk,
    SensorScripted,
    aggregate,
    run_episodes,
)


def _lay(start=(2.0, 4.0), heading=0.0, goal=(6.0, 4.0), obstacles=()):
    return Layout(0, "test", start, heading, goal, (2.0, 7.0), tuple(obstacles))


def test_layout_deterministic_and_splits_disjoint():
    assert make_layout(5, "train") == make_layout(5, "train")
    assert make_layout(5, "train").hash() != make_layout(6, "train").hash()
    tr = {make_layout(i, "train").hash() for i in range(300)}
    te = {make_layout(i, "test").hash() for i in range(300)}
    va = {make_layout(i, "val").hash() for i in range(300)}
    assert not (tr & te) and not (tr & va) and not (te & va)
    with pytest.raises(ValueError):
        make_layout(0, "bogus")
    with pytest.raises(ValueError):
        make_layout(1_000_000, "train")


def test_layout_constraints():
    for i in range(100):
        lay = make_layout(i, "test")
        assert math.dist(lay.start, lay.goal) >= 4.0
        for cx, cy, r in lay.obstacles:
            assert math.dist((cx, cy), lay.start) >= r + 0.8 - 1e-9
            assert math.dist((cx, cy), lay.goal) >= r + A.GOAL_R + 0.5 - 1e-9


def test_clock_constants_consistent():
    assert A.CONTROL_DT == pytest.approx(A.PHYSICS_DT * A.SUBSTEPS)
    assert A.N_OBS == len(A.CHANNELS) == 10
    flat = sorted(i for g in A.MODALITIES.values() for i in g)
    assert flat == list(range(A.N_OBS))


def test_forward_motion_and_reward_formula():
    env = NavEnv(_lay())
    env.reset()
    obs, r, term, trunc, info = env.step((1.0, 0.0))  # full forward, no turn
    dist_moved = A.V_MAX * A.CONTROL_DT
    assert env.state[0] == pytest.approx(2.0 + dist_moved, abs=1e-9)
    assert env.state[1] == pytest.approx(4.0, abs=1e-9)
    assert env.state[2] == pytest.approx(0.0)
    assert r == pytest.approx(dist_moved - 0.01)  # progress straight toward goal, step cost
    assert not term and not trunc
    assert len(obs) == A.N_OBS
    assert obs[7] == pytest.approx(1.0)  # prop_v: previous forward speed / V_MAX


def test_turn_rate_ccw_positive_and_action_clipping():
    env = NavEnv(_lay())
    env.reset()
    env.step((-1.0, 1.0))  # zero speed, max left turn
    assert env.state[2] == pytest.approx(A.W_MAX * A.CONTROL_DT)
    assert env.state[:2] == (2.0, 4.0)
    env2 = NavEnv(_lay())
    env2.reset()
    env2.step((50.0, -50.0))  # clipped to bounds
    assert env2.state[2] == pytest.approx(-A.W_MAX * A.CONTROL_DT)
    assert env2.last_v == pytest.approx(A.V_MAX)


def test_wall_collision_clamps_and_flags():
    env = NavEnv(_lay(start=(7.85, 4.0), heading=0.0))
    env.reset()
    _, r, *_ = env.step((1.0, 0.0))
    assert env.state[0] <= A.ARENA - A.AGENT_R + 1e-12
    assert env.contact == 1.0 and env.collision_steps == 1
    assert r < 0  # collision penalty and no progress


def test_obstacle_collision_rejects_move():
    lay = _lay(start=(2.0, 4.0), obstacles=[(2.5, 4.0, 0.3)])
    env = NavEnv(lay)
    env.reset()
    for _ in range(5):
        env.step((1.0, 0.0))
    assert math.hypot(env.state[0] - 2.5, env.state[1] - 4.0) >= 0.3 + A.AGENT_R - 1e-9
    assert env.collision_steps >= 1


def test_success_criterion_terminates_at_goal_radius():
    env = NavEnv(_lay(start=(5.0, 4.0), goal=(6.0, 4.0)))
    env.reset()
    steps, term = 0, False
    while not term and steps < 50:
        _, r, term, trunc, info = env.step((1.0, 0.0))
        steps += 1
    assert term and env.success and info["success"] and not trunc
    # goal radius 0.5, distance 1.0 -> needs travel > 0.5; speed 0.15/step -> 4 steps
    assert steps == 4
    assert math.dist(env.state[:2], (6.0, 4.0)) < A.GOAL_R
    assert r > 4.0  # success bonus included
    with pytest.raises(RuntimeError):
        env.step((0.0, 0.0))


def test_truncation_at_max_steps_not_success():
    env = NavEnv(_lay(start=(2.0, 4.0), goal=(7.0, 7.0)), max_steps=5)
    env.reset()
    for _i in range(5):
        _, _, term, trunc, _ = env.step((-1.0, 0.0))  # stand still
        assert term is False
    assert trunc is True and not env.success


def test_observation_geometry_odor_and_beacon():
    # goal straight ahead: beacon bearing 0 => sin~0, cos~1; antennae symmetric
    lay = _lay()
    env = NavEnv(lay, EvalCondition())
    obs = env.reset()
    assert abs(obs[2]) < 0.3 and obs[3] > 0.7  # noise sigma 0.05
    assert abs(obs[0] - obs[1]) < 0.2
    # goal to the LEFT (CCW): left antenna closer => odor_L > odor_R
    lay_left = _lay(goal=(2.0, 7.5), heading=0.0)
    vals = []
    for s in range(20):
        e = NavEnv(lay_left, noise_seed=s)
        o = e.reset()
        vals.append(o[0] - o[1])
    assert np.mean(vals) > 0.1
    # conflict condition moves the beacon to the decoy; odor still points to the goal
    e = NavEnv(_lay(goal=(6.0, 4.0)), EvalCondition(conflict=True))
    o = e.reset()
    assert o[3] < 0.6  # decoy at (2,7) is at ~90 deg => cos small


def test_contrast_scales_vision_only():
    lay = _lay(obstacles=[(3.0, 4.0, 0.3)])
    hi = np.mean([NavEnv(lay, EvalCondition(contrast=1.0), noise_seed=s).reset()[4:7] for s in range(40)], 0)
    lo = np.mean([NavEnv(lay, EvalCondition(contrast=0.3), noise_seed=s).reset()[4:7] for s in range(40)], 0)
    assert lo[1] < hi[1] and lo[1] == pytest.approx(0.3 * hi[1], abs=0.05)


def test_dropout_channel_zeros_and_is_constant_per_episode():
    cond = EvalCondition(dropout=DropoutSpec(0.5, "channel", "episode"))
    env = NavEnv(make_layout(3, "test"), cond, mask_seed=7)
    obs = env.reset()
    dropped = [i for i, k in enumerate(env._keep) if not k]
    assert dropped and all(obs[i] == 0.0 for i in dropped)
    masks = {tuple(env._keep)}
    for _ in range(10):
        obs, *_ = env.step((0.5, 0.2))
        masks.add(tuple(env._keep))
        assert all(obs[i] == 0.0 for i in dropped)  # dropped channels exactly zero every step
    assert len(masks) == 1  # per-episode mask is held


def test_dropout_per_step_changes_and_modality_groups():
    rng_env = NavEnv(make_layout(4, "test"), EvalCondition(dropout=DropoutSpec(0.5, "channel", "step")))
    rng_env.reset()
    seen = set()
    for _ in range(20):
        rng_env.step((0.3, 0.1))
        seen.add(tuple(rng_env._keep))
    assert len(seen) > 1
    spec = DropoutSpec(0.5, "modality", "episode")
    import random

    for s in range(30):
        keep = spec.draw(random.Random(s))
        for idx in A.MODALITIES.values():
            assert len({keep[i] for i in idx}) == 1  # all-or-nothing per modality
    assert DropoutSpec(0.0).draw(random.Random(0)) == [True] * A.N_OBS
    assert DropoutSpec(1.0).draw(random.Random(0)) == [False] * A.N_OBS
    with pytest.raises(ValueError):
        DropoutSpec(1.5)
    with pytest.raises(ValueError):
        DropoutSpec(0.2, "bogus")


def test_dropout_rate_matches_level():
    spec = DropoutSpec(0.25)
    import random

    n = sum(sum(not k for k in spec.draw(random.Random(s))) for s in range(2000))
    assert n / (2000 * A.N_OBS) == pytest.approx(0.25, abs=0.02)


def test_masks_and_noise_paired_across_controllers():
    lays = [make_layout(i, "test") for i in range(8)]
    cond = EvalCondition(dropout=DropoutSpec(0.5))
    e1 = [NavEnv(lay, cond, noise_seed=1, mask_seed=2) for lay in lays]
    e2 = [NavEnv(lay, cond, noise_seed=1, mask_seed=2) for lay in lays]
    for a, b in zip(e1, e2, strict=True):
        assert a._keep == b._keep
        assert a.reset() == b.reset()
    # the mask stream is independent of the noise stream: changing only noise keeps masks
    fresh = NavEnv(lays[0], cond, noise_seed=1, mask_seed=2)
    e3 = NavEnv(lays[0], cond, noise_seed=99, mask_seed=2)
    assert e3._keep == fresh._keep and e3.reset() != fresh.reset()


def test_reference_controllers_behave():
    lays = [make_layout(i, "test") for i in range(80)]
    nominal = EvalCondition()
    t, _ = run_episodes(PrivilegedTeacher(), lays, nominal)
    s, _ = run_episodes(SensorScripted(), lays, nominal)
    rnd, _ = run_episodes(RandomWalk(0), lays, nominal)
    assert aggregate(t)["success_rate"] >= 0.9
    assert aggregate(s)["success_rate"] >= 0.85
    assert aggregate(rnd)["success_rate"] <= 0.25
    drop, _ = run_episodes(SensorScripted(), lays, EvalCondition(dropout=DropoutSpec(0.5)))
    assert aggregate(drop)["success_rate"] < aggregate(s)["success_rate"]  # dropout hurts


def test_episode_metrics_definitions():
    env = NavEnv(_lay(start=(2.0, 4.0), goal=(6.0, 4.0)))
    env.reset()
    while not env.done:
        env.step((1.0, 0.0))
    m = env.episode_metrics()
    assert m["success"] and 0.0 < m["spl"] <= 1.0
    assert m["spl"] == pytest.approx((4.0 - A.GOAL_R) / max(4.0 - A.GOAL_R, m["path_len"]))
    assert m["heading_err_deg"] == pytest.approx(0.0, abs=1e-6)
    assert m["collision_rate"] == 0.0 and m["recovery_s"] == [] and m["unrecovered"] == 0


def test_recovery_time_definition():
    env = NavEnv(_lay())
    env.reset()
    env.t = 10
    env.success = False
    # collision starts at idx 2 with distance 4.0; recovers when distance <= 3.8 at idx 6
    env.dist_trace = [5.0, 4.5, 4.0, 4.0, 3.95, 3.9, 3.79, 3.5]
    env.collision_trace = [0, 0, 1, 1, 0, 0, 0, 0]
    m = env.episode_metrics()
    assert m["recovery_s"] == [pytest.approx(4 * A.CONTROL_DT)]
    assert m["unrecovered"] == 0
    env.dist_trace = [5.0, 4.5, 4.0, 4.0, 3.99, 3.99]
    env.collision_trace = [0, 0, 1, 1, 0, 0]
    m = env.episode_metrics()
    assert m["recovery_s"] == [] and m["unrecovered"] == 1


# ---- controller interface, privileged separation, provenance, offline ------------------------
class _SpyNeural(NeuralController):
    def __init__(self, model):
        super().__init__(model)
        self.seen_envs = []

    def _act(self, obs, envs):
        self.seen_envs.append(envs)
        return super()._act(obs, envs)


def _tiny_gru():
    from flycircuit.models.baselines import GRUBaseline

    return GRUBaseline(A.N_OBS, A.N_ACT, 8, seed=0)


def test_neural_controller_never_receives_privileged_env():
    ctl = _SpyNeural(_tiny_gru())
    assert ctl.privileged is False
    lays = [make_layout(i, "test") for i in range(3)]
    run_episodes(ctl, lays, EvalCondition())
    assert ctl.seen_envs and all(e is None for e in ctl.seen_envs)
    assert PrivilegedTeacher.privileged is True


def test_swappable_controller_interface_all_models():
    import torch

    from flycircuit.graph.synthetic import make_synthetic_graph
    from flycircuit.models.baselines import EncoderDecoderOnly, GRUBaseline, random_sparse_rnn
    from flycircuit.models.sparse_rnn import ModelSpec, SparseConnectomeRNN

    g = make_synthetic_graph(30, 0.15, seed=1)  # SYNTHETIC graph: interface test only
    spec = ModelSpec(n_in=A.N_OBS, n_out=A.N_ACT, trainable_encoder=True)
    models = [
        SparseConnectomeRNN.sparse_from_graph(g, spec),
        random_sparse_rnn(g, spec, seed=0),
        GRUBaseline(A.N_OBS, A.N_ACT, 8),
        EncoderDecoderOnly(A.N_OBS, A.N_ACT, 8),
    ]
    lays = [make_layout(i, "val") for i in range(2)]
    for m in models:
        metrics, prov = run_episodes(NeuralController(m), lays, EvalCondition(), )
        assert len(metrics) == 2 and prov["action_source"] == "model_forward"
    assert torch.is_tensor(models[0].raw_gain)


def test_action_provenance_logged_and_deterministic():
    lays = [make_layout(i, "val") for i in range(4)]
    out = []
    for _ in range(2):
        ctl = NeuralController(_tiny_gru())
        _, prov = run_episodes(ctl, lays, EvalCondition())
        out.append(prov)
    assert out[0]["calls"] > 0 and out[0]["actions"] == out[0]["calls"] * 4
    assert out[0]["external_service"] is False
    assert out[0]["action_hash"] == out[1]["action_hash"]  # same seeds => same actions
    assert out[0]["controller"] == "gru"


def test_policy_execution_is_offline_and_llm_free(monkeypatch):
    def _blocked(*a, **k):
        raise AssertionError("network access attempted during policy execution")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    lays = [make_layout(i, "val") for i in range(3)]
    for ctl in (NeuralController(_tiny_gru()), SensorScripted(), PrivilegedTeacher()):
        metrics, prov = run_episodes(ctl, lays, EvalCondition())
        assert len(metrics) == 3 and prov["external_service"] is False
    for mod in ("anthropic", "openai"):
        assert mod not in sys.modules
    # and no LLM client is a declared dependency of the project
    pp = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pp.read_text().lower()
    for bad in ("anthropic", "openai", "langchain", "litellm"):
        assert bad not in text
    # no module in the environment package imports a network/LLM client
    for f in (Path(A.__file__).parent).glob("*.py"):
        src = f.read_text()
        for bad in ("import requests", "import httpx", "import openai", "import anthropic",
                    "urllib.request", "import socket"):
            assert bad not in src, f"{f.name} imports {bad}"


def test_dagger_labels_are_teacher_actions_and_beta1_executes_teacher():
    lays = [make_layout(i, "train") for i in range(4)]
    ctl = NeuralController(_tiny_gru())
    tch = PrivilegedTeacher()
    metrics, _, (x, y, m) = run_episodes(ctl, lays, EvalCondition(), record=True, beta=1.0, teacher=tch)
    assert x.shape[2] == A.N_OBS and y.shape[2] == A.N_ACT and m.shape == x.shape[:2]
    assert m.sum() == sum(mm["steps"] for mm in metrics)  # mask counts valid steps
    # with beta=1 the executed trajectory equals pure teacher rollout => same outcomes
    ref, _ = run_episodes(PrivilegedTeacher(), lays, EvalCondition())
    assert [a["success"] for a in ref] == [a["success"] for a in metrics]
    assert [a["steps"] for a in ref] == [a["steps"] for a in metrics]


def test_base_controller_clips_actions():
    class Big(BatchController):
        kind = "big"

        def _act(self, obs, envs):
            return np.full((obs.shape[0], 2), 9.0, dtype=np.float32)

    c = Big()
    c.reset(2)
    assert np.all(c.act(np.zeros((2, A.N_OBS), np.float32)) == 1.0)
