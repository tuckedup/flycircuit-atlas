"""A training-state snapshot must remain owned after the live learner is reset."""
import copy

import numpy as np
import pytest
import torch

from flycircuit.eval import learning_v2 as V
from flycircuit.tasks.graph_loader import make_learning_synthetic


@pytest.mark.parametrize("arm", ["bptt", "plast_dec"])
def test_in_memory_snapshot_survives_full_reset_and_restores_predictions(arm):
    proto = copy.deepcopy(V.load_protocol())
    proto["task"].update(n_train=40, n_val=16, n_test=16)
    base = make_learning_synthetic(n_kc=24, n_mbon=4, n_aux=4, seed=0)
    task = V.L.make_task(proto, 7)
    agent, _ = V.make_agent_v2(proto, base, {}, arm, "connectome", 0.01, 7, task)
    V.L.train_phase(agent, task, 0, 40, V.L._rngs(7, 0))
    if agent.plastic is not None:
        agent.plastic.gain.add_(0.001)
    baseline_accuracy = agent.evaluate("val", 0)
    snapshot = agent.checkpoint_payload()
    expected = copy.deepcopy(snapshot)
    agent.reset("full")
    for name, value in expected["model"].items():
        assert torch.equal(snapshot["model"][name], value), f"snapshot alias: {name}"
    if agent.plastic is not None:
        assert torch.equal(snapshot["plastic_gain"], expected["plastic_gain"])
    V._restore(agent, snapshot)
    for name, value in expected["model"].items():
        assert torch.equal(agent.model.state_dict()[name], value)
    assert np.array_equal(agent.value.v, expected["critic_v"])
    assert agent.evaluate("val", 0) == baseline_accuracy


def test_association_reset_probe_restores_current_contingency_score():
    proto = copy.deepcopy(V.load_protocol())
    proto["task"].update(n_train=40, n_val=16, n_test=16)
    proto["training"].update(trials_per_phase=40, reversal_trials=40, delay_trials=10,
                             acquisition_window=5, perseveration_window=10)
    base = make_learning_synthetic(n_kc=24, n_mbon=4, n_aux=4, seed=0)
    task = V.L.make_task(proto, 7)
    agent, _ = V.make_agent_v2(proto, base, {}, "bptt", "connectome", 0.01, 7, task)
    result = V.run_association_v2(agent, task, proto, 7, ["val"])
    assert result["reset_probe_val_acc_current_contingency"]["none"] == result["val_acc_reversed"]
    assert agent.evaluate("val", 0, reversed_=True) == result["val_acc_reversed"]
