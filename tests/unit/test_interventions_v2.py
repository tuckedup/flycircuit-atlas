"""Screen v2 contracts: sign-policy loader switch, trained-instance forward == agent forward,
clamp/pairing semantics on the trained path, two-level aggregation, exclusion, closed-loop honesty.
Offline: uses the SYNTHETIC KC/MBON circuit (no real data); real-data checks skip when absent."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flycircuit.config import load_yaml, project_root
from flycircuit.interventions import trained as T
from flycircuit.interventions.rng import RngStreams
from flycircuit.interventions.spec import InterventionSpec


def _proto():
    p = load_yaml(project_root() / "configs" / "experiments" / "learning_protocol.yaml")
    p["training"]["trials_per_phase"] = 40
    p["task"].update(n_train=40, n_val=16, n_test=16)
    return p


@pytest.fixture(scope="module")
def inst():
    from flycircuit.eval.learning import _rngs, build_sparse_model, make_task, train_phase
    from flycircuit.plasticity.agent import LearningAgent
    from flycircuit.tasks.graph_loader import make_learning_synthetic

    proto = _proto()
    b = make_learning_synthetic(n_kc=60, n_mbon=6, n_aux=12, seed=0)
    task = make_task(proto, 3)
    model = build_sparse_model(proto, b, 3, task.spec.n_in)
    agent = LearningAgent(model, "bptt", task, 0.003, seed=3, logit_cap=2.0)
    train_phase(agent, task, 0, 40, _rngs(3, 0))
    w = model.edge_values().detach()
    for p in model.parameters():
        p.requires_grad_(False)
    ins = T.Instance("mem", 3, "bptt", "synthetic", model, agent.f_mu, agent.f_sd, w, task,
                     task.response_window(), float("nan"), {})
    return ins, agent, b


def test_trained_forward_matches_agent_forward(inst):
    ins, agent, _ = inst
    cues, _ = T.heldout_cues(ins.task)
    x = torch.as_tensor(ins.task.make_input(cues, 0, np.random.default_rng(1)))
    mine = T.branch_logits(ins, x, None, RngStreams(0, 0))
    ref, _ = agent._forward(x, False, 0.0, False, None)
    assert torch.allclose(mine, ref, atol=1e-5)


def _strong(ins):
    from dataclasses import replace

    # recurrent drive matters; identity z-scoring so the +-5 clip of the tiny-activity synthetic
    # calibration cannot saturate the readout and hide interventions
    return replace(ins, w_eff=ins.w_eff * 5.0, f_mu=torch.zeros_like(ins.f_mu),
                   f_sd=torch.ones_like(ins.f_sd))


def _x(ins):
    cues, _ = T.heldout_cues(ins.task)
    return torch.as_tensor(ins.task.make_input(cues, 0, np.random.default_rng(1)))


def test_baseline_deterministic_and_clamp_excite_of_hub_changes_output(inst):
    ins, _, _ = inst
    strong, x = _strong(ins), _x(ins)
    hub = int(np.argmax(np.bincount(ins.model.pre.numpy(), minlength=ins.model.n)))
    base = T.branch_logits(strong, x, None, RngStreams(0, 0))
    assert torch.equal(base, T.branch_logits(strong, x, None, RngStreams(0, 0)))
    cl = T.branch_logits(strong, x, InterventionSpec("clamp_excite", (hub,), dose=0.9, t_start=0),
                         RngStreams(0, 0))
    assert float((base - cl).abs().max()) > 1e-6


def test_disruption_stream_does_not_shift_baseline(inst):
    ins, _, _ = inst
    cues, _ = T.heldout_cues(ins.task)
    s = RngStreams(5, 0)
    x = torch.as_tensor(ins.task.make_input(cues, 0, s.stream("obs_noise")))
    spec = InterventionSpec("transient_disruption", (int(ins.model.readout_idx[1]),), dose=1.0,
                            t_start=0)
    T.branch_logits(ins, x, spec, RngStreams(5, 0))
    x2 = torch.as_tensor(ins.task.make_input(cues, 0, RngStreams(5, 0).stream("obs_noise")))
    assert torch.equal(x, x2)  # intervention randomness never perturbs the shared input stream


def test_clamp_silence_differs_from_edge_deletion_on_readout_node(inst):
    """Clamping a READOUT node to 0 changes what the decoder sees; deleting its outgoing edges
    does not (it has none to the readout) - distinct interventions must give distinct outputs."""
    ins, _, _ = inst
    strong, x = _strong(ins), _x(ins)
    tgt = int(ins.model.readout_idx[0])
    a = T.branch_logits(strong, x, InterventionSpec("clamp_silence", (tgt,), dose=0.0, t_start=0),
                        RngStreams(0, 0))
    b = T.branch_logits(strong, x, InterventionSpec("delete_outgoing", (tgt,), dose=1.0, t_start=0),
                        RngStreams(0, 0))
    base = T.branch_logits(strong, x, None, RngStreams(0, 0))
    assert float((a - b).abs().max()) > 1e-6
    assert float((a - base).abs().max()) > 1e-6


def test_instance_effects_and_aggregate_two_level(inst):
    ins, _, b = inst
    g = b.graph
    specs = [InterventionSpec("clamp_silence", (int(i),), dose=0.0, t_start=0, level="neuron",
                              label=f"n{i}", family="clamp_silence", intervention_id=f"c:{i}")
             for i in range(3)]
    per = [T.instance_effects(ins, specs, {}, n_rep=3, seed=0) for _ in range(3)]
    rows = T.aggregate(specs, per, g, seed=0, n_controls=0)
    assert len(rows) == 3
    r = rows[0]
    assert r["n_instances"] == 3 and "within_instance_sd_acc" in r
    # identical instances -> zero between-instance variance -> p is 1.0 (zero effect) or nan (constant)
    assert r["effect_acc_sd_between"] == pytest.approx(0.0, abs=1e-12)


def test_across_requires_instances_not_repetitions():
    out = T._across(np.array([0.1]), 0)
    assert np.isnan(out["p"])  # one instance cannot support a test, however many noise reps it had
    out = T._across(np.array([0.1, 0.12, 0.09, 0.11]), 0)
    assert out["p"] < 0.01 and out["sign_consistency"] == 1.0


def test_bad_checkpoint_format_rejected(tmp_path):
    p = tmp_path / "x.pt"
    torch.save({"format": "other/9"}, p)
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        T.load_instance(p)


def test_sign_variant_unknown_name_errors():
    with pytest.raises(ValueError, match="unknown sign variant"):
        T.sign_vector(np.array([1], dtype=np.int64), "nope")


def test_legacy_vs_v2_sign_coverage_on_real_pilot():
    from flycircuit.interventions.real import circuit_dir, load_circuit_graph

    if not (circuit_dir("mushroom_body_right_pilot") / "internal_ids.parquet").exists():
        pytest.skip("real pilot selection not ingested")
    _, info_old, s_old = load_circuit_graph("mushroom_body_right_pilot", sign_policy="legacy_top_nt")
    g, info_new, s_new = load_circuit_graph("mushroom_body_right_pilot")
    assert (s_old != 0).sum() < 100                      # v1 regime: ~1.8% usable
    assert (s_new != 0).sum() > 1300                     # v2: known_nt covers ~99%
    mod = np.asarray(g.node_attrs["modulatory"], dtype=bool)
    assert mod.sum() > 0
    assert info_new["sign_policy"].startswith("v2.0")
    assert info_old["sign_policy"] == "legacy_top_nt"
    # dopamine neurons keep fast sign 0 under the default (separate modulatory channel)
    assert (s_new[mod] == 0).all()


def test_variant_table_covers_legacy_and_v2_on_real_pilot():
    from flycircuit.interventions.real import circuit_dir, load_circuit_graph
    from flycircuit.interventions.sensitivity import variant_table

    if not (circuit_dir("mushroom_body_right_pilot") / "internal_ids.parquet").exists():
        pytest.skip("real pilot selection not ingested")
    g, _, _ = load_circuit_graph("mushroom_body_right_pilot")
    tab = variant_table(g, ["v2_default", "unknown_inhibitory"])
    assert {"v2_default", "unknown_inhibitory", "legacy_top_nt+unknown_excitatory",
            "legacy_top_nt+unknown_zero"} <= set(tab)
    for v in tab.values():
        assert set(np.unique(v["sign"])) <= {-1.0, 0.0, 1.0}
    n_legacy = (tab["legacy_top_nt+unknown_zero"]["sign"] != 0).sum()
    assert n_legacy < 100 < (tab["v2_default"]["sign"] != 0).sum()


def test_sweep_summary_is_per_instance_and_uncertainty_split():
    import json

    rows = []
    for tgt in range(4):
        for dose in (0.25, 0.5, 1.0):
            eff = [dose * 0.1 * (1 if tgt % 2 else -1)] * 3 + [dose * 0.12]
            rows.append({"kind": "disconnect_both", "dose": dose, "t_start": 0, "t_end": -1,
                         "per_instance_effect_acc": json.dumps(eff)})
    tab, dr = T.summarize_sweep(rows, seed=0)
    assert len(tab) == 3 and all(t["n_instances"] == 4 and t["n_targets"] == 4 for t in tab)
    assert tab[0]["mean_abs_effect_pp"] < tab[2]["mean_abs_effect_pp"]  # larger dose, larger |effect|
    assert dr and dr[0]["spearman_dose_vs_abs_effect"] == pytest.approx(1.0)
    prim = [{"family": "clamp_silence", "effect_acc_sd_between": 0.02,
             "within_instance_sd_acc": 0.01}]
    u = T.uncertainty_split(prim)
    assert u["median_ratio_between_over_noise"] == pytest.approx(2.0)
