"""Interventions: pairing, silencing semantics, distinct targets, multiplicity, controls."""

import numpy as np
import pytest
import torch

from flycircuit.interventions.engine import (
    SimConfig,
    edited_edge_values,
    effect_metrics,
    run_branch,
)
from flycircuit.interventions.inference import benjamini_hochberg, effect_summary, holm
from flycircuit.interventions.real import synthetic_screen_graph
from flycircuit.interventions.rng import RngStreams
from flycircuit.interventions.screen import build_model, run_screen
from flycircuit.interventions.spec import InterventionSpec
from flycircuit.interventions.targets import (
    dedupe_units,
    degrees,
    eligible_units,
    matched_random_spec,
    plan_screen,
)


@pytest.fixture(scope="module")
def setup():
    g = synthetic_screen_graph(n=120, deg=8, seed=1)
    sign = np.where(np.random.default_rng(1).random(g.n) < 0.2, -1.0, 1.0).astype("float32")
    return g, build_model(g, sign, {"init_gain": 0.1})


CFG = SimConfig(steps=30)


def test_engine_matches_model_forward_without_intervention(setup):
    g, model = setup
    ev = __import__("flycircuit.interventions.engine", fromlist=["make_events"]).make_events(
        model, RngStreams(0, 0), CFG)
    x = ev["stim"] + ev["obs_noise"]
    st = model.init_state(1)
    st["h"] = ev["h0"].clone()
    with torch.no_grad():
        y_ref, _ = model(x, st)
        y = run_branch(model, RngStreams(0, 0), CFG, None)["y"]
    assert torch.allclose(y, y_ref, atol=1e-5)


def test_sham_equals_baseline_and_streams_are_paired(setup):
    _, model = setup
    a = run_branch(model, RngStreams(3, 2), CFG, None)["y"]
    b = run_branch(model, RngStreams(3, 2), CFG, None)["y"]
    assert torch.equal(a, b)
    c = run_branch(model, RngStreams(3, 5), CFG, None)["y"]  # different run -> different events
    assert not torch.equal(a, c)


def test_independent_streams_not_shifted_by_other_consumption():
    s1, s2 = RngStreams(0, 0), RngStreams(0, 0)
    s1.stream("disruption").standard_normal(1000)  # intervention branch eats draws
    assert np.array_equal(s1.stream("stimulus").random(5), s2.stream("stimulus").random(5))


def test_disruption_draws_do_not_change_shared_events(setup):
    g, model = setup
    spec = InterventionSpec("transient_disruption", (10, 11, 12), dose=0.5, t_start=5)
    s = RngStreams(0, 0)
    base_ev = __import__("flycircuit.interventions.engine", fromlist=["make_events"]).make_events(model, s, CFG)
    run_branch(model, RngStreams(0, 0), CFG, spec)
    again = __import__("flycircuit.interventions.engine", fromlist=["make_events"]).make_events(
        model, RngStreams(0, 0), CFG)
    assert torch.equal(base_ev["stim"], again["stim"])


def test_clamp_silence_node_cannot_regain_activity_despite_bias_and_input(setup):
    g, model = setup
    import copy

    m = copy.deepcopy(model)
    tgt = (20, 21)
    with torch.no_grad():
        m.bias[list(tgt)] = 50.0  # huge bias: would saturate tanh if not clamped
    spec = InterventionSpec("clamp_silence", tgt, dose=0.0, t_start=0)
    out = run_branch(m, RngStreams(0, 0), CFG, spec)
    assert float(out["clamped_trace"].abs().max()) == 0.0
    free = run_branch(m, RngStreams(0, 0), CFG, None)
    assert float(free["mean_abs_h"][list(tgt)].min()) > 0.1  # without clamp they are active


def test_clamp_is_released_after_window(setup):
    _, model = setup
    import copy

    m = copy.deepcopy(model)
    with torch.no_grad():
        m.bias[30] = 5.0
    spec = InterventionSpec("clamp_silence", (30,), dose=0.0, t_start=2, t_end=10)
    out = run_branch(m, RngStreams(0, 0), CFG, spec)
    assert out["clamped_trace"].shape[0] == 8  # steps 2..9 only


def test_clamp_and_edge_deletion_are_different_interventions(setup):
    g, model = setup
    tgt = tuple(int(i) for i in np.flatnonzero(~g.node_masks["sensory_input"])[:5] + 20)
    base = run_branch(model, RngStreams(0, 0), CFG, None)["y"]
    ya = run_branch(model, RngStreams(0, 0), CFG, InterventionSpec("clamp_silence", tgt, t_start=0))["y"]
    yb = run_branch(model, RngStreams(0, 0), CFG, InterventionSpec("delete_outgoing", tgt, t_start=0))["y"]
    assert not torch.allclose(ya, yb)
    assert not torch.allclose(ya, base) or not torch.allclose(yb, base)


def test_edge_edit_masks_correct_edges(setup):
    _, model = setup
    tgt = (7,)
    w0 = model.edge_values().detach()
    w_out = edited_edge_values(model, w0, InterventionSpec("delete_outgoing", tgt))
    w_in = edited_edge_values(model, w0, InterventionSpec("delete_incoming", tgt))
    pre, post = model.pre.numpy(), model.post.numpy()
    assert (w_out[pre == 7] == 0).all() and (w_out[pre != 7] == w0[pre != 7]).all()
    assert (w_in[post == 7] == 0).all() and (w_in[post != 7] == w0[post != 7]).all()
    w_g = edited_edge_values(model, w0, InterventionSpec("gain_scale_incoming", tgt, dose=3.0))
    assert torch.allclose(w_g[post == 7], 3 * w0[post == 7])


def test_intervention_effect_zero_for_sham_nonzero_for_real(setup):
    _, model = setup
    base = run_branch(model, RngStreams(0, 0), CFG, None)["y"]
    assert effect_metrics(base, base, 0)["deviation"] == 0.0
    spec = InterventionSpec("drive_excitation", (40, 41, 42), dose=3.0, t_start=0)
    assert effect_metrics(base, run_branch(model, RngStreams(0, 0), CFG, spec)["y"], 0)["deviation"] > 0


def test_closed_loop_feedback_changes_trajectory(setup):
    _, model = setup
    spec = InterventionSpec("drive_excitation", (40, 41, 42), dose=3.0, t_start=0)
    cl = SimConfig(steps=30, closed_loop=True)
    yo = run_branch(model, RngStreams(0, 0), CFG, spec)["y"]
    yc = run_branch(model, RngStreams(0, 0), cl, spec)["y"]
    assert not torch.allclose(yo, yc)


def test_duplicate_targets_rejected():
    with pytest.raises(ValueError):
        InterventionSpec("clamp_silence", (1, 1))
    with pytest.raises(ValueError):
        InterventionSpec("not_a_kind", (1,))


def test_screen_plan_has_no_duplicate_targets_and_reports_shortfall():
    g = synthetic_screen_graph(n=60, deg=6)
    units, _ = eligible_units(g, "neuron")
    specs, plan = plan_screen(g, units, n_interventions=120, min_distinct=100)
    keys = [(s.kind, frozenset(s.targets)) for s in specs]
    assert len(keys) == len(set(keys))
    assert plan["planned_distinct_targets"] <= len(units) < 100
    assert not plan["meets_distinct_target_goal"] and "only" in plan["note"]
    all_ids = {int(i) for s in specs for i in s.targets}
    assert all_ids <= set(range(g.n))  # no invented ids


def test_screen_reaches_goal_when_enough_targets():
    g = synthetic_screen_graph(n=300, deg=8)
    units, _ = eligible_units(g, "neuron")
    specs, plan = plan_screen(g, units, 120, 100)
    assert plan["meets_distinct_target_goal"] and len(specs) == 120
    primary = [s for s in specs if s.kind == "clamp_silence"]
    assert len({s.targets for s in primary}) == len(primary) >= 100


def test_eligibility_excludes_sensory_readout_and_reports_missing_attribute():
    g = synthetic_screen_graph(n=80, deg=6)
    units, rep = eligible_units(g, "neuron")
    nodes = {u.nodes[0] for u in units}
    assert not nodes & set(np.flatnonzero(g.node_masks["sensory_input"]))
    assert not nodes & set(np.flatnonzero(g.node_masks["action_readout"]))
    _, rep2 = eligible_units(g, "region")
    assert "error" in rep2 and rep2["n_units"] == 0  # actionable, not silently empty
    tu, _ = eligible_units(g, "type", max_size=50)
    assert tu and all(len(u.nodes) >= 2 for u in tu)
    d, dropped = dedupe_units(tu + tu)
    assert dropped == len(tu)


def test_matched_random_control_properties(setup):
    g, _ = setup
    ind, outd = degrees(g)
    pool = np.flatnonzero(~(g.node_masks["sensory_input"] | g.node_masks["action_readout"]))
    spec = InterventionSpec("clamp_silence", tuple(int(i) for i in pool[:6]), intervention_id="x")
    c1 = matched_random_spec(g, spec, pool, 0, 0)
    c2 = matched_random_spec(g, spec, pool, 0, 0)
    assert c1.targets == c2.targets  # reproducible
    assert not set(c1.targets) & set(spec.targets)
    assert len(c1.targets) == len(spec.targets) and len(set(c1.targets)) == len(c1.targets)
    assert c1.kind == spec.kind and c1.dose == spec.dose and c1.matched_to == "x"


def test_holm_and_bh_known_values_and_nan_handling():
    p = np.array([0.01, 0.04, 0.03, 0.005, np.nan])
    assert np.allclose(holm(p)[:4], [0.03, 0.06, 0.06, 0.02], atol=1e-9)
    assert np.isnan(holm(p)[4])
    b = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.005]))
    assert np.allclose(b, [0.02, 0.04, 0.04, 0.02])


def test_effect_summary_cases():
    s = effect_summary([0.0, 0.0, 0.0])
    assert s["p_value"] == 1.0
    s = effect_summary([1.0, 1.1, 0.9, 1.2, 1.0, 0.95])
    assert s["p_value"] < 0.01 and s["ci_lo"] < s["mean"] < s["ci_hi"]
    assert np.isnan(effect_summary([1.0, 2.0])["p_value"])


def test_run_screen_end_to_end_small(setup):
    g, model = setup
    units, _ = eligible_units(g, "neuron")
    specs, plan = plan_screen(g, units, n_interventions=6, min_distinct=4, extra_kinds=())
    per_run, summ = run_screen(g, model, specs, n_runs=4, seed=0, sim=SimConfig(steps=20), n_controls=1)
    assert len(summ) == 6 and all(d["n_runs"] == 4 for d in summ)
    assert all("p_holm" in d and "vs_control_mean" in d for d in summ)
    assert {r["role"] for r in per_run} == {"target", "control0"}
    # clamp enforced exactly in every run
    assert all(r.get("clamp_max_abs_dev", 0.0) == 0.0 for r in per_run if r["role"] == "target")
