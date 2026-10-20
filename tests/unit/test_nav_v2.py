"""Navigation v2 contracts: cue semantics, memory demand, ambiguity handling, leakage, selection rules."""

import numpy as np
import pytest

from flycircuit.envs.arena import DropoutSpec
from flycircuit.envs.arena_v2 import (
    CTX_STEPS,
    N_OBS_V2,
    EvalCondV2,
    NavEnvV2,
    layout_set,
    make_layout_v2,
)
from flycircuit.envs.controllers import aggregate, run_episodes
from flycircuit.envs.controllers_v2 import CtxMemory, CtxNoMemory, OdorOnly
from flycircuit.eval import navigation as nv1
from flycircuit.eval import navigation_v2 as nv


def _lay(trusted):
    for i in range(200):
        lay = make_layout_v2(i, "val")
        if lay.trusted == trusted:
            return lay
    raise AssertionError


def test_cue_assignment_follows_trusted_latent():
    lo, lv = _lay("odor"), _lay("vision")
    eo = NavEnvV2(lo, EvalCondV2(conflict=True))
    assert eo.odor_src == lo.goal and eo.beacon == lo.decoy
    ev = NavEnvV2(lv, EvalCondV2(conflict=True))
    assert ev.odor_src == lv.decoy and ev.beacon == lv.goal
    en = NavEnvV2(lo, EvalCondV2(conflict=False))
    assert en.odor_src == lo.goal and en.beacon == lo.goal


def test_trusted_is_balanced_and_deterministic():
    t = [make_layout_v2(i, "val").trusted for i in range(400)]
    assert 0.4 < np.mean([x == "odor" for x in t]) < 0.6
    assert t == [make_layout_v2(i, "val").trusted for i in range(400)]


@pytest.mark.parametrize("mode,pattern", [
    ("pulse", lambda t, s: s if t < CTX_STEPS else 0.0),
    ("persistent", lambda t, s: s),
    ("hidden", lambda t, s: 0.0),
])
def test_ctx_channel_schedule(mode, pattern):
    lay = _lay("vision")
    e = NavEnvV2(lay, EvalCondV2(ctx=mode))
    obs = e.reset()
    assert len(obs) == N_OBS_V2
    for t in range(CTX_STEPS + 5):
        assert obs[10] == pattern(t, -1.0)
        obs, *_ = e.step((0.0, 0.0))


def test_ctx_zero_without_conflict_and_never_dropped():
    lay = _lay("odor")
    assert NavEnvV2(lay, EvalCondV2(conflict=False)).reset()[10] == 0.0
    e = NavEnvV2(lay, EvalCondV2(dropout=DropoutSpec(1.0)))
    obs = e.reset()
    assert obs[10] == 1.0 and all(o == 0.0 for o in obs[:10])


def test_layout_splits_disjoint_including_ood_and_train_seeds():
    proto = nv.load_protocol()
    seeds = proto["seeds"]["search"] + proto["seeds"]["final"] + proto["seeds"]["dev_pilot"]
    out = nv.assert_no_leakage(proto, seeds)
    assert out["shared"] == 0
    ood = nv.eval_layouts(proto, "test", ood=True)
    assert min(len(lay.obstacles) for lay in ood) >= 6  # genuinely out-of-distribution density
    assert max(len(lay.obstacles) for lay in layout_set("train", 0, 200)) <= 5


def test_benchmark_is_learnable_but_not_trivial_without_memory():
    """Scripted oracle-from-observation solves it; single-cue and memoryless controllers cannot; hidden ctx
    is capped near chance (intrinsically ambiguous)."""
    lays = layout_set("val", 0, 60)

    def sr(ctl, **kw):
        m, _ = run_episodes(ctl, lays, EvalCondV2(**kw), noise_seed=1, mask_seed=2, env_cls=NavEnvV2)
        return aggregate(m)["success_rate"]

    assert sr(CtxMemory()) >= 0.9
    assert sr(OdorOnly()) <= 0.7
    assert sr(CtxNoMemory()) <= 0.7                       # needs memory under 'pulse'
    assert sr(CtxNoMemory(), ctx="persistent") >= 0.9     # ... but not when ctx persists
    assert sr(CtxMemory(), ctx="hidden") <= 0.7           # ambiguity ceiling
    assert sr(OdorOnly(), conflict=False) >= 0.9          # non-conflict episodes remain solvable


def test_policy_execution_is_local_scripted_or_model_only():
    m, prov = run_episodes(CtxMemory(), layout_set("val", 0, 3), EvalCondV2(), env_cls=NavEnvV2)
    assert prov["external_service"] is False and prov["action_source"] == "scripted"


def test_choose_lrs_equal_budget_and_tiebreak():
    grid, seeds = [0.001, 0.01], [1, 2]
    done = {f"a|{lr}|{s}": {"val_success": 0.5, "diverged": False} for lr in grid for s in seeds}
    ch = nv.choose_lrs(done, ["a"], grid, seeds)
    assert ch["a"]["lr"] == 0.001 and ch["a"]["grid_edge"]  # tie -> smaller lr
    done.pop("a|0.01|2")
    with pytest.raises(RuntimeError, match="incomplete search"):
        nv.choose_lrs(done, ["a"], grid, seeds)


def test_diverged_trials_excluded_from_selection():
    grid, seeds = [0.01, 0.03], [1]
    done = {"a|0.01|1": {"val_success": 0.4, "diverged": False},
            "a|0.03|1": {"val_success": 0.9, "diverged": True}}
    assert nv.choose_lrs(done, ["a"], grid, seeds)["a"]["lr"] == 0.01


def _fake_models(proto, conn, nul):
    out = []
    cn = [c["name"] for c in proto["evaluation"]["conditions"]]
    for s, (a, b) in enumerate(zip(conn, nul, strict=True)):
        for cond, v in (("connectome_pilot_mb", a), ("null_v2_swap", b)):
            out.append({"condition": cond, "seed": 300 + s, "lr": 0.01, "grid_edge": False,
                        "training": {"diverged": False},
                        "eval": {c: {"success_rate": v} for c in cn}})
    return out


def test_summary_separates_success_from_advantage_and_skips_ambiguous():
    proto = nv.load_protocol()
    refs = {"ctx_memory": {"conflict_pulse": {"success_rate": 1.0}, "conflict_hidden": {"success_rate": 0.5}}}
    same = nv.summarize_v2(_fake_models(proto, [0.8, 0.82, 0.78, 0.81, 0.79], [0.8, 0.82, 0.78, 0.81, 0.79]),
                           proto, refs)
    assert same["question_1_task_success"]["connectome_pilot_mb"]["estimate"] == pytest.approx(0.8)
    assert same["question_1_task_success"]["connectome_pilot_mb"]["fraction_of_scripted_oracle"] == pytest.approx(0.8)
    # high absolute success but zero difference vs null => NOT an advantage
    assert "no advantage" in same["question_2_anatomical_advantage"]["verdict"]
    assert "conflict_hidden" not in same["question_2_anatomical_advantage"]["contrasts_exploratory"]["null_v2_swap"]
    better = nv.summarize_v2(_fake_models(proto, [0.9, 0.91, 0.92, 0.9, 0.93], [0.5, 0.52, 0.51, 0.5, 0.49]),
                             proto, refs)
    assert "supported" in better["question_2_anatomical_advantage"]["verdict"]
    worse = nv.summarize_v2(_fake_models(proto, [0.5, 0.52, 0.51, 0.5, 0.49], [0.9, 0.91, 0.92, 0.9, 0.93]),
                            proto, refs)
    assert "worse" in worse["question_2_anatomical_advantage"]["verdict"]


def test_protocol_freeze_detects_edits(tmp_path):
    (tmp_path / "configs/experiments").mkdir(parents=True)
    src = nv.load_protocol()
    import yaml

    p = tmp_path / nv.PROTOCOL_REL
    p.write_text(yaml.safe_dump(src))
    nv.freeze_protocol(tmp_path)
    assert nv.verify_protocol(tmp_path)["match"]
    src["training"]["tbptt_steps"] = 1
    p.write_text(yaml.safe_dump(src))
    assert not nv.verify_protocol(tmp_path)["match"]


def test_v1_protocol_untouched_by_v2():
    v = nv1.verify_protocol()
    assert v["frozen"] and v["match"]  # v1 hash still matches => v1 config unchanged
