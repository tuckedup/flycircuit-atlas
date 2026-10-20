"""Contracts of study v3: equal search budgets, no test access outside confirm, seed hygiene,
stratified masks, crossed bootstrap, freeze-hash scope. Offline, no flygym, no real data."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from flycircuit.bodies import stages_v3 as B
from flycircuit.eval import learning_v3 as L3


# ---------------------------------------------------------------- body: stratified masks
@pytest.mark.parametrize("n", [96, 108, 120, 144])
def test_balanced_masks_exact_channel_rates(n):
    m = B.balanced_masks(n, 0.25, seed=3)
    assert m.shape == (n, 10)
    assert (m.sum(0) == n // 4).all(), "every channel dropped in exactly 25% of episodes"
    assert set(np.unique(m.sum(1))) <= {2, 3}
    assert m.sum() == n * 10 // 4
    assert np.array_equal(m, B.balanced_masks(n, 0.25, seed=3)), "deterministic given seed"
    assert not np.array_equal(m, B.balanced_masks(n, 0.25, seed=4))


def test_balanced_masks_rejects_unbalanceable_n():
    with pytest.raises(ValueError):
        B.balanced_masks(50, 0.25)   # 12.5 episodes per channel


def test_eval_episodes_carry_explicit_masks_and_fresh_indices():
    m = B.balanced_masks(96, 0.25, seed=1)
    eps = B.eval_episodes("test", 5000, m, "scripted")
    assert eps[0]["idx"] == 5000 and eps[-1]["idx"] == 5095
    assert all(len(e["mask"]) == 10 for e in eps)
    assert len({e["key"] for e in eps}) == 96
    assert min(e["idx"] for e in eps) >= 5000, "no v1 test index (0-23) is reused"


# ---------------------------------------------------------------- body: crossed bootstrap
def test_crossed_bootstrap_includes_episode_uncertainty():
    rng = np.random.default_rng(0)
    # 5 models, all identical on the same 24 episodes: model-only bootstrap would give a zero-width interval
    ep = (rng.random(24) < 0.6).astype(float)
    mat = np.tile(ep, (5, 1))
    est, lo, hi, _, _ = B.crossed_boot([mat], lambda m: float(np.mean(m[0])), n_boot=500, seed=1)
    assert est == pytest.approx(ep.mean())
    assert hi - lo > 0.1, "episode sampling must widen the interval"
    assert lo <= est <= hi


def test_crossed_bootstrap_paired_difference_of_equal_models_covers_zero():
    rng = np.random.default_rng(1)
    a = (rng.random((5, 120)) < 0.5).astype(float)
    est, lo, hi, p, _ = B.crossed_boot([a, a.copy()], lambda m: float(np.mean(m[0]) - np.mean(m[1])),
                                       n_boot=300, seed=2)
    assert est == 0.0 and lo == 0.0 and hi == 0.0 and p == 1.0


def test_holm_is_monotone_and_bounded():
    h = B.holm({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5})
    assert h["a"] == pytest.approx(0.004)
    vals = [h[k] for k in ("a", "b", "c", "d")]
    assert vals == sorted(vals) and max(vals) <= 1.0


# ---------------------------------------------------------------- body: freeze scope
def test_train_hash_ignores_sample_size_but_eval_hash_does_not(monkeypatch):
    p = B.proto()
    monkeypatch.setattr(B, "proto", lambda: p)
    h_train0 = B.train_hash()
    p96 = B.effective_eval_protocol({"n_test_episodes": 96})
    p120 = B.effective_eval_protocol({"n_test_episodes": 120})
    assert B.train_hash() == h_train0, "deriving N must not invalidate the training freeze"
    assert p['evaluation']['n_test_episodes'] is None, "raw YAML must stay immutable"
    assert B.effective_eval_hash(p96) != B.effective_eval_hash(p120)


def test_body_seeds_disjoint_from_v1():
    p = B.proto()
    mine = set(p["seeds"]["train"]) | set(p["seeds"]["search"])
    assert not (mine & B.FORBIDDEN_SEEDS)


# ---------------------------------------------------------------- learning: seeds, budgets, test guard
def test_learning_v3_seed_hygiene_passes_and_detects_overlap():
    proto = L3.load_protocol()
    L3.check_seed_hygiene(proto)
    bad = dict(proto, seeds=dict(proto["seeds"], confirmatory=[100201] + proto["seeds"]["confirmatory"][1:]))
    with pytest.raises(ValueError):
        L3.check_seed_hygiene(bad)


def test_run_one_v3_refuses_test_split_outside_confirm():
    proto = L3.load_protocol()
    with pytest.raises(RuntimeError):
        L3.run_one_v3(proto, None, {}, "bptt", "connectome", "std", 0.001, 1, "sgd", None,
                      ("val", "test"), ("association",), "search")


def _fake_factory(calls):
    def fake(proto, base, packs, arm, family, mode, lr, seed, optimizer, decoder_lr, splits, protocols, stage):
        calls.append({"arm": arm, "family": family, "optimizer": optimizer, "lr": lr, "dlr": decoder_lr,
                      "seed": seed, "splits": tuple(splits)})
        peak = {"sgd": 3e-5, "rmsprop": 1e-5}.get(optimizer, 1e-3) if arm.startswith("plast") else 1e-3
        score = 0.9 - 0.02 * (np.log(lr / peak)) ** 2
        if decoder_lr is not None:
            score -= 0.005 * (np.log(decoder_lr / 3e-3)) ** 2
        return {"arm": arm, "family": family, "mode": mode, "lr": lr, "seed": seed, "optimizer": optimizer,
                "decoder_lr": decoder_lr, "status": "ok", "elapsed_s": 0.0,
                "continual": {"final_mean_acc_val": float(score)}}
    return fake


def test_search_gives_every_kind_exactly_12_evaluations_on_validation_only(monkeypatch, tmp_path):
    # This unit isolates the search allocator. The standalone resume fixtures
    # exercise the real source/input manifests and ledger with an artificial repo.
    calls: list[dict] = []
    monkeypatch.setattr(L3, "run_one_v3", _fake_factory(calls))
    def folder(s):
        d = tmp_path / s
        d.mkdir(exist_ok=True)
        return d
    def read(s):
        p = folder(s) / 'records.jsonl'
        return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
    def append(s, record):
        with (folder(s) / 'records.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
    monkeypatch.setattr(L3, "stage_dir", folder)
    monkeypatch.setattr(L3, "ensure_stage_manifest", lambda *a: None)
    monkeypatch.setattr(L3, "read_records", read)
    monkeypatch.setattr(L3, "_append", append)
    monkeypatch.setattr(L3.I, "atomic_json", lambda p, v: p.write_text(json.dumps(v, sort_keys=True)))
    proto = L3.load_protocol()
    sel = L3.run_search(proto, None, {}, "search", log=lambda *_: None)
    assert set(sel) == {f"{a}|{k}" for a, k in L3.KINDS}
    for name, e in sel.items():
        assert e["n_evals"] == 12, name
        assert e["complete"]
    assert len(calls) == 288, "12 kinds x 12 DISTINCT settings x 2 validation seeds"
    assert set(Counter((c['arm'], 'null' if c['family'].startswith('null') else c['family'])
                       for c in calls).values()) == {24}
    full_keys = {(c['arm'], c['family'], c['optimizer'], c['lr'], c['dlr'], c['seed']) for c in calls}
    assert len(full_keys) == 288
    assert all(c["splits"] == ("val",) for c in calls), "search must never touch the test split"
    # exactly 2 validation seeds per configuration, from the declared disjoint search seeds
    assert {c["seed"] for c in calls} == set(proto["search"]["seeds"])
    # plast_dec searched its OWN decoder lr (not just inherited): distinct decoder values appear
    dlrs = {c["dlr"] for c in calls if c["arm"] == "plast_dec"}
    assert len(dlrs) >= 6
    pd = sel["plast_dec|connectome"]
    assert len(pd["scores_plastic_lr"]) == 6 and len(pd["scores_decoder_lr"]) == 6
    assert pd["decoder_lr"] is not None and pd["optimizer"] == sel["plast_fixed|connectome"]["optimizer"]
    # fixed-decoder plastic arms split 12 evaluations across both optimizer options
    pf = sel["plast_fixed|connectome"]
    assert {o: v["n_evals"] for o, v in pf["by_optimizer"].items()} == {"sgd": 6, "rmsprop": 6}
    # resumable: a second call performs no new run
    n = len(calls)
    L3.run_search(proto, None, {}, "search", log=lambda *_: None)
    assert len(calls) == n


def test_conditions_include_all_required_comparators_and_factorial_nulls():
    proto = L3.load_protocol()
    conds = L3.conditions(proto)
    fams = {(a, f, m) for a, f, m, _ in conds}
    for need in [("bptt", "connectome", "std"), ("bptt", "gru", "std"), ("bptt", "gru_c", "std"),
                 ("bptt", "encdec", "std"), ("decoder_only", "kc_static", "std"),
                 ("plast_dec", "connectome", "std"), ("plast_modoff", "connectome", "std")]:
        assert need in fams
    seeds = proto["seeds"]["confirmatory"]
    sel = {f"{a}|{k}": {"optimizer": "sgd", "lr": 0.001, "decoder_lr": 0.003 if a == "plast_dec" else None}
           for a, k in L3.KINDS}
    sel["plast_fixed|connectome"] = {"optimizer": "sgd", "lr": 1e-5, "decoder_lr": None}
    jobs = L3.build_jobs(proto, sel, seeds)
    nulls_bptt = [j for j in jobs if j[0] == "bptt" and j[1].startswith("null") and j[2] == "std"]
    assert len(nulls_bptt) == proto["seeds"]["null_graphs"] * len(seeds)
    assert len(seeds) == 8 and len({j for j in jobs}) == len(jobs)
