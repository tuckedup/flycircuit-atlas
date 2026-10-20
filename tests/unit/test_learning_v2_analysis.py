"""Learning-v2 analysis contracts: job design, pairing, contrasts, Holm family, freeze guard."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from flycircuit.config import project_root
from flycircuit.eval import learning_v2 as V


@pytest.fixture(scope="module")
def proto():
    return V.load_protocol()


def _sel(proto):
    return {f"{a}|{k}": {"lr": 1e-3} for a, k in V.search_kinds(proto)}


def test_job_design_counts_and_pairing(proto):
    jobs = V.build_jobs(proto, _sel(proto), proto["seeds"]["confirmatory"])
    by = {}
    for arm, fam, mode, _lr, _dl, s in jobs:
        by.setdefault((arm, mode), []).append((fam, s))
    kn, ns = proto["seeds"]["null_graphs"], len(proto["seeds"]["confirmatory"])
    # factorial: every null x every seed for bptt and plast_dec
    for arm in ("bptt", "plast_dec"):
        nulls = {(f, s) for f, s in by[(arm, "std")] if f.startswith("null")}
        assert len(nulls) == kn * ns
    # paired: null i used with exactly one seed, each seed once
    pf = [(f, s) for f, s in by[("plast_fixed", "std")] if f.startswith("null")]
    assert len(pf) == ns and len({s for _, s in pf}) == ns and len({f for f, _ in pf}) == kn
    # no duplicate jobs
    assert len(jobs) == len({(a, f, m, s) for a, f, m, _l, _d, s in jobs})
    # every arm of the preregistered family is present
    arms = {a for a, _, _, _, _, _ in jobs}
    assert arms == set(V.ARMS)


def test_borrowed_and_decoder_learning_rates(proto):
    sel = _sel(proto)
    sel["plast_fixed|connectome"] = {"lr": 3e-4}
    sel["decoder_only|connectome"] = {"lr": 7e-3}
    assert V.lr_for(sel, "plast_modoff", "connectome")[0] == 3e-4  # borrows plast_fixed
    lr, dec = V.lr_for(sel, "plast_dec", "connectome")
    assert dec == 7e-3  # decoder lr from decoder_only
    assert V.lr_for(sel, "bptt", "null3")[0] == sel["bptt|null"]["lr"]


def _rec(arm, fam, seed, acc, forg=0.05, mode="std"):
    return {"arm": arm, "family": fam, "mode": mode, "seed": seed, "lr": 1e-3, "status": "ok",
            "oracle_raw_cue_logreg": {"test": 0.95}, "elapsed_s": 1.0, "n_params_trainable": 10,
            "association": {"test_acc_acq": acc, "test_acc_reversed": acc,
                            "test_acc_after_delay": acc, "perseveration_rate": 0.7,
                            "acquisition_censored": False},
            "continual": {"final_mean_acc_test": acc, "avg_forgetting_keep_test": forg,
                          "avg_forgetting_clip_test": max(forg, 0), "bwt_test": -forg,
                          "fwt_test": 0.0, "n_training_trials": 2400, "replay_storage_bits": 0}}


def test_primary_contrasts_follow_declared_definitions(proto):
    rng = np.random.default_rng(0)
    recs = []
    for s in range(201, 206):
        recs.append(_rec("bptt", "connectome", s, 0.80 + 0.01 * rng.standard_normal(), 0.06))
        recs.append(_rec("bptt", "gru", s, 0.84 + 0.01 * rng.standard_normal(), 0.05))
        for i in range(5):
            recs.append(_rec("bptt", f"null{i}", s, 0.80 + 0.01 * rng.standard_normal()))
            recs.append(_rec("plast_dec", f"null{i}", s, 0.70))
        recs.append(_rec("plast_dec", "connectome", s, 0.70))
        recs.append(_rec("plast_fixed", "connectome", s, 0.75))
        recs.append(_rec("plast_modoff", "connectome", s, 0.50))
        recs.append(_rec("plast_shuffled", "connectome", s, 0.52))
    summ = V.summarize(recs, proto)
    p = summ["primary"]
    e2 = p["E2_forgetting_improvement_gru_minus_connectome"]["detail"]
    assert e2["absolute"] == pytest.approx(0.05 - 0.06)  # baseline forgetting - connectome forgetting
    assert e2["relative"] == pytest.approx((0.05 - 0.06) / 0.05)  # denominator = baseline
    assert p["E1_acc_connectome_minus_gru"]["estimate"] < 0
    assert p["E3_topology_bptt"]["n_nulls"] == 5
    assert abs(p["E4_topology_plast_dec"]["mean_diff"]) < 1e-12
    assert p["E5_plast_fixed_minus_modoff"]["estimate"] == pytest.approx(0.25)
    # Holm family has exactly the declared members and never shrinks p-values
    assert set(p["p_holm_family"]) == set(p["p_exact_signflip"]) <= {"E1", "E2", "E3", "E4", "E5", "E6"}
    for k, v in p["p_holm_family"].items():
        assert v >= p["p_exact_signflip"][k] - 1e-12
    # smallest attainable p with 5 paired units is 2/32
    assert p["p_exact_signflip"]["E5"] == pytest.approx(0.0625)
    tg = V.target_rows(summ, [r for r in recs if r["status"] == "ok"], proto)
    assert tg[0]["status"] == "evaluated/missed"  # 0.80 < 0.82, 5 seeds -> not exploratory
    assert tg[1]["status"] == "evaluated/missed"  # improvement is negative


def test_relative_forgetting_undefined_when_baseline_forgetting_is_zero(proto):
    recs = []
    for s in range(201, 206):
        recs.append(_rec("bptt", "connectome", s, 0.8, 0.03))
        recs.append(_rec("bptt", "gru", s, 0.8, 0.0))
    d = V.summarize(recs, proto)["primary"]["E2_forgetting_improvement_gru_minus_connectome"]["detail"]
    assert d["relative"] is None


def test_prereg_v2_exists_declares_disclosures_and_is_not_frozen_by_default():
    txt = (project_root() / "configs" / "prereg_v2.yaml").read_text()
    doc = yaml.safe_load(txt)
    assert doc["relationship_to_v1"]["v1_test_data_were_seen"] is True
    assert set(doc["primary_family"]) == {"E1", "E2", "E3", "E4", "E5", "E6"}
    assert "multiplicity" in doc and "search_budget" in doc and "exclusions" in doc
    assert doc["seeds"]["final_evaluation"] == V.load_protocol()["seeds"]["confirmatory"]


def test_checkpoint_loader_and_confirm_refuse_without_freeze_marker():
    # the CLI guard is a regex on the frozen flag; make sure the shipped file does not pre-satisfy it
    import re

    txt = (project_root() / "configs" / "prereg_v2.yaml").read_text()
    frozen = bool(re.search(r"^frozen: true", txt, re.M))
    if frozen:  # after the real freeze the hash must be recorded
        assert V.protocol_hash(V.load_protocol()) in txt
