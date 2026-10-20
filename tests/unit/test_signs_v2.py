import numpy as np
import pytest

from flycircuit.graph.signs import (
    SignPolicy,
    assign_signs,
    parse_known_nt,
    policy_grid,
)


@pytest.mark.parametrize(
    "raw,expect",
    [
        ("acetylcholine; sNPF; acetylcholine, sNPF", ("acetylcholine", "single")),
        ("acetylcholine; acetylcholine", ("acetylcholine", "single")),
        ("gaba-negative", (None, "none_classical")),
        ("gaba, nitric oxide", ("gaba", "single")),
        ("glutamate, gaba; glutamate", (None, "ambiguous")),
        ("acetylcholine-negative, glutamate-negative, gaba-negative, dopamine-negative", (None, "none_classical")),
        ("acetylcholine; gaba-negative", ("acetylcholine", "single")),
        (None, (None, "missing")),
        (float("nan"), (None, "missing")),
        ("", (None, "missing")),
    ],
)
def test_parse_known_nt(raw, expect):
    assert parse_known_nt(raw) == expect


def _arr():
    # Kenyon-like (known ACh, predicted dopamine), MBON glutamate, DAN dopamine, unknown, histamine
    known = ["acetylcholine; acetylcholine", "glutamate", "dopamine", None, "histamine", "gaba-negative"]
    top = ["dopamine", "glutamate", "dopamine", "gaba", "acetylcholine", "acetylcholine"]
    conf = [0.64, 0.9, 0.7, 0.3, 0.8, 0.9]
    cls = ["Kenyon_Cell", "MBON", "DAN", "x", "x", "x"]
    return known, top, conf, cls


def test_known_beats_predicted_and_kc_is_cholinergic():
    a = assign_signs(*_arr())
    assert a.sign[0] == 1 and a.evidence[0] == "known"  # not dopamine despite top_nt
    assert a.sign[5] == 1 and a.evidence[5] == "predicted"  # '-negative' is not evidence


def test_monoamines_are_modulatory_not_plain_excitatory():
    a = assign_signs(*_arr())
    assert a.modulatory[2] and a.sign[2] == 0
    legacy = assign_signs(*_arr(), SignPolicy(monoamine_mode="legacy_excitatory"))
    assert legacy.sign[2] == 1 and legacy.modulatory[2]


def test_low_confidence_prediction_is_unknown():
    a = assign_signs(*_arr())
    assert a.evidence[3] == "unknown" and a.sign[3] == 0


def test_predicted_only_reproduces_dopamine_kc_artifact():
    a = assign_signs(*_arr(), SignPolicy(name="predicted_only"))
    assert a.nt[0] == "dopamine" and a.modulatory[0] and a.sign[0] == 0
    fixed = assign_signs(*_arr(), SignPolicy(name="predicted_only", kc_literature_prior=True))
    assert fixed.sign[0] == 1 and fixed.evidence[0] == "prior"


def test_glutamate_variants_and_histamine_prior():
    base = assign_signs(*_arr())
    assert base.sign[1] == -1 and base.sign[4] == -1 and base.evidence[4] == "prior"
    assert assign_signs(*_arr(), SignPolicy(glutamate="excitatory")).sign[1] == 1
    assert assign_signs(*_arr(), SignPolicy(glutamate="unknown")).sign[1] == 0


def test_unknown_fill_policies_and_random_is_seeded():
    k = _arr()
    ex = assign_signs(*k, SignPolicy(unknown_fill="excitatory"))
    assert ex.sign[3] == 1 and ex.evidence[3] == "unknown"
    r1 = assign_signs(*k, SignPolicy(unknown_fill="random_fixed", fill_seed=3))
    r2 = assign_signs(*k, SignPolicy(unknown_fill="random_fixed", fill_seed=3))
    assert np.array_equal(r1.sign, r2.sign) and r1.hash() == r2.hash()
    assert abs(int(r1.sign[3])) == 1


def test_policy_validation_and_grid():
    with pytest.raises(ValueError):
        SignPolicy(glutamate="maybe")
    grid = policy_grid()
    assert {"v2_default", "glu_excitatory", "glu_unknown", "unknown_random_fixed"} <= set(grid)
    tags = {p.tag() for p in grid.values()}
    assert len(tags) == len(grid)  # every variant is distinguishable in provenance
