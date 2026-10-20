"""PROPOSAL (not frozen, not used by v3): validation selection rule for a future prospective study.

Motivation (measured on v3 search records): the mean absolute difference between the two search seeds'
continual scores was 0.058; for bptt|connectome nine learning rates between 0.001 and 0.003 lay within
one standard error of the best, so ``argmax`` of noisy 2-seed means is close to arbitrary. v2 and v3 chose
lr 0.01 vs 0.00172 with the same objective. This module is pure Python/NumPy so it can be unit-tested
without training anything.

Conventions (must be identical for every arm, including GRU / encoder-decoder / zero-edge controls):
  * score = weighted mean of validation acquisition accuracy and validation continual final accuracy;
    both components are always reported next to the score;
  * among settings within one standard error of the best mean, choose the SMALLEST learning rate
    (conservative tie rule fixed in advance);
  * log-spaced refinement stops when the neighbouring rates differ by less than ``min_ratio``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def composite_score(acq: float, cont: float, w_acq: float = 0.5) -> float:
    if not (0.0 <= w_acq <= 1.0):
        raise ValueError("w_acq must be in [0, 1]")
    if not (math.isfinite(acq) and math.isfinite(cont)):
        return float("-inf")
    return w_acq * acq + (1.0 - w_acq) * cont


def mean_and_se(values: Sequence[float]) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    if v.size == 0 or not np.isfinite(v).all():
        return float("-inf"), float("inf")
    if v.size == 1:
        return float(v[0]), float("inf")      # one seed gives no uncertainty estimate
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(v.size))


def one_se_choice(table: Mapping[float, Sequence[float]], min_seeds: int = 4) -> dict:
    """``table``: lr -> per-seed composite scores. Returns the choice plus the evidence behind it."""
    if not table:
        raise ValueError("empty search table")
    short = {lr: len(v) for lr, v in table.items() if len(v) < min_seeds}
    if short:
        raise ValueError(f"need >= {min_seeds} search seeds per setting; got {short}")
    stats = {lr: mean_and_se(v) for lr, v in table.items()}
    best = max(sorted(stats), key=lambda k: (stats[k][0], -k))
    thr = stats[best][0] - stats[best][1]
    candidates = sorted(lr for lr, (m, _) in stats.items() if m >= thr)
    chosen = candidates[0]
    lrs = sorted(table)
    return {
        "chosen_lr": chosen, "best_lr": best, "best_mean": stats[best][0], "best_se": stats[best][1],
        "n_indistinguishable": len(candidates), "indistinguishable_lrs": candidates,
        "chosen_equals_best": chosen == best,
        "optimum_at_edge": best in (lrs[0], lrs[-1]) or chosen in (lrs[0], lrs[-1]),
        "stats": {repr(float(k)): {"mean": m, "se": s} for k, (m, s) in sorted(stats.items())},
    }


def two_group_simplicity(setting: tuple[float, float], base_lr: float) -> tuple[float, ...]:
    """Prospective conservative ordering for a two-learning-rate setting.

    Rates are normalized by the arm's v3 anchor.  First minimize the larger
    multiplier, then their geometric mean, with numeric rates as deterministic
    final ties.  This avoids silently treating ``core_lr`` as more important
    than ``decoder_lr``.
    """
    core, dec = map(float, setting)
    if not (math.isfinite(core) and math.isfinite(dec) and core > 0 and dec > 0 and base_lr > 0):
        raise ValueError("learning rates and base_lr must be positive and finite")
    rc, rd = core / base_lr, dec / base_lr
    return max(rc, rd), math.sqrt(rc * rd), core, dec


def one_se_choice_2d(
    table: Mapping[tuple[float, float], Sequence[float]], base_lr: float, min_seeds: int = 4
) -> dict:
    """One-SE selection with the predeclared two-group simplicity ordering."""
    if not table:
        raise ValueError("empty search table")
    short = {repr(k): len(v) for k, v in table.items() if len(v) != min_seeds}
    if short:
        raise ValueError(f"need exactly {min_seeds} search seeds per setting; got {short}")
    stats = {tuple(map(float, k)): mean_and_se(v) for k, v in table.items()}
    if any(not (math.isfinite(m) and math.isfinite(se)) for m, se in stats.values()):
        raise ValueError("all per-setting scores must be finite")
    best_mean = max(m for m, _ in stats.values())
    best = min((k for k, (m, _) in stats.items() if m == best_mean),
               key=lambda k: two_group_simplicity(k, base_lr))
    threshold = stats[best][0] - stats[best][1]
    candidates = [k for k, (m, _) in stats.items() if m >= threshold]
    chosen = min(candidates, key=lambda k: two_group_simplicity(k, base_lr))
    return {
        "chosen": chosen,
        "best": best,
        "best_mean": stats[best][0],
        "best_se": stats[best][1],
        "threshold": threshold,
        "n_indistinguishable": len(candidates),
        "indistinguishable": sorted(candidates, key=lambda k: two_group_simplicity(k, base_lr)),
        "chosen_equals_best": chosen == best,
        "stats": {f"{k[0]:.17g}|{k[1]:.17g}": {"mean": m, "se": se}
                  for k, (m, se) in sorted(stats.items())},
    }


def next_lr(scores: Mapping[float, float], min_ratio: float = 1.25, edge_factor: float = 3.0):
    """Next learning rate to try, or ``None`` when refinement would only split hairs below min_ratio."""
    lrs = sorted(scores)
    best = max(lrs, key=lambda k: (scores[k], -k))
    i = lrs.index(best)
    if i == 0:
        return best / edge_factor
    if i == len(lrs) - 1:
        return best * edge_factor
    nb = lrs[i - 1] if scores[lrs[i - 1]] >= scores[lrs[i + 1]] else lrs[i + 1]
    if max(best, nb) / min(best, nb) < min_ratio:
        return None
    return float(math.sqrt(best * nb))
