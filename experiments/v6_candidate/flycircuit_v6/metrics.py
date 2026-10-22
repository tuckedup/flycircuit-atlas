"""Standard-library metrics. Independent units are model seeds, never episodes."""

from __future__ import annotations

import math
import random
from statistics import fmean, stdev


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a numeric probability")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("probabilities must be finite and in [0,1]")
    return value


def continual(matrix):
    """R[i][j] is accuracy on task j after phase i; future-task cells are unused."""
    n = len(matrix)
    if n < 2 or any(len(row) != n for row in matrix):
        raise ValueError("need a square task matrix of order >=2")
    # Reject nonfinite future cells too: raw matrix artifacts have a strict schema.
    r = [[probability(x) for x in row] for row in matrix]
    losses = [max(r[i][j] for i in range(j, n - 1)) - r[-1][j] for j in range(n - 1)]
    return {
        "final_average_accuracy": fmean(r[-1]),
        "mean_seen_task_accuracy": fmean(fmean(r[i][: i + 1]) for i in range(n)),
        "learned_task_accuracy": fmean(r[i][i] for i in range(n)),
        "forgetting_keep": fmean(losses),
        "forgetting_clip": fmean(max(0.0, x) for x in losses),
        "backward_transfer": fmean(r[-1][j] - r[j][j] for j in range(n - 1)),
        "forward_transfer": None,
        "forward_transfer_note": "requires separate reference models; not inferred from R",
    }


def robustness_curve(points, required_rates):
    """Trapezoidal success AUC on a COMPLETE predeclared grid from 0 to 1.

    AUC depends on this grid and mask distribution. Neither is interchangeable
    with another paper's sensory modalities, environment or success rule.
    """
    grid = [probability(x) for x in required_rates]
    if len(grid) < 2 or grid != sorted(set(grid)) or grid[0] != 0 or grid[-1] != 1:
        raise ValueError("rates must be strictly increasing and include 0 and 1")
    pairs = [(probability(x), probability(y)) for x, y in points]
    if sorted(x for x, _ in pairs) != grid:
        raise ValueError("curve has missing, extra or duplicate dropout rates")
    values = dict(pairs)
    auc = sum(
        (grid[i + 1] - grid[i]) * (values[grid[i + 1]] + values[grid[i]]) / 2
        for i in range(len(grid) - 1)
    )
    clean = values[0.0]
    return {
        "robustness_auc": auc,
        "clean_success": clean,
        "success_at_full_dropout": values[1.0],
        "worst_declared_rate_success": min(values.values()),
        "retained_auc_fraction": auc / clean if clean else None,
        "rates": grid,
    }


def spl(success, path_length, shortest_path_length):
    """SPL requires a verified shortest feasible path, not a straight-line guess."""
    if not isinstance(success, bool):
        raise ValueError("success must be boolean")
    if shortest_path_length is None:
        raise ValueError(
            "shortest feasible path is unavailable; report SPL as not measured"
        )
    actual, shortest = float(path_length), float(shortest_path_length)
    if (
        not math.isfinite(actual)
        or actual < 0
        or not math.isfinite(shortest)
        or shortest <= 0
    ):
        raise ValueError(
            "actual path must be nonnegative and shortest path positive; both finite"
        )
    if success and actual + 1e-9 < shortest:
        raise ValueError(
            "successful path cannot be shorter than the certified shortest path"
        )
    return (shortest / max(shortest, actual)) if success else 0.0


def paired_seed_difference(candidate, baseline, draws=2000, seed=0):
    """Mean candidate-baseline and percentile bootstrap across paired MODEL seeds.

    Small-n intervals are exploratory; no automatic SOTA/significance verdict.
    """
    if not candidate or set(candidate) != set(baseline):
        raise ValueError("exactly the same nonempty model-seed identities are required")
    if (
        isinstance(draws, bool)
        or not isinstance(draws, int)
        or not 100 <= draws <= 100000
    ):
        raise ValueError("bootstrap draws must be an integer in [100,100000]")
    ids = sorted(candidate)
    diffs = []
    for key in ids:
        values = [float(candidate[key]), float(baseline[key])]
        if not all(math.isfinite(x) for x in values):
            raise ValueError("nonfinite seed metric")
        diffs.append(values[0] - values[1])
    result = {
        "mean_difference": fmean(diffs),
        "n_model_seeds": len(ids),
        "paired_differences": dict(zip(ids, diffs, strict=True)),
        "se": stdev(diffs) / math.sqrt(len(ids)) if len(ids) > 1 else None,
        "ci95": None,
        "inference": "exploratory paired-model-seed bootstrap",
    }
    if len(ids) >= 2:
        rng = random.Random(seed)
        boots = sorted(fmean(rng.choice(diffs) for _ in diffs) for _ in range(draws))
        result["ci95"] = [
            boots[int(0.025 * (draws - 1))],
            boots[int(0.975 * (draws - 1))],
        ]
    result["small_n_warning"] = len(ids) < 10
    return result


def episode_success(records, expected_ids):
    """One model, one frozen condition. Execution failures remain in denominator."""
    expected = set(expected_ids)
    if not expected or len(expected) != len(expected_ids):
        raise ValueError("expected IDs must be nonempty and unique")
    seen, successes, failures = set(), 0, 0
    for row in records:
        identity = row["episode_id"]
        if identity in seen or identity not in expected:
            raise ValueError("duplicate or unexpected episode identity")
        seen.add(identity)
        if row["status"] != "ok":
            failures += 1
        else:
            if not isinstance(row["success"], bool):
                raise ValueError("successful execution requires a boolean outcome")
            successes += int(row["success"])
    if seen != expected:
        raise ValueError("incomplete episode matrix; do not shrink the denominator")
    return {
        "successes": successes,
        "n_episodes": len(expected),
        "execution_failures": failures,
        "success_rate": successes / len(expected),
    }
