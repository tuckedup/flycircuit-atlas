"""Statistics respecting the unit of independence, plus continual-learning metrics (spec §5, §12).

Independent units are trained-model replicates (seeds) -- never frames/trials within one model.
Nested designs (null graph -> training seeds) use a two-stage cluster bootstrap.

Task matrix convention: ``R[i, j]`` = held-out score (fraction in [0,1]) on task ``j`` after
learning phase ``i`` (phase i trains task i). T tasks, T phases.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(values, stat=np.mean, n_boot: int = 10000, ci: float = 0.95, seed: int = 0) -> dict:
    """Percentile bootstrap over independent units (one value per trained model)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return {"estimate": float(stat(v)) if n else float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n_units": n, "note": "need >=2 independent units for an interval"}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.apply_along_axis(stat, 1, v[idx])
    a = (1 - ci) / 2
    return {"estimate": float(stat(v)), "lo": float(np.quantile(boots, a)),
            "hi": float(np.quantile(boots, 1 - a)), "n_units": n, "ci": ci, "n_boot": n_boot,
            "note": "small-n percentile interval is exploratory and potentially unstable" if n < 10 else ""}


def paired_bootstrap_diff(a, b, n_boot: int = 10000, ci: float = 0.95, seed: int = 0) -> dict:
    """Bootstrap of mean(a-b) over paired independent units (same seed/split index)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have equal shape")
    return bootstrap_ci(a - b, np.mean, n_boot, ci, seed)


def nested_bootstrap(groups: dict, n_boot: int = 5000, ci: float = 0.95, seed: int = 0) -> dict:
    """Two-stage cluster bootstrap. ``groups``: {graph_id: [score per training seed]}.
    Resample graphs, then seeds within each sampled graph; statistic = mean of graph means."""
    keys = list(groups)
    arrs = [np.asarray(groups[k], float) for k in keys]
    if len(keys) < 2:
        return {"estimate": float(np.mean(arrs[0])) if arrs else float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n_graphs": len(keys), "note": "need >=2 graphs"}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        gi = rng.integers(0, len(keys), size=len(keys))
        means = [arrs[g][rng.integers(0, len(arrs[g]), size=len(arrs[g]))].mean() for g in gi]
        boots[b] = np.mean(means)
    a = (1 - ci) / 2
    est = float(np.mean([x.mean() for x in arrs]))
    return {"estimate": est, "lo": float(np.quantile(boots, a)), "hi": float(np.quantile(boots, 1 - a)),
            "n_graphs": len(keys), "seeds_per_graph": [len(x) for x in arrs], "ci": ci,
            "design": "graph-then-seed cluster bootstrap"}


# ---- continual learning ------------------------------------------------------------------
def _check_matrix(R) -> np.ndarray:
    R = np.asarray(R, float)
    if R.ndim != 2 or R.shape[0] != R.shape[1] or R.shape[0] < 2:
        raise ValueError("task matrix must be square with T>=2")
    return R


def forgetting_per_task(R, negative_policy: str = "keep") -> np.ndarray:
    """f_j = max_{l in [j, T-2]} R[l, j] - R[T-1, j] for j < T-1 (final task excluded: it has no
    later learning exposure). ``negative_policy`` (predeclare!): 'keep' allows negative values
    (net improvement); 'clip' sets negatives to 0."""
    if negative_policy not in ("keep", "clip"):
        raise ValueError("negative_policy must be 'keep' or 'clip'")
    R = _check_matrix(R)
    t = R.shape[0]
    f = np.array([R[j:t - 1, j].max() - R[t - 1, j] for j in range(t - 1)])
    return np.maximum(f, 0.0) if negative_policy == "clip" else f


def average_forgetting(R, negative_policy: str = "keep") -> float:
    return float(forgetting_per_task(R, negative_policy).mean())


def backward_transfer(R) -> float:
    """BWT = mean_{j<T-1} (R[T-1, j] - R[j, j]). Negative = forgetting."""
    R = _check_matrix(R)
    t = R.shape[0]
    return float(np.mean([R[t - 1, j] - R[j, j] for j in range(t - 1)]))


def forward_transfer(R, reference) -> float:
    """FWT = mean_{j>=1} (R[j-1, j] - reference[j]); ``reference[j]`` = score on task j of a
    reference model with no training on previous tasks. Required: raises if absent."""
    if reference is None:
        raise ValueError("forward transfer requires per-task reference measurements")
    R = _check_matrix(R)
    ref = np.asarray(reference, float)
    if ref.shape != (R.shape[0],):
        raise ValueError("reference must have one score per task")
    return float(np.mean([R[j - 1, j] - ref[j] for j in range(1, R.shape[0])]))


def forgetting_improvement(baseline_forgetting: float, connectome_forgetting: float,
                           target_abs: float = 0.07) -> dict:
    """improvement = baseline forgetting - connectome forgetting (fractions).
    Relative change = improvement / baseline_forgetting (denominator = baseline); undefined (None)
    when baseline forgetting is 0."""
    absd = baseline_forgetting - connectome_forgetting
    rel = absd / baseline_forgetting if baseline_forgetting != 0 else None
    return {"absolute": absd, "percentage_points": 100.0 * absd, "relative": rel,
            "relative_denominator": "baseline_forgetting",
            "meets_target_abs": bool(absd >= target_abs), "target_abs": target_abs}


def target_status(measured: float | None, threshold: float, higher_is_better: bool = True,
                  exploratory: bool = True, blocked: bool = False) -> str:
    """Status vocabulary: not run | blocked | exploratory | evaluated/met | evaluated/missed."""
    if blocked:
        return "blocked"
    if measured is None:
        return "not run"
    if exploratory:
        return "exploratory"
    ok = measured >= threshold if higher_is_better else measured <= threshold
    return "evaluated/met" if ok else "evaluated/missed"
