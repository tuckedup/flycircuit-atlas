"""Effect estimation across independent runs + multiplicity correction.

Correction is applied to a defined FAMILY of p-values only (never to CIs or effect sizes).
Uncertainty here is across independent stimulus/initial-state/noise draws ("runs") of ONE fixed
model instance. It is not uncertainty over trained models or biological replicates.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from flycircuit.eval.stats import bootstrap_ci


def effect_summary(values, seed: int = 0, n_boot: int = 4000) -> dict:
    v = np.asarray(values, dtype=float)
    n = len(v)
    out = {"mean": float(v.mean()) if n else float("nan"),
           "sd": float(v.std(ddof=1)) if n > 1 else float("nan"), "n_runs": n}
    ci = bootstrap_ci(v, np.mean, n_boot=n_boot, seed=seed)
    out["ci_lo"], out["ci_hi"] = ci["lo"], ci["hi"]
    out["ci_note"] = ci.get("note", "")
    if n < 3:
        out["p_value"], out["p_note"] = float("nan"), "need >=3 runs for a test"
    elif np.allclose(v, v[0]):
        # zero variance: t-test undefined. Exactly-zero effect -> p=1; constant nonzero -> flagged.
        out["p_value"] = 1.0 if abs(v[0]) < 1e-12 else float("nan")
        out["p_note"] = "zero variance across runs"
    else:
        out["p_value"] = float(sps.ttest_1samp(v, 0.0).pvalue)
        out["p_note"] = "two-sided one-sample t-test vs 0 (exploratory, assumes ~normal)"
    return out


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values; NaN entries are excluded from the family size."""
    p = np.asarray(p, dtype=float)
    adj = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return adj
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx], kind="stable")]
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def benjamini_hochberg(p) -> np.ndarray:
    """BH step-up adjusted p-values (FDR); NaN entries excluded."""
    p = np.asarray(p, dtype=float)
    adj = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return adj
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx], kind="stable")]
    run = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        run = min(run, p[i] * m / (rank + 1))
        adj[i] = min(1.0, run)
    return adj


def adjust_families(families, p) -> tuple[np.ndarray, np.ndarray]:
    """Holm and BH within each family label (array of labels aligned to p)."""
    fam = np.asarray(families)
    p = np.asarray(p, dtype=float)
    h, b = np.full_like(p, np.nan), np.full_like(p, np.nan)
    for f in np.unique(fam):
        m = fam == f
        h[m], b[m] = holm(p[m]), benjamini_hochberg(p[m])
    return h, b
