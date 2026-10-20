#!/usr/bin/env python3
"""Power / precision calculation for CONFIRMATION_PROTOCOL_V2_DRAFT.md. numpy only (no scipy). No training, no data access.

Test: one-sided t-test of H0: mu <= threshold, alpha .05 (the 95% lower t-bound must be >= threshold).
Power = P(lower bound >= threshold) for n seeds, true mean mu, true per-seed SD sigma.
Analytic: E over chi-square of 1-Phi(t_crit*sqrt(V/df) - delta); Monte-Carlo cross-check with the actual t statistic.
SD uncertainty: pilot sample SD (4 seeds, df=3); 90% chi-square interval; also power averaged over
sigma^2 = df*s^2/chi2_df (scaled inverse chi-square, flat-prior style; an assumption, stated).
Normal per-seed data assumed (acquisition is bounded <=1 and on a 64-item-like grid; ceiling effects ignored).
Run: python3 power_calc_v2.py   (writes power_calc_v2_output.json next to this file)
"""
import json
import math
from pathlib import Path

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz
RNG = np.random.default_rng(20261019)
NS = [8, 12, 16, 20, 24, 32, 40]
ALPHA = 0.05
CHI2_3_05, CHI2_3_95 = 0.351846, 7.814728      # chi-square df=3 quantiles (0.05, 0.95)

ACQ = dict(thr=0.82, pilot_sd=0.0691, means=[0.80, 0.82, 0.824, 0.85, 0.88])
GAP = dict(thr=0.07, pilot_sd=0.0273 * 2.0, means=[-0.03, 0.0, 0.03, 0.07, 0.10])  # SE .0273 x sqrt(4)


def phi(x):
    z = np.asarray(x, dtype=float) / math.sqrt(2.0)      # Abramowitz-Stegun 7.1.26 erf, |err| < 1.5e-7
    sgn = np.sign(z)
    a = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return 0.5 * (1.0 + sgn * y)


def chi2_pdf(v, df):
    return np.exp((df / 2 - 1) * np.log(v) - v / 2 - (df / 2) * math.log(2) - math.lgamma(df / 2))


def t_cdf(t, df):
    xs = np.linspace(-80, t, 40001)
    pdf = np.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi) * (1 + xs ** 2 / df) ** (-(df + 1) / 2)
    return float(_trapz(pdf, xs))


_TC = {}


def tcrit(df, alpha=ALPHA):
    key = (df, alpha)
    if key in _TC:
        return _TC[key]
    if df > 60:  # Cornish-Fisher expansion of the t quantile (accurate to ~1e-4 for df>60)
        zq = {0.05: 1.6448536, 0.025: 1.9599640}[alpha]
        _TC[key] = zq + (zq ** 3 + zq) / (4 * df) + (5 * zq ** 5 + 16 * zq ** 3 + 3 * zq) / (96 * df ** 2)
        return _TC[key]
    if key not in _TC:
        lo, hi = 0.0, 20.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if t_cdf(mid, df) < 1 - alpha:
                lo = mid
            else:
                hi = mid
        _TC[key] = (lo + hi) / 2
    return _TC[key]


def power_analytic(n, delta_raw, sigma, alpha=ALPHA):
    df = n - 1
    tc = tcrit(df, alpha)
    v = np.linspace(1e-6, df + 40 * math.sqrt(2 * df) + 20, 6001)
    d = math.sqrt(n) * delta_raw / sigma
    return float(_trapz(chi2_pdf(v, df) * (1 - phi(tc * np.sqrt(v / df) - d)), v))


def power_mc(n, delta_raw, sigma, reps=20000, alpha=ALPHA):
    x = RNG.normal(delta_raw, sigma, size=(reps, n))
    t = x.mean(1) / (x.std(1, ddof=1) / math.sqrt(n))
    return float((t > tcrit(n - 1, alpha)).mean())


def power_posterior_sd(n, delta_raw, s, reps=20000, alpha=ALPHA, df0=3):
    sig = np.sqrt(df0 * s * s / RNG.chisquare(df0, size=reps))
    z = RNG.normal(0.0, 1.0, size=(reps, n)) * sig[:, None] + delta_raw
    t = z.mean(1) / (z.std(1, ddof=1) / math.sqrt(n))
    return float((t > tcrit(n - 1, alpha)).mean())


def required_n(delta_raw, sigma, target=0.80, nmax=5000, alpha=ALPHA):
    if delta_raw <= 0:
        return None
    if power_analytic(nmax, delta_raw, sigma, alpha) < target:
        return None
    lo, hi = 3, nmax                      # power is monotone in n for delta>0
    while lo < hi:
        mid = (lo + hi) // 2
        if power_analytic(mid, delta_raw, sigma, alpha) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def block(spec):
    s = spec["pilot_sd"]
    sds = {"pilot_lo90": s * math.sqrt(3 / CHI2_3_95), "pilot": s, "pilot_hi90": s * math.sqrt(3 / CHI2_3_05),
           "sensitivity_0.10": 0.10}
    out = {"threshold": spec["thr"], "pilot_sd": s, "sd_levels": sds, "power": {}, "power_posterior_sd": {},
           "ci_halfwidth95_two_sided": {}}
    for sd_name, sd in sds.items():
        for mu in spec["means"]:
            for n in NS:
                out["power"][f"{sd_name}|mu={mu}|n={n}"] = {
                    "analytic": round(power_analytic(n, mu - spec["thr"], sd), 4),
                    "mc": round(power_mc(n, mu - spec["thr"], sd), 4)}
        for n in NS:
            out["ci_halfwidth95_two_sided"][f"{sd_name}|n={n}"] = round(tcrit(n - 1, 0.025) * sd / math.sqrt(n), 4)
    for mu in spec["means"]:
        for n in NS:
            out["power_posterior_sd"][f"mu={mu}|n={n}"] = round(power_posterior_sd(n, mu - spec["thr"], s), 4)
    return out


def main():
    res = {"alpha_one_sided": ALPHA, "ns": NS, "tcrit_95_onesided": {str(n): round(tcrit(n - 1), 4) for n in NS},
           "acquisition": block(ACQ), "gap": block(GAP)}
    req = {}
    for mu in (0.85, 0.824, 0.88):
        for sd_name, sd in res["acquisition"]["sd_levels"].items():
            req[f"acq mu={mu} {sd_name}(sd={sd:.4f})"] = required_n(mu - 0.82, sd)
    for mu in (0.10, 0.07, 0.03, 0.0, -0.033):
        for sd_name, sd in res["gap"]["sd_levels"].items():
            req[f"gap>=0.07 mu={mu} {sd_name}(sd={sd:.4f})"] = required_n(mu - 0.07, sd)
    for mu in (0.07, 0.03):
        for sd_name, sd in res["gap"]["sd_levels"].items():
            req[f"gap>0 (weaker claim) mu={mu} {sd_name}(sd={sd:.4f})"] = required_n(mu, sd)
    res["required_n_80pct"] = req
    Path(__file__).with_name("power_calc_v2_output.json").write_text(json.dumps(res, indent=1))

    for nm, key, spec in (("ACQUISITION vs 0.82", "acquisition", ACQ), ("GAP vs 0.07", "gap", GAP)):
        b = res[key]
        print(f"\n=== {nm}: P(95% one-sided t lower bound >= threshold); analytic (MC) ===")
        for sd_name, sd in b["sd_levels"].items():
            print(f"-- SD {sd_name} = {sd:.4f}")
            print("   mu \\ n  " + "  ".join(f"{n:>11d}" for n in NS))
            for mu in spec["means"]:
                row = [f"{b['power'][f'{sd_name}|mu={mu}|n={n}']['analytic']:.3f}({b['power'][f'{sd_name}|mu={mu}|n={n}']['mc']:.3f})" for n in NS]
                print(f"   {mu:>6}  " + "  ".join(f"{r:>11s}" for r in row))
        print("-- power averaged over pilot-SD uncertainty (MC)")
        for mu in spec["means"]:
            print(f"   {mu:>6}  " + "  ".join(f"{b['power_posterior_sd'][f'mu={mu}|n={n}']:>11.3f}" for n in NS))
        print("-- two-sided 95% CI half-width (precision only)")
        for sd_name in b["sd_levels"]:
            print(f"   {sd_name:>18}  " + "  ".join(f"{b['ci_halfwidth95_two_sided'][f'{sd_name}|n={n}']:>11.4f}" for n in NS))
    print("\n=== required n for 80% power (None = >5000) ===")
    for k, v in req.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
