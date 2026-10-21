"""Summarize stage W1_encoder_E8 from its JSONL (ENCODER_DIAGNOSTIC_PLAN.md + AMENDMENT 1). Stdlib+numpy only."""
from _paths import CANON as _CANON  # noqa: F401,E402
from _paths import ROOT as _ROOT  # noqa: F401,E402
import json
import sys

import numpy as np

OUT = (_ROOT + "/results_v4/learning_readout_v4_1")
JL = f"{OUT}/W1_encoder_E8_rows.jsonl"
T7 = 2.365
CELLS = [f"{f}|W76" for f in ["connectome", "null0", "null1", "null2", "null3", "null4", "random_reach"]] + \
    ["kc_static|native1385", "kc_static_sub|K76"]
CTRL = [f"{f}|W76" for f in ["null0", "null1", "null2", "null3", "null4", "random_reach"]]
SEEDS = list(range(1101, 1109))


def ci(d):
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), n
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    t = T7 if n == 8 else T7  # t(7) per spec; n<8 is flagged in output
    return m, se, m - t * se, m + t * se, n


def fmt(c):
    m, se, lo, hi, n = c
    return f"mean {m:+.4f} SE {se:.4f} CI95[{lo:+.4f},{hi:+.4f}] n={n}"


def main():
    rows = [json.loads(line) for line in open(JL) if line.strip()]
    ok = {(r["family"], r["condition"], r["seed"]): r for r in rows if not r.get("failed")}
    failed = [r for r in rows if r.get("failed")]
    print(f"rows {len(rows)} ok {len(ok)} failed {len(failed)} (expected 144)")
    for cell in CELLS:
        k = sum(1 for r in failed if r["family"] == cell)
        if k:
            print(f"  failures {cell}: {k}  e.g. {next(r['error'] for r in failed if r['family'] == cell)}")

    def val(cell, cond, seed, key):
        r = ok.get((cell, cond, seed))
        return None if r is None else r.get(key)

    def paired(cell, key="ctx0_val"):
        out = {}
        for s in SEEDS:
            a, b = val(cell, "E8", s, key), val(cell, "baseline", s, key)
            if a is not None and b is not None:
                out[s] = a - b
        return out

    print("\n== Paired seed differences E8 - baseline ==")
    gains = {}
    for key in ["ctx0_val", "percontext_val", "eff_rank"]:
        print(f"-- {key}")
        for cell in CELLS:
            d = paired(cell, key)
            if key == "ctx0_val":
                gains[cell] = d
            print(f"  {cell:24s} {fmt(ci(list(d.values())))}")
    print("-- train-val gap (ctx0_train - ctx0_val), E8 - baseline")
    for cell in CELLS:
        d = []
        for s in SEEDS:
            vals = [val(cell, c, s, k) for c in ("E8", "baseline") for k in ("ctx0_train", "ctx0_val")]
            if None not in vals:
                d.append((vals[0] - vals[1]) - (vals[2] - vals[3]))
        print(f"  {cell:24s} {fmt(ci(d))}")

    print("\n== Primary contrast P = connectome gain - mean gain over null0-4 + random_reach (paired by seed) ==")
    pseeds = [s for s in SEEDS if s in gains["connectome|W76"] and all(s in gains[c] for c in CTRL)]
    P = [gains["connectome|W76"][s] - np.mean([gains[c][s] for c in CTRL]) for s in pseeds]
    Pc = ci(P)
    print("  P:", fmt(Pc))

    print("\n== Gate G-E1 (Amendment item 6) ==")
    gc = ci(list(gains["connectome|W76"].values()))
    c1 = gc[4] > 0 and gc[0] >= 0.03 and gc[2] > 0
    c2 = len(P) > 0 and Pc[0] > 0
    sat = []
    for s in SEEDS:
        a, b = val("connectome|W76", "E8", s, "sat_pre_kc"), val("connectome|W76", "baseline", s, "sat_pre_kc")
        if a is not None and b is not None:
            sat.append(a - b)
    sm = float(np.mean(sat)) if sat else float("nan")
    c3 = bool(sat) and sm <= 0.05
    print(f"  (i) connectome gain >= +0.03 and CI lower > 0: {'PASS' if c1 else 'FAIL'}  [{fmt(gc)}]")
    print(f"  (ii) P point estimate > 0: {'PASS' if c2 else 'FAIL'}  [{fmt(Pc)}; design underpowered below ~0.05]")
    print(f"  (iii) frac |pre|>2 (KC, connectome cell) E8-baseline <= +0.05: {'PASS' if c3 else 'FAIL'}  "
          f"[mean diff {sm:+.4f}]")
    allsat = {c: np.mean([val(c, "E8", s, "sat_pre_kc") - val(c, "baseline", s, "sat_pre_kc")
                          for s in SEEDS if val(c, "E8", s, "sat_pre_kc") is not None
                          and val(c, "baseline", s, "sat_pre_kc") is not None] or [np.nan]) for c in CELLS}
    print("      (info) saturation E8-baseline per cell:", {k: round(float(v), 4) for k, v in allsat.items()})
    complete = len(ok) == 144
    print(f"  completeness: {len(ok)}/144 ok rows, {len(failed)} failed -> "
          f"{'complete' if complete else 'INCOMPLETE: gate not valid'}")
    verdict = c1 and c2 and c3 and complete
    print(f"  G-E1 overall: {'PASS (proceed to PREREGISTRATION of trained pilot only)' if verdict else 'FAIL / STOP'}")

    print("\n== Descriptive (uncorrected; measurement only): saturation / context / reachability ==")
    keys = ["sat_pre_kc", "sat_h_kc", "sat_h_ro", "kc_ctx_count", "reach_le5", "reach_le7", "eff_rank",
            "silent_frac", "clamp_frac", "n_feat"]
    print(f"{'cell':24s} {'cond':9s} " + " ".join(f"{k:>11s}" for k in keys))
    for cell in CELLS:
        for cond in ("baseline", "E8"):
            cols = []
            for k in keys:
                v = [r[k] for s in SEEDS for r in [ok.get((cell, cond, s))]
                     if r is not None and r.get(k) is not None]
                cols.append(f"{np.mean(v):11.4f}" if v else f"{'null':>11s}")
            print(f"{cell:24s} {cond:9s} " + " ".join(cols))
    print("\nNotes: pre-activation is input-only drive x@encoder at window steps; reachability is null for kc_static "
          "(no edges); reach = fraction of readout nodes with min hop <=5 / <=7 from sensory_idx.")


if __name__ == "__main__":
    main()
