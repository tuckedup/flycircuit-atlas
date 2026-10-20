"""Summarize stage W2_sign_policy from its JSONL (W2_PREREG.md). Stdlib+numpy only."""
import json

import numpy as np

OUT = "<PROJECT_ROOT>/results_v4/learning_readout_v4_1"
JL = f"{OUT}/W2_sign_policy_rows.jsonl"
T7 = 2.365
FAMS = ["connectome", "null0", "null1", "null2", "null3", "null4", "random_reach"]
CELLS = [f"{f}|W76" for f in FAMS] + [f"{f}|W25" for f in FAMS] + ["kc_static|native1385"]
CTRL = [f"{f}|W76" for f in FAMS[1:]]
SEEDS = list(range(1301, 1309))
N_EXPECTED = 240


def ci(d):
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), n
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    return m, se, m - T7 * se, m + T7 * se, n


def fmt(c):
    m, se, lo, hi, n = c
    return f"mean {m:+.4f} SE {se:.4f} CI95[{lo:+.4f},{hi:+.4f}] n={n}"


def main():
    rows = [json.loads(line) for line in open(JL) if line.strip()]
    ok = {(r["family"], r["condition"], r["seed"]): r for r in rows if not r.get("failed")}
    failed = [r for r in rows if r.get("failed")]
    print(f"rows {len(rows)} ok {len(ok)} failed {len(failed)} (expected {N_EXPECTED})")
    for cell in CELLS:
        k = [r for r in failed if r["family"] == cell]
        if k:
            print(f"  failures {cell}: {len(k)}  e.g. {k[0]['error']}")

    def val(cell, cond, seed, key):
        r = ok.get((cell, cond, seed))
        return None if r is None else r.get(key)

    def paired(cell, key="ctx0_val"):
        out = {}
        for s in SEEDS:
            a, b = val(cell, "policy", s, key), val(cell, "baseline", s, key)
            if a is not None and b is not None:
                out[s] = a - b
        return out

    print("\n== Paired seed differences policy(excitatory) - baseline(zero) ==")
    gains = {}
    for key in ["ctx0_val", "percontext_val", "eff_rank", "silent_frac"]:
        print(f"-- {key}")
        for cell in CELLS:
            d = paired(cell, key)
            if key == "ctx0_val":
                gains[cell] = d
            print(f"  {cell:24s} {fmt(ci(list(d.values())))}")
    print("-- train-val gap (ctx0_train - ctx0_val), policy - baseline")
    for cell in CELLS:
        d = []
        for s in SEEDS:
            v = [val(cell, c, s, k) for c in ("policy", "baseline") for k in ("ctx0_train", "ctx0_val")]
            if None not in v:
                d.append((v[0] - v[1]) - (v[2] - v[3]))
        print(f"  {cell:24s} {fmt(ci(d))}")

    kc = gains["kc_static|native1385"]
    kcmax = max((abs(v) for v in kc.values()), default=float("nan"))
    print(f"\n== Sanity control kc_static native1385: max |paired gain| = {kcmax:.2e} "
          f"({'OK (unaffected)' if kc and kcmax < 1e-9 else 'NONZERO: pipeline problem' if kc else 'no data'})")

    print("\n== Primary contrast P = connectome W76 gain - mean gain over null0-4 + random_reach at W76 (paired by seed) ==")
    pseeds = [s for s in SEEDS if s in gains["connectome|W76"] and all(s in gains[c] for c in CTRL)]
    P = [gains["connectome|W76"][s] - np.mean([gains[c][s] for c in CTRL]) for s in pseeds]
    Pc = ci(P)
    print("  P:", fmt(Pc))

    print("\n== Gate G-E2 (W2_PREREG.md section 4) ==")
    gc = ci(list(gains["connectome|W76"].values()))
    c1 = gc[4] > 0 and gc[0] >= 0.03 and gc[2] > 0
    c2 = len(P) > 0 and Pc[0] > 0
    sd = paired("connectome|W76", "silent_frac")
    drop = -float(np.mean(list(sd.values()))) if sd else float("nan")
    c3 = bool(sd) and drop >= 0.05
    print(f"  (i) connectome W76 gain >= +0.03 and CI lower > 0: {'PASS' if c1 else 'FAIL'}  [{fmt(gc)}]")
    print(f"  (ii) P point estimate > 0: {'PASS' if c2 else 'FAIL'}  [{fmt(Pc)}; underpowered below ~0.05]")
    print(f"  (iii) connectome W76 silent fraction falls >= 0.05 absolute: {'PASS' if c3 else 'FAIL'}  [fall {drop:+.4f}]")
    complete = len(ok) == N_EXPECTED
    print(f"  completeness: {len(ok)}/{N_EXPECTED} ok rows, {len(failed)} failed -> "
          f"{'complete' if complete else 'INCOMPLETE: gate not valid'}")
    verdict = c1 and c2 and c3 and complete
    print(f"  G-E2 overall: {'PASS (proceed to PREREGISTRATION of trained pilot only)' if verdict else 'FAIL / STOP'}")

    print("\n== Descriptive (uncorrected; measurement only) ==")
    keys = ["silent_frac", "eff_rank", "clamp_frac", "ctx0_val", "ctx0_train", "n_unknown_sign", "n_feat"]
    print(f"{'cell':24s} {'cond':9s} " + " ".join(f"{k:>14s}" for k in keys))
    for cell in CELLS:
        for cond in ("baseline", "policy"):
            cols = []
            for k in keys:
                v = [r[k] for s in SEEDS for r in [ok.get((cell, cond, s))] if r is not None and r.get(k) is not None]
                cols.append(f"{np.mean(v):14.4f}" if v else f"{'null':>14s}")
            print(f"{cell:24s} {cond:9s} " + " ".join(cols))
    print("\nNotes: sign-policy change is a graph/sign-policy scientific change; nulls share the sign vector, so no anatomical "
          "claim; untrained frozen features; 8 seeds underpowered.")


if __name__ == "__main__":
    main()
