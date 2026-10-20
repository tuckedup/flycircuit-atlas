"""Summarize stage W3_trained_w76 (W3_PREREG.md). Stdlib + numpy only; absolute paths; read-only except W3_summary.json."""
import json
import os

import numpy as np

OUT = "<PROJECT_ROOT>/results_v4/learning_readout_v4_1"
ROWS = f"{OUT}/W3_trained_w76_rows.jsonl"
CHOICES = f"{OUT}/W3_trained_w76_choices.json"
FAMS = ["connectome", "null0", "null1", "null2", "random_reach", "gru", "encdec"]
NULLS = ["null0", "null1", "null2"]
CHECK = [1501, 1502, 1503, 1504]


def ms(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return None, None, 0
    se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None
    return float(v.mean()), se, len(v)


def f(x):
    return "NA" if x is None else f"{x:+.4f}" if isinstance(x, float) and x < 0 else (f"{x:.4f}" if isinstance(x, float) else str(x))


def main():
    rows = [json.loads(line) for line in open(ROWS) if line.strip()] if os.path.exists(ROWS) else []
    ch = json.load(open(CHOICES)) if os.path.exists(CHOICES) else {}
    ok = [r for r in rows if r["status"] == "ok"]
    summ = {"n_rows": len(rows), "n_failed": len(rows) - len(ok), "families": {}}
    print(f"rows {len(rows)} ok {len(ok)} failed {len(rows) - len(ok)}  (expected 280); VALIDATION ONLY exploratory; no gate status changed")
    per = {}
    for fam in FAMS:
        c = ch.get(fam, {})
        if c.get("core_lr") is None:
            print(f"{fam}: no chosen setting")
            continue
        rs = {r["seed"]: r for r in ok if r["family"] == fam and r["w3_phase"] == "check" and r["core_lr"] == c["core_lr"]
              and r["decoder_lr"] == c["decoder_lr"]}
        d = {"core_lr": c["core_lr"], "decoder_lr": c["decoder_lr"], "n_indist": c["n_indistinguishable"],
             "chosen_equals_best": c["chosen_equals_best"], "grid_edge": c["optimum_at_grid_edge"],
             "acq": {s: rs[s]["association"]["val_acc_acq"] for s in sorted(rs)},
             "cont": {s: rs[s]["continual"]["final_mean_acc_val"] for s in sorted(rs)},
             "forget": {s: rs[s]["continual"]["avg_forgetting_keep_val"] for s in sorted(rs)}}
        for k in ("acq", "cont", "forget"):
            d[k + "_mean_se_n"] = ms(list(d[k].values()))
        d["acq_ge_0_82_descriptive"] = bool(d["acq_mean_se_n"][0] is not None and d["acq_mean_se_n"][0] >= 0.82)
        per[fam] = d
        summ["families"][fam] = d
        print(f"{fam:12s} core={c['core_lr']:.6g} dec={c['decoder_lr']:.6g} indist={c['n_indistinguishable']} edge={c['optimum_at_grid_edge']} "
              f"acq {f(d['acq_mean_se_n'][0])}(se {f(d['acq_mean_se_n'][1])}) cont {f(d['cont_mean_se_n'][0])}(se {f(d['cont_mean_se_n'][1])}) "
              f"forget {f(d['forget_mean_se_n'][0])}(se {f(d['forget_mean_se_n'][1])}) n={d['acq_mean_se_n'][2]} acq>=0.82:{d['acq_ge_0_82_descriptive']}")
        print("   per-seed acq", d["acq"], "\n   per-seed cont", d["cont"], "\n   per-seed forget", d["forget"])
    # signed GRU-minus-connectome forgetting gap, paired by check seed (no abs)
    if "gru" in per and "connectome" in per:
        seeds = [s for s in CHECK if s in per["gru"]["forget"] and s in per["connectome"]["forget"]]
        gap = [per["gru"]["forget"][s] - per["connectome"]["forget"][s] for s in seeds]
        m, se, n = ms(gap)
        summ["gru_minus_connectome_forgetting"] = {"per_seed": dict(zip(seeds, gap)), "mean": m, "se": se, "n": n,
                                                   "descriptive_ge_0_07": bool(m is not None and m >= 0.07)}
        print(f"SIGNED gru-minus-connectome forgetting gap: mean {f(m)} se {f(se)} n={n}  >=0.07: {m is not None and m >= 0.07}  per-seed {gap}")
    # connectome minus nulls / random_reach acquisition, paired by seed
    if "connectome" in per:
        a = per["connectome"]["acq"]
        if all(n in per for n in NULLS):
            s_ = [s for s in CHECK if s in a and all(s in per[n]["acq"] for n in NULLS)]
            dn = [a[s] - float(np.mean([per[n]["acq"][s] for n in NULLS])) for s in s_]
            m, se, n = ms(dn)
            summ["connectome_minus_mean_null012_acq"] = {"per_seed": dict(zip(s_, dn)), "mean": m, "se": se, "n": n}
            print(f"connectome - mean(null0..2) acquisition: mean {f(m)} se {f(se)} n={n}")
        if "random_reach" in per:
            s_ = [s for s in CHECK if s in a and s in per["random_reach"]["acq"]]
            dr = [a[s] - per["random_reach"]["acq"][s] for s in s_]
            m, se, n = ms(dr)
            summ["connectome_minus_random_reach_acq"] = {"per_seed": dict(zip(s_, dr)), "mean": m, "se": se, "n": n}
            print(f"connectome - random_reach acquisition: mean {f(m)} se {f(se)} n={n}")
    print("NOTE: three nulls are NOT graph-level inference; descriptive, 4 seeds, each family at its own selected setting.")
    tmp = f"{OUT}/W3_summary.json.tmp"
    json.dump(summ, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, f"{OUT}/W3_summary.json")


if __name__ == "__main__":
    main()
