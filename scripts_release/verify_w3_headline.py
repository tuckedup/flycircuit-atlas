"""Recompute and cross-check the W3 headline numbers (no training, stdlib only, <1 s).

Usage (from the repository root):
    python scripts_release/verify_w3_headline.py [--root DIR]

Checks, each reported as a failure line if violated:
  1. Row inventory: exactly 280 rows, no duplicate (phase, family, core_lr, decoder_lr, seed) keys, no failed rows,
     and the full design present (7 families x 9 settings x search seeds 1401-1404; 7 families x check seeds 1501-1504).
  2. Check rows use the setting recorded in W3_trained_w76_choices.json for their family.
  3. Recomputed per-seed values, means and SEs equal W3_summary.json (written by the original run).
  4. Trainable parameter counts equal the figures quoted in the docs (7,551 / 9,252 / 8,956).
  5. Documentation drift: README.md and RESULTS_REPORT.md quote the recomputed headline values and table entries, and no
     longer contain the retracted "matched GRU" / "matched controls" wording.
Exit status 0 only if every check passes.
"""
import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

FAMILIES = ["connectome", "null0", "null1", "null2", "random_reach", "gru", "encdec"]
TABLE_NAMES = {"Connectome W76": "connectome", "Null 0": "null0", "Null 1": "null1", "Null 2": "null2",
               "Random-reach graph": "random_reach", "GRU (hidden 37)": "gru", "Encoder-decoder": "encdec"}
SEARCH_SEEDS, CHECK_SEEDS = [1401, 1402, 1403, 1404], [1501, 1502, 1503, 1504]
EXPECTED_PARAMS = {"connectome": 7551, "gru": 9252, "encdec": 8956}
BANNED = [r"matched\s+GRU", r"matched\s+controls?", r"GRU-matched"]


def mean_se(values):
    v = list(values)
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    return m, sd / math.sqrt(len(v))


def verify(root: Path):
    """Return (failures, facts). failures is a list of strings; empty means every check passed."""
    fails = []
    d = root / "results_v4" / "learning_readout_v4_1"
    rows = [json.loads(x) for x in (d / "W3_trained_w76_rows.jsonl").read_text().splitlines() if x.strip()]
    choices = json.loads((d / "W3_trained_w76_choices.json").read_text())
    summary = json.loads((d / "W3_summary.json").read_text())

    # 1. inventory ------------------------------------------------------------------------------------------
    if len(rows) != 280:
        fails.append(f"row count {len(rows)} != 280")
    keys = Counter((r.get("w3_phase"), r["family"], r["core_lr"] if "core_lr" in r else r["lr"], r["decoder_lr"], r["seed"])
                   for r in rows)
    for k, n in keys.items():
        if n > 1:
            fails.append(f"duplicate row key {k} x{n}")
    bad = [r for r in rows if r.get("status") != "ok"]
    if bad:
        fails.append(f"{len(bad)} rows with status != ok")
    phases = Counter(r.get("w3_phase") for r in rows)
    if phases.get("check") != 28 or phases.get("search") != 252:
        fails.append(f"phase counts {dict(phases)} != search 252 / check 28")
    for fam in FAMILIES:
        s = [r for r in rows if r["family"] == fam and r.get("w3_phase") == "search"]
        settings = {(r["core_lr"], r["decoder_lr"]) for r in s}
        if len(settings) != 9:
            fails.append(f"{fam}: {len(settings)} search settings, expected 9")
        for st in settings:
            seeds = sorted(r["seed"] for r in s if (r["core_lr"], r["decoder_lr"]) == st)
            if seeds != SEARCH_SEEDS:
                fails.append(f"{fam} setting {st}: search seeds {seeds}")
        c = sorted(r["seed"] for r in rows if r["family"] == fam and r.get("w3_phase") == "check")
        if c != CHECK_SEEDS:
            fails.append(f"{fam}: check seeds {c}, expected {CHECK_SEEDS}")

    check = [r for r in rows if r.get("w3_phase") == "check" and r.get("status") == "ok"]
    # 2. check rows use the chosen setting ------------------------------------------------------------------
    for r in check:
        ch = choices.get(r["family"])
        if ch is None or (r["core_lr"], r["decoder_lr"]) != (ch["core_lr"], ch["decoder_lr"]):
            fails.append(f"{r['family']} seed {r['seed']}: check row setting differs from choices.json")

    acq, cont, forget, params = defaultdict(dict), defaultdict(dict), defaultdict(dict), {}
    for r in check:
        acq[r["family"]][r["seed"]] = r["association"]["val_acc_acq"]
        cont[r["family"]][r["seed"]] = r["continual"]["final_mean_acc_val"]
        forget[r["family"]][r["seed"]] = r["continual"]["avg_forgetting_keep_val"]
        params[r["family"]] = r["n_params_trainable"]

    if not all(sorted(acq[f]) == CHECK_SEEDS for f in FAMILIES):
        fails.append("check rows incomplete: statistics, summary and documentation cross-checks skipped")
        return fails, None

    # 3. against the original run's own summary -----------------------------------------------------------
    for fam in FAMILIES:
        s = summary["families"].get(fam)
        if not s:
            fails.append(f"{fam}: missing from W3_summary.json")
            continue
        for name, got in (("acq", acq[fam]), ("forget", forget[fam])):
            for seed, v in got.items():
                if abs(s[name][str(seed)] - v) > 1e-12:
                    fails.append(f"{fam} {name} seed {seed}: summary {s[name][str(seed)]} != rows {v}")
        for name, got in (("acq_mean_se_n", acq[fam]), ("cont_mean_se_n", cont[fam]), ("forget_mean_se_n", forget[fam])):
            m, se = mean_se(got.values())
            if abs(s[name][0] - m) > 1e-9 or abs(s[name][1] - se) > 1e-9:
                fails.append(f"{fam} {name}: summary {s[name][:2]} != recomputed {[m, se]}")
    gap = {s: forget["gru"][s] - forget["connectome"][s] for s in forget["connectome"]}
    gm, gse = mean_se(gap.values())
    if abs(summary["gru_minus_connectome_forgetting"]["mean"] - gm) > 1e-9:
        fails.append("forgetting gap differs from W3_summary.json")

    # 4. parameter counts ---------------------------------------------------------------------------------
    for fam, n in EXPECTED_PARAMS.items():
        if params.get(fam) != n:
            fails.append(f"{fam} trainable params {params.get(fam)} != {n}")

    # 5. documentation drift ------------------------------------------------------------------------------
    best_seed = max(acq["connectome"], key=acq["connectome"].get)
    best, cmean = acq["connectome"][best_seed], mean_se(acq["connectome"].values())[0]
    readme = (root / "README.md").read_text()
    report = (root / "RESULTS_REPORT.md").read_text()
    for need in (f"{best * 100:.2f} %", f"{cmean * 100:.2f} %", f"seed {best_seed}"):
        if need not in readme:
            fails.append(f"README.md does not quote {need!r}")
    if f"{gm * 100:+.1f}".replace("-", "−") not in readme:
        fails.append("README.md does not quote the recomputed forgetting gap")
    if gm >= 0.07:
        fails.append("forgetting gap now meets +0.07: documentation claims must be revisited")
    seed_pct = " / ".join(f"{acq['connectome'][s] * 100:.1f}" for s in CHECK_SEEDS)
    if seed_pct not in readme.replace(", ", " / "):
        fails.append(f"README.md per-seed list differs from recomputed {seed_pct}")
    for fn, text in (("README.md", readme), ("RESULTS_REPORT.md", report), ("ROADMAP.md", (root / "ROADMAP.md").read_text())):
        for pat in BANNED:
            if re.search(pat, text, re.I):
                fails.append(f"{fn} contains retracted wording /{pat}/")
    for n in ("7,551", "9,252", "8,956"):
        if n not in report:
            fails.append(f"RESULTS_REPORT.md does not quote parameter count {n}")
    table = {}
    for line in report.splitlines():
        cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
        if cells and cells[0] in TABLE_NAMES and len(cells) == 5:
            table[TABLE_NAMES[cells[0]]] = cells
    for fam in FAMILIES:
        if fam not in table:
            fails.append(f"RESULTS_REPORT.md table lacks a row for {fam}")
            continue
        _, a, per, cf, fg = table[fam]
        m, se = mean_se(acq[fam].values())
        if not a.startswith(f"{m:.3f} ({se:.3f})"):
            fails.append(f"report {fam} acquisition cell {a!r} != {m:.3f} ({se:.3f})")
        want = " / ".join(f"{acq[fam][s]:.3f}" for s in CHECK_SEEDS)
        if per != want:
            fails.append(f"report {fam} per-seed cell {per!r} != {want!r}")
        if cf != f"{mean_se(cont[fam].values())[0]:.3f}":
            fails.append(f"report {fam} continual cell {cf!r}")
        if not fg.startswith(f"{mean_se(forget[fam].values())[0]:.3f}"):
            fails.append(f"report {fam} forgetting cell {fg!r}")

    facts = {"best_seed": best_seed, "best": best, "mean": cmean, "gap": gm, "gap_se": gse,
             "acq": {f: mean_se(acq[f].values()) for f in FAMILIES}}
    return fails, facts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args(argv)
    fails, f = verify(Path(args.root))
    if f is not None:
        print("W3 validation-check acquisition (seeds 1501-1504, validation split, NOT held-out test)")
        for fam, (m, se) in f["acq"].items():
            print(f"  {fam:13s} mean {m:.4f} (SE {se:.4f})")
        print(f"connectome selected run : seed {f['best_seed']}, acquisition {f['best']:.6f}")
        print(f"connectome 4-seed mean  : {f['mean']:.8f}")
        print(f"signed forgetting gap (GRU minus connectome): {f['gap']:+.4f} (SE {f['gap_se']:.4f}); target >= +0.07 -> "
              f"{'met' if f['gap'] >= 0.07 else 'NOT met'}")
    for x in fails:
        print("FAIL:", x)
    print("verification:", "PASSED" if not fails else f"FAILED ({len(fails)} problems)")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
