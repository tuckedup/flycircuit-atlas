"""Recompute the W3 headline numbers from the raw per-run rows (no training, stdlib only, runs in <1 s).

Usage (from the repository root):
    python scripts_release/verify_w3_headline.py

Reads results_v4/learning_readout_v4_1/W3_trained_w76_rows.jsonl and prints, for every family, the
per-seed validation-check acquisition, the mean, the best single seed, and the signed forgetting gap.
Exits non-zero if any headline value stated in README.md differs from the recomputed one.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROWS = Path(__file__).resolve().parents[1] / "results_v4" / "learning_readout_v4_1" / "W3_trained_w76_rows.jsonl"
FAMILIES = ["connectome", "null0", "null1", "null2", "random_reach", "gru", "encdec"]

rows = [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]
check = [r for r in rows if r["w3_phase"] == "check" and r["status"] == "ok"]
assert len(rows) == 280, f"expected 280 rows, got {len(rows)}"
assert sum(r["status"] != "ok" for r in rows) == 0

acq, forget = defaultdict(dict), defaultdict(dict)
for r in check:
    acq[r["family"]][r["seed"]] = r["association"]["val_acc_acq"]
    forget[r["family"]][r["seed"]] = r["continual"]["avg_forgetting_keep_val"]


def mean_se(d):
    v = list(d.values())
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    return m, sd / math.sqrt(len(v))


print("W3 validation-check acquisition (seeds 1501-1504, split = validation, NOT held-out test)")
for f in FAMILIES:
    m, se = mean_se(acq[f])
    per = ", ".join(f"{s}:{v:.4f}" for s, v in sorted(acq[f].items()))
    print(f"  {f:13s} mean {m:.4f} (SE {se:.4f})  best seed {max(acq[f].values()):.4f}  [{per}]")

best_seed = max(acq["connectome"], key=acq["connectome"].get)
gap = {s: forget["gru"][s] - forget["connectome"][s] for s in forget["connectome"]}
gap_m, gap_se = mean_se(gap)
print(f"\nconnectome selected run : seed {best_seed}, acquisition {acq['connectome'][best_seed]:.6f}")
print(f"connectome 4-seed mean  : {mean_se(acq['connectome'])[0]:.8f}")
print(f"signed forgetting gap (GRU minus connectome): {gap_m:+.4f} (SE {gap_se:.4f}); target >= +0.07 -> "
      f"{'met' if gap_m >= 0.07 else 'NOT met'}")

ok = (best_seed == 1502 and abs(acq["connectome"][1502] - 0.890625) < 1e-9
      and abs(mean_se(acq["connectome"])[0] - 0.82421875) < 1e-9 and gap_m < 0.07)
print("\nREADME headline values reproduced:", ok)
sys.exit(0 if ok else 1)
