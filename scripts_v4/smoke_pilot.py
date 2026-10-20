"""One setting per arm (seed 501, validation only) to catch plumbing bugs before the long pilot.
Writes nothing to results; prints scores, param-group sizes and gradient diagnostics."""
import json
import time

import flycircuit

assert "flycircuit_debug_copy" in flycircuit.__file__
from flycircuit.eval import learning_v4 as P

proto, base, packs = P._setup()
lrs = P.v3_lrs()
print("estimate", json.dumps(P.estimate()), "v3 lrs", lrs, flush=True)
for fam in P.ARMS:
    t0 = time.time()
    r = P.run_setting(proto, base, packs, fam, lrs[fam], lrs[fam] * 3, 501)
    print(fam, r["status"][:60], "acq/cont", P.scores(r) if P.ok(r) else None, "groups",
          r.get("param_groups"), "diag", r["diag"], f"{time.time() - t0:.1f}s", flush=True)
    assert P.ok(r)
    assert r["diag"]["n"] > 0
# rule logic on the records format with synthetic rows
recs = []
for fam in P.ARMS:
    for i, core in enumerate(P.grid_for(lrs[fam])):
        for j, dec in enumerate(P.grid_for(lrs[fam])):
            for sd in P.SEARCH_SEEDS:
                recs.append({"arm": "bptt", "family": fam, "status": "ok", "core_lr": core, "decoder_lr": dec,
                             "seed": sd, "association": {"val_acc_acq": 0.6 + 0.02 * i + 0.001 * sd % 7},
                             "continual": {"final_mean_acc_val": 0.7 - 0.01 * i + 0.02 * j}})
for fam in P.ARMS:
    print(fam, {rule: P.choose(recs, fam, rule) for rule in "ABCD"})
print("smoke ok")
