from _paths import CANON as _CANON  # noqa: F401,E402
from _paths import ROOT as _ROOT  # noqa: F401,E402
import collections
import json
import statistics as s

D = (_ROOT + "/results_v4/learning_readout_v4_1/")
g = collections.defaultdict(list)
for line in open(D + "W0_width_probe_rows.jsonl"):
    r = json.loads(line)
    g[r["family"]].append(r)
print("rows", sum(len(v) for v in g.values()), "cells", len(g))
for f, v in g.items():
    print("%-26s n=%d nfeat=%4d ctx0 %.3f (sd %.3f) pc %.3f train0 %.3f rank %.1f clamp %.3f" % (
        f, len(v), v[0]["n_feat"], s.mean(x["ctx0_val"] for x in v), s.stdev(x["ctx0_val"] for x in v),
        s.mean(x["percontext_val"] for x in v), s.mean(x["ctx0_train"] for x in v),
        s.mean(x["eff_rank"] for x in v), s.mean(x["clamp_frac"] for x in v)))
