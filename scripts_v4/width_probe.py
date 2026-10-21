"""Stage W0 readout-width probe. See results_v4/learning_readout_v4_1/PREREG_WIDTH_PROBE.md. Absolute paths only."""
from _paths import CANON as _CANON  # noqa: F401,E402
from _paths import ROOT as _ROOT  # noqa: F401,E402
import hashlib
import json
import os
import time

import numpy as np
import torch

from flycircuit.eval import learning as L
from flycircuit.eval import learning_v2 as V2
from flycircuit.eval import learning_v3 as V3
from flycircuit.tasks.graph_loader import CircuitBundle

OUT = (_ROOT + "/results_v4/learning_readout_v4_1")
STAGE = "W0_width_probe"
torch.set_num_threads(2)
proto = V3.load_protocol()
base = V2.load_base(proto, "flywire")
packs = V2.load_packs(base, proto, "flywire")
g = base.graph
n = g.n
sens = set(map(int, base.sensory_idx))
r25 = [int(i) for i in base.readout_idx]
indeg = np.bincount(g.post, minlength=n)
contacts = np.bincount(g.post, weights=g.count, minlength=n)
elig = [i for i in range(n) if i not in sens and i not in set(r25) and indeg[i] > 0]
elig.sort(key=lambda i: (-contacts[i], i))
allns = [i for i in range(n) if i not in sens and i not in set(r25)]
SETS = {"W25": r25, "W64": r25 + elig[:39], "W76": r25 + elig, "W115": r25 + allns}
assert len(SETS["W64"]) == 64 and len(SETS["W76"]) == 76 and len(SETS["W115"]) == 115
manifest = {"stage": STAGE, "sets": SETS, "n_elig": len(elig), "edge_hash": g.edge_hash(),
            "sets_hash": hashlib.sha256(json.dumps(SETS).encode()).hexdigest()}
os.makedirs(OUT, exist_ok=True)
json.dump(manifest, open(f"{OUT}/{STAGE}_manifest.json", "w"), indent=1)


def with_ro(b, ro):
    return CircuitBundle(b.graph, b.sign, b.sensory_idx, np.asarray(ro, dtype=base.readout_idx.dtype),
                         b.label, b.notes, b.modulatory)


@torch.no_grad()
def feats(agent, task, split, ctx, rng):
    x = torch.as_tensor(task.make_input(task.cues[(split, ctx)], ctx, rng))
    win = agent.window
    m = agent.model
    if agent.sparse:
        w = m.edge_values().detach()
        st = m.init_state(x.shape[1])
        zs = []
        for t in range(x.shape[0]):
            _, st = m.step(x[t], st, w)
            if win.start <= t < win.stop:
                zs.append(agent._zscore(st["h"][:, m.readout_idx]))
        return torch.stack(zs).mean(0)
    cap = {}
    dec = m.dec if hasattr(m, "dec") else m.net[2]
    h = dec.register_forward_hook(lambda mod, i, o: cap.__setitem__("in", i[0]))
    m(x)
    h.remove()
    return cap["in"][win].mean(0)


def probe(xt, xv, yt, yv):
    mu, sd = xt.mean(0), xt.std(0) + 1e-6
    w, b = V2._fit_logreg((xt - mu) / sd, yt)
    va = float(((((xv - mu) / sd) @ w + b > 0).long() == yv).float().mean())
    ta = float(((((xt - mu) / sd) @ w + b > 0).long() == yt.long()).float().mean())
    return va, ta


def stats(xt):
    c = torch.cov(xt.T) if xt.shape[1] > 1 else torch.ones(1, 1)
    ev = torch.linalg.eigvalsh(c).clamp_min(0)
    pr = float(ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-12))
    return dict(eff_rank=pr, silent_frac=float((xt.std(0) < 1e-6).float().mean()),
                clamp_frac=float((xt.abs() >= 5 - 1e-6).float().mean()))


def run_cell(fam, label, seed, ro=None, sub=None):
    task = L.make_task(proto, seed)
    b = base if ro is None else with_ro(base, ro)
    agent, _ = V2.make_agent_v2(proto, b, packs, "bptt", fam, 1e-3, seed, task)
    rng = np.random.default_rng(seed)
    rec = dict(stage=STAGE, family=label, seed=seed)
    pc = []
    for ctx in range(task.spec.n_ctx):
        xt = feats(agent, task, "train", ctx, rng)
        xv = feats(agent, task, "val", ctx, rng)
        if sub:
            idx = np.sort(np.random.default_rng([seed, 99, sub]).choice(xt.shape[1], sub, replace=False))
            xt, xv = xt[:, idx], xv[:, idx]
        yt = torch.as_tensor(task.labels[("train", ctx)]).float().reshape(-1)
        yv = torch.as_tensor(task.labels[("val", ctx)]).reshape(-1)
        va, ta = probe(xt, xv, yt, yv)
        pc.append((va, ta))
        if ctx == 0:
            rec.update(ctx0_val=va, ctx0_train=ta, **stats(xt))
    rec.update(n_feat=int(xt.shape[1]), percontext_val=float(np.mean([p[0] for p in pc])),
               percontext_train=float(np.mean([p[1] for p in pc])),
               oracle_val=V2.oracle_ceiling(task, ("val",))["val"])
    return rec


SEEDS = list(range(801, 809))
cells = []
for sname, ro in SETS.items():
    for fam in ["connectome", "null0", "null1", "null2", "null3", "null4", "random_reach"]:
        cells.append((fam, f"{fam}|{sname}", ro, None))
for K in (25, 64, 76):
    cells.append(("kc_static", f"kc_static_sub|K{K}", None, K))
    cells.append(("encdec", f"encdec_sub|K{K}", None, K))
cells += [("kc_static", "kc_static|native1385", None, None), ("gru", "gru|native37", None, None),
          ("encdec", "encdec|native407", None, None)]
rows = []
t0 = time.time()
fh = open(f"{OUT}/{STAGE}_rows.jsonl", "a")
for fam, label, ro, sub in cells:
    for seed in SEEDS:
        r = run_cell(fam, label, seed, ro, sub)
        rows.append(r)
        fh.write(json.dumps(r) + "\n")
        fh.flush()
    print(label, round(time.time() - t0), "s", flush=True)
tmp = f"{OUT}/{STAGE}_rows.json.tmp"
json.dump(rows, open(tmp, "w"), indent=1)
os.replace(tmp, f"{OUT}/{STAGE}_rows.json")
print("done", round(time.time() - t0), "s")
