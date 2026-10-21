"""Validation-only, no-RL diagnostic: how well can a LINEAR readout of each UNTRAINED model's z-scored
response-window features separate the association task? Separates representation quality from learning.
Forward passes + L2 logistic regression only; train split fits, val split scores. Seeds 701-708."""
from _paths import CANON as _CANON  # noqa: F401,E402
from _paths import ROOT as _ROOT  # noqa: F401,E402
import json, sys, time
import numpy as np, torch
from flycircuit.eval import learning_v2 as V2, learning_v3 as V3, learning as L

torch.set_num_threads(2)
proto = V3.load_protocol()
base = V2.load_base(proto, "flywire")
packs = V2.load_packs(base, proto, "flywire")
FAMS = ["connectome", "kc_static", "random_reach", "null0", "null1", "null2", "null3", "null4", "gru", "encdec"]
SEEDS = list(range(701, 709))
out = []

@torch.no_grad()
def feats(agent, task, split, ctx, rng):
    cues = task.cues[(split, ctx)]
    x = torch.as_tensor(task.make_input(cues, ctx, rng))
    win = agent.window
    m = agent.model
    if agent.sparse:
        w = m.edge_values().detach()
        st = m.init_state(x.shape[1]); zs = []
        for t in range(x.shape[0]):
            _, st = m.step(x[t], st, w)
            if win.start <= t < win.stop:
                zs.append(agent._zscore(st["h"][:, m.readout_idx]))
        return torch.stack(zs).mean(0)
    # dense models: use the model's own pre-decoder hidden via a forward hook on the decoder input
    cap = {}
    dec = m.dec if hasattr(m, "dec") else m.net[2]
    h = dec.register_forward_hook(lambda mod, i, o: cap.__setitem__("in", i[0]))
    m(x); h.remove()
    return cap["in"][win].mean(0)

t0 = time.time()
for fam in FAMS:
    for seed in SEEDS:
        task = L.make_task(proto, seed)
        agent, info = V2.make_agent_v2(proto, base, packs, "bptt", fam, 1e-3, seed, task)
        rng = np.random.default_rng(seed)
        accs, trn = [], []
        for ctx in range(task.spec.n_ctx):
            xt = feats(agent, task, "train", ctx, rng); xv = feats(agent, task, "val", ctx, rng)
            yt = torch.as_tensor(task.labels[("train", ctx)]).float().reshape(-1)
            yv = torch.as_tensor(task.labels[("val", ctx)]).reshape(-1)
            mu, sd = xt.mean(0), xt.std(0) + 1e-6
            w, b = V2._fit_logreg((xt - mu) / sd, yt)
            accs.append(float(((((xv - mu) / sd) @ w + b > 0).long() == yv).float().mean()))
            trn.append(float(((((xt - mu) / sd) @ w + b > 0).long() == yt.long()).float().mean()))
        r = dict(family=fam, seed=seed, val_acc=float(np.mean(accs)), train_acc=float(np.mean(trn)),
                 n_feat=int(xt.shape[1]), oracle_val=V2.oracle_ceiling(task, ("val",))["val"])
        out.append(r); print(json.dumps(r), flush=True)
json.dump(out, open((_ROOT + "/diag_out/readout_ceiling.json"), "w"), indent=1)
print("done", round(time.time() - t0), "s")
