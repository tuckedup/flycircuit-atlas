"""Label-free structural check: which W76 readout nodes are silent in the untrained connectome model, and why."""
import numpy as np
import torch

from flycircuit.eval import learning as L
from flycircuit.eval import learning_v2 as V2
from flycircuit.eval import learning_v3 as V3
from flycircuit.tasks.graph_loader import CircuitBundle

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
w76 = r25 + elig
b = CircuitBundle(g, base.sign, base.sensory_idx, np.asarray(w76, dtype=base.readout_idx.dtype), base.label,
                  base.notes, base.modulatory)
seed = 1201
task = L.make_task(proto, seed)
agent, _ = V2.make_agent_v2(proto, b, packs, "bptt", "connectome", 1e-3, seed, task)
m = agent.model
x = torch.as_tensor(task.make_input(task.cues[("train", 0)], 0, np.random.default_rng(seed)))
with torch.no_grad():
    w = m.edge_values().detach()
    st = m.init_state(x.shape[1])
    hs = []
    for t in range(x.shape[0]):
        _, st = m.step(x[t], st, w)
        if agent.window.start <= t < agent.window.stop:
            hs.append(st["h"][:, m.readout_idx])
    h = torch.stack(hs).mean(0)
sd = h.std(0).numpy()
mean_abs = h.abs().mean(0).numpy()
sign = np.asarray(base.sign)
# incoming edges from non-zero-sign (non-silent) presynaptic nodes
pre_sign = sign[g.pre]
rows = []
for k, node in enumerate(w76):
    inc = np.where(g.post == node)[0]
    rows.append((k, node, k < 25, float(sd[k]), float(mean_abs[k]), int(indeg[node]), int((pre_sign[inc] != 0).sum()),
                 int(sign[node])))
print("slot node orig25 sd mean|h| indeg n_inc_from_nonzero_sign_pre own_sign")
for r in rows:
    if r[3] < 1e-6:
        print(*r)
sil = [r for r in rows if r[3] < 1e-6]
print("silent", len(sil), "of", len(rows), "; silent in original 25:", sum(r[2] for r in sil))
print("silent with indeg>0 but zero non-zero-sign presynaptic edges:", sum(1 for r in sil if r[5] > 0 and r[6] == 0))
print("non-silent features with zero nonzero-sign inputs:", sum(1 for r in rows if r[3] >= 1e-6 and r[6] == 0))
print("global sign counts (-1,0,+1):", [(int(k), int(v)) for k, v in zip(*np.unique(sign, return_counts=True))])
