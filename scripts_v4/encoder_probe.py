"""Stage W1_encoder_E8 encoder-bundle probe (UNTRAINED, frozen features).
Spec: results_v4/learning_readout_v4_1/ENCODER_DIAGNOSTIC_PLAN.md incl. AMENDMENT 1. Absolute paths only.
Run: python encoder_probe.py            (full probe)
     python encoder_probe.py --selftest (tiny CPU unit check, no model construction)
feats/probe/stats/run_cell logic and W76 set construction are copied from scripts_v4/width_probe.py.
"""
import hashlib
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch

from flycircuit.eval import learning as L

ROOT = "<PROJECT_ROOT>"
OUT = f"{ROOT}/results_v4/learning_readout_v4_1"
STAGE = "W1_encoder_E8"
SEEDS = list(range(1101, 1109))
CONDS = ["baseline", "E8"]
WALL_LIMIT = 12 * 60
MAX_FAIL_FRAC = 0.05
N_ROWS_TOTAL = 144
GRAPH_FAMS = ["connectome", "null0", "null1", "null2", "null3", "null4", "random_reach"]
torch.set_num_threads(2)

# ---------------------------------------------------------------------------------------------
# Encoder wrapper. learning.build_sparse_model calls _fixed_encoder as a module global of
# flycircuit.eval.learning; learning_v2._kc_static calls it as L._fixed_encoder (module attribute).
# Both therefore resolve to the replacement below once L._fixed_encoder is reassigned.
# ---------------------------------------------------------------------------------------------
_ORIG_FIXED_ENCODER = L._fixed_encoder
FLAG = {"E8": False}
E8_EXTRA = 4
E8_W = float(np.float32(2.0 / np.sqrt(8.0)))


def e8_matrix(enc_base: np.ndarray, seed: int) -> np.ndarray:
    """Baseline 4-sets (nonzero rows of each column) + 4 distinct extra rows from the other n_in-4
    channels, drawn from SeedSequence([seed,5]); all 8 weights 2/sqrt(8)."""
    n_in, n_sens = enc_base.shape
    rng = np.random.default_rng(np.random.SeedSequence([seed, 5]))
    out = np.zeros_like(enc_base, dtype=np.float32)
    for j in range(n_sens):
        rows = np.flatnonzero(enc_base[:, j])
        rest = np.setdiff1d(np.arange(n_in), rows)
        extra = rng.choice(rest, size=E8_EXTRA, replace=False)
        out[np.concatenate([rows, extra]), j] = np.float32(2.0 / np.sqrt(8.0))
    return out


def _wrapped_fixed_encoder(model, fan_in, seed, n_cue):
    _ORIG_FIXED_ENCODER(model, fan_in, seed, n_cue)
    if FLAG["E8"]:
        base_np = model.encoder.detach().cpu().numpy().copy()
        new = e8_matrix(base_np, seed)
        with torch.no_grad():
            model.encoder.copy_(torch.as_tensor(new, dtype=model.encoder.dtype))
        model.encoder.requires_grad_(False)


def install_wrapper():
    L._fixed_encoder = _wrapped_fixed_encoder


def selftest():
    class M(torch.nn.Module):
        def __init__(self, n_in, n_sens):
            super().__init__()
            self.encoder = torch.nn.Parameter(torch.zeros(n_in, n_sens))

    n_in, n_sens, seed = 19, 60, 1101
    install_wrapper()
    m0, m1, m2 = M(n_in, n_sens), M(n_in, n_sens), M(n_in, n_sens)
    _ORIG_FIXED_ENCODER(m0, 4, seed, 16)
    FLAG["E8"] = False
    L._fixed_encoder(m1, 4, seed, 16)
    assert torch.equal(m0.encoder, m1.encoder) and m0.encoder.dtype == m1.encoder.dtype
    assert m0.encoder.detach().numpy().tobytes() == m1.encoder.detach().numpy().tobytes()
    FLAG["E8"] = True
    L._fixed_encoder(m2, 4, seed, 16)
    FLAG["E8"] = False
    b, e = m0.encoder.detach().numpy(), m2.encoder.detach().numpy()
    assert e.shape == b.shape and m2.encoder.dtype == m0.encoder.dtype and not m2.encoder.requires_grad
    for j in range(n_sens):
        rb, re_ = set(np.flatnonzero(b[:, j])), set(np.flatnonzero(e[:, j]))
        assert rb <= re_ and len(re_) == 8 and len(rb) == 4
        assert np.allclose(e[list(re_), j], 0.7071067811865476, atol=1e-6)
        assert abs(np.linalg.norm(e[:, j]) - 2.0) < 1e-5
    print("SELFTEST PASS: baseline bit-identical; E8 superset, 8 nonzeros of 0.7071, col L2 norm 2")


# ---------------------------------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest():
    files = {
        "scripts_v4/encoder_probe.py": os.path.abspath(__file__),
        "results_v4/learning_readout_v4_1/PREREG_WIDTH_PROBE.md": f"{OUT}/PREREG_WIDTH_PROBE.md",
        "results_v4/learning_readout_v4_1/ENCODER_DIAGNOSTIC_PLAN.md": f"{OUT}/ENCODER_DIAGNOSTIC_PLAN.md",
        "results_v4/learning_readout_v4_1/seed_registry.json": f"{OUT}/seed_registry.json",
        "scripts_v4/width_probe.py": f"{ROOT}/scripts_v4/width_probe.py",
        "src/flycircuit/eval/learning.py": f"{ROOT}/src/flycircuit/eval/learning.py",
        "src/flycircuit/eval/learning_v2.py": f"{ROOT}/src/flycircuit/eval/learning_v2.py",
        "src/flycircuit/models/sparse_rnn.py": f"{ROOT}/src/flycircuit/models/sparse_rnn.py",
    }
    man = {"stage": STAGE, "written": time.strftime("%Y-%m-%d %H:%M:%S"), "torch_threads": 2,
           "seeds": SEEDS, "conditions": CONDS,
           "sha256": {k: sha256(v) for k, v in files.items()}}
    path = f"{OUT}/{STAGE}_manifest.json"
    k = 0
    while os.path.exists(path):
        k += 1
        path = f"{OUT}/{STAGE}_manifest_resume{k}.json"
    json.dump(man, open(path, "w"), indent=1)
    return man


# ---------------------------------------------------------------------------------------------
def reach_fracs(model, max_hop=7):
    """Depth-limited multi-source BFS from sensory_idx over directed edges pre->post.
    Returns (frac readout nodes with min hop <=5, <=7); None for edgeless (kc_static)."""
    pre, post = model.pre.cpu().numpy(), model.post.cpu().numpy()
    if len(pre) == 0:
        return None, None
    n = model.n
    order = np.argsort(pre, kind="stable")
    pre_s, post_s = pre[order], post[order]
    start = np.searchsorted(pre_s, np.arange(n + 1))
    dist = np.full(n, -1, dtype=np.int64)
    src = model.sensory_idx.cpu().numpy()
    dist[src] = 0
    frontier = np.unique(src)
    for hop in range(1, max_hop + 1):
        nxt = []
        for u in frontier:
            nb = post_s[start[u]:start[u + 1]]
            nb = nb[dist[nb] < 0]
            if len(nb):
                dist[nb] = hop
                nxt.append(np.unique(nb))
        if not nxt:
            break
        frontier = np.unique(np.concatenate(nxt))
    ro = dist[model.readout_idx.cpu().numpy()]
    return float(((ro >= 0) & (ro <= 5)).mean()), float(((ro >= 0) & (ro <= 7)).mean())


@torch.no_grad()
def desc_stats(agent, task, proto, seed):
    """Untrained descriptive stats on the ctx0 train batch with a SEPARATE rng (probe rng untouched).
    Pre-activation = INPUT-ONLY drive x@encoder at window steps (recurrent term excluded, documented)."""
    m = agent.model
    rng = np.random.default_rng([seed, 77])
    x = torch.as_tensor(task.make_input(task.cues[("train", 0)], 0, rng))
    win = agent.window
    w = m.edge_values().detach()
    st = m.init_state(x.shape[1])
    pre_abs, hk, hr = [], [], []
    for t in range(x.shape[0]):
        _, st = m.step(x[t], st, w)
        if win.start <= t < win.stop:
            pre_abs.append((x[t] @ m.encoder).abs())
            hk.append(st["h"][:, m.sensory_idx].abs())
            hr.append(st["h"][:, m.readout_idx].abs())
    pre_abs, hk, hr = torch.cat(pre_abs), torch.cat(hk), torch.cat(hr)
    n_cue = proto["task"]["n_cue"]
    enc = m.encoder
    ctx_cnt = float((enc[n_cue:] != 0).float().sum(0).mean())
    r5, r7 = reach_fracs(m)
    return dict(sat_pre_kc=float((pre_abs > 2).float().mean()), sat_h_kc=float((hk > 0.95).float().mean()),
                sat_h_ro=float((hr > 0.95).float().mean()), kc_ctx_count=ctx_cnt,
                reach_le5=r5, reach_le7=r7, preact_note="input-only drive x@enc at window steps")


def main():
    from flycircuit.eval import learning_v2 as V2
    from flycircuit.eval import learning_v3 as V3
    from flycircuit.tasks.graph_loader import CircuitBundle

    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    selftest()
    install_wrapper()
    write_manifest()

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
    W76 = r25 + elig
    assert len(W76) == 76
    try:
        w0 = json.load(open(f"{OUT}/W0_width_probe_manifest.json"))
        assert [int(i) for i in w0["sets"]["W76"]] == W76, "W76 differs from W0 manifest"
    except FileNotFoundError:
        print("warning: W0 manifest not found for W76 cross-check", flush=True)

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
        raise RuntimeError("dense model not supported in W1")

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

    def run_cell(fam, label, cond, seed, ro=None, sub=None):
        FLAG["E8"] = cond == "E8"
        try:
            task = L.make_task(proto, seed)
            b = base if ro is None else with_ro(base, ro)
            agent, _ = V2.make_agent_v2(proto, b, packs, "bptt", fam, 1e-3, seed, task)
            rng = np.random.default_rng(seed)
            rec = dict(stage=STAGE, condition=cond, family=label, seed=seed)
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
                       percontext_train=float(np.mean([p[1] for p in pc])))
            rec.update(desc_stats(agent, task, proto, seed))
            for k in ("ctx0_val", "ctx0_train", "eff_rank", "percontext_val"):
                if not np.isfinite(rec[k]):
                    raise ValueError(f"non-finite {k}")
            return rec
        except Exception as ex:  # recorded, counted, never dropped
            return dict(stage=STAGE, condition=cond, family=label, seed=seed, failed=True,
                        error=f"{type(ex).__name__}: {ex}")
        finally:
            FLAG["E8"] = False

    cells = [(fam, f"{fam}|W76", W76, None) for fam in GRAPH_FAMS]
    cells.append(("kc_static", "kc_static|native1385", None, None))
    cells.append(("kc_static", "kc_static_sub|K76", None, 76))

    jl = f"{OUT}/{STAGE}_rows.jsonl"
    done = set()
    if os.path.exists(jl):
        for line in open(jl):
            line = line.strip()
            if line:
                r = json.loads(line)
                done.add((r["family"], r["condition"], r["seed"]))
    n_fail = sum(1 for line in (open(jl) if os.path.exists(jl) else []) if '"failed": true' in line)
    fh = open(jl, "a")
    aborted = None
    t_run = time.time()
    for fam, label, ro, sub in cells:
        for seed in SEEDS:
            for cond in CONDS:
                if (label, cond, seed) in done:
                    continue
                if time.time() - t_run > WALL_LIMIT:
                    aborted = "wall limit 12 min"
                    break
                r = run_cell(fam, label, cond, seed, ro, sub)
                if r.get("failed"):
                    n_fail += 1
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                if n_fail > MAX_FAIL_FRAC * N_ROWS_TOTAL:
                    aborted = f"failures {n_fail} > 5% of {N_ROWS_TOTAL}"
                    break
            if aborted:
                break
        print(label, round(time.time() - t0), "s", flush=True)
        if aborted:
            break
    fh.close()
    rows = [json.loads(line) for line in open(jl) if line.strip()]
    final = dict(stage=STAGE, aborted=aborted, n_rows=len(rows), n_failed=sum(1 for r in rows if r.get("failed")),
                 expected_rows=N_ROWS_TOTAL, rows=rows)
    tmp = f"{OUT}/{STAGE}_rows.json.tmp"
    json.dump(final, open(tmp, "w"), indent=1)
    os.replace(tmp, f"{OUT}/{STAGE}_rows.json")
    print("done" if not aborted else f"ABORTED: {aborted}", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
