"""Stage W2_sign_policy unknown-sign policy probe (UNTRAINED, frozen features).
Spec: results_v4/learning_readout_v4_1/W2_PREREG.md. Absolute paths only.
Run: python sign_probe.py            (full probe; NOT launched by the author)
     python sign_probe.py --selftest (tiny CPU unit check, no data load)
Intervention: proto["model"]["unknown_sign_policy"] 'zero' (baseline) vs 'excitatory' (existing supported value in
sparse_rnn.SIGN_POLICIES). Done on a deep copy of the protocol; no existing source is edited.
feats/probe/stats and W76 set construction are copied from scripts_v4/width_probe.py (not imported).
"""
from _paths import CANON as _CANON  # noqa: F401,E402
from _paths import ROOT as _ROOT  # noqa: F401,E402
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

from flycircuit.eval import learning as L
from flycircuit.models.sparse_rnn import SIGN_POLICIES, SparseConnectomeRNN, ModelSpec, resolve_signs

ROOT = _ROOT
OUT = f"{ROOT}/results_v4/learning_readout_v4_1"
STAGE = "W2_sign_policy"
SEEDS = list(range(1301, 1309))
CONDS = ["baseline", "policy"]
POLICY = {"baseline": "zero", "policy": "excitatory"}
WALL_LIMIT = 12 * 60
MAX_FAIL_FRAC = 0.05
GRAPH_FAMS = ["connectome", "null0", "null1", "null2", "null3", "null4", "random_reach"]
N_CELLS = len(GRAPH_FAMS) * 2 + 1
N_ROWS_TOTAL = N_CELLS * len(CONDS) * len(SEEDS)  # 240
assert N_ROWS_TOTAL == 240
T_PROCESS = time.time()
torch.set_num_threads(2)


def selftest():
    assert "excitatory" in SIGN_POLICIES and "zero" in SIGN_POLICIES
    rng = np.random.default_rng(0)
    n = 60
    sign = rng.choice([-1, 0, 1], size=n, p=[0.1, 0.3, 0.6]).astype(np.float32)
    zero = resolve_signs(sign, n, "zero")
    exc = resolve_signs(sign, n, "excitatory")
    assert np.array_equal(zero, sign), "baseline (zero) sign vector must equal base.sign"
    assert zero.dtype == exc.dtype == np.float32
    unk = sign == 0
    assert unk.any() and np.array_equal(exc[~unk], sign[~unk]), "policy must not touch known entries"
    assert (exc[unk] == 1.0).all(), "policy must set unknown entries to +1"
    assert int((exc != zero).sum()) == int(unk.sum()), "policy changes only zero entries"
    # model-level: sign_node buffer reflects policy via the same constructor used by learning.build_sparse_model
    pre = rng.integers(0, n, 200)
    post = rng.integers(0, n, 200)
    cnt = np.ones(200, dtype=np.float32)
    spec = ModelSpec(n_in=19, n_out=2, dt=1.0, tau=5.0, prior="sqrt", init_gain=0.005, nonneg_gain=True, seed=1301)
    ms = {}
    for cond, pol in POLICY.items():
        try:
            ms[cond] = SparseConnectomeRNN(n, pre, post, cnt, spec, sign=sign, unknown_sign_policy=pol,
                                           sensory_idx=np.arange(5), readout_idx=np.arange(5, 10))
        except Exception as ex:  # prior name may differ from this synthetic spec; fall back to sign-vector check only
            print("note: synthetic model construction skipped:", type(ex).__name__, ex)
            ms = {}
            break
    if ms:
        sb, sp = ms["baseline"].sign_node.numpy(), ms["policy"].sign_node.numpy()
        assert np.array_equal(sb, sign) and np.array_equal(sp, exc)
    # protocol copy leaves original untouched
    proto = {"model": {"unknown_sign_policy": "zero"}}
    p2 = copy.deepcopy(proto)
    p2["model"]["unknown_sign_policy"] = POLICY["policy"]
    assert proto["model"]["unknown_sign_policy"] == "zero" and p2["model"]["unknown_sign_policy"] == "excitatory"
    print("SELFTEST PASS: baseline sign == base.sign; policy changes only zero entries (-> +1); protocol deep-copied")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest():
    files = {
        "scripts_v4/sign_probe.py": os.path.abspath(__file__),
        "results_v4/learning_readout_v4_1/W2_PREREG.md": f"{OUT}/W2_PREREG.md",
        "results_v4/learning_readout_v4_1/seed_registry.json": f"{OUT}/seed_registry.json",
        "results_v4/learning_readout_v4_1/W0_width_probe_manifest.json": f"{OUT}/W0_width_probe_manifest.json",
        "scripts_v4/width_probe.py": f"{ROOT}/scripts_v4/width_probe.py",
        "scripts_v4/encoder_probe.py": f"{ROOT}/scripts_v4/encoder_probe.py",
        "src/flycircuit/eval/learning.py": f"{ROOT}/src/flycircuit/eval/learning.py",
        "src/flycircuit/eval/learning_v2.py": f"{ROOT}/src/flycircuit/eval/learning_v2.py",
        "src/flycircuit/models/sparse_rnn.py": f"{ROOT}/src/flycircuit/models/sparse_rnn.py",
        "configs/experiments/learning_v3_protocol.yaml": f"{ROOT}/configs/experiments/learning_v3_protocol.yaml",
    }
    man = {"stage": STAGE, "written": time.strftime("%Y-%m-%d %H:%M:%S"), "torch_threads": 2,
           "seeds": SEEDS, "conditions": CONDS, "policy_values": POLICY, "expected_rows": N_ROWS_TOTAL,
           "sha256": {k: sha256(v) for k, v in files.items()}}
    path = f"{OUT}/{STAGE}_manifest.json"
    k = 0
    while os.path.exists(path):
        k += 1
        path = f"{OUT}/{STAGE}_manifest_resume{k}.json"
    json.dump(man, open(path, "w"), indent=1)
    return man


def main():
    from flycircuit.eval import learning_v2 as V2
    from flycircuit.eval import learning_v3 as V3
    from flycircuit.tasks.graph_loader import CircuitBundle

    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    selftest()
    write_manifest()  # hashes BEFORE any compute

    proto = V3.load_protocol()
    assert proto["model"]["unknown_sign_policy"] == "zero"
    protos = {}
    for cond, pol in POLICY.items():
        p = copy.deepcopy(proto)
        p["model"]["unknown_sign_policy"] = pol
        protos[cond] = p
    base = V2.load_base(proto, "flywire")
    packs = V2.load_packs(base, proto, "flywire")
    g = base.graph
    n = g.n
    # real-data baseline-path check: zero policy reproduces base.sign exactly; policy changes only the unknown nodes
    bs = np.asarray(base.sign, dtype=np.float32)
    assert np.array_equal(resolve_signs(bs, n, "zero"), bs)
    ex = resolve_signs(bs, n, "excitatory")
    assert int((ex != bs).sum()) == int((bs == 0).sum()) and (ex[bs == 0] == 1).all()
    n_unknown = int((bs == 0).sum())
    sens = set(map(int, base.sensory_idx))
    r25 = [int(i) for i in base.readout_idx]
    indeg = np.bincount(g.post, minlength=n)
    contacts = np.bincount(g.post, weights=g.count, minlength=n)
    elig = [i for i in range(n) if i not in sens and i not in set(r25) and indeg[i] > 0]
    elig.sort(key=lambda i: (-contacts[i], i))
    SETS = {"W25": r25, "W76": r25 + elig}
    assert len(SETS["W25"]) == 25 and len(SETS["W76"]) == 76
    try:
        w0 = json.load(open(f"{OUT}/W0_width_probe_manifest.json"))
        for k in SETS:
            assert [int(i) for i in w0["sets"][k]] == SETS[k], f"{k} differs from W0 manifest"
    except FileNotFoundError:
        print("warning: W0 manifest not found for set cross-check", flush=True)
    print(f"unknown-sign nodes: {n_unknown} of {n}", flush=True)

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
        raise RuntimeError("dense model not supported in W2")

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

    def run_cell(fam, label, cond, seed, ro=None):
        try:
            task = L.make_task(protos[cond], seed)
            b = base if ro is None else with_ro(base, ro)
            agent, _ = V2.make_agent_v2(protos[cond], b, packs, "bptt", fam, 1e-3, seed, task)
            sn = agent.model.sign_node.detach().cpu().numpy()
            if fam != "kc_static":
                chk = resolve_signs(np.asarray(b.sign, dtype=np.float32), n, POLICY[cond])
                assert np.array_equal(sn, chk), "model sign_node != expected policy vector"
            rng = np.random.default_rng(seed)
            rec = dict(stage=STAGE, condition=cond, family=label, seed=seed, n_unknown_sign=int((sn == 0).sum()))
            pc = []
            for ctx in range(task.spec.n_ctx):
                xt = feats(agent, task, "train", ctx, rng)
                xv = feats(agent, task, "val", ctx, rng)
                yt = torch.as_tensor(task.labels[("train", ctx)]).float().reshape(-1)
                yv = torch.as_tensor(task.labels[("val", ctx)]).reshape(-1)
                va, ta = probe(xt, xv, yt, yv)
                pc.append((va, ta))
                if ctx == 0:
                    rec.update(ctx0_val=va, ctx0_train=ta, **stats(xt))
            rec.update(n_feat=int(xt.shape[1]), percontext_val=float(np.mean([p[0] for p in pc])),
                       percontext_train=float(np.mean([p[1] for p in pc])))
            for k in ("ctx0_val", "ctx0_train", "eff_rank", "percontext_val"):
                if not np.isfinite(rec[k]):
                    raise ValueError(f"non-finite {k}")
            return rec
        except Exception as ex:  # recorded, counted, never dropped
            return dict(stage=STAGE, condition=cond, family=label, seed=seed, failed=True,
                        error=f"{type(ex).__name__}: {ex}")

    cells = [(fam, f"{fam}|{wn}", SETS[wn]) for wn in ("W76", "W25") for fam in GRAPH_FAMS]
    cells.append(("kc_static", "kc_static|native1385", None))
    assert len(cells) == N_CELLS

    jl = f"{OUT}/{STAGE}_rows.jsonl"
    done = set()
    n_fail = 0
    if os.path.exists(jl):
        for line in open(jl):
            line = line.strip()
            if line:
                r = json.loads(line)
                done.add((r["family"], r["condition"], r["seed"]))
                n_fail += 1 if r.get("failed") else 0
    fh = open(jl, "a")
    aborted = None
    t_run = time.time()
    for fam, label, ro in cells:
        for seed in SEEDS:
            for cond in CONDS:
                if (label, cond, seed) in done:
                    continue
                if time.time() - T_PROCESS > WALL_LIMIT:  # includes data load
                    aborted = "wall limit 12 min (incl. data load)"
                    break
                r = run_cell(fam, label, cond, seed, ro)
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
                 expected_rows=N_ROWS_TOTAL, n_unknown_sign_nodes=n_unknown, rows=rows)
    tmp = f"{OUT}/{STAGE}_rows.json.tmp"
    json.dump(final, open(tmp, "w"), indent=1)
    os.replace(tmp, f"{OUT}/{STAGE}_rows.json")
    print("done" if not aborted else f"ABORTED: {aborted}", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
