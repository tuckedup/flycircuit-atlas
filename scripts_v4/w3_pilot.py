"""Stage W3 trained validation-only exploratory pilot of the W76 widened readout.
Spec: results_v4/learning_readout_v4_1/W3_PREREG.md (written before any run). Absolute paths only.
Run (guarded, NOT launched by the author):
    scripts_v4/run_guarded.sh W3 $FLYCIRCUIT_COPY/scripts_v4/w3_pilot.py
Selftest (no training, builds models only):  w3_pilot.py --selftest
Reuses the reviewed v4 learner (flycircuit.eval.learning_v4: run_setting, make_agent_v4 param groups). Single change:
readout set W25 -> W76 for sparse families; gru/encdec use the ORIGINAL base (same architecture as the v4 pilot).
"""
from _paths import CANON as _CANON  # noqa: F401,E402
from _paths import ROOT as _ROOT  # noqa: F401,E402
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = _ROOT
OUT = f"{ROOT}/results_v4/learning_readout_v4_1"
STAGE = "W3_trained_w76"
ROWS = f"{OUT}/{STAGE}_rows.jsonl"
CHOICES = f"{OUT}/{STAGE}_choices.json"
FINAL = f"{OUT}/{STAGE}_final.json"
MANIFEST = f"{OUT}/{STAGE}_manifest.json"
FAMILIES = ["connectome", "gru", "encdec", "null0", "null1", "null2", "random_reach"]
SPARSE = {"connectome", "null0", "null1", "null2", "random_reach"}
SEARCH_SEEDS = [1401, 1402, 1403, 1404]
CHECK_SEEDS = [1501, 1502, 1503, 1504]
RATIOS = [1 / 3, 1.0, 3.0]
WALL_LIMIT_S = 120 * 60
N_SEARCH = len(FAMILIES) * 9 * len(SEARCH_SEEDS)  # 252
N_CHECK = len(FAMILIES) * len(CHECK_SEEDS)  # 28
N_TOTAL = N_SEARCH + N_CHECK  # 280
MAX_FAIL = int(0.05 * N_TOTAL)  # abort if failures > 14
EXPECT_DECODER_N = {"sparse": 76 * 2, "gru": 76, "encdec": 816}  # gru/encdec = recorded v4 pilot values
T_PROCESS = time.time()
torch.set_num_threads(2)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def canon(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=1, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def anchors():
    """v3 control selection (source of the v4 pilot anchors). null*/random_reach use v3 bptt|null / bptt|random_reach."""
    from flycircuit.eval import learning_v4 as P
    sel = json.load(open(str(P.ORIG / "results/learning_v3/search/selection.json")))
    a = {}
    for f in FAMILIES:
        a[f] = float(sel["bptt|null" if f.startswith("null") else f"bptt|{f}"]["lr"])
    return a


def grid(c):
    return [float(f"{c * r:.6g}") for r in RATIOS]


def key(family, core, dec, seed):
    return json.dumps([family, float(core), float(dec), int(seed)], separators=(",", ":"))


def search_plan(anc):
    return [(f, c, d, s) for f in FAMILIES for c in grid(anc[f]) for d in grid(anc[f]) for s in SEARCH_SEEDS]


def build_w76(base, w0_path=f"{OUT}/W0_width_probe_manifest.json"):
    """Same construction as scripts_v4/width_probe.py; asserted equal to the W0 manifest set."""
    g = base.graph
    n = g.n
    sens = set(map(int, base.sensory_idx))
    r25 = [int(i) for i in base.readout_idx]
    indeg = np.bincount(g.post, minlength=n)
    contacts = np.bincount(g.post, weights=g.count, minlength=n)
    s25 = set(r25)
    elig = [i for i in range(n) if i not in sens and i not in s25 and indeg[i] > 0]
    elig.sort(key=lambda i: (-contacts[i], i))
    r76 = r25 + elig
    assert len(r25) == 25 and len(r76) == 76 and r76[:25] == r25
    w0 = json.load(open(w0_path))
    assert [int(i) for i in w0["sets"]["W76"]] == r76, "W76 differs from W0 manifest"
    assert [int(i) for i in w0["sets"]["W25"]] == r25, "W25 differs from W0 manifest"
    return r25, r76, hashlib.sha256(json.dumps(r76).encode()).hexdigest()


def with_ro(base, ro):
    from flycircuit.tasks.graph_loader import CircuitBundle
    b = base
    return CircuitBundle(b.graph, b.sign, b.sensory_idx, np.asarray(ro, dtype=base.readout_idx.dtype),
                         b.label, b.notes, b.modulatory)


def base_for(family, base25, base76):
    return base76 if family in SPARSE else base25  # gru/encdec: readout-independent, original base


def selftest():
    from flycircuit.eval import learning as L
    from flycircuit.eval import learning_v2 as V2
    from flycircuit.eval import learning_v4 as P
    from flycircuit.eval import learning_v4_select as S
    t0 = time.time()
    cwd0 = os.getcwd()
    anc = anchors()
    plan = search_plan(anc)
    assert len(plan) == N_SEARCH == 252 and len({key(*p) for p in plan}) == 252 and N_TOTAL == 280
    reg = json.load(open(f"{OUT}/seed_registry.json"))["used"]
    assert reg["W3_search_committed"] == SEARCH_SEEDS and reg["W3_check_committed"] == CHECK_SEEDS
    prior = {s for k, v in reg.items() if not k.startswith("W3") for s in v}
    assert not (set(SEARCH_SEEDS) | set(CHECK_SEEDS)) & prior, "W3 seeds already used"
    assert not (set(SEARCH_SEEDS) | set(CHECK_SEEDS)) & P.USED
    assert anc["null0"] == anc["null1"] == anc["null2"]
    # simplicity order sanity: anchor is simpler than 3x, 1/3 simpler than anchor
    c = anc["connectome"]
    assert S.two_group_simplicity((c, c), c) < S.two_group_simplicity((3 * c, c), c)
    proto, base25, packs = P._setup()
    r25, r76, h76 = build_w76(base25)
    base76 = with_ro(base25, r76)
    assert len(base76.readout_idx) == 76 and len(base25.readout_idx) == 25
    n_in = proto["task"]["n_in"] if "n_in" in proto["task"] else L.make_task(proto, 1401).spec.n_in
    # built models actually use the W76 bundle
    for fam in ("connectome", "null0", "null1", "null2", "random_reach"):
        m, b, _, _ = V2.build_model_v2(proto, base_for(fam, base25, base76), packs, fam, 1401, n_in)
        assert len(m.readout_idx) == 76, (fam, len(m.readout_idx))
        assert m.readout_idx.tolist() == r76 and m.readout_idx[:25].tolist() == r25, fam
        assert tuple(m.decoder.shape) == (76, 2), (fam, tuple(m.decoder.shape))
        assert len(b.readout_idx) == 76
    m25, *_ = V2.build_model_v2(proto, base25, packs, "connectome", 1401, n_in)
    assert len(m25.readout_idx) == 25 and tuple(m25.decoder.shape) == (25, 2)
    # gru/encdec unaffected: same objects/shapes/weights as the W25-base build used by the v4 pilot
    for fam, hid in (("gru", 37), ("encdec", 407)):
        mw, bw, _, info_w = V2.build_model_v2(proto, base_for(fam, base25, base76), packs, fam, 1501, n_in)
        mo, bo, _, info_o = V2.build_model_v2(proto, base25, packs, fam, 1501, n_in)
        assert bw is None and info_w == info_o == {"hidden": hid}, (fam, info_w, info_o)
        sw, so = mw.state_dict(), mo.state_dict()
        assert sw.keys() == so.keys() and all(torch.equal(sw[k], so[k]) for k in sw), fam
        _, _, _, info_leak = V2.build_model_v2(proto, base76, packs, fam, 1501, n_in)
        print(f"  note: {fam} matched hidden with W25 base={info_o['hidden']}, with W76 base would be {info_leak['hidden']}")
    # agent-level: param groups from the v4 learner see the W76 decoder (no training)
    task = L.make_task(proto, 1401)
    P.CUR["core"], P.CUR["dec"] = grid(c)[1], grid(c)[1]
    ag, meta = V2.make_agent_v2(proto, base76, packs, "bptt", "connectome", P.CUR["core"], 1401, task, P.CUR["dec"])
    assert meta["param_groups"]["decoder_n"] == EXPECT_DECODER_N["sparse"] == 152, meta["param_groups"]
    assert len(ag.model.readout_idx) == 76
    ag2, meta2 = V2.make_agent_v2(proto, base25, packs, "bptt", "gru", P.CUR["core"], 1401, task, P.CUR["dec"])
    assert meta2["param_groups"]["decoder_n"] == EXPECT_DECODER_N["gru"], meta2["param_groups"]
    # cwd independence: all paths absolute under ROOT
    os.chdir("/")
    try:
        for p in (OUT, ROWS, CHOICES, FINAL, MANIFEST):
            assert os.path.isabs(p) and p.startswith(ROOT)
        assert os.path.exists(f"{OUT}/W0_width_probe_manifest.json") and os.path.exists(f"{OUT}/W3_PREREG.md")
        assert build_w76(base25)[2] == h76
    finally:
        os.chdir(cwd0)
    print(f"SELFTEST PASS ({time.time() - t0:.0f}s): W76 bundle (76 slots, first 25 = original, sha {h76[:12]} == W0 manifest) "
          f"used by built models (readout_idx 76, decoder (76,2), agent decoder_n 152); gru/encdec built from W25 base, "
          f"weights identical; 252 search + 28 check; seeds unused; cwd independent")


def write_manifest(anc, h76):
    from flycircuit.eval import learning_v4 as P
    files = {
        "scripts_v4/w3_pilot.py": os.path.abspath(__file__),
        "scripts_v4/summarize_w3.py": f"{ROOT}/scripts_v4/summarize_w3.py",
        "scripts_v4/run_guarded.sh": f"{ROOT}/scripts_v4/run_guarded.sh",
        "scripts_v4/width_probe.py": f"{ROOT}/scripts_v4/width_probe.py",
        "results_v4/learning_readout_v4_1/W3_PREREG.md": f"{OUT}/W3_PREREG.md",
        "results_v4/learning_readout_v4_1/seed_registry.json": f"{OUT}/seed_registry.json",
        "results_v4/learning_readout_v4_1/W0_width_probe_manifest.json": f"{OUT}/W0_width_probe_manifest.json",
        "results_v4/learning_v4/pilot_manifest.json": str(P.MANIFEST),
        "src/flycircuit/eval/learning_v4.py": f"{ROOT}/src/flycircuit/eval/learning_v4.py",
        "src/flycircuit/eval/learning_v4_select.py": f"{ROOT}/src/flycircuit/eval/learning_v4_select.py",
        "src/flycircuit/eval/learning_v2.py": f"{ROOT}/src/flycircuit/eval/learning_v2.py",
        "src/flycircuit/eval/learning.py": f"{ROOT}/src/flycircuit/eval/learning.py",
        "configs/experiments/learning_v3_protocol.yaml": f"{ROOT}/configs/experiments/learning_v3_protocol.yaml",
    }
    plan = search_plan(anc)
    man = {"stage": STAGE, "written": time.strftime("%Y-%m-%d %H:%M:%S"), "torch_threads": 2,
           "families": FAMILIES, "anchors": anc, "ratios": RATIOS, "search_seeds": SEARCH_SEEDS,
           "check_seeds": CHECK_SEEDS, "w76_sha256": h76, "expected_rows": N_TOTAL,
           "search_plan_sha256": canon([list(p) for p in plan]), "wall_limit_s": WALL_LIMIT_S,
           "max_failures": MAX_FAIL, "v4_pilot_manifest_sha256": P.manifest_sha256(),
           "sha256": {k: sha_file(v) for k, v in files.items()}}
    core = {k: v for k, v in man.items() if k != "written"}
    path, k = MANIFEST, 0
    while os.path.exists(path):
        prev = json.load(open(path))
        if {kk: vv for kk, vv in prev.items() if kk != "written"} != core:
            raise RuntimeError(f"inputs changed since manifest {path}; refusing to resume (no reruns / no drift)")
        k += 1
        path = f"{OUT}/{STAGE}_manifest_resume{k}.json"
    atomic_json(path, man)
    return man, canon(core)


def select(rows, anc):
    """Predeclared rule: composite one-SE over 4 search seeds, two_group_simplicity order. Deterministic."""
    from flycircuit.eval import learning_v4_select as S
    out = {}
    for f in FAMILIES:
        t = {}
        for r in rows:
            if r["family"] == f and r["w3_phase"] == "search" and r["status"] == "ok":
                acq = r["association"]["val_acc_acq"]
                cont = r["continual"]["final_mean_acc_val"]
                t.setdefault((r["core_lr"], r["decoder_lr"]), []).append(S.composite_score(acq, cont))
        t = {k: v for k, v in t.items() if len(v) == len(SEARCH_SEEDS)}
        if not t:
            out[f] = {"chosen": None, "reason": "no setting with 4 ok seeds"}
            continue
        o = S.one_se_choice_2d(t, anc[f], min_seeds=len(SEARCH_SEEDS))
        g = grid(anc[f])
        out[f] = {"core_lr": o["chosen"][0], "decoder_lr": o["chosen"][1], "best": list(o["best"]),
                  "best_mean": o["best_mean"], "best_se": o["best_se"], "threshold": o["threshold"],
                  "n_indistinguishable": o["n_indistinguishable"], "chosen_equals_best": o["chosen_equals_best"],
                  "n_eligible_settings": len(t), "optimum_at_grid_edge": bool(
                      o["chosen"][0] in (g[0], g[-1]) or o["chosen"][1] in (g[0], g[-1]) or
                      o["best"][0] in (g[0], g[-1]) or o["best"][1] in (g[0], g[-1]))}
    return out


def main():
    from flycircuit.eval import learning_v4 as P
    if not str(P.__file__).startswith(ROOT):
        raise RuntimeError("must import flycircuit from the isolated copy (PYTHONPATH)")
    os.makedirs(OUT, exist_ok=True)
    anc = anchors()
    P.prepare_manifest()
    proto, base25, packs = P._setup()
    r25, r76, h76 = build_w76(base25)
    base76 = with_ro(base25, r76)
    man, core_hash = write_manifest(anc, h76)  # hashes BEFORE any compute
    print(f"manifest written core={core_hash[:12]} w76={h76[:12]}", flush=True)

    rows = P.read(__import__("pathlib").Path(ROWS))
    done = {key(r["family"], r["core_lr"], r["decoder_lr"], r["seed"]) for r in rows}
    n_fail = sum(1 for r in rows if r["status"] != "ok")
    aborted = None
    t_start = time.time()

    def run_cell(phase, fam, core, dec, seed):
        nonlocal n_fail, aborted
        k = key(fam, core, dec, seed)
        if k in done:
            return
        if time.time() - T_PROCESS > WALL_LIMIT_S:
            aborted = "wall limit 120 min"
            return
        r = P.run_setting(proto, base_for(fam, base25, base76), packs, fam, core, dec, seed)
        want = EXPECT_DECODER_N["sparse" if fam in SPARSE else fam]
        if r["status"] == "ok" and r.get("param_groups", {}).get("decoder_n") != want:
            r["status"] = f"failed: decoder_n {r.get('param_groups')} != {want} (readout set not as planned)"
        r.update(w3_stage=STAGE, w3_phase=phase, w3_manifest_core_sha256=core_hash,
                 readout_set="W76" if fam in SPARSE else "none", n_readout=76 if fam in SPARSE else None)
        P.append(__import__("pathlib").Path(ROWS), r)
        rows.append(r)
        done.add(k)
        if r["status"] != "ok":
            n_fail += 1
            if n_fail > MAX_FAIL:
                aborted = f"failures {n_fail} > 5% of {N_TOTAL}"
        sc = (r["association"]["val_acc_acq"], r["continual"]["final_mean_acc_val"]) if r["status"] == "ok" else None
        print(f"{phase} {fam:12s} core={core:<10g} dec={dec:<10g} seed={seed} {r['status'][:30]} "
              f"{r['elapsed_s']:.1f}s {sc} t={time.time() - t_start:.0f}s", flush=True)

    for fam, core, dec, seed in search_plan(anc):
        run_cell("search", fam, core, dec, seed)
        if aborted:
            break
    choices = None
    if not aborted:
        choices = select(rows, anc)
        if os.path.exists(CHOICES) and json.load(open(CHOICES)) != json.loads(json.dumps(choices)):
            raise RuntimeError("choices differ from earlier W3_choices.json (nondeterminism or drift)")
        atomic_json(CHOICES, choices)
        print("choices", json.dumps({f: (c.get("core_lr"), c.get("decoder_lr")) for f, c in choices.items()}), flush=True)
        for fam in FAMILIES:
            c = choices[fam]
            if c.get("core_lr") is None:
                continue
            for seed in CHECK_SEEDS:
                run_cell("check", fam, c["core_lr"], c["decoder_lr"], seed)
                if aborted:
                    break
            if aborted:
                break
    final = dict(stage=STAGE, aborted=aborted, n_rows=len(rows), n_failed=n_fail, expected_rows=N_TOTAL,
                 choices=choices, elapsed_s=time.time() - T_PROCESS, manifest_core_sha256=core_hash)
    atomic_json(FINAL, final)
    print("done" if not aborted else f"ABORTED: {aborted}", f"{time.time() - T_PROCESS:.0f}s", flush=True)
    return 0 if not aborted else 2


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        sys.exit(main())
