"""FlyGym M4-M6 stage driver (main env). Physics runs in the isolated FlyGym env via subprocess.

  python -m flycircuit.bodies.stages freeze|refs|collect|search|train|eval|report
Stages are resumable. Launch heavy stages through scripts/linux/heavyjob.sh (job registry + lock).
Protocol: configs/experiments/flygym_protocol.yaml (flygym_nav_v1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from flycircuit.bodies import jobs as J

BUDGET_TICKS = 180        # 1.8 s of simulated time per episode, 10 ms control tick
RADIUS_MM = 1.5
RES = J.ROOT / "results/flygym"
PROTO_PATH = J.ROOT / "configs/experiments/flygym_protocol.yaml"
HASH_PATH = J.ROOT / "configs/experiments/flygym_protocol.sha256"
CONDITIONS = ["connectome_pilot_mb", "null_double_edge_swap", "random_sparse",
              "gru_param_matched", "encdec_param_matched"]
UNTRAINED = ["connectome_pilot_mb", "gru_param_matched"]


# ------------------------------------------------------------------ protocol
def proto() -> dict:
    return yaml.safe_load(PROTO_PATH.read_text())


def proto_hash() -> str:
    p = proto()
    p.pop("amendments", None)
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()


def require_frozen() -> str:
    if not HASH_PATH.exists() or HASH_PATH.read_text().split()[0] != proto_hash():
        raise SystemExit("flygym protocol not frozen or modified since freeze; run `stages freeze` "
                         "(and add an amendment) before train/eval")
    return proto_hash()


def run_rollouts(name: str, eps: list[dict], workers: int) -> Path:
    out = RES / name
    jf = J.write_jobs(out / "jobs.json", eps, BUDGET_TICKS, RADIUS_MM)
    env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", OMP_NUM_THREADS="1")
    r = subprocess.run([str(J.FLYGYM_PY), "-I", str(J.ROLLOUT), str(jf), str(out), str(workers)], env=env)
    if r.returncode != 0:
        raise SystemExit(f"rollout failed rc={r.returncode}")
    return out


# ------------------------------------------------------------------ stages
def cmd_freeze(a):
    HASH_PATH.write_text(proto_hash() + "\n")
    print("frozen", proto_hash())


def cmd_refs(a):
    eps = []
    for pol in ("teacher", "scripted", "straight"):
        eps += J.episodes(a.split, range(a.n), pol)
    out = run_rollouts(f"refs_{a.split}", eps, a.workers)
    rows = J.load_results(out)
    summ = {p: J.summarize([r for r in rows if r["policy"] == p]) for p in ("teacher", "scripted", "straight")}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


def cmd_collect(a):
    P = proto()["training"]["data"]
    lv = P["action_noise_levels"]
    per = P["n_train_episodes"] // len(lv)
    eps, k = [], 0
    for an in lv:
        eps += J.episodes("train", range(k, k + per), "teacher", action_noise=an, record=True, tag="bc")
        k += per
    eps += J.episodes("val", range(P["n_val_episodes"]), "teacher", action_noise=0.3, record=True, tag="bcval")
    out = run_rollouts("bc_data", eps, a.workers)
    print("recorded", len(list(out.glob("rec_*.npz"))), "episodes")


def load_dataset(prefix: str):
    d = RES / "bc_data"
    files = sorted(d.glob(f"rec_{prefix}*.npz"))
    if not files:
        raise FileNotFoundError(f"no recordings for {prefix} in {d}; run `stages collect` first")
    T = BUDGET_TICKS
    X = np.zeros((T, len(files), 10), np.float32)
    Y = np.zeros((T, len(files), 2), np.float32)
    M = np.zeros((T, len(files)), np.float32)
    for i, f in enumerate(files):
        z = np.load(f)
        n = min(T, len(z["x"]))
        X[:n, i], Y[:n, i], M[:n, i] = z["x"][:n], z["y"][:n], 1.0
    return X, Y, M


def _graph():
    from flycircuit.eval.navigation import load_mb_graph, load_protocol
    g, sign, meta = load_mb_graph()
    return g, sign, meta, load_protocol()


def train_one(model, data, lr: float, epochs: int, seed: int, tcfg: dict, threads: int = 4):
    import torch

    from flycircuit.models.train import detach_state
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    X, Y, M = (torch.as_tensor(a) for a in data)
    levels = np.array(tcfg["dropout_levels_in_training"])
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = np.random.default_rng(seed * 101 + 7)
    n = X.shape[1]
    losses = []
    for _ep in range(epochs):
        perm = rng.permutation(n)
        for b0 in range(0, n, tcfg["batch_sequences"]):
            idx = perm[b0:b0 + tcfg["batch_sequences"]]
            lv = rng.choice(levels, size=len(idx))
            mask = torch.as_tensor((rng.random((len(idx), 10)) < lv[:, None]).astype(np.float32))
            xb, yb, mb = X[:, idx] * (1 - mask)[None], Y[:, idx], M[:, idx]
            state = None
            for s in range(0, X.shape[0], tcfg["tbptt_steps"]):
                xc, yc, mc = (a[s:s + tcfg["tbptt_steps"]] for a in (xb, yb, mb))
                if float(mc.sum()) == 0:
                    break
                opt.zero_grad(set_to_none=True)
                out, state = model(xc, state)
                loss = ((((torch.tanh(out) - yc) ** 2).sum(-1)) * mc).sum() / mc.sum()
                if not torch.isfinite(loss):
                    return {"diverged": True, "loss": float("nan")}
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                opt.step()
                state = detach_state(state)
                losses.append(float(loss.detach()))
    return {"diverged": False, "loss": float(np.mean(losses[-20:])) if losses else float("nan")}


def val_loss(model, data, level: float = 0.25) -> float:
    import torch
    X, Y, M = (torch.as_tensor(a) for a in data)
    rng = np.random.default_rng(999)
    mask = torch.as_tensor((rng.random((X.shape[1], 10)) < level).astype(np.float32))
    with torch.no_grad():
        out, _ = model(X * (1 - mask)[None])
        return float((((torch.tanh(out) - Y) ** 2).sum(-1) * M).sum() / M.sum())


def _positive_minutes(value):
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("minutes must be positive and finite")
    return number


def _deadline(args):
    minutes = getattr(args, "max_minutes", None)
    return None if minutes is None else time.monotonic() + 60 * minutes


def cmd_search(a):
    from flycircuit.eval.navigation import build_model
    deadline = _deadline(a)
    require_frozen()
    P = proto()
    g, sign, meta, nproto = _graph()
    tr, vl = load_dataset("train"), load_dataset("val")
    S = P["training"]["lr_search"]
    out = RES / "search.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    for cond in CONDITIONS:
        for lr in S["grid"]:
            key = f"{cond}|{lr}"
            if key in res:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                print(f"search time slice complete; {len(res)} trials saved; resume the same command", flush=True)
                return 75
            t0 = time.time()
            model, _ = build_model(cond, S["seed"], g, sign, nproto)
            r = train_one(model, tr, lr, S["epochs"], S["seed"], P["training"], a.threads)
            r["val_loss"] = val_loss(model, vl) if not r["diverged"] else float("nan")
            r["seconds"] = time.time() - t0
            res[key] = r
            out.write_text(json.dumps(res, indent=2))
            print(key, r, flush=True)
    best = {}
    for cond in CONDITIONS:
        c = {lr: res[f"{cond}|{lr}"]["val_loss"] for lr in S["grid"]}
        ok = {k: v for k, v in c.items() if np.isfinite(v)}
        best[cond] = {"lr": min(ok, key=ok.get), "grid_edge": min(ok, key=ok.get) in (S["grid"][0], S["grid"][-1]),
                      "val_loss_by_lr": c}
    (RES / "best_lr.json").write_text(json.dumps(best, indent=2))
    print(json.dumps(best, indent=2))


def cmd_train(a):
    from flycircuit.bodies.export import export_policy
    from flycircuit.eval.navigation import build_model
    deadline = _deadline(a)
    require_frozen()
    P = proto()
    g, sign, meta, nproto = _graph()
    tr, vl = load_dataset("train"), load_dataset("val")
    best = json.loads((RES / "best_lr.json").read_text())
    mdir = RES / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    log_p = RES / "train_log.json"
    log = json.loads(log_p.read_text()) if log_p.exists() else {}
    for seed in P["seeds"]["train"]:
        for cond in CONDITIONS:
            key = f"{cond}_s{seed}"
            if key in log:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                print(f"training time slice complete; {len(log)} models saved; resume the same command", flush=True)
                return 75
            t0 = time.time()
            model, info = build_model(cond, seed, g, sign, nproto)
            if cond in UNTRAINED:
                export_policy(model, mdir / f"untrained_{key}.npz")
                info_u = {"val_loss_untrained": val_loss(model, vl)}
            else:
                info_u = {}
            r = train_one(model, tr, best[cond]["lr"], P["training"]["epochs"], seed, P["training"], a.threads)
            if r["diverged"]:
                log[key] = {"diverged": True}
            else:
                export_policy(model, mdir / f"{key}.npz")
                log[key] = {**r, **info_u, "lr": best[cond]["lr"], "val_loss": val_loss(model, vl),
                            "train_seconds": time.time() - t0, "param_report": info["param_report"],
                            "null_stats": info.get("null_stats"),
                            "reach_readout": info.get("null_reach_readout_fraction",
                                                      info.get("random_reach_readout_fraction"))}
            log_p.write_text(json.dumps(log, indent=2, default=str))
            print(key, {k: v for k, v in log[key].items() if k != "param_report"}, flush=True)


def cmd_eval(a):
    P = proto()
    require_frozen()
    E = P["evaluation"]
    mdir = RES / "models"
    eps = []
    for pol in ("teacher", "scripted", "straight"):
        for dr in (0.0, 0.25):
            eps += J.episodes("test", range(E["n_test_episodes"]), pol, dropout=dr)
        eps += J.episodes("test_ood", range(E["n_test_ood_episodes"]), pol)
    for seed in P["seeds"]["train"]:
        for cond in CONDITIONS:
            f = mdir / f"{cond}_s{seed}.npz"
            if not f.exists():
                print("missing (diverged?)", f)
                continue
            for dr in (0.0, 0.25):
                eps += J.episodes("test", range(E["n_test_episodes"]), str(f), dropout=dr)
            eps += J.episodes("test_ood", range(E["n_test_ood_episodes"]), str(f))
        for cond in UNTRAINED:
            f = mdir / f"untrained_{cond}_s{seed}.npz"
            if f.exists():
                eps += J.episodes("test", range(E["n_test_episodes"]), str(f), dropout=0.0)
    out = run_rollouts("eval", eps, a.workers)
    print("done", len(J.load_results(out)), "episodes")


def cmd_report(a):
    from flycircuit.eval.stats import bootstrap_ci, paired_bootstrap_diff
    P = proto()
    rows = J.load_results(RES / "eval")
    if not rows:
        raise SystemExit("no eval results")

    def label(r):
        pol = r["policy"]
        if pol in ("teacher", "scripted", "straight"):
            return pol, None
        stem = Path(r["policy_path"]).stem
        untr = stem.startswith("untrained_")
        stem = stem.replace("untrained_", "")
        cond, s = stem.rsplit("_s", 1)
        return ("untrained_" + cond if untr else cond), int(s)

    from collections import defaultdict
    cell = defaultdict(list)       # (cond, seed, split, dropout) -> rows
    for r in rows:
        c, s = label(r)
        cell[(c, s, r["split"], r["dropout"])].append(r)
    table, per_seed = {}, defaultdict(dict)
    for (c, s, sp, dr), rs in cell.items():
        succ = float(np.mean([x["success"] for x in rs]))
        spl = float(np.mean([x["spl"] for x in rs]))
        per_seed[(c, sp, dr)][s] = {"success": succ, "spl": spl, "n_episodes": len(rs),
                                    "n_status_failed": sum(x["status"] != "ok" for x in rs)}
    for (c, sp, dr), d in sorted(per_seed.items(), key=str):
        vals = [v["success"] for v in d.values()]
        spl = [v["spl"] for v in d.values()]
        table[f"{c}|{sp}|dr{dr}"] = {
            "n_units": len(vals), "success": bootstrap_ci(vals) if len(vals) >= 2 else {"estimate": vals[0]},
            "spl_mean": float(np.mean(spl)), "per_seed_success": {str(k): v["success"] for k, v in d.items()},
            "n_episodes_per_unit": next(iter(d.values()))["n_episodes"]}
    diffs = {}
    for ref in ("gru_param_matched", "null_double_edge_swap", "random_sparse", "encdec_param_matched"):
        for sp, dr in (("test", 0.0), ("test", 0.25), ("test_ood", 0.0)):
            a_, b_ = per_seed.get(("connectome_pilot_mb", sp, dr)), per_seed.get((ref, sp, dr))
            if a_ and b_:
                seeds = sorted(set(a_) & set(b_))
                diffs[f"connectome-minus-{ref}|{sp}|dr{dr}"] = paired_bootstrap_diff(
                    [a_[s]["success"] for s in seeds], [b_[s]["success"] for s in seeds])
    for c in UNTRAINED:
        for dr in (0.0,):
            a_, b_ = per_seed.get((c, "test", dr)), per_seed.get(("untrained_" + c, "test", dr))
            if a_ and b_:
                seeds = sorted(set(a_) & set(b_))
                diffs[f"trained-minus-untrained|{c}|test|dr{dr}"] = paired_bootstrap_diff(
                    [a_[s]["success"] for s in seeds], [b_[s]["success"] for s in seeds])
    out = {"protocol_id": P["protocol_id"], "protocol_hash": proto_hash(), "table": table,
           "paired_differences": diffs,
           "limitations": P["caveats"] + ["n=5 trained models per condition: intervals exploratory",
                                          "24 test + 12 OOD targets per model; episodes within a model are not independent replicates"]}
    (RES / "report.json").write_text(json.dumps(out, indent=2))
    lines = ["# FlyGym body-in-the-loop (flygym_nav_v1)", "", "| condition | split | dropout | n models | success [95% CI] | SPL |", "|---|---|---|---|---|---|"]
    for k, v in table.items():
        c, sp, dr = k.split("|")
        s = v["success"]
        ci = f"{s['estimate']:.3f}" + (f" [{s['lo']:.3f}, {s['hi']:.3f}]" if "lo" in s and s["lo"] == s["lo"] else "")
        lines.append(f"| {c} | {sp} | {dr} | {v['n_units']} | {ci} | {v['spl_mean']:.3f} |")
    lines += ["", "## Paired differences (mean over models)", ""]
    for k, v in diffs.items():
        lines.append(f"- {k}: {v['estimate']:+.3f}" + (f" [{v['lo']:+.3f}, {v['hi']:+.3f}]" if v["lo"] == v["lo"] else ""))
    (RES / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("freeze", cmd_freeze), ("refs", cmd_refs), ("collect", cmd_collect), ("search", cmd_search),
                     ("train", cmd_train), ("eval", cmd_eval), ("report", cmd_report)):
        s = sub.add_parser(name)
        s.add_argument("--split", default="train")
        s.add_argument("--n", type=int, default=12)
        s.add_argument("--workers", type=int, default=5)
        s.add_argument("--threads", type=int, default=4)
        if name in ("search", "train"):
            s.add_argument("--max-minutes", type=_positive_minutes, default=None,
                           help="pause between completed trials/models; exit 75 and resume without changing epoch budgets")
        s.set_defaults(fn=fn)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
