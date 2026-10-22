"""Learning study v3 (protocol ``learning-v3.0``): prospective follow-up to v1/v2 (both preserved).

Reuses the v2 task, agents, models and statistics (``eval.learning_v2``) and changes only what v2's audit
identified (see ``configs/prereg_v3.yaml``):
  * no borrowed learning rates: ``plast_dec`` searches its own plastic AND decoder lr (equal 12-evaluation budget);
  * plastic arms get an equal-budget ``sgd``/``rmsprop`` option search (gradient arms have one Adam lr);
  * 8 independent seeds, fresh disjoint from every earlier seed list;
  * fewer, relevant conditions (the v2 interface controls stay the v2 result).

Stages (``results/learning_v3/<stage>/``): ``search`` (VALIDATION only) -> ``pilot`` (VALIDATION only,
engineering sanity, changes nothing) -> ``freeze`` -> ``confirm`` (test split touched here and only here).

  python -m flycircuit.eval.learning_v3 search|pilot|freeze|confirm|summary [--synthetic]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from flycircuit import v3_integrity as I
from flycircuit.config import project_root
from flycircuit.eval import learning_v2 as V2
from flycircuit.eval import stats as S

PROTOCOL_VERSION = "learning-v3.0"
ROOT = project_root()
PROTO_PATH = ROOT / "configs" / "experiments" / "learning_v3_protocol.yaml"
HASH_PATH = ROOT / "configs" / "experiments" / "learning_v3_protocol.sha256"
PREREG_PATH = ROOT / "configs" / "prereg_v3.yaml"
PROVENANCE = ROOT / "results/learning_v3/provenance"
FREEZE_MANIFEST = PROVENANCE / "confirmation_freeze.json"
LEGACY_LEDGER = PROVENANCE / "legacy_search_eligibility.json"
LEGACY_LEDGER_SHA256 = "d83b9d3c389e54ff76f3c81f8d75e223dc1ba165e621407b7ace3efe954d8d1a"
SCRATCH_SEEDS_FORBIDDEN = {1, 2, 7, 11, 12, 21, 22, 101, 102, 103, 104, 105, 201, 202, 203, 204, 205, 206,
                           207, 208, 300, 301, 302, 303, 304, 305, 900, 100201, 100202, 100203, 100204,
                           100205}

KINDS = [("decoder_only", "connectome"), ("decoder_only", "kc_static"),
         ("bptt", "connectome"), ("bptt", "random_reach"), ("bptt", "gru"), ("bptt", "gru_c"),
         ("bptt", "encdec"), ("bptt", "null"),
         ("plast_fixed", "connectome"), ("plast_shuffled", "connectome"),
         ("plast_dec", "connectome"), ("plast_dec", "null")]
LR_FROM = {"plast_modoff": "plast_fixed"}      # isolated factor: no own search (as in v2)


# ---- protocol -------------------------------------------------------------------------------------
def load_protocol(path: Path | None = None) -> dict:
    return yaml.safe_load((path or PROTO_PATH).read_text())


def protocol_hash(proto: dict) -> str:
    return V2.protocol_hash(proto)


def check_seed_hygiene(proto: dict) -> None:
    cohorts = [proto["seeds"]["confirmatory"], proto["seeds"]["pilot"], proto["search"]["seeds"]]
    if any(len(x) != len(set(x)) for x in cohorts) or sum(map(len, cohorts)) != len(set(sum(cohorts, []))):
        raise ValueError("search, pilot and confirmation seeds must be unique and mutually disjoint")
    mine = set(proto["seeds"]["confirmatory"]) | set(proto["seeds"]["pilot"]) | set(proto["search"]["seeds"])
    bad = mine & SCRATCH_SEEDS_FORBIDDEN
    if bad:
        raise ValueError(f"v3 seeds overlap earlier studies: {sorted(bad)}")


def conditions(proto: dict) -> list[tuple[str, str, str, str]]:
    """(arm, family, mode, pairing). 'factorial' = every null x every seed."""
    kn = proto["seeds"]["null_graphs"]
    nulls = [f"null{i}" for i in range(kn)]
    out: list[tuple[str, str, str, str]] = []
    for f in ("connectome", "random_reach", "gru", "gru_c", "encdec"):
        out.append(("bptt", f, "std", "single"))
    out += [("bptt", n, "std", "factorial") for n in nulls]
    out += [("decoder_only", "connectome", "std", "single"), ("decoder_only", "kc_static", "std", "single"),
            ("plast_fixed", "connectome", "std", "single"), ("plast_modoff", "connectome", "std", "single"),
            ("plast_shuffled", "connectome", "std", "single"), ("plast_dec", "connectome", "std", "single")]
    out += [("plast_dec", n, "std", "factorial") for n in nulls]
    out += [("bptt", f, "replay", "single") for f in ("connectome", "gru", "encdec")]
    out.append(("plast_dec", "connectome", "replay", "single"))
    return out


# ---- storage ----------------------------------------------------------------------------------------
def stage_dir(stage: str) -> Path:
    d = ROOT / "results" / "learning_v3" / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(r: dict) -> str:
    # Preserve full float identity; :g previously rounded distinct configurations.
    return json.dumps([r['arm'], r['family'], r['mode'], r.get('optimizer', 'sgd'),
                       float(r['lr']), None if r.get('decoder_lr') is None else float(r['decoder_lr']),
                       r['seed']], separators=(",", ":"), allow_nan=False)


def read_records(stage: str) -> list[dict]:
    p = stage_dir(stage) / "records.jsonl"
    records = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    if not records:
        return records
    proto = load_protocol()
    legacy = legacy_record_hashes() if stage == "search" else set()
    I.validate_record_identity(records, protocol_hash(proto), legacy)
    inputs_id = I.canonical_hash(runtime_inputs(stage.endswith("_synthetic")))
    keys = [_key(r) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate records in {stage}; preserve and investigate rather than overwriting")
    for r in records:
        if r.get("stage") != stage:
            raise ValueError("record stage identity does not match its storage namespace")
        expected_splits = ["val", "test"] if stage == "confirm" else ["val"]
        if r.get("splits") != expected_splits:
            raise ValueError("record split identity differs from the prospective stage")
        if r.get("input_id") != inputs_id and I.canonical_hash(r) not in legacy:
            raise ValueError("record scientific inputs differ from the current immutable inventory")
    return records


def legacy_record_hashes() -> set[str]:
    if not LEGACY_LEDGER.is_file() or I.file_hash(LEGACY_LEDGER) != LEGACY_LEDGER_SHA256:
        raise ValueError("exact pre-correction validation eligibility ledger is missing or modified")
    doc = json.loads(LEGACY_LEDGER.read_text())
    if doc.get("allowed_stage") != "search" or doc.get("allowed_splits") != ["val"]:
        raise ValueError("legacy ledger must be limited to validation search")
    return set(doc["record_hashes"])


@lru_cache(maxsize=2)
def runtime_inputs(synthetic=False) -> dict:
    return {"circuit": "synthetic"} if synthetic else I.circuit_inputs(
        load_protocol()["circuit"]["name"], include_group=True)


def prereg_spec_hash() -> str:
    doc = yaml.safe_load(PREREG_PATH.read_text())
    changing = {"status", "frozen", "frozen_utc", "protocol_hash", "selection_sha256"}
    return I.canonical_hash({k: v for k, v in doc.items() if k not in changing})


def stage_identity(stage: str, proto: dict) -> dict:
    identity = {"phase": stage, "protocol_hash": protocol_hash(proto),
                "inputs": runtime_inputs(stage.endswith("_synthetic")),
                "runtime": I.runtime_identity(), "prereg_spec_hash": prereg_spec_hash()}
    if not stage.endswith("_synthetic"):
        identity["legacy_ledger_sha256"] = LEGACY_LEDGER_SHA256
    if not stage.startswith("search"):
        suffix = "_synthetic" if stage.endswith("_synthetic") else ""
        identity["selection_sha256"] = I.file_hash(stage_dir("search" + suffix) / "selection.json")
    return identity


def ensure_stage_manifest(stage: str, proto: dict):
    path = PROVENANCE / (stage + ".json")
    # Existing confirmation data must never be retroactively given a freeze.
    if stage == "confirm" and not path.exists() and read_records(stage):
        raise ValueError("confirmation records predate their stage manifest")
    read_records(stage)  # validates any resumed cache before creating a manifest
    configs = [PROTO_PATH, PREREG_PATH]
    if not stage.endswith("_synthetic"):
        configs.append(LEGACY_LEDGER)
    if not stage.startswith("search"):
        suffix = "_synthetic" if stage.endswith("_synthetic") else ""
        configs.append(stage_dir("search" + suffix) / "selection.json")
    return I.freeze_manifest(path, stage_identity(stage, proto), configs)


def json_finite(value):
    """Search failures remain in raw records; nonfinite selection scores are null."""
    if isinstance(value, dict):
        return {k: json_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_finite(v) for v in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def _append(stage: str, rec: dict) -> None:
    source_id, _ = I._unchanged_source()
    if rec.get("source_id") != source_id:
        raise ValueError("run source changed before the result was persisted")
    with open(stage_dir(stage) / "records.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---- one run ---------------------------------------------------------------------------------------
def run_one_v3(proto, base, packs, arm, family, mode, lr, seed, optimizer, decoder_lr, splits,
               protocols, stage) -> dict:
    """Delegates to the v2 runner with the plastic optimizer set per run; stamps v3 identity."""
    if "test" in splits and stage != "confirm":
        raise RuntimeError("test split may only be evaluated in the frozen 'confirm' stage")
    source_id, _ = I._unchanged_source()
    p = copy.deepcopy(proto)
    p["plasticity"]["optimizer"] = optimizer if arm.startswith("plast") else "sgd"
    rec = V2.run_one_v2(p, base, packs, arm, family, mode, lr, seed, splits=tuple(splits),
                        protocols=tuple(protocols), stage=None, save_ckpt=False, decoder_lr=decoder_lr)
    rec.update(protocol=PROTOCOL_VERSION, protocol_hash=protocol_hash(proto), stage=stage,
               optimizer=p["plasticity"]["optimizer"], source_id=source_id,
               input_id=I.canonical_hash(runtime_inputs(stage.endswith("_synthetic"))))
    return rec


# ---- equal-budget search (validation only) ------------------------------------------------------------
def _next_lr(scores: dict[float, float]) -> float:
    lrs = sorted(scores)
    best = max(lrs, key=lambda k: (scores[k], -k))
    i = lrs.index(best)
    if i == 0:
        nxt = best / 3.0
    elif i == len(lrs) - 1:
        nxt = best * 3.0
    else:
        nb = lrs[i - 1] if scores[lrs[i - 1]] >= scores[lrs[i + 1]] else lrs[i + 1]
        nxt = float(np.sqrt(best * nb))
    if any(abs(np.log(nxt / k)) < 1e-9 for k in lrs):  # duplicate: step outward instead
        nxt = lrs[-1] * 3.0
    return float(nxt)


def search_1d(evalfn, grid: list[float], n_total: int, exclude=()) -> dict[float, float]:
    """Evaluate n_total NEW coordinates; cached anchors cannot consume the budget."""
    if n_total <= 0 or not grid or any(not np.isfinite(k) or k <= 0 for k in grid):
        raise ValueError("search requires a positive budget and finite positive initial rates")
    excluded = set(exclude)
    scores: dict[float, float] = {}
    for lr in grid:
        if len(scores) == n_total:
            break
        if lr not in excluded and lr not in scores:
            scores[lr] = evalfn(lr)
    while len(scores) < n_total:
        nxt = _next_lr(scores) if scores else min(grid) / 3.0
        # Deterministic prospective extension if a proposal lands on a cached anchor.
        while nxt in excluded or nxt in scores:
            nxt *= 3.0
        if not np.isfinite(nxt) or nxt <= 0:
            raise ValueError("learning-rate extension left the finite positive domain")
        scores[nxt] = evalfn(nxt)
    return scores


def _best(scores: dict[float, float]) -> float:
    return max(sorted(scores), key=lambda k: (scores[k], -k))


def _edge(scores: dict[float, float]) -> bool:
    lrs = sorted(scores)
    return _best(scores) in (lrs[0], lrs[-1])


def make_evaluator(proto, base, packs, stage, arm, fam, optimizer, decoder_lr, vary, fixed_lr, log, done,
                   name):
    """Returns f(x) = mean validation score over the search seeds; ``vary`` in {'lr','decoder_lr'}."""
    seeds = proto["search"]["seeds"]

    def f(x: float) -> float:
        lr = x if vary == "lr" else fixed_lr
        dlr = decoder_lr if vary == "lr" else x
        vals = []
        for sd in seeds:
            probe = {"arm": arm, "family": fam, "mode": "std", "optimizer": optimizer, "lr": lr,
                     "decoder_lr": dlr, "seed": sd}
            r = done.get(_key(probe))
            if r is None:
                V2._check_deadline()
                r = run_one_v3(proto, base, packs, arm, fam, "std", lr, sd, optimizer, dlr, ("val",),
                               ("continual",), stage)
                _append(stage, r)
                done[_key(r)] = r
                log(f"search {name:28s} opt={optimizer:7s} lr={lr:<9g} dlr={dlr} seed={sd} "
                    f"{r['status']} {r['elapsed_s']:.1f}s")
            if r["status"] == "ok":
                vals.append(r["continual"]["final_mean_acc_val"])
        return float(np.mean(vals)) if len(vals) == len(seeds) and np.isfinite(vals).all() else float("-inf")

    return f


def validate_search_group(records, proto, arm, family):
    """Count distinct full configurations with exactly the paired planned seeds."""
    group = [r for r in records if r["arm"] == arm and r["family"] == family]
    keys = [_key(r) for r in group]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate search records for {arm}|{family}")
    settings = {}
    for r in group:
        if r["mode"] != "std":
            raise ValueError("search records must use the standard mode")
        setting = _key({**r, "seed": None})
        settings.setdefault(setting, set()).add(r["seed"])
    if (len(settings) != proto["search"]["budget_total_lr_evals"]
            or any(s != set(proto["search"]["seeds"]) for s in settings.values())):
        raise ValueError(f"incomplete/unbalanced unique search budget for {arm}|{family}")
    return len(settings)


def run_search(proto, base, packs, stage="search", log=print) -> dict:
    check_seed_hygiene(proto)
    ensure_stage_manifest(stage, proto)
    s = proto["search"]
    n_total = s["budget_total_lr_evals"]
    existing = read_records(stage)
    done = {_key(r): r for r in existing}
    if len(done) != len(existing):
        raise ValueError("duplicate search run keys; refuse ambiguous cached results")
    sel_path = stage_dir(stage) / "selection.json"
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    for arm, kind in KINDS:
        name = f"{arm}|{kind}"
        if sel.get(name, {}).get("complete"):
            validate_search_group(list(done.values()), proto, arm, "null0" if kind == "null" else kind)
            # Reconstruct the selected setting from the same cached validation
            # scores. A checkpoint's `complete` field is not evidence that its
            # selected rate or claimed budget was calculated correctly.
        fam = "null0" if kind == "null" else kind
        entry: dict = {"arm": arm, "kind": kind}
        if arm in ("bptt", "decoder_only"):
            f = make_evaluator(proto, base, packs, stage, arm, fam, "sgd", None, "lr", None, log, done, name)
            sc = search_1d(f, s["initial_grid"]["gradient"], n_total)
            entry.update(optimizer="sgd", lr=_best(sc), decoder_lr=None, n_evals=len(sc),
                         optimum_at_edge=_edge(sc), scores={repr(float(k)): v for k, v in sorted(sc.items())})
        elif arm in ("plast_fixed", "plast_shuffled"):
            per = n_total // len(s["plastic_optimizer_options"])
            best = None
            entry["by_optimizer"] = {}
            for opt in s["plastic_optimizer_options"]:
                f = make_evaluator(proto, base, packs, stage, arm, fam, opt, None, "lr", None, log, done, name)
                sc = search_1d(f, s["initial_grid"][f"plastic_{opt}"], per)
                entry["by_optimizer"][opt] = {"scores": {repr(float(k)): v for k, v in sorted(sc.items())},
                                              "lr": _best(sc), "best_score": sc[_best(sc)],
                                              "optimum_at_edge": _edge(sc), "n_evals": len(sc)}
                if best is None or sc[_best(sc)] > best[1]:   # tie -> first option (sgd)
                    best = (opt, sc[_best(sc)])
            opt = best[0]
            entry.update(optimizer=opt, lr=entry["by_optimizer"][opt]["lr"], decoder_lr=None,
                         n_evals=sum(v["n_evals"] for v in entry["by_optimizer"].values()),
                         optimum_at_edge=entry["by_optimizer"][opt]["optimum_at_edge"])
        elif arm == "plast_dec":
            pf = sel.get("plast_fixed|connectome")
            dec0 = sel.get("decoder_only|connectome")
            if not pf or not dec0:
                raise RuntimeError("plast_dec search needs plast_fixed|connectome and decoder_only|connectome")
            opt, dlr0 = pf["optimizer"], dec0["lr"]
            n_p, n_d = s["plast_dec_split"]["plastic_lr_evals"], s["plast_dec_split"]["decoder_lr_evals"]
            f1 = make_evaluator(proto, base, packs, stage, arm, fam, opt, dlr0, "lr", None, log, done, name)
            sp = search_1d(f1, s["initial_grid"][f"plastic_{opt}"], n_p)
            lr_best = _best(sp)
            f2 = make_evaluator(proto, base, packs, stage, arm, fam, opt, None, "decoder_lr", lr_best, log,
                                done, name)
            sd = search_1d(f2, s["initial_grid"]["gradient"], n_d, exclude=(dlr0,))
            # The plastic-coordinate anchor already exists. It remains eligible for
            # selection but is not counted again as a new decoder configuration.
            decoder_scores = {dlr0: sp[lr_best], **sd}
            entry.update(optimizer=opt, lr=lr_best, decoder_lr=_best(decoder_scores), n_evals=len(sp) + len(sd),
                         optimum_at_edge=bool(_edge(sp) or _edge(decoder_scores)),
                         cached_decoder_anchor={"lr": dlr0, "score": sp[lr_best]},
                         scores_plastic_lr={repr(float(k)): v for k, v in sorted(sp.items())},
                         scores_decoder_lr={repr(float(k)): v for k, v in sorted(sd.items())},
                         decoder_lr_start_point=dlr0)
        validate_search_group(list(done.values()), proto, arm, fam)
        entry["complete"] = True
        sel[name] = entry
        I.atomic_json(sel_path, json_finite(sel))
    return sel


def lr_for(sel: dict, arm: str, family: str) -> tuple[str, float, float | None]:
    src = LR_FROM.get(arm, arm)
    e = sel[f"{src}|{V2.kind_of(family)}"]
    return e["optimizer"], e["lr"], e.get("decoder_lr")


def build_jobs(proto: dict, sel: dict, seeds: list[int]) -> list[tuple]:
    kn = proto["seeds"]["null_graphs"]
    jobs = []
    for arm, fam, mode, pairing in conditions(proto):
        opt, lr, dlr = lr_for(sel, arm, fam)
        for si, sd in enumerate(seeds):
            if V2.is_null(fam) and pairing == "paired" and int(fam.lstrip("gnul")) != si % kn:
                continue
            jobs.append((arm, fam, mode, opt, lr, dlr, sd))
    return jobs


# ---- pilot / confirm ----------------------------------------------------------------------------------
def run_stage(stage, proto, base, packs, sel, seeds, splits, protocols, subset=None, log=print):
    ensure_stage_manifest(stage, proto)
    done = {_key(r) for r in read_records(stage)}
    jobs = build_jobs(proto, sel, seeds)
    if subset:
        jobs = [j for j in jobs if (j[0], j[1], j[2]) in subset]
    for n, (arm, fam, mode, opt, lr, dlr, sd) in enumerate(jobs):
        probe = {"arm": arm, "family": fam, "mode": mode, "optimizer": opt, "lr": lr, "decoder_lr": dlr,
                 "seed": sd}
        if _key(probe) in done:
            continue
        V2._check_deadline()
        r = run_one_v3(proto, base, packs, arm, fam, mode, lr, sd, opt, dlr, splits, protocols, stage)
        _append(stage, r)
        done.add(_key(r))
        a = r.get("association", {}).get("test_acc_acq", r.get("association", {}).get("val_acc_acq"))
        log(f"[{n + 1}/{len(jobs)}] {arm:14s} {fam:13s} {mode:6s} opt={opt:7s} seed={sd} {r['status']} "
            f"{r['elapsed_s']:.1f}s acc_acq={a}")
    return read_records(stage)


PILOT_SUBSET = {("bptt", "connectome", "std"), ("bptt", "gru", "std"), ("bptt", "encdec", "std"),
                ("decoder_only", "kc_static", "std"), ("plast_fixed", "connectome", "std"),
                ("plast_dec", "connectome", "std")}


def validate_stage_coverage(stage, proto, sel, records):
    seeds = proto["seeds"]["pilot" if stage == "pilot" else "confirmatory"]
    jobs = build_jobs(proto, sel, seeds)
    if stage == "pilot":
        jobs = [j for j in jobs if (j[0], j[1], j[2]) in PILOT_SUBSET]
    expected = {_key({"arm": a, "family": f, "mode": m, "optimizer": o,
                      "lr": lr, "decoder_lr": dlr, "seed": sd}) for a, f, m, o, lr, dlr, sd in jobs}
    actual = [_key(r) for r in records]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError(f"{stage} lacks the exact planned condition/seed matrix")
    failed, invalid = [], []
    split = "val" if stage == "pilot" else "test"
    n = proto["task"]["n_ctx"]
    for r in records:
        if r["status"] != "ok":
            failed.append(r)
            continue
        try:
            c = r["continual"]
            matrix = np.asarray(c[f"R_{split}"], dtype=float)
            endpoints = [r["association"][f"{split}_acc_acq"], c[f"final_mean_acc_{split}"],
                         c[f"avg_forgetting_keep_{split}"]]
            if (matrix.shape != (n, n) or not np.isfinite(matrix).all() or
                    not ((matrix >= 0) & (matrix <= 1)).all() or not np.isfinite(endpoints).all()):
                raise ValueError("invalid endpoint values")
            mean = float(matrix[-1].mean())
            forget = float(np.mean([matrix[j:-1, j].max() - matrix[-1, j] for j in range(n - 1)]))
            if (not 0 <= endpoints[0] <= 1 or abs(mean - endpoints[1]) > 1e-10 or
                    abs(forget - endpoints[2]) > 1e-10):
                raise ValueError("endpoint arithmetic mismatch")
        except (KeyError, ValueError, TypeError, IndexError):
            invalid.append(r)
    return {"complete": True, "expected_runs": len(expected), "failed_runs": len(failed) + len(invalid),
            "failed_execution_runs": len(failed), "invalid_endpoint_runs": len(invalid)}


def freeze_identity(proto):
    identity = stage_identity("confirmation_freeze", proto)
    identity.update(search_records_sha256=I.file_hash(stage_dir("search") / "records.jsonl"),
                    pilot_records_sha256=I.file_hash(stage_dir("pilot") / "records.jsonl"))
    return identity


def freeze() -> dict:
    proto = load_protocol()
    check_seed_hygiene(proto)
    if not FREEZE_MANIFEST.exists() and read_records("confirm"):
        raise SystemExit("confirm records already exist; cannot freeze after seeing confirmatory data")
    sel_path = stage_dir("search") / "selection.json"
    if not sel_path.exists():
        raise SystemExit("search selection missing; run `search` first")
    sel = json.loads(sel_path.read_text())
    missing = [f"{a}|{k}" for a, k in KINDS if not sel.get(f"{a}|{k}", {}).get("complete")]
    if missing:
        raise SystemExit(f"search incomplete: {missing}")
    records = read_records("search")
    for arm, kind in KINDS:
        fam = "null0" if kind == "null" else kind
        validate_search_group(records, proto, arm, fam)
        opt, lr, dlr = lr_for(sel, arm, fam)
        chosen = {_key({"arm": arm, "family": fam, "mode": "std", "optimizer": opt,
                       "lr": lr, "decoder_lr": dlr, "seed": sd}) for sd in proto["search"]["seeds"]}
        selected = [r for r in records if _key(r) in chosen]
        if (len(selected) != len(chosen) or any(r["status"] != "ok" or not np.isfinite(
                r["continual"]["final_mean_acc_val"]) for r in selected)):
            raise SystemExit(f"selected configuration lacks complete valid search seeds: {arm}|{kind}")
    pilot = read_records("pilot")
    coverage = validate_stage_coverage("pilot", proto, sel, pilot)
    rep = pilot_report(pilot)
    if coverage["failed_runs"] or rep.get("failures"):
        raise SystemExit("pilot has failed or nonfinite units; preserve them and investigate before freezing")
    h = protocol_hash(proto)
    sel_sha = hashlib.sha256(sel_path.read_bytes()).hexdigest()
    # Create once BEFORE confirmation. Re-entry verifies the same durable manifest
    # and recovers legacy marker/prereg writes without resetting the timestamp.
    manifest = I.freeze_manifest(FREEZE_MANIFEST, freeze_identity(proto),
                                 [PROTO_PATH, PREREG_PATH, sel_path,
                                  LEGACY_LEDGER, stage_dir("search") / "records.jsonl", stage_dir("pilot") / "records.jsonl"])
    now = manifest["frozen_utc"]
    marker = f"{h} {sel_sha}\n"
    if HASH_PATH.exists() and HASH_PATH.read_text() != marker:
        raise SystemExit("existing learning freeze marker differs; preserve and investigate")
    HASH_PATH.write_text(marker)
    txt = PREREG_PATH.read_text()
    txt = re.sub(r"^frozen: .*$", "frozen: true", txt, flags=re.M)
    txt = re.sub(r'^frozen_utc: .*$', f'frozen_utc: "{now}"', txt, flags=re.M)
    txt = re.sub(r'^protocol_hash: .*$', f'protocol_hash: "{h}"', txt, flags=re.M)
    txt = re.sub(r'^selection_sha256: .*$', f'selection_sha256: "{sel_sha}"', txt, flags=re.M)
    txt = re.sub(r"^status: .*$", "status: frozen", txt, flags=re.M)
    PREREG_PATH.write_text(txt)
    return {"hash": h, "selection_sha256": sel_sha, "frozen_utc": now}


def require_frozen(proto: dict) -> dict:
    if not HASH_PATH.exists():
        raise SystemExit("v3 not frozen; run `freeze` after search")
    h, sel_sha = HASH_PATH.read_text().split()
    sel_path = stage_dir("search") / "selection.json"
    if h != protocol_hash(proto):
        raise SystemExit("protocol changed since freeze (hash mismatch); record an amendment")
    if hashlib.sha256(sel_path.read_bytes()).hexdigest() != sel_sha:
        raise SystemExit("selection.json changed since freeze")
    I.require_manifest(FREEZE_MANIFEST, freeze_identity(proto))
    return json.loads(sel_path.read_text())


# ---- summary -----------------------------------------------------------------------------------------
def summarize_v3(records: list[dict], proto: dict) -> dict:
    ok = [r for r in records if r["status"] == "ok"]
    summ = V2.summarize(records, proto)
    summ["protocol"] = PROTOCOL_VERSION
    n_seeds = len(proto["seeds"]["confirmatory"])
    if "primary" in summ:
        summ["primary"]["family_note"] = (
            f"Holm over the prereg family E1-E6 (exact paired sign-flip p; nulls: per-null means as units). "
            f"Seed contrasts: n={n_seeds}, minimum two-sided p={2.0 / 2 ** n_seeds:.7f}. "
            f"E3/E4 null-graph contrasts: n={proto['seeds']['null_graphs']}, minimum two-sided "
            f"p={2.0 / 2 ** proto['seeds']['null_graphs']:.7f}.")
    # descriptive interface control: connectome bptt vs zero-edge kc_static decoder-only
    a = V2._vals(ok, "bptt", "connectome", V2.ACC)
    b = V2._vals(ok, "decoder_only", "kc_static", V2.ACC)
    ss, x, y = V2._paired(a, b)
    if ss:
        summ["exploratory_interface_control"] = {
            "bptt_connectome_minus_decoder_only_kc_static_acc": {**S.paired_bootstrap_diff(x, y, seed=21),
                                                                 "seeds": ss}}
    # collapse monitor (the v2 plast_dec failure signature: exactly 0.500 on every seed)
    for arm in ("plast_dec", "plast_fixed"):
        v = list(V2._vals(ok, arm, "connectome", V2.ACC).values())
        if v:
            summ.setdefault("collapse_monitor", {})[arm] = {"n": len(v), "all_exactly_0.5": bool(
                all(abs(t - 0.5) < 1e-12 for t in v)), "mean": float(np.mean(v))}
    return summ


def write_summary(stage: str, proto: dict) -> dict:
    try:
        return _write_summary_impl(stage, proto)
    except (Exception, SystemExit) as exc:
        I.atomic_json(stage_dir(stage) / "summary.json", {"complete": False, "status": "blocked",
                      "stage": stage, "error": str(exc), "targets": []})
        raise


def _write_summary_impl(stage: str, proto: dict) -> dict:
    sel = require_frozen(proto)
    recs = read_records(stage)
    coverage = validate_stage_coverage(stage, proto, sel, recs)
    if coverage["failed_runs"]:
        blocked = {"protocol": PROTOCOL_VERSION, "stage": stage, "coverage": coverage,
                   "complete": False, "status": "blocked", "n_records": len(recs),
                   "n_failed": coverage["failed_runs"], "targets": [],
                   "note": "every failed attempt is retained; no complete final comparison can be certified"}
        I.atomic_json(stage_dir(stage) / "summary.json", blocked)
        return blocked
    summ = summarize_v3(recs, proto)
    ok = [r for r in recs if r["status"] == "ok"]
    summ["targets"] = V2.target_rows(summ, ok, proto)
    summ["stage"], summ["n_records"] = stage, len(recs)
    summ["complete"], summ["coverage"] = True, coverage
    I.atomic_json(stage_dir(stage) / "summary.json", summ)
    return summ


def pilot_report(records: list[dict]) -> dict:
    out: dict = {}
    for r in records:
        if r["status"] != "ok":
            out.setdefault("failures", []).append({"cond": f"{r['arm']}|{r['family']}", "status": r["status"]})
            continue
        endpoints = [r["association"]["val_acc_acq"], r["continual"]["final_mean_acc_val"],
                     r["continual"]["avg_forgetting_keep_val"]]
        if not np.isfinite(endpoints).all():
            out.setdefault("failures", []).append({"cond": f"{r['arm']}|{r['family']}", "status": "nonfinite pilot endpoint"})
            continue
        k = f"{r['arm']}|{r['family']}|{r.get('optimizer')}"
        out.setdefault(k, []).append({"seed": r["seed"], "val_acc_acq": r["association"]["val_acc_acq"],
                                      "val_final_mean_acc": r["continual"]["final_mean_acc_val"],
                                      "val_forgetting_keep": r["continual"]["avg_forgetting_keep_val"],
                                      "elapsed_s": r["elapsed_s"]})
    return out


# ---- CLI ---------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="learning_v3")
    p.add_argument("cmd", choices=["search", "pilot", "freeze", "confirm", "summary", "estimate"])
    p.add_argument("--synthetic", action="store_true", help="engineering check on the synthetic circuit")
    p.add_argument("--max-minutes", type=float, default=None)
    a = p.parse_args(argv)
    if a.synthetic and a.cmd in {"freeze", "confirm", "summary"}:
        p.error("synthetic engineering runs cannot freeze, confirm or summarize the production study")
    if a.max_minutes:
        V2.set_time_budget(a.max_minutes)
    proto = load_protocol()
    check_seed_hygiene(proto)
    circuit = "synthetic" if a.synthetic else "flywire"
    if a.cmd == "freeze":
        print(json.dumps(freeze()))
        return 0
    if a.cmd == "summary":
        stage = "confirm"
        s = write_summary(stage, proto)
        print(json.dumps({"n_ok": s.get("n_ok", 0), "n_failed": s["n_failed"], "targets": s["targets"]}, indent=2,
                         default=str))
        return 0 if s.get("complete") else 4
    if a.cmd == "confirm":
        require_frozen(proto)  # reject invalid provenance before loading scientific inputs/models
    stage_sfx = "_synthetic" if a.synthetic else ""
    if a.cmd in {"search", "pilot", "confirm"}:
        ensure_stage_manifest(a.cmd + stage_sfx, proto)
    if a.cmd == "estimate":
        sel = json.loads((stage_dir("search" + stage_sfx) / "selection.json").read_text())
        print(len(build_jobs(proto, sel, proto["seeds"]["confirmatory"])), "confirm runs")
        return 0
    base = V2.load_base(proto, circuit)
    packs = V2.load_packs(base, proto, circuit)
    try:
        if a.cmd == "search":
            sel = run_search(proto, base, packs, "search" + stage_sfx)
            print(json.dumps({k: {"optimizer": v["optimizer"], "lr": v["lr"], "decoder_lr": v.get("decoder_lr"),
                                  "n_evals": v["n_evals"], "edge": v["optimum_at_edge"]}
                              for k, v in sel.items()}, indent=1))
        elif a.cmd == "pilot":
            sel = json.loads((stage_dir("search" + stage_sfx) / "selection.json").read_text())
            recs = run_stage("pilot" + stage_sfx, proto, base, packs, sel, proto["seeds"]["pilot"],
                             ("val",), ("association", "continual"), subset=PILOT_SUBSET)
            rep = pilot_report(recs)
            I.atomic_json(stage_dir("pilot" + stage_sfx) / "pilot_report.json", rep)
            print(json.dumps(rep, indent=1)[:3000])
            if rep.get("failures"):
                return 4
        elif a.cmd == "confirm":
            sel = require_frozen(proto)
            recs = run_stage("confirm", proto, base, packs, sel, proto["seeds"]["confirmatory"],
                             ("val", "test"), ("association", "continual"))
            summ = write_summary("confirm", proto)
            if not summ.get("complete"):
                return 4
    except V2.TimeBudget:
        print("time slice complete; rerun the same command to resume", flush=True)
        return 75
    return 0


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"elapsed {time.time() - t0:.0f}s", file=sys.stderr)
    sys.exit(rc)
