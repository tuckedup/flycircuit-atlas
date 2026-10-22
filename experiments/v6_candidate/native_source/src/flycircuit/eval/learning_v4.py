"""PROPOSAL / PILOT (validation split ONLY, never frozen, never a study result): does a selection rule that
scores acquisition AND continual retention, with >=4 search seeds, a one-SE rule and per-group learning
rates change which settings are chosen, and does it change connectome-vs-GRU on validation?

Design (equal budget for every arm): arms bptt x {connectome, gru, encdec}. Per arm a 5x5 grid of
(core_lr, decoder_lr) with values c*{1/9,1/3,1,3,9} for BOTH axes, where c is the arm's v3-selected lr
(so the diagonal core==decoder is the ordinary single-lr search, a subset of the same grid). "core" =
recurrent gains / GRU cell + encoder; "decoder" = readout. 4 search seeds (501-504), none used before.
Rules compared on the SAME search records:
  A  v3 rule, single lr (diagonal): argmax continual final_mean_acc_val
  B  new rule, single lr (diagonal): one-SE on composite = mean(val_acc_acq, final_mean_acc_val)
  C  new rule, 2 groups (full grid)
  D  v3 rule, 2 groups (full grid): argmax continual only        (separates groups from the score)
Chosen settings are then re-evaluated on 4 FRESH validation seeds (601-604) so the comparison is not
in-sample. Nothing here touches the test split, the original results, or frozen v3 source: process-local
monkeypatches only (V2.make_agent_v2 wrapper, torch clip_grad_norm_ recorder).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from flycircuit.config import project_root
from flycircuit.eval import learning_v2 as V2
from flycircuit.eval import learning_v3 as V3
from flycircuit.eval import learning_v4_select as S

COPY = Path(__file__).resolve().parents[3]
OUT = COPY / "results_v4" / "learning_v4"
ORIG = Path("/home/atulp/projects/flycircuit")
MANIFEST = OUT / "pilot_manifest.json"
LEGACY = OUT / "legacy_unmanifested"
_MANIFEST_CACHE: dict = {}
ARMS = ["connectome", "gru", "encdec"]
RATIOS = [1 / 9, 1 / 3, 1.0, 3.0, 9.0]
SEARCH_SEEDS = [501, 502, 503, 504]
CHECK_SEEDS = [601, 602, 603, 604]
USED = {0, 1, 2, 11, 12, 13, 14, 21, 22, 23, 24, 100, 101, 102, 103, 104, 105, 201, 202, 203, 204, 205,
        206, 207, 208, 300, 301, 302, 303, 304, 411, 412, 413, *range(100201, 100206),
        *range(100401, 100409)}
assert not (set(SEARCH_SEEDS) | set(CHECK_SEEDS)) & USED

CUR = {"core": None, "dec": None}
DIAG: list[float] = []
CLIP_MAX = {"v": 1.0}


def v3_lrs() -> dict:
    sel = json.loads((ORIG / "results/learning_v3/search/selection.json").read_text())
    return {a: sel[f"bptt|{a}"]["lr"] for a in ARMS}


def grid_for(c: float) -> list[float]:
    return [float(f"{c * r:.6g}") for r in RATIOS]


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _canonical_hash(value) -> str:
    return _sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode())


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _coord(family: str, core: float, dec: float, seed: int) -> dict:
    return {"arm": "bptt", "family": family, "core_lr": float(core),
            "decoder_lr": float(dec), "seed": int(seed)}


def expected_search() -> list[dict]:
    anchors = v3_lrs()
    return [_coord(fam, core, dec, seed) for fam in ARMS
            for core in grid_for(anchors[fam]) for dec in grid_for(anchors[fam])
            for seed in SEARCH_SEEDS]


def _assert_runtime() -> None:
    if COPY not in Path(__file__).resolve().parents:
        raise RuntimeError(f"v4 pilot must import from isolated copy, got {__file__}")
    if project_root().resolve() != ORIG.resolve():
        raise RuntimeError("set FLYCIRCUIT_HOME=/home/atulp/projects/flycircuit for canonical inputs")
    if Path(V3.__file__).resolve() != COPY / "src/flycircuit/eval/learning_v3.py":
        raise RuntimeError("learning_v3 was not imported from the isolated copy")


def _manifest_identity() -> dict:
    _assert_runtime()
    sources = {str(p.relative_to(COPY)): _sha_file(p)
               for p in sorted((COPY / "src/flycircuit").rglob("*.py"))}
    configs = {
        str(V3.PROTO_PATH): _sha_file(V3.PROTO_PATH),
        str(ORIG / "configs/experiments/learning_v2_protocol.yaml"):
            _sha_file(ORIG / "configs/experiments/learning_v2_protocol.yaml"),
    }
    selection = ORIG / "results/learning_v3/search/selection.json"
    inputs = V3.runtime_inputs(False)
    planned = expected_search()
    return {
        "pilot": "learning_v4_validation_only",
        "scope": "validation only; exploratory pilot; not a study result",
        "source_files": sources,
        "source_tree_sha256": _canonical_hash(sources),
        "config_files": configs,
        "effective_protocol_hash": V3.protocol_hash(V3.load_protocol()),
        "canonical_input_inventory": inputs,
        "canonical_input_inventory_sha256": _canonical_hash(inputs),
        "control_selection": str(selection),
        "control_selection_sha256": _sha_file(selection),
        "search_seeds": SEARCH_SEEDS,
        "check_seeds": CHECK_SEEDS,
        "expected_search_coordinates": planned,
        "expected_search_coordinates_sha256": _canonical_hash(planned),
        "expected_search_n": len(planned),
        "selection_rules": {
            "A": "diagonal; maximal continual mean; exact ties use lower lr",
            "B": "diagonal; composite one-SE; lower lr",
            "C": "full grid; composite one-SE; normalized max, geometric mean, core, decoder",
            "D": "full grid; maximal continual mean; exact ties use C simplicity ordering",
        },
    }


def _archive_unmanifested() -> list[dict]:
    archived = []
    for name in ("search_records.jsonl", "check_records.jsonl", "choices.json", "pilot_report.json"):
        src = OUT / name
        if not src.exists():
            continue
        data = src.read_bytes()
        digest = _sha_bytes(data)
        LEGACY.mkdir(parents=True, exist_ok=True)
        dst = LEGACY / f"{src.stem}.{digest[:16]}{src.suffix}"
        if dst.exists() and dst.read_bytes() != data:
            raise RuntimeError(f"legacy archive collision: {dst}")
        if not dst.exists():
            os.replace(src, dst)
        else:
            src.unlink()
        if dst.read_bytes() != data:
            raise RuntimeError(f"legacy archive byte verification failed: {dst}")
        archived.append({"source": str(src), "archive": str(dst), "sha256": digest,
                         "bytes": len(data)})
    if archived:
        journal = {"reason": "pre-manifest v4 artifacts are preserved but ineligible",
                   "archived_utc": datetime.now(UTC).isoformat(), "files": archived}
        _atomic_json(LEGACY / "migration.json", journal)
    return archived


def _protected_signatures(identity: dict) -> dict[str, tuple[int, int]]:
    paths = list(identity.get("source_files", {})) + list(identity.get("config_files", {}))
    if identity.get("control_selection"):
        paths.append(identity["control_selection"])
    def resolved(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else COPY / path
    return {p: (resolved(p).stat().st_size, resolved(p).stat().st_mtime_ns) for p in paths}


def prepare_manifest() -> dict:
    """Create/validate provenance before the first eligible result is written."""
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cache_key = str(MANIFEST.resolve())
        cached = _MANIFEST_CACHE.get(cache_key)
        if cached is not None:
            if _sha_file(MANIFEST) != cached["manifest_sha256"]:
                raise RuntimeError("pilot manifest changed during this process")
            if _protected_signatures(doc["identity"]) != cached["signatures"]:
                raise RuntimeError("protected pilot source/config/selection changed during this process")
            return doc
        identity = _manifest_identity()
        if doc.get("identity") != identity:
            raise RuntimeError("pilot manifest identity differs from current source/config/data/plan")
        _MANIFEST_CACHE[cache_key] = {"manifest_sha256": _sha_file(MANIFEST),
                                      "signatures": _protected_signatures(identity)}
        return doc
    identity = _manifest_identity()
    archived = _archive_unmanifested()
    doc = {"schema": 1, "created_utc": datetime.now(UTC).isoformat(),
           "identity": identity, "legacy_migration": archived}
    _atomic_json(MANIFEST, doc)
    _MANIFEST_CACHE[str(MANIFEST.resolve())] = {"manifest_sha256": _sha_file(MANIFEST),
                                                "signatures": _protected_signatures(identity)}
    return doc


def manifest_sha256() -> str:
    if not MANIFEST.exists():
        raise RuntimeError("pilot manifest must exist before records are read or written")
    prepare_manifest()
    return _sha_file(MANIFEST)


def revalidate_manifest_full() -> None:
    """Recompute canonical data inventory and all protected hashes at command end."""
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    V3.runtime_inputs.cache_clear()
    current = _manifest_identity()
    if doc.get("identity") != current:
        raise RuntimeError("pilot source/config/data/selection/plan changed during the command")


def _split(agent):
    dec_ids = {id(p) for p in agent.decoder_params()}
    core = [p for p in agent.params if id(p) not in dec_ids]
    dec = [p for p in agent.params if id(p) in dec_ids]
    return core, dec


_orig_make = V2.make_agent_v2


def make_agent_v4(proto, base, packs, arm, family, lr, seed, task, decoder_lr=None):
    agent, meta = _orig_make(proto, base, packs, arm, family, CUR["core"], seed, task, decoder_lr)
    if arm != "bptt":
        raise ValueError("pilot covers the bptt learner only")
    core, dec = _split(agent)
    if not core or not dec:
        raise RuntimeError("parameter-group split produced an empty group")
    agent.opt = torch.optim.Adam([{"params": core, "lr": CUR["core"]}, {"params": dec, "lr": CUR["dec"]}])
    meta["param_groups"] = {"core_n": int(sum(p.numel() for p in core)),
                            "decoder_n": int(sum(p.numel() for p in dec)),
                            "core_lr": CUR["core"], "decoder_lr": CUR["dec"]}
    return agent, meta


_orig_clip = torch.nn.utils.clip_grad_norm_


def _recording_clip(parameters, max_norm, *a, **kw):
    n = _orig_clip(parameters, max_norm, *a, **kw)
    DIAG.append(float(n))
    CLIP_MAX["v"] = float(max_norm)
    return n


def diag_summary() -> dict:
    g = np.asarray(DIAG, dtype=float)
    if g.size == 0:
        return {"n": 0}
    fin = g[np.isfinite(g)]
    return {"n": int(g.size), "nonfinite": int(g.size - fin.size),
            "clip_fraction": float((fin > CLIP_MAX["v"]).mean()) if fin.size else None,
            "grad_norm_median": float(np.median(fin)) if fin.size else None,
            "grad_norm_p95": float(np.percentile(fin, 95)) if fin.size else None}


def _key(r: dict) -> str:
    return json.dumps([r["arm"], r["family"], float(r["core_lr"]),
                       float(r["decoder_lr"]), int(r["seed"])],
                      separators=(",", ":"), allow_nan=False)


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError(f"truncated JSONL (missing final newline): {path}")
    out = []
    for n, raw in enumerate(data.splitlines(), 1):
        if not raw.strip():
            raise ValueError(f"blank JSONL row at {path}:{n}")
        try:
            row = json.loads(raw)
        except Exception as e:
            raise ValueError(f"malformed JSONL at {path}:{n}: {e}") from e
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSONL row at {path}:{n}")
        out.append(row)
    return out


def append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def validate_records(records: list[dict], expected: list[dict], phase: str,
                     manifest_id: str, protocol_hash: str | None = None) -> dict:
    """Fail closed on cache identity and return explicit completion status."""
    exp = {_key(x) for x in expected}
    if len(exp) != len(expected):
        raise ValueError(f"duplicate coordinates in expected {phase} plan")
    seen: set[str] = set()
    n_ok = n_failed = 0
    for i, r in enumerate(records, 1):
        try:
            k = _key(r)
        except Exception as e:
            raise ValueError(f"malformed {phase} coordinate at row {i}: {e}") from e
        if k not in exp:
            raise ValueError(f"unknown {phase} coordinate at row {i}: {k}")
        if k in seen:
            raise ValueError(f"duplicate {phase} coordinate at row {i}: {k}")
        seen.add(k)
        if r.get("pilot") != "learning_v4_validation_only":
            raise ValueError(f"wrong pilot identity at {phase} row {i}")
        if r.get("splits") != ["val"]:
            raise ValueError(f"split identity must be exactly ['val'] at {phase} row {i}")
        if r.get("pilot_manifest_sha256") != manifest_id:
            raise ValueError(f"manifest provenance mismatch at {phase} row {i}")
        if r.get("mode") != "std" or r.get("stage") is not None:
            raise ValueError(f"run-mode identity mismatch at {phase} row {i}")
        if float(r.get("lr", float("nan"))) != float(r["core_lr"]):
            raise ValueError(f"core learning-rate identity mismatch at {phase} row {i}")
        if protocol_hash is not None and r.get("protocol_hash") != protocol_hash:
            raise ValueError(f"protocol provenance mismatch at {phase} row {i}")
        status = r.get("status")
        if status == "ok":
            try:
                endpoints = scores(r)
                forgetting = r["continual"]["avg_forgetting_keep_val"]
            except Exception as e:
                raise ValueError(f"missing endpoint at {phase} row {i}: {e}") from e
            if not all(math.isfinite(float(x)) for x in (*endpoints, forgetting)):
                raise ValueError(f"nonfinite endpoint at {phase} row {i}")
            n_ok += 1
        elif isinstance(status, str) and status.startswith("failed:"):
            n_failed += 1
        else:
            raise ValueError(f"unknown status at {phase} row {i}: {status!r}")
    missing = sorted(exp - seen)
    return {"phase": phase, "expected": len(exp), "records": len(records), "ok": n_ok,
            "failed": n_failed, "missing": len(missing), "complete": not missing and not n_failed,
            "missing_keys": missing[:10]}


def read_validated(path: Path, expected: list[dict], phase: str) -> tuple[list[dict], dict]:
    rows = read(path)
    manifest_id = manifest_sha256()
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return rows, validate_records(rows, expected, phase, manifest_id,
                                  doc["identity"]["effective_protocol_hash"])


def run_setting(proto, base, packs, family, core, dec, seed) -> dict:
    CUR["core"], CUR["dec"] = core, dec
    DIAG.clear()
    r = V2.run_one_v2(proto, base, packs, "bptt", family, "std", core, seed, splits=("val",),
                      protocols=("association", "continual"), stage=None, save_ckpt=False, decoder_lr=dec)
    r.update(core_lr=core, decoder_lr=dec, diag=diag_summary(), pilot="learning_v4_validation_only",
             pilot_manifest_sha256=manifest_sha256())
    return r


def ok(r: dict) -> bool:
    return r.get("status") == "ok"


def scores(r: dict) -> tuple[float, float]:
    return r["association"]["val_acc_acq"], r["continual"]["final_mean_acc_val"]


def estimate() -> dict:
    cost = {"connectome": 20.0, "gru": 12.0, "encdec": 5.0}   # v3 pilot/confirm elapsed_s, assoc+continual
    n = len(RATIOS) ** 2 * len(SEARCH_SEEDS)
    s = sum(cost[a] * n for a in ARMS)
    return {"search_runs": n * len(ARMS), "search_est_min": s / 60,
            "check_runs_max": 4 * len(CHECK_SEEDS) * len(ARMS),
            "check_est_min_max": sum(cost[a] * 4 * len(CHECK_SEEDS) for a in ARMS) / 60}


def _setup():
    torch.set_num_threads(2)
    torch.nn.utils.clip_grad_norm_ = _recording_clip
    V2.make_agent_v2 = make_agent_v4
    proto = V3.load_protocol()
    base = V2.load_base(proto, "flywire")
    packs = V2.load_packs(base, proto, "flywire")
    return proto, base, packs


def cmd_search(log=print) -> None:
    prepare_manifest()
    proto, base, packs = _setup()
    path = OUT / "search_records.jsonl"
    plan = expected_search()
    rows, before = read_validated(path, plan, "search")
    done = {_key(r) for r in rows}
    base_lr = v3_lrs()
    log(f"estimate {estimate()}  v3 lrs {base_lr}  coverage_before={before}")
    for fam in ARMS:
        G = grid_for(base_lr[fam])
        for core in G:
            for dec in G:
                for sd in SEARCH_SEEDS:
                    probe = {"arm": "bptt", "family": fam, "core_lr": core, "decoder_lr": dec, "seed": sd}
                    if _key(probe) in done:
                        continue
                    r = run_setting(proto, base, packs, fam, core, dec, sd)
                    append(path, r)
                    done.add(_key(r))
                    a = scores(r) if ok(r) else None
                    log(f"search {fam:10s} core={core:<9g} dec={dec:<9g} seed={sd} {r['status'][:20]} "
                        f"{r['elapsed_s']:.1f}s acq/cont={a}")
    _, after = read_validated(path, plan, "search")
    log(f"search coverage {after}")
    revalidate_manifest_full()


def table(records, fam, diag_only: bool, composite: bool) -> dict:
    t: dict = {}
    for r in records:
        if r["family"] != fam or not ok(r):
            continue
        if diag_only and r["core_lr"] != r["decoder_lr"]:
            continue
        acq, cont = scores(r)
        t.setdefault((r["core_lr"], r["decoder_lr"]), []).append(
            S.composite_score(acq, cont) if composite else cont)
    return t


def choose(records, fam, rule: str, base_lr: float | None = None) -> dict:
    if rule not in "ABCD":
        raise ValueError(f"unknown rule: {rule}")
    diag_only = rule in ("A", "B")
    composite = rule in ("B", "C")
    t = table(records, fam, diag_only, composite)
    keys = sorted(t)
    expected_n = len(RATIOS) if diag_only else len(RATIOS) ** 2
    if len(keys) != expected_n or any(len(t[k]) != len(SEARCH_SEEDS) for k in keys):
        raise ValueError(f"{fam} rule {rule} lacks exact {expected_n} settings x "
                         f"{len(SEARCH_SEEDS)} seeds")
    if any(not all(math.isfinite(float(x)) for x in t[k]) for k in keys):
        raise ValueError(f"{fam} rule {rule} has nonfinite selection scores")
    base_lr = float(v3_lrs()[fam] if base_lr is None else base_lr)
    if rule in ("A", "D"):
        means = {k: float(np.mean(v)) for k, v in t.items()}
        top = max(means.values())
        tied = [k for k in keys if means[k] == top]
        best = min(tied, key=lambda k: S.two_group_simplicity(k, base_lr))
        return {"rule": rule, "core_lr": best[0], "decoder_lr": best[1], "mean_score": means[best],
                "n_settings": len(keys), "tie_rule": "two_group_simplicity"}
    if rule == "B":
        flat = {k[0]: v for k, v in t.items()}
        out = S.one_se_choice(flat, min_seeds=len(SEARCH_SEEDS))
        k = (out["chosen_lr"], out["chosen_lr"])
        mean = out["stats"][repr(float(out["chosen_lr"]))]["mean"]
    else:
        out = S.one_se_choice_2d(t, base_lr, min_seeds=len(SEARCH_SEEDS))
        k = out["chosen"]
        mean = out["stats"][f"{k[0]:.17g}|{k[1]:.17g}"]["mean"]
    return {"rule": rule, "core_lr": k[0], "decoder_lr": k[1], "mean_score": mean,
            "n_settings": len(keys), "n_indistinguishable": out["n_indistinguishable"],
            "chosen_equals_best": out["chosen_equals_best"],
            "tie_rule": "lower_lr" if rule == "B" else "two_group_simplicity"}


def _require_complete(status: dict) -> None:
    if not status["complete"]:
        raise RuntimeError(f"{status['phase']} cache is incomplete or failed: {status}")


def expected_check(choices: dict) -> list[dict]:
    out = []
    for fam in ARMS:
        uniq = {(float(c["core_lr"]), float(c["decoder_lr"])) for c in choices[fam].values()}
        out.extend(_coord(fam, core, dec, seed) for core, dec in sorted(uniq) for seed in CHECK_SEEDS)
    return out


def cmd_check(log=print) -> None:
    prepare_manifest()
    proto, base, packs = _setup()
    recs, search_status = read_validated(OUT / "search_records.jsonl", expected_search(), "search")
    _require_complete(search_status)
    path = OUT / "check_records.jsonl"
    anchors = v3_lrs()
    choices = {fam: {rule: choose(recs, fam, rule, anchors[fam]) for rule in "ABCD"} for fam in ARMS}
    _atomic_json(OUT / "choices.json", choices)
    plan = expected_check(choices)
    rows, before = read_validated(path, plan, "check")
    done = {_key(r) for r in rows}
    log(f"check coverage_before={before}")
    for fam in ARMS:
        uniq = {(c["core_lr"], c["decoder_lr"]) for c in choices[fam].values()}
        for core, dec in sorted(uniq):
            for sd in CHECK_SEEDS:
                probe = {"arm": "bptt", "family": fam, "core_lr": core, "decoder_lr": dec, "seed": sd}
                if _key(probe) in done:
                    continue
                r = run_setting(proto, base, packs, fam, core, dec, sd)
                append(path, r)
                done.add(_key(r))
                log(f"check {fam:10s} core={core:<9g} dec={dec:<9g} seed={sd} {r['status'][:20]} "
                    f"{r['elapsed_s']:.1f}s")
    _, after = read_validated(path, plan, "check")
    log(f"check coverage {after}")
    revalidate_manifest_full()


def _mse(v):
    return S.mean_and_se(v)


def forgetting_improvement_status(gru_minus_connectome: float | None) -> bool | None:
    """Descriptive target is a positive GRU-minus-connectome difference, not magnitude."""
    return None if gru_minus_connectome is None else bool(gru_minus_connectome >= 0.07)


def cmd_report() -> dict:
    prepare_manifest()
    sr, search_status = read_validated(OUT / "search_records.jsonl", expected_search(), "search")
    _require_complete(search_status)
    choices = json.loads((OUT / "choices.json").read_text())
    anchors = v3_lrs()
    expected_choices = {fam: {rule: choose(sr, fam, rule, anchors[fam]) for rule in "ABCD"}
                        for fam in ARMS}
    if choices != expected_choices:
        raise RuntimeError("choices.json differs from deterministic selection over manifested search")
    cr, check_status = read_validated(OUT / "check_records.jsonl", expected_check(choices), "check")
    _require_complete(check_status)
    rep: dict = {"scope": "VALIDATION ONLY pilot; no test split; not a study result", "estimate": estimate(),
                 "seeds": {"search": SEARCH_SEEDS, "check": CHECK_SEEDS}, "v3_lrs": v3_lrs(),
                 "coverage": {"search": search_status, "check": check_status},
                 "n_search_records": len(sr), "n_search_failed": sum(not ok(r) for r in sr), "arms": {}}
    # search-grid noise and structure
    for fam in ARMS:
        t = table(sr, fam, False, True)
        means = {k: float(np.mean(v)) for k, v in t.items()}
        se = {k: S.mean_and_se(v)[1] for k, v in t.items()}
        best = max(means, key=means.get)
        rep["arms"][fam] = {
            "grid": grid_for(v3_lrs()[fam]),
            "best_composite_setting": {"core": best[0], "decoder": best[1], "mean": means[best], "se": se[best]},
            "n_within_1se_of_best_full_grid": sum(m >= means[best] - se[best] for m in means.values()),
            "mean_seed_se_over_grid": float(np.mean(list(se.values()))),
            "choices": choices[fam], "check": {}}
        by_rule = {}
        for rule, c in choices[fam].items():
            rows = [r for r in cr if r["family"] == fam and ok(r) and r["core_lr"] == c["core_lr"]
                    and r["decoder_lr"] == c["decoder_lr"]]
            acq = [scores(r)[0] for r in rows]
            cont = [scores(r)[1] for r in rows]
            frg = [r["continual"]["avg_forgetting_keep_val"] for r in rows]
            by_rule[rule] = {"core_lr": c["core_lr"], "decoder_lr": c["decoder_lr"], "n": len(rows),
                             "acq": _mse(acq), "cont": _mse(cont), "forget_keep": _mse(frg),
                             "per_seed_acq": {r["seed"]: scores(r)[0] for r in rows},
                             "per_seed_cont": {r["seed"]: scores(r)[1] for r in rows},
                             "per_seed_forget": {r["seed"]: r["continual"]["avg_forgetting_keep_val"]
                                                 for r in rows},
                             "descriptive_acquisition_ge_0_82": bool(np.mean(acq) >= 0.82),
                             "diag_clip_fraction": float(np.mean([r["diag"].get("clip_fraction") or 0 for r in rows])) if rows else None,
                             "diag_grad_norm_median": float(np.median([r["diag"].get("grad_norm_median") or 0 for r in rows])) if rows else None,
                             "diag_nonfinite": int(sum(r["diag"].get("nonfinite", 0) for r in rows))}
        rep["arms"][fam]["check"] = by_rule
    # connectome minus GRU, paired by fresh seed, per rule
    gap = {}
    forget_gap = {}
    for rule in "ABCD":
        out = {}
        for key in ("per_seed_acq", "per_seed_cont"):
            c = rep["arms"]["connectome"]["check"][rule][key]
            g = rep["arms"]["gru"]["check"][rule][key]
            d = [c[s] - g[s] for s in c if s in g]
            out[key] = {"mean_diff_conn_minus_gru": float(np.mean(d)) if d else None,
                        "se": S.mean_and_se(d)[1] if len(d) > 1 else None, "n": len(d)}
        c = rep["arms"]["connectome"]["check"][rule]["per_seed_forget"]
        g = rep["arms"]["gru"]["check"][rule]["per_seed_forget"]
        d = [g[s] - c[s] for s in c if s in g]
        mean_d = float(np.mean(d)) if d else None
        forget_gap[rule] = {
            "mean_diff_gru_minus_conn": mean_d,
            "se": S.mean_and_se(d)[1] if len(d) > 1 else None,
            "n": len(d),
            "descriptive_gru_minus_conn_improvement_ge_0_07": forgetting_improvement_status(mean_d),
        }
        gap[rule] = out
    rep["connectome_minus_gru_fresh_val_seeds"] = gap
    rep["gru_minus_connectome_forgetting_fresh_val_seeds"] = forget_gap
    rep["descriptive_thresholds"] = {
        "scope": "validation-only descriptive statuses; not confirmatory",
        "acquisition": ">= 0.82 within each arm/rule fresh-seed mean",
        "forgetting_improvement": ">= 0.07 for paired GRU-minus-connectome fresh-seed mean; "
                                  "negative differences do not pass",
    }
    revalidate_manifest_full()
    _atomic_json(OUT / "pilot_report.json", rep)
    return rep


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["estimate", "prepare", "search", "check", "report", "all"])
    a = p.parse_args(argv)
    t0 = time.time()
    if a.cmd == "estimate":
        print(json.dumps(estimate(), indent=1))
    if a.cmd == "prepare":
        print(json.dumps(prepare_manifest(), indent=1)[:6000])
    if a.cmd in ("search", "all"):
        cmd_search()
    if a.cmd in ("check", "all"):
        cmd_check()
    if a.cmd in ("report", "all"):
        print(json.dumps(cmd_report(), indent=1, default=float)[:6000])
    print(f"elapsed {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
