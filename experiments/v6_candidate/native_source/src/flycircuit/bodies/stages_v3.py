"""FlyGym body-in-the-loop study v3 (protocol ``flygym_nav_v3``), main-env stage driver.

  python -m flycircuit.bodies.stages_v3 freeze|collect|search|train|power|freeze-eval|eval|report

Differences from v1 (``bodies/stages.py``, untouched): matched DART-style training data (scripted controller under
dropout, teacher labels) for every condition; annotation_v2 signs and cached v2 nulls; a large STRATIFIED test set with
masks independent of geometry (channels are correlated, each dropped in exactly 25% of episodes); N from VALIDATION and
frozen before any test episode; crossed model x episode bootstrap; per-channel breakdown; trained weights persisted
(git-ignored, results/flygym_v3/models). Physics runs in the isolated FlyGym env via subprocess (fly_rollout.py).
Every heavy stage must be launched through scripts/linux/heavyjob.sh.
"""
from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from flycircuit import v3_integrity as I
from flycircuit.bodies import jobs as J
from flycircuit.bodies import stages as V1

ROOT = J.ROOT
RES = ROOT / "results/flygym_v3"
V1RES = ROOT / "results/flygym"
PROTO_PATH = ROOT / "configs/experiments/flygym_v3_protocol.yaml"
NAV_PROTO_PATH = ROOT / "configs/experiments/nav_protocol.yaml"
PREREG_PATH = ROOT / "configs/prereg_v3.yaml"
V1_PROTO_PATH = ROOT / "configs/experiments/flygym_protocol.yaml"
PROVENANCE = ROOT / "results/provenance/study_v3"
TRAIN_MANIFEST = PROVENANCE / "flygym_training.json"
SEARCH_MANIFEST = PROVENANCE / "flygym_search_cache.json"
MODEL_MANIFEST = PROVENANCE / "flygym_model_cache.json"
EVAL_MANIFEST = PROVENANCE / "flygym_evaluation.json"
BUDGET_TICKS, RADIUS_MM, N_CH = 180, 1.5, 10
TRAIN_THREADS, ROLLOUT_WORKERS = 2, 1
WORKER_ENV = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "OMP_NUM_THREADS": "1",
              "CUDA_VISIBLE_DEVICES": ""}
CONDITIONS = ["connectome_pilot_mb", "null_v2_plain", "random_sparse_reachable", "gru_param_matched",
              "encdec_param_matched"]
UNTRAINED = ["connectome_pilot_mb", "gru_param_matched"]
FORBIDDEN_SEEDS = {0, 301, 302, 303, 304, 305, 300}


# ------------------------------------------------------------------ protocol + hashes
def proto() -> dict:
    return yaml.safe_load(PROTO_PATH.read_text())


def _hash(p: dict, drop: tuple[str, ...]) -> str:
    q = {k: v for k, v in p.items() if k not in drop}
    return hashlib.sha256(json.dumps(q, sort_keys=True).encode()).hexdigest()


def train_hash() -> str:
    return _hash(proto(), ("amendments", "evaluation"))


def _config_paths() -> list[Path]:
    return [PROTO_PATH, NAV_PROTO_PATH, PREREG_PATH, V1_PROTO_PATH]


@functools.lru_cache(maxsize=1)
def _circuit_inventory() -> dict[str, str]:
    return I.circuit_inputs()


def _training_identity() -> dict:
    prereg = yaml.safe_load(PREREG_PATH.read_text())
    changing = {"status", "frozen", "frozen_utc", "protocol_hash", "selection_sha256"}
    return {"study": "flygym_nav_v3", "phase": "training", "train_hash": train_hash(),
            "raw_protocol_sha256": I.file_hash(PROTO_PATH),
            "teacher_data_protocol_sha256": I.file_hash(V1_PROTO_PATH),
            "runtime": I.runtime_identity(), "circuit_inputs": _circuit_inventory(),
            "prereg_spec_hash": I.canonical_hash({k: v for k, v in prereg.items() if k not in changing}),
            "navigation_protocol_sha256": I.file_hash(NAV_PROTO_PATH)}


def require_train_frozen() -> str:
    try:
        I.require_manifest(TRAIN_MANIFEST, _training_identity())
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"v3 training spec is not validly frozen: {exc}; run `freeze`") from exc
    return train_hash()


def effective_eval_protocol(power: dict | None = None) -> dict:
    """Return the frozen effective protocol, including validation-derived N.

    Consumers must use this instead of reading ``n_test_episodes`` directly
    from the immutable YAML. With no argument, the full saved power record and
    its upstream cache identities are validated first.
    """
    if power is None:
        power = require_power_record()
    p = copy.deepcopy(proto())
    declared = p["evaluation"].get("n_test_episodes")
    derived = power.get("n_test_episodes")
    if type(derived) is not int or derived not in range(96, 145, 12):
        raise ValueError("power record has no valid predeclared test N")
    if declared not in (None, derived):
        raise ValueError("protocol N conflicts with the validation-derived power record")
    p["evaluation"] = {**p["evaluation"], "n_test_episodes": derived}
    return p


def effective_eval_hash(p: dict | None = None) -> str:
    """Hash an effective protocol; validate and load it when omitted."""
    if p is None:
        p = effective_eval_protocol()
    return _hash(p, ("amendments",))


def require_eval_frozen() -> dict:
    require_train_frozen()
    power = require_power_record()
    p = effective_eval_protocol(power)
    try:
        I.require_manifest(EVAL_MANIFEST, _evaluation_identity(p, power))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"v3 evaluation is not validly frozen: {exc}; run `freeze-eval`") from exc
    return p


# ------------------------------------------------------------------ stratified masks
def balanced_masks(n: int, p: float = 0.25, n_ch: int = N_CH, seed: int = 0) -> np.ndarray:
    """(n, n_ch) boolean masks: every channel is dropped in exactly n*p episodes; each episode drops 2 or 3
    channels when p*n_ch = 2.5 (half/half); no repeated channel inside an episode."""
    slots = n * p * n_ch
    per_ch = n * p
    if abs(per_ch - round(per_ch)) > 1e-9 or abs(slots - round(slots)) > 1e-9:
        raise ValueError("n*p must be an integer so that each channel is dropped equally often")
    per_ch = int(round(per_ch))
    rng = np.random.default_rng(seed)
    mean = p * n_ch
    lo, hi = math.floor(mean), math.ceil(mean)
    n_hi = int(round((mean - lo) * n)) if hi != lo else 0
    sizes = np.array([hi] * n_hi + [lo] * (n - n_hi))
    if sizes.sum() != per_ch * n_ch:
        raise ValueError("inconsistent mask design")
    # Gale-Ryser style construction: each episode takes the channels with the most remaining quota (random
    # tie-breaks, random episode order), which always realises a feasible degree sequence.
    rng.shuffle(sizes)
    remaining = np.full(n_ch, per_ch)
    m = np.zeros((n, n_ch), bool)
    for e in rng.permutation(n):
        order = np.lexsort((rng.random(n_ch), -remaining))      # most remaining first, random ties
        ch = order[:sizes[e]]
        if (remaining[ch] <= 0).any():
            raise RuntimeError("could not construct balanced masks")
        m[e, ch] = True
        remaining[ch] -= 1
    assert (remaining == 0).all() and (m.sum(0) == per_ch).all() and (m.sum(1) == sizes).all()
    return m


def eval_episodes(split: str, idx_start: int, masks: np.ndarray, policy: str, tag: str = "") -> list[dict]:
    out = []
    for k, m in enumerate(masks):
        i = idx_start + k
        out.append(dict(key=f"{split}{i}_{tag or Path(policy).stem}_strat", split=split, idx=i, policy=policy,
                        dropout=0.25, mask=[int(x) for x in m], action_noise=0.0, record=False))
    return out


# ------------------------------------------------------------------ physics
def _json(path: Path, label: str):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}: {exc}") from exc


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


@functools.lru_cache(maxsize=1)
def _flygym_runtime_identity() -> dict:
    code = ("import importlib.metadata as m,json,sys; "
            "p=sorted((d.metadata.get('Name',''),d.version) for d in m.distributions()); "
            "print(json.dumps({'python':sys.version,'executable':sys.executable,'packages':p}))")
    try:
        r = subprocess.run([str(J.FLYGYM_PY), "-I", "-c", code], capture_output=True, text=True,
                           timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("isolated FlyGym runtime identity timed out after 30 seconds") from exc
    except OSError as exc:
        raise ValueError(f"cannot start isolated FlyGym runtime identity: {exc}") from exc
    if r.returncode != 0:
        detail = (r.stderr or r.stdout).strip()[:300]
        raise ValueError(f"cannot identify isolated FlyGym runtime (rc={r.returncode}): {detail}")
    try:
        raw = json.loads(r.stdout)
        if set(raw) != {"python", "executable", "packages"} or not isinstance(raw["packages"], list):
            raise ValueError("unexpected metadata fields")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("isolated FlyGym runtime returned malformed identity metadata") from exc
    return {"python": raw["python"], "executable": raw["executable"],
            "installed_distributions_sha256": I.canonical_hash(raw["packages"]),
            "python_executable_sha256": I.file_hash(J.FLYGYM_PY)}


def _rollout_payload(eps: list[dict]) -> dict:
    return {"np_policy_py": str(J.NP_POLICY), "budget_ticks": BUDGET_TICKS,
            "success_radius_mm": RADIUS_MM, "episodes": eps}


def _policy_paths(eps: list[dict]) -> list[Path]:
    return sorted({Path(e["policy"]) for e in eps if e["policy"] not in ("teacher", "scripted", "straight")})


def _rollout_identity(name: str, eps: list[dict]) -> dict:
    source_id, _ = I._unchanged_source()
    bundles = _policy_paths(eps)
    missing = [p for p in bundles if not p.is_file()]
    if missing:
        raise ValueError(f"rollout policy bundle missing: {missing[0]}")
    payload = _rollout_payload(eps)
    return {"study": "flygym_nav_v3", "stage": name, "source_id": source_id,
            "workers": ROLLOUT_WORKERS,
            "main_runtime": I.runtime_identity(), "flygym_runtime": _flygym_runtime_identity(),
            "worker_environment": WORKER_ENV,
            "worker_files": I.input_inventory([J.ROLLOUT, J.NP_POLICY]),
            "policy_bundles": I.input_inventory(bundles), "jobs_sha256": I.canonical_hash(payload)}


def _validate_rollout_rows(eps: list[dict], rows: list[dict]) -> None:
    expected = {e["key"]: e for e in eps}
    if len(expected) != len(eps):
        raise ValueError("planned rollout keys are not unique")
    seen = set()
    for row in rows:
        key = row.get("key")
        if key not in expected or key in seen:
            raise ValueError(f"unknown or duplicate cached rollout key: {key!r}")
        seen.add(key)
        ep = expected[key]
        policy = Path(ep["policy"]).stem if str(ep["policy"]).endswith(".npz") else ep["policy"]
        exact = {"split": ep["split"], "idx": ep["idx"], "policy": policy,
                 "policy_path": ep["policy"], "dropout": ep.get("dropout", 0.0),
                 "action_noise": ep.get("action_noise", 0.0)}
        if any(row.get(k) != v for k, v in exact.items()):
            raise ValueError(f"cached rollout metadata differs from its job: {key}")
        mask = row.get("mask")
        if (not isinstance(mask, list) or len(mask) != N_CH or
                not all(type(x) is int and x in (0, 1) for x in mask)):
            raise ValueError(f"invalid cached rollout mask: {key}")
        if "mask" in ep and mask != ep["mask"]:
            raise ValueError(f"cached rollout mask differs from its job: {key}")
        if row.get("status") not in ("ok", "nan_action", "nan_state"):
            raise ValueError(f"unknown cached execution status: {key}")
        if row.get("success") not in (False, True):
            raise ValueError(f"invalid cached success value: {key}")


def _validate_recordings(out: Path, eps: list[dict], rows: list[dict]) -> None:
    expected = {f"rec_{r['key']}.npz" for r in rows if next(e for e in eps if e["key"] == r["key"]).get("record")}
    actual = {p.name for p in out.glob("rec_*.npz")}
    if actual != expected:
        raise ValueError("recorded episode files do not exactly match completed rollout rows")
    for path in sorted(out.glob("rec_*.npz")):
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {"x", "y"} or z["x"].ndim != 2 or z["y"].ndim != 2:
                raise ValueError(f"invalid recorded episode bundle: {path}")
            if (z["x"].shape[1:] != (N_CH,) or z["y"].shape[1:] != (2,) or
                    len(z["x"]) != len(z["y"]) or not 0 < len(z["x"]) <= BUDGET_TICKS or
                    not np.isfinite(z["x"]).all() or not np.isfinite(z["y"]).all()):
                raise ValueError(f"invalid recorded episode arrays: {path}")


def _require_rollout_cache(name: str, eps: list[dict], complete_required: bool) -> list[dict]:
    out = RES / name
    identity = _rollout_identity(name, eps)
    if _json(out / "manifest.json", "rollout manifest") != identity:
        raise ValueError(f"rollout cache identity changed: {out}")
    if _json(out / "jobs.json", "rollout jobs") != _rollout_payload(eps):
        raise ValueError(f"rollout jobs changed: {out}")
    rows = J.load_results(out)
    _validate_rollout_rows(eps, rows)
    _validate_recordings(out, eps, rows)
    completion = out / "complete.json"
    if completion.exists():
        expected = {"identity_sha256": I.canonical_hash(identity),
                    "results_sha256": I.file_hash(out / "results.jsonl"), "n_rows": len(rows)}
        if len(rows) != len(eps) or _json(completion, "rollout completion") != expected:
            raise ValueError(f"rollout completion record changed: {out}")
    elif complete_required or len(rows) == len(eps):
        raise ValueError(f"rollout cache lacks a valid completion record: {out}")
    return rows


def _require_complete_rollout_cache(name: str, eps: list[dict]) -> list[dict]:
    return _require_rollout_cache(name, eps, complete_required=True)


def run_rollouts(name: str, eps: list[dict], workers: int) -> Path:
    if workers != ROLLOUT_WORKERS:
        raise ValueError(f"v3 rollout worker count is frozen at {ROLLOUT_WORKERS}")
    out = RES / name
    jf, mf, donef = out / "jobs.json", out / "manifest.json", out / "complete.json"
    payload, identity = _rollout_payload(eps), _rollout_identity(name, eps)
    if mf.exists():
        if _json(mf, "rollout manifest") != identity:
            raise ValueError(f"rollout cache identity changed: {out}")
    else:
        if jf.exists() or (out / "results.jsonl").exists() or list(out.glob("rec_*.npz")):
            raise ValueError(f"unmanifested rollout cache exists: {out}")
        I.atomic_json(mf, identity)
    if jf.exists() and _json(jf, "rollout jobs") != payload:
        raise ValueError(f"rollout jobs changed: {jf}")
    if not jf.exists():
        I.atomic_json(jf, payload)
    rows = J.load_results(out)
    _validate_rollout_rows(eps, rows)
    _validate_recordings(out, eps, rows)
    if donef.exists():
        complete = _json(donef, "rollout completion")
        if (len(rows) != len(eps) or complete != {"identity_sha256": I.canonical_hash(identity),
                "results_sha256": I.file_hash(out / "results.jsonl"), "n_rows": len(rows)}):
            raise ValueError(f"rollout completion record is stale or corrupt: {donef}")
        return out
    if len(rows) == len(eps):
        I.atomic_json(donef, {"identity_sha256": I.canonical_hash(identity),
                              "results_sha256": I.file_hash(out / "results.jsonl"), "n_rows": len(rows)})
        return out
    env = dict(os.environ, **WORKER_ENV)
    r = subprocess.run([str(J.FLYGYM_PY), "-I", str(J.ROLLOUT), str(jf), str(out), str(workers)], env=env)
    if r.returncode != 0:
        raise SystemExit(f"rollout failed rc={r.returncode}")
    rows = J.load_results(out)
    _validate_rollout_rows(eps, rows)
    _validate_recordings(out, eps, rows)
    if len(rows) != len(eps):
        raise ValueError(f"rollout worker returned success with only {len(rows)}/{len(eps)} rows")
    I.atomic_json(donef, {"identity_sha256": I.canonical_hash(identity),
                          "results_sha256": I.file_hash(out / "results.jsonl"), "n_rows": len(rows)})
    return out


# ------------------------------------------------------------------ data
def load_files(files: list[Path]):
    T = BUDGET_TICKS
    X = np.zeros((T, len(files), N_CH), np.float32)
    Y = np.zeros((T, len(files), 2), np.float32)
    M = np.zeros((T, len(files)), np.float32)
    for i, f in enumerate(files):
        with np.load(f, allow_pickle=False) as z:
            if set(z.files) != {"x", "y"} or z["x"].shape[1:] != (N_CH,) or z["y"].shape[1:] != (2,):
                raise ValueError(f"invalid training recording: {f}")
            n = len(z["x"])
            if n != len(z["y"]) or not 0 < n <= T or not np.isfinite(z["x"]).all() or not np.isfinite(z["y"]).all():
                raise ValueError(f"invalid training recording arrays: {f}")
            X[:n, i], Y[:n, i], M[:n, i] = z["x"], z["y"], 1.0
    return X, Y, M


def _cat(a, b):
    return tuple(np.concatenate([x, y], axis=1) for x, y in zip(a, b, strict=True))


def _expected_dataset_paths() -> dict[str, list[Path]]:
    v1eps, k = [], 0
    levels = V1.proto()["training"]["data"]["action_noise_levels"]
    per = 300 // len(levels)
    for noise in levels:
        v1eps += J.episodes("train", range(k, k + per), "teacher", action_noise=noise, record=True, tag="bc")
        k += per
    v1eps += J.episodes("val", range(48), "teacher", action_noise=0.3, record=True, tag="bcval")
    d3eps = _dart_episodes()
    return {"teacher_train": [V1RES / "bc_data" / f"rec_{e['key']}.npz" for e in v1eps if e["split"] == "train"],
            "teacher_val": [V1RES / "bc_data" / f"rec_{e['key']}.npz" for e in v1eps if e["split"] == "val"],
            "dart_train": [RES / "bc_dart" / f"rec_{e['key']}.npz" for e in d3eps if e["split"] == "train"],
            "dart_val": [RES / "bc_dart" / f"rec_{e['key']}.npz" for e in d3eps if e["split"] == "val"]}


def _dart_episodes() -> list[dict]:
    data, eps, k = proto()["training"]["data"], [], 300
    levels = data["dart_dropout_levels"]
    if data["dart_episodes"] % len(levels):
        raise ValueError("DART episode count must divide equally across declared dropout levels")
    per = data["dart_episodes"] // len(levels)
    for dropout in levels:
        eps += J.episodes("train", range(k, k + per), "scripted", dropout=dropout,
                          action_noise=data["dart_action_noise"], record=True, tag="dart")
        k += per
    eps += J.episodes("val", range(100, 100 + data["dart_val_episodes"]), "scripted", dropout=0.25,
                      action_noise=data["dart_action_noise"], record=True, tag="dartval")
    return eps


@functools.lru_cache(maxsize=1)
def dataset_inventory() -> tuple[dict[str, str], dict[str, int]]:
    dart_rows = _require_complete_rollout_cache("bc_dart", _dart_episodes())
    failed = [row["key"] for row in dart_rows if row["status"] != "ok"]
    if failed:
        raise ValueError(f"DART collection contains failed executions and cannot feed training: {failed[:3]}")
    groups = _expected_dataset_paths()
    by_folder: dict[Path, set[Path]] = defaultdict(set)
    for paths in groups.values():
        by_folder[paths[0].parent].update(paths)
    for folder, expected in by_folder.items():
        actual = set(folder.glob("rec_*.npz"))
        if actual != expected:
            raise ValueError(f"dataset folder must contain exactly its {len(expected)} planned recordings: {folder}")
    files = [p for paths in groups.values() for p in paths]
    return I.input_inventory(files), {name: len(paths) for name, paths in groups.items()}


def datasets():
    groups = _expected_dataset_paths()
    _, counts = dataset_inventory()
    t1, v1v, t3, v3 = (groups[k] for k in ("teacher_train", "teacher_val", "dart_train", "dart_val"))
    return _cat(load_files(t1), load_files(t3)), _cat(load_files(v1v), load_files(v3)), \
        counts


# ------------------------------------------------------------------ models
_CACHE: dict = {}


def graph_bundle():
    if "g" not in _CACHE:
        from flycircuit.eval import navigation as nv1
        from flycircuit.eval import navigation_v2 as nv2
        from flycircuit.graph.signs import SignPolicy, load_assignment_for_ids
        if nv1.N_OBS != N_CH:
            raise RuntimeError(f"v1 interface has {nv1.N_OBS} channels, expected {N_CH}")
        g, _s, meta = nv1.load_mb_graph("mushroom_body_right_pilot", n_sensory=N_CH)
        sa = load_assignment_for_ids(g.root_ids, SignPolicy())
        sign = sa.sign.astype(np.int8)
        meta["sign_policy"], meta["sign_assignment_hash"] = SignPolicy().tag(), sa.hash()
        nulls = nv2.load_nulls(g, "plain")
        _CACHE["g"] = (g, sign, meta, nv1.load_protocol(), nulls, nv2)
    return _CACHE["g"]


def build_model_v3(cond: str, seed: int, seed_index: int):
    import torch

    from flycircuit.eval import navigation as nv1
    from flycircuit.models.baselines import (
        EncoderDecoderOnly,
        GRUBaseline,
        match_encdec_hidden,
        match_gru_hidden,
    )
    from flycircuit.models.sparse_rnn import ModelSpec, SparseConnectomeRNN
    g, sign, meta, nproto, nulls, nv2 = graph_bundle()
    mi = nproto["model_interface"]
    torch.manual_seed(seed)
    spec = ModelSpec(n_in=N_CH, n_out=nv1.N_ACT, dt=mi["dt"], tau=mi["tau"], prior=mi["prior"],
                     trainable_encoder=mi["trainable_encoder"], trainable_decoder=mi["trainable_decoder"],
                     init_gain=mi["init_gain"], seed=seed)

    def sparse(gr):
        return SparseConnectomeRNN.sparse_from_graph(gr, spec, sign=sign, unknown_sign_policy="zero")

    ref = sparse(g)
    rep = ref.param_report()
    info: dict = {"condition": cond, "seed": seed, "signs": meta["sign_policy"]}
    if cond == "connectome_pilot_mb":
        model = ref
    elif cond == "null_v2_plain":
        ng = nulls[seed_index % len(nulls)]
        info["null_index"] = seed_index % len(nulls)
        info["null_reach_readout_fraction"] = nv1.reach_fraction(ng)
        model = sparse(ng)
        model.kind = "null_double_edge_swap"
    elif cond == "random_sparse_reachable":
        rg, rinfo = nv2.random_reachable(g, seed)
        info["random"] = rinfo
        model = sparse(rg)
        model.kind = "random_sparse_rnn"
    elif cond == "gru_param_matched":
        model = GRUBaseline(N_CH, nv1.N_ACT, match_gru_hidden(rep["trainable_params"], N_CH, nv1.N_ACT), seed)
    elif cond == "encdec_param_matched":
        model = EncoderDecoderOnly(N_CH, nv1.N_ACT,
                                   match_encdec_hidden(rep["trainable_params"], N_CH, nv1.N_ACT), seed)
    else:
        raise ValueError(cond)
    info["param_report"] = model.param_report()
    return model, info


# ------------------------------------------------------------------ bounded cache identities
def _search_identity() -> dict:
    inventory, counts = dataset_inventory()
    p = proto()
    return {"study": "flygym_nav_v3", "phase": "search_cache", "train_hash": train_hash(),
            "training_freeze_sha256": I.file_hash(TRAIN_MANIFEST), "runtime": I.runtime_identity(),
            "circuit_inputs": _circuit_inventory(), "datasets": inventory, "dataset_counts": counts,
            "search": p["training"]["lr_search"], "training_threads": TRAIN_THREADS}


def _ensure_search_manifest() -> dict:
    identity = _search_identity()
    if not SEARCH_MANIFEST.exists() and any((RES / name).exists() for name in ("search.json", "best_lr.json")):
        raise ValueError("unmanifested search cache exists; preserve it and record a prospective amendment")
    return I.freeze_manifest(SEARCH_MANIFEST, identity, _config_paths())


def _expected_search_keys(p: dict) -> dict[str, tuple[str, float]]:
    return {f"{cond}|{lr}": (cond, lr) for cond in CONDITIONS for lr in p["training"]["lr_search"]["grid"]}


def _validate_search_results(p: dict, results: dict, complete: bool = False) -> None:
    expected = _expected_search_keys(p)
    if not isinstance(results, dict) or not set(results).issubset(expected):
        raise ValueError("search cache has unknown or malformed coordinates")
    if complete and set(results) != set(expected):
        raise ValueError(f"search cache has {len(results)}/{len(expected)} planned coordinates")
    for key, row in results.items():
        if (not isinstance(row, dict) or type(row.get("diverged")) is not bool or
                row.get("status") not in ("ok", "diverged", "failed")):
            raise ValueError(f"malformed search result: {key}")
        value = row.get("val_loss")
        if row["status"] != "ok":
            if value is not None:
                raise ValueError(f"failed search result must have null validation loss: {key}")
        elif not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"accepted search result has nonfinite validation loss: {key}")


def _validate_best(p: dict, results: dict, best: dict) -> None:
    _validate_search_results(p, results, complete=True)
    if not isinstance(best, dict) or set(best) != set(CONDITIONS) | {"_data", "search_sha256"}:
        raise ValueError("selected-rate file has the wrong conditions")
    if best["search_sha256"] != I.file_hash(RES / "search.json"):
        raise ValueError("selected rates do not bind the exact search cache")
    _, counts = dataset_inventory()
    if best["_data"] != counts:
        raise ValueError("selected rates were computed with different dataset counts")
    grid = p["training"]["lr_search"]["grid"]
    for cond in CONDITIONS:
        vals = {lr: results[f"{cond}|{lr}"]["val_loss"] for lr in grid
                if results[f"{cond}|{lr}"]["val_loss"] is not None}
        if not vals:
            raise ValueError(f"no finite validation result for {cond}")
        selected = min(vals, key=vals.get)
        entry = best[cond]
        if (entry.get("lr") != selected or not np.isfinite(entry.get("lr", np.nan)) or
                entry.get("val_loss_by_lr") != {str(lr): results[f"{cond}|{lr}"]["val_loss"] for lr in grid}):
            raise ValueError(f"invalid selected validation rate for {cond}")


def _model_identity(p: dict) -> dict:
    inventory, counts = dataset_inventory()
    best_path, search_path = RES / "best_lr.json", RES / "search.json"
    best, search = _json(best_path, "selected rates"), _json(search_path, "search results")
    _validate_best(p, search, best)
    return {"study": "flygym_nav_v3", "phase": "model_cache", "train_hash": train_hash(),
            "training_freeze_sha256": I.file_hash(TRAIN_MANIFEST),
            "search_freeze_sha256": I.file_hash(SEARCH_MANIFEST), "runtime": I.runtime_identity(),
            "circuit_inputs": _circuit_inventory(), "datasets": inventory, "dataset_counts": counts,
            "search_sha256": I.file_hash(search_path), "selected_rate_sha256": I.file_hash(best_path),
            "training_threads": TRAIN_THREADS}


def _ensure_model_manifest(p: dict) -> dict:
    identity = _model_identity(p)
    has_models = (RES / "train_log.json").exists() or bool(list((RES / "models").glob("*.npz")))
    if not MODEL_MANIFEST.exists() and has_models:
        raise ValueError("unmanifested model cache exists; refusing to adopt or overwrite it")
    return I.freeze_manifest(MODEL_MANIFEST, identity, _config_paths())


def _validate_policy(path: Path, digest: str | None = None) -> str:
    if not path.is_file():
        raise ValueError(f"saved policy missing: {path}")
    actual = I.file_hash(path)
    if digest is not None and actual != digest:
        raise ValueError(f"saved policy hash changed: {path}")
    try:
        with np.load(path, allow_pickle=False) as z:
            if not z.files or "kind" not in z.files:
                raise ValueError("policy has no kind")
            for name in z.files:
                value = z[name]
                if value.dtype.kind in "fc" and not np.isfinite(value).all():
                    raise ValueError(f"nonfinite array {name}")
    except Exception as exc:
        raise ValueError(f"saved policy is unreadable: {path}: {exc}") from exc
    return actual


def _require_sha256(value, field: str, key: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(ch not in "0123456789abcdef" for ch in value)):
        raise ValueError(f"training record lacks a valid {field}: {key}")
    return value


def _training_keys(p: dict) -> set[str]:
    return {f"{cond}_s{seed}" for seed in p["seeds"]["train"] for cond in CONDITIONS}


def _validate_train_log(p: dict, log: dict, require_full: bool = False) -> dict[str, str]:
    expected, source_id = _training_keys(p), I._unchanged_source()[0]
    if not isinstance(log, dict) or not set(log).issubset(expected) or (require_full and set(log) != expected):
        raise ValueError("training log has unknown, malformed, or incomplete model keys")
    declared = {}
    context_sha = I.file_hash(MODEL_MANIFEST)
    for key, row in log.items():
        if (not isinstance(row, dict) or row.get("source_id") != source_id or
                row.get("model_manifest_sha256") != context_sha or row.get("train_hash") != train_hash() or
                row.get("status") not in ("ok", "diverged", "failed")):
            raise ValueError(f"training record identity is invalid: {key}")
        cond = key.rsplit("_s", 1)[0]
        trained = RES / "models" / f"{key}.npz"
        if row["status"] == "ok":
            digest = _require_sha256(row.get("policy_sha256"), "policy_sha256", key)
            declared[trained.name] = _validate_policy(trained, digest)
            if not isinstance(row.get("val_loss"), (int, float)) or not np.isfinite(row["val_loss"]):
                raise ValueError(f"trained model has nonfinite validation loss: {key}")
        elif trained.exists():
            raise ValueError(f"failed/diverged model unexpectedly has a trained policy: {trained}")
        untrained = RES / "models" / f"untrained_{key}.npz"
        if cond in UNTRAINED and (row.get("untrained_policy_sha256") is not None or untrained.exists()):
            digest = _require_sha256(row.get("untrained_policy_sha256"), "untrained_policy_sha256", key)
            declared[untrained.name] = _validate_policy(untrained, digest)
        elif cond in UNTRAINED and row["status"] in ("ok", "diverged"):
            raise ValueError(f"completed training record lacks its untrained control: {key}")
    actual = {path.name for path in (RES / "models").glob("*.npz")} if (RES / "models").exists() else set()
    if actual != set(declared):
        raise ValueError("model directory contains missing or unbound policy bundles")
    return declared


def _export_new_policy(export_policy, model, path: Path) -> str:
    if path.exists():
        raise ValueError(f"refusing to overwrite existing policy: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".v3-policy-", suffix=".npz", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        export_policy(model, temporary)
        digest = _validate_policy(temporary)
        os.replace(temporary, path)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


# ------------------------------------------------------------------ stages
def cmd_freeze(a):
    p = proto()
    bad = set(p["seeds"]["train"]) | set(p["seeds"]["search"]) | set(p["seeds"]["power_models"])
    if bad & FORBIDDEN_SEEDS:
        raise SystemExit(f"seed overlap with v1 body study: {sorted(bad & FORBIDDEN_SEEDS)}")
    stale_manifests = any(path.exists() for path in (SEARCH_MANIFEST, MODEL_MANIFEST, EVAL_MANIFEST))
    if not TRAIN_MANIFEST.exists() and ((RES.exists() and any(RES.iterdir())) or stale_manifests):
        raise SystemExit("v3 artifacts already exist without a training freeze; refusing retrospective freeze")
    manifest = I.freeze_manifest(TRAIN_MANIFEST, _training_identity(), _config_paths())
    print("frozen training spec", train_hash(), "at", manifest["frozen_utc"])


def cmd_collect(a):
    require_train_frozen()
    eps = _dart_episodes()
    out = run_rollouts("bc_dart", eps, a.workers)
    failed = [row["key"] for row in J.load_results(out) if row["status"] != "ok"]
    if failed:
        raise ValueError(f"DART collection preserved {len(failed)} failed executions; training is blocked")
    print("recorded", len(list(out.glob("rec_*.npz"))), "DART episodes")


def _val_loss(model, vl):
    return V1.val_loss(model, vl, 0.25)


def cmd_search(a):
    require_train_frozen()
    if a.threads != TRAIN_THREADS:
        raise ValueError(f"v3 training/search thread count is frozen at {TRAIN_THREADS}")
    P = proto()
    tr, vl, counts = datasets()
    _ensure_search_manifest()
    S = P["training"]["lr_search"]
    out = RES / "search.json"
    RES.mkdir(parents=True, exist_ok=True)
    res = _json(out, "search results") if out.exists() else {}
    _validate_search_results(P, res)
    deadline = V1._deadline(a)
    for cond in CONDITIONS:
        for lr in S["grid"]:
            key = f"{cond}|{lr}"
            if key in res:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                print("time slice complete; resume", flush=True)
                return 75
            t0 = time.time()
            try:
                model, _ = build_model_v3(cond, S["seed"], 0)
                r = V1.train_one(model, tr, lr, S["epochs"], S["seed"], P["training"], a.threads)
                r["status"] = "diverged" if r["diverged"] else "ok"
                r["val_loss"] = _val_loss(model, vl) if r["status"] == "ok" else None
            except Exception as exc:
                r = {"diverged": False, "status": "failed", "val_loss": None, "error": repr(exc)}
            r["seconds"] = time.time() - t0
            res[key] = r
            _validate_search_results(P, res)
            I.atomic_json(out, res)
            print(key, r, flush=True)
    _validate_search_results(P, res, complete=True)
    best = {}
    for cond in CONDITIONS:
        c = {lr: res[f"{cond}|{lr}"]["val_loss"] for lr in S["grid"]}
        ok = {k: v for k, v in c.items() if v is not None and np.isfinite(v)}
        if not ok:
            raise ValueError(f"all planned validation searches failed for {cond}; no rate accepted")
        b = min(ok, key=ok.get)
        best[cond] = {"lr": b, "grid_edge": b in (S["grid"][0], S["grid"][-1]),
                      "val_loss_by_lr": {str(k): v for k, v in c.items()}}
    best["_data"] = counts
    best["search_sha256"] = I.file_hash(out)
    I.atomic_json(RES / "best_lr.json", best)
    _validate_best(P, res, best)
    print(json.dumps({k: (best[k]["lr"], best[k]["grid_edge"]) for k in CONDITIONS}))


def cmd_train(a):
    from flycircuit.bodies.export import export_policy
    require_train_frozen()
    if a.threads != TRAIN_THREADS:
        raise ValueError(f"v3 training/search thread count is frozen at {TRAIN_THREADS}")
    P = proto()
    tr, vl, _ = datasets()
    _ensure_search_manifest()
    best = _json(RES / "best_lr.json", "selected rates")
    search = _json(RES / "search.json", "search results")
    _validate_best(P, search, best)
    _ensure_model_manifest(P)
    mdir = RES / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    log_p = RES / "train_log.json"
    log = _json(log_p, "training log") if log_p.exists() else {}
    _validate_train_log(P, log)
    deadline = V1._deadline(a)
    for si, seed in enumerate(P["seeds"]["train"]):
        for cond in CONDITIONS:
            key = f"{cond}_s{seed}"
            if key in log:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                print(f"time slice complete; {len(log)} models saved; resume", flush=True)
                return 75
            t0 = time.time()
            model, info, info_u, untrained_tmp = None, {}, {}, None
            untrained_sha, policy_sha = None, None
            try:
                model, info = build_model_v3(cond, seed, si)
                if cond in UNTRAINED:
                    fd, name = tempfile.mkstemp(prefix=".v3-untrained-", suffix=".npz", dir=mdir)
                    os.close(fd)
                    untrained_tmp = Path(name)
                    export_policy(model, untrained_tmp)
                    _validate_policy(untrained_tmp)
                    info_u = {"val_loss_untrained": _val_loss(model, vl)}
                r = V1.train_one(model, tr, best[cond]["lr"], P["training"]["epochs"], seed,
                                 P["training"], a.threads)
                status = "diverged" if r["diverged"] else "ok"
                if status == "ok":
                    val_loss = _val_loss(model, vl)
                    if not np.isfinite(val_loss):
                        raise ValueError("trained model has nonfinite validation loss")
                    policy_sha = _export_new_policy(export_policy, model, mdir / f"{key}.npz")
            except Exception as exc:
                r, status = {"diverged": False, "error": repr(exc)}, "failed"
            if untrained_tmp is not None:
                untrained_path = mdir / f"untrained_{key}.npz"
                if untrained_path.exists():
                    raise ValueError(f"refusing to overwrite existing policy: {untrained_path}")
                untrained_sha = I.file_hash(untrained_tmp)
                os.replace(untrained_tmp, untrained_path)
            source_id, _ = I._unchanged_source()
            record = {**r, **info_u, "status": status, "source_id": source_id, "train_hash": train_hash(),
                      "model_manifest_sha256": I.file_hash(MODEL_MANIFEST), "policy_sha256": policy_sha,
                      "untrained_policy_sha256": untrained_sha, "lr": best[cond]["lr"],
                      "train_seconds": time.time() - t0, "param_report": info.get("param_report"),
                      "null_index": info.get("null_index"), "random": info.get("random")}
            if status == "ok":
                record["val_loss"] = val_loss
            log[key] = record
            log = _json_value(log)
            I.atomic_json(log_p, log)
            _validate_train_log(P, log)
            print(key, {k: v for k, v in log[key].items() if k != "param_report"}, flush=True)
    _validate_train_log(P, log, require_full=True)


def _label(r):
    pol = r["policy"]
    if pol in ("teacher", "scripted", "straight"):
        return pol, None
    stem = Path(r["policy_path"]).stem
    untr = stem.startswith("untrained_")
    cond, s = stem.replace("untrained_", "").rsplit("_s", 1)
    return ("untrained_" + cond if untr else cond), int(s)


def _model_log(p: dict, require_full: bool = True) -> dict:
    I.require_manifest(MODEL_MANIFEST, _model_identity(p))
    log = _json(RES / "train_log.json", "training log")
    _validate_train_log(p, log, require_full=require_full)
    return log


def _require_successful_model_matrix(p: dict) -> dict:
    log = _model_log(p, require_full=True)
    failed = {key: row["status"] for key, row in log.items() if row["status"] != "ok"}
    if failed:
        raise ValueError(f"full trained policy matrix unavailable: {failed}")
    return log


def _power_episodes(p: dict) -> tuple[np.ndarray, list[dict]]:
    e = p["evaluation"]
    masks = balanced_masks(e["n_power_val_episodes"], e["dropout"], seed=1234)
    eps = []
    for seed in p["seeds"]["power_models"]:
        for cond in CONDITIONS:
            path = RES / "models" / f"{cond}_s{seed}.npz"
            eps += eval_episodes("val", 3000, masks, str(path))
    eps += eval_episodes("val", 3000, masks, "scripted")
    return masks, eps


def _validate_power_rows(p: dict, masks: np.ndarray, rows: list[dict]) -> None:
    n = p["evaluation"]["n_power_val_episodes"]
    expected = {(c, seed, 3000 + k) for seed in p["seeds"]["power_models"]
                for c in CONDITIONS for k in range(n)}
    expected |= {("scripted", None, 3000 + k) for k in range(n)}
    seen, geometry = set(), {}
    for row in rows:
        cond, seed = _label(row)
        unit = cond, seed, row.get("idx")
        if unit not in expected or unit in seen:
            raise ValueError(f"unexpected or duplicate power unit: {unit}")
        seen.add(unit)
        k = row["idx"] - 3000
        if row.get("split") != "val" or row.get("mask") != masks[k].astype(int).tolist():
            raise ValueError(f"power split/mask mismatch: {unit}")
        if row.get("dropout") != p["evaluation"]["dropout"] or row.get("action_noise") != 0.0:
            raise ValueError(f"power perturbation metadata mismatch: {unit}")
        geom = row.get("target_dist_mm"), row.get("target_bearing_deg")
        if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in geom):
            raise ValueError(f"power target geometry invalid: {unit}")
        if row["idx"] in geometry and geometry[row["idx"]] != geom:
            raise ValueError(f"power target geometry is not paired: {unit}")
        geometry[row["idx"]] = geom
    if seen != expected:
        raise ValueError(f"power matrix incomplete: {len(seen)}/{len(expected)} units")


def _derive_power_report(p: dict, rows: list[dict]) -> dict:
    by = defaultdict(list)
    for row in rows:
        cond, _ = _label(row)
        by[cond].append(float(row["status"] == "ok" and row["success"]))
    trained = [by[c] for c in CONDITIONS]
    p_hat = float(np.mean([np.mean(values) for values in trained]))
    h = 0.08
    n_rule = math.ceil(1.96 ** 2 * p_hat * (1 - p_hat) / h ** 2)
    n_test = int(min(144, max(96, math.ceil(n_rule / 12) * 12)))
    return {"p_hat_pooled_trained_val": p_hat,
            "success_by_condition": {c: float(np.mean(by[c])) for c in [*CONDITIONS, "scripted"]},
            "n_power_val_episodes": p["evaluation"]["n_power_val_episodes"],
            "n_rule_unclipped": n_rule, "n_test_episodes": n_test,
            "achieved_episode_sampling_half_width": 1.96 * math.sqrt(p_hat * (1 - p_hat) / n_test),
            "cap_binds": n_rule > 144,
            "execution_failures_counted_as_zero": sum(r["status"] != "ok" for r in rows),
            "note": "VALIDATION episodes only (val idx 3000+); no test episode has been run"}


def _power_identity(p: dict) -> dict:
    out = RES / "power_val"
    return {"study": "flygym_nav_v3", "phase": "power", "train_hash": train_hash(),
            "source_id": I._unchanged_source()[0], "runtime": I.runtime_identity(),
            "model_manifest_sha256": I.file_hash(MODEL_MANIFEST),
            "selected_rate_sha256": I.file_hash(RES / "best_lr.json"),
            "rollout_manifest_sha256": I.file_hash(out / "manifest.json"),
            "rollout_completion_sha256": I.file_hash(out / "complete.json"),
            "results_sha256": I.file_hash(out / "results.jsonl")}


def require_power_record() -> dict:
    p = proto()
    _require_successful_model_matrix(p)
    masks, eps = _power_episodes(p)
    out = RES / "power_val"
    if not (out / "manifest.json").is_file() or _json(out / "manifest.json", "power rollout manifest") != _rollout_identity("power_val", eps):
        raise ValueError("power rollout manifest missing or changed")
    rows = J.load_results(out)
    _validate_rollout_rows(eps, rows)
    _validate_power_rows(p, masks, rows)
    complete = _json(out / "complete.json", "power completion")
    expected_complete = {"identity_sha256": I.canonical_hash(_rollout_identity("power_val", eps)),
                         "results_sha256": I.file_hash(out / "results.jsonl"), "n_rows": len(rows)}
    if complete != expected_complete:
        raise ValueError("power rollout completion record changed")
    saved = _json(RES / "power.json", "power record")
    report = _derive_power_report(p, rows)
    expected = {**report, "identity": _power_identity(p)}
    if saved != expected:
        raise ValueError("power record differs from the complete validation matrix")
    effective_eval_protocol(saved)
    return saved


def cmd_power(a):
    """VALIDATION-only variance estimate -> test sample size (rule pre-registered in the protocol)."""
    require_train_frozen()
    P = proto()
    _require_successful_model_matrix(P)
    masks, eps = _power_episodes(P)
    if (RES / "power.json").exists():
        rep = require_power_record()
        print(json.dumps(rep, indent=2))
        return
    out = run_rollouts("power_val", eps, a.workers)
    rows = J.load_results(out)
    _validate_power_rows(P, masks, rows)
    rep = {**_derive_power_report(P, rows), "identity": _power_identity(P)}
    I.atomic_json(RES / "power.json", rep)
    print(json.dumps(rep, indent=2))


def _evaluation_identity(p: dict, power: dict) -> dict:
    log = _require_successful_model_matrix(proto())
    policy_paths = []
    for key in sorted(log):
        policy_paths.append(RES / "models" / f"{key}.npz")
        if key.rsplit("_s", 1)[0] in UNTRAINED:
            policy_paths.append(RES / "models" / f"untrained_{key}.npz")
    inventory, counts = dataset_inventory()
    return {"study": "flygym_nav_v3", "phase": "evaluation", "runtime": I.runtime_identity(),
            "effective_eval_hash": effective_eval_hash(p), "n_test_episodes": p["evaluation"]["n_test_episodes"],
            "training_freeze_sha256": I.file_hash(TRAIN_MANIFEST),
            "model_manifest_sha256": I.file_hash(MODEL_MANIFEST),
            "training_log_sha256": I.file_hash(RES / "train_log.json"),
            "selected_rate_sha256": I.file_hash(RES / "best_lr.json"),
            "power_record_sha256": I.file_hash(RES / "power.json"),
            "power_results_sha256": power["identity"]["results_sha256"],
            "circuit_inputs": _circuit_inventory(), "datasets": inventory, "dataset_counts": counts,
            "policy_bundles": I.input_inventory(policy_paths),
            "test_masks_sha256": I.canonical_hash(balanced_masks(
                p["evaluation"]["n_test_episodes"], p["evaluation"]["dropout"], seed=20261017).astype(int).tolist())}


def cmd_freeze_eval(a):
    require_train_frozen()
    power = require_power_record()
    p = effective_eval_protocol(power)
    if not EVAL_MANIFEST.exists() and ((RES / "eval").exists() or (RES / "test_masks.npy").exists()):
        raise SystemExit("evaluation artifacts exist without a freeze; refusing retrospective freeze")
    manifest = I.freeze_manifest(EVAL_MANIFEST, _evaluation_identity(p, power), _config_paths())
    print("frozen evaluation spec", effective_eval_hash(p), "N =", p["evaluation"]["n_test_episodes"],
          "at", manifest["frozen_utc"])


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".v3-array-", suffix=".npy", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        np.save(temporary, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _eval_episodes(p: dict, masks: np.ndarray) -> list[dict]:
    e, eps = p["evaluation"], []
    for policy in ("teacher", "scripted", "straight"):
        eps += eval_episodes("test", e["test_idx_start"], masks, policy)
    for seed in p["seeds"]["train"]:
        for cond in CONDITIONS:
            eps += eval_episodes("test", e["test_idx_start"], masks,
                                 str(RES / "models" / f"{cond}_s{seed}.npz"))
        for cond in UNTRAINED:
            eps += eval_episodes("test", e["test_idx_start"], masks[:e["n_untrained_episodes"]],
                                 str(RES / "models" / f"untrained_{cond}_s{seed}.npz"))
    return eps


def cmd_eval(a):
    P = require_eval_frozen()
    _require_successful_model_matrix(proto())
    E = P["evaluation"]
    n = E["n_test_episodes"]
    masks = balanced_masks(n, E["dropout"], seed=20261017)
    mask_path = RES / "test_masks.npy"
    if mask_path.exists():
        saved_masks = np.load(mask_path, allow_pickle=False)
        if not np.array_equal(saved_masks, masks):
            raise ValueError("saved test masks differ from the frozen stratified design")
    else:
        _atomic_npy(mask_path, masks)
    eps = _eval_episodes(P, masks)
    out = run_rollouts("eval", eps, a.workers)
    print("done", len(J.load_results(out)), "episodes")


# ------------------------------------------------------------------ crossed bootstrap + report
def crossed_boot(mats: list[np.ndarray], stat, n_boot: int = 4000, seed: int = 0):
    """mats: success matrices (models x episodes). Resample model rows AND episode columns jointly.
    stat(list of resampled matrices) -> float. Returns (estimate, lo, hi, p_two_sided_vs_zero, draws)."""
    rng = np.random.default_rng(seed)
    m, e = mats[0].shape
    est = stat(mats)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        ri, ci = rng.integers(0, m, m), rng.integers(0, e, e)
        draws[b] = stat([x[np.ix_(ri, ci)] for x in mats])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((np.sum(draws <= 0) + 1) / (n_boot + 1), (np.sum(draws >= 0) + 1) / (n_boot + 1))
    return float(est), float(lo), float(hi), float(min(p, 1.0)), draws


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    out, prev, m = {}, 0.0, len(items)
    for i, (k, p) in enumerate(items):
        prev = max(prev, min(1.0, (m - i) * p))
        out[k] = prev
    return out


def channel_boot(mat: np.ndarray, dropped: np.ndarray, n_boot=1000, seed=0):
    """Dropped-minus-kept, conditional on the fixed exposure counts in this design.

    Resample model rows jointly and episode indices separately WITHIN the two
    exposure strata. The exposure label thus stays attached to every outcome.
    This descriptive contrast is not a causal single-channel intervention: other
    dropped channels co-vary under the constrained mask design.
    """
    mat, dropped = np.asarray(mat), np.asarray(dropped)
    if (mat.ndim != 2 or not mat.shape[0] or not np.isfinite(mat).all()
            or dropped.shape != (mat.shape[1],) or not np.isin(dropped, [0, 1]).all()):
        raise ValueError("invalid success matrix or exposure labels")
    d, k = np.flatnonzero(dropped), np.flatnonzero(~dropped.astype(bool))
    if not len(d) or not len(k) or n_boot <= 0:
        raise ValueError("both exposure strata and a positive draw count are required")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        ri = rng.integers(0, mat.shape[0], mat.shape[0])
        di, ki = rng.choice(d, len(d)), rng.choice(k, len(k))
        draws[b] = mat[np.ix_(ri, di)].mean() - mat[np.ix_(ri, ki)].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(mat[:, d].mean() - mat[:, k].mean()), float(lo), float(hi), draws


def validate_eval_rows(P: dict, masks: np.ndarray, rows: list[dict]) -> dict:
    """Require exact policy x seed x episode coverage before calculating endpoints."""
    E = P["evaluation"]
    n, start, nu = E["n_test_episodes"], E["test_idx_start"], E["n_untrained_episodes"]
    seeds = P["seeds"]["train"]
    if not seeds or len(set(seeds)) != len(seeds) or not (0 < nu <= n):
        raise ValueError("invalid planned seeds or episode counts")
    if (masks.shape != (n, N_CH) or not np.isin(masks, [0, 1]).all()
            or not np.array_equal(masks, balanced_masks(n, E["dropout"], seed=20261017))):
        raise ValueError("test masks violate the frozen stratified design")
    expected = {(c, s, start + k) for c in CONDITIONS for s in seeds for k in range(n)}
    expected |= {("untrained_" + c, s, start + k) for c in UNTRAINED for s in seeds for k in range(nu)}
    expected |= {(c, None, start + k) for c in ("teacher", "scripted", "straight") for k in range(n)}
    seen, geometry, failed = set(), {}, []
    for r in rows:
        policy, policy_path = r.get("policy"), r.get("policy_path")
        if not isinstance(policy, str) or not isinstance(policy_path, str):
            raise ValueError("missing policy identity")
        if ((policy in ("teacher", "scripted", "straight") and policy_path != policy)
                or (policy not in ("teacher", "scripted", "straight")
                    and Path(policy_path).stem != policy)):
            raise ValueError("contradictory policy label and policy_path")
        c, s = _label(r)
        idx = r["idx"]
        if type(idx) is not int:
            raise ValueError("episode idx must be an integer")
        key = c, s, idx
        if key not in expected or key in seen:
            raise ValueError(f"unexpected or duplicate evaluation unit: {key}")
        seen.add(key)
        if r.get("split") != "test" or not np.array_equal(r.get("mask"), masks[idx - start]):
            raise ValueError(f"split/mask mismatch: {key}")
        if r.get("dropout") != E["dropout"] or r.get("action_noise") != 0.0:
            raise ValueError(f"dropout/action noise mismatch: {key}")
        if r.get("success") not in (False, True, 0, 1) or r.get("status") not in ("ok", "nan_action", "nan_state"):
            raise ValueError(f"invalid outcome/status: {key}")
        if not np.isfinite(r.get("spl", np.nan)) or not 0 <= r["spl"] <= 1:
            raise ValueError(f"invalid SPL: {key}")
        g = (r.get("target_dist_mm"), r.get("target_bearing_deg"))
        if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in g):
            raise ValueError(f"missing/nonfinite target geometry: {key}")
        if idx in geometry and geometry[idx] != g:
            raise ValueError(f"unpaired target geometry: {key}")
        geometry[idx] = g
        if r["status"] != "ok":
            failed.append({"condition": c, "seed": s, "idx": idx, "status": r["status"]})
    missing = sorted(expected - seen, key=lambda v: (v[0], -1 if v[1] is None else v[1], v[2]))
    return {"complete": not missing, "expected_units": len(expected), "observed_units": len(seen),
            "missing_units": [list(k) for k in missing], "execution_failures": failed,
            "execution_valid": not failed, "planned_seed_ids": seeds}


def write_report_files(report, markdown):
    """Replace reports without truncating an existing (possibly aliased) inode."""
    payloads = [(RES / "report.json", json.dumps(report, indent=2, allow_nan=False)),
                (RES / "report.md", markdown)]
    pending = []
    try:
        for dest, content in payloads:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=RES,
                                             prefix=".v3-report-", delete=False) as stream:
                pending.append((Path(stream.name), dest))
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        for temporary, dest in pending:
            os.replace(temporary, dest)
    finally:
        for temporary, _ in pending:
            temporary.unlink(missing_ok=True)


def cmd_report(a):
    try:
        return _report_impl(a)
    except (Exception, SystemExit) as exc:
        blocked = {"complete": False, "status": "blocked", "error": str(exc),
                   "owner_ambition_0.68": {"status": "blocked", "connectome_success": None}}
        write_report_files(blocked, "# Evaluation blocked\n\n" + str(exc) + "\n")
        raise


def _report_impl(a):
    P = require_eval_frozen()
    E = P["evaluation"]
    n, start = E["n_test_episodes"], E["test_idx_start"]
    masks = np.load(RES / "test_masks.npy")
    rows = _require_rollout_cache("eval", _eval_episodes(P, masks), complete_required=False)
    coverage = validate_eval_rows(P, masks, rows)
    if not coverage["complete"]:
        out = {"protocol_id": P["protocol_id"], "complete": False, "status": "blocked",
               "evaluation_state": "incomplete",
               "coverage": coverage, "table": {}, "contrasts": {},
               "owner_ambition_0.68": {"connectome_success": None, "status": "blocked"}}
        write_report_files(out,
            f"# Evaluation incomplete\n\n{coverage['observed_units']}/{coverage['expected_units']} units saved. "
            "No final endpoint or met/missed claim is certified.\n")
        print("evaluation incomplete; resume missing planned episodes")
        return 3
    mats: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    spl: dict[str, list] = defaultdict(list)
    n_fail = 0
    refs: dict[str, np.ndarray] = {}
    for r in rows:
        c, s = _label(r)
        k = r["idx"] - start
        if r["status"] != "ok":
            n_fail += 1
        # Retain every original row; execution failures contribute zero under the
        # predeclared policy, even if a malformed worker recorded success=True.
        outcome = float(r["status"] == "ok" and r["success"])
        if c in ("teacher", "scripted", "straight"):
            refs.setdefault(c, np.full(n, np.nan))[k] = outcome
            continue
        if c.startswith("untrained_"):
            mats[c].setdefault(s, np.full(E["n_untrained_episodes"], np.nan))[k] = outcome
            continue
        mats[c].setdefault(s, np.full(n, np.nan))[k] = outcome
        spl[c].append(r["spl"] if r["status"] == "ok" else 0.0)
    seeds = P["seeds"]["train"]
    # Coverage has proved that every actual seed ID exists for every condition.
    M = {c: np.stack([mats[c][s] for s in seeds]) for c in CONDITIONS}
    failed_conditions = {r["condition"] for r in coverage["execution_failures"]}
    complete = {c: bool(not np.isnan(v).any() and c not in failed_conditions) for c, v in M.items()}
    table = {}
    for c, mat in M.items():
        est, lo, hi, _, _ = crossed_boot([mat], lambda m: float(np.mean(m[0])), seed=1)
        per_model = mat.mean(1)
        table[c] = {"n_models": mat.shape[0], "n_episodes": n, "success": est, "ci95_crossed": [lo, hi],
                    "per_model_success": per_model.tolist(), "model_sd": float(per_model.std(ddof=1)) if len(per_model) > 1 else None,
                    "episode_sd": float(mat.mean(0).std(ddof=1)), "spl_mean": float(np.mean(spl[c])),
                    "complete": complete[c]}
    ps, contr = {}, {}
    conn = M.get("connectome_pilot_mb")
    for ref in ("gru_param_matched", "null_v2_plain", "random_sparse_reachable", "encdec_param_matched"):
        if conn is None or ref not in M:
            continue
        est, lo, hi, p, _ = crossed_boot([conn, M[ref]], lambda m: float(np.mean(m[0]) - np.mean(m[1])), seed=2)
        contr[f"connectome-minus-{ref}"] = {"estimate": est, "ci95_crossed": [lo, hi], "p_boot": p}
        ps[f"connectome-minus-{ref}"] = p
    for k_, v in holm(ps).items():
        contr[k_]["p_holm"] = v
    # trained minus untrained on the shared first n_untrained episodes
    nu = E["n_untrained_episodes"]
    for c in UNTRAINED:
        u = mats.get("untrained_" + c)
        if u and c in M:
            um = np.stack([u[s] for s in seeds])
            est, lo, hi, p, _ = crossed_boot([M[c][:, :nu], um], lambda m: float(np.mean(m[0]) - np.mean(m[1])), seed=3)
            contr[f"trained-minus-untrained|{c}"] = {"estimate": est, "ci95_crossed": [lo, hi], "n_episodes": nu}
    # per-channel breakdown: success when channel c dropped minus kept
    chan = {}
    names = P["interface"]["obs_channels"]
    for c, mat in M.items():
        chan[c] = {}
        for ch in range(N_CH):
            d = masks[:, ch].astype(bool)

            est, lo, hi, _ = channel_boot(mat, d, n_boot=1000, seed=10 + ch)
            chan[c][names[ch]] = {"dropped_minus_kept": est, "ci95_crossed": [lo, hi],
                                  "success_when_dropped": float(mat[:, d].mean()), "n_dropped_episodes": int(d.sum())}
    refsum = {k: {"success": float(np.nanmean(v)), "n": int(np.sum(~np.isnan(v)))} for k, v in refs.items()}
    conn_s = table.get("connectome_pilot_mb", {}).get("success")
    status = ("blocked" if n_fail else
              ("evaluated/met" if conn_s >= 0.68 else "evaluated/missed"))
    out = {"protocol_id": P["protocol_id"], "train_hash": train_hash(),
           "eval_hash": effective_eval_hash(P),
           "complete": coverage["execution_valid"], "coverage": coverage, "paired_seed_ids": seeds,
           "n_test_episodes": n, "n_failed_status": n_fail, "table": table, "contrasts": contr,
           "per_channel_dropout_breakdown": chan, "references": refsum,
           "per_channel_interval_method": "model rows crossed with episode resampling within dropped/kept strata; descriptive association, not a causal channel intervention",
           "owner_ambition_0.68": {"connectome_success": conn_s, "status": status,
                                   "note": "owner's ambition, not a protocol gate; says nothing about anatomy"},
           "limitations": P["caveats"] + ["5 models per condition: model-level interval exploratory-grade",
                                          "episodes are shared across models (crossed), not independent of each other"]}
    L = [f"# FlyGym v3 ({P['protocol_id']}) -- stratified 25% dropout, N={n} test episodes", "",
         "| condition | models | success [95% crossed CI] | SPL | model SD |", "|---|---|---|---|---|"]
    for c, v in table.items():
        L.append(f"| {c} | {v['n_models']} | {v['success']:.3f} [{v['ci95_crossed'][0]:.3f}, {v['ci95_crossed'][1]:.3f}] "
                 f"| {v['spl_mean']:.3f} | {v['model_sd']} |")
    L += ["", "## References (same episodes and masks)", ""] + [f"- {k}: {v['success']:.3f} (n={v['n']})" for k, v in refsum.items()]
    L += ["", "## Connectome contrasts (crossed bootstrap; Holm over four)", ""]
    for k, v in contr.items():
        L.append(f"- {k}: {v['estimate']:+.3f} [{v['ci95_crossed'][0]:+.3f}, {v['ci95_crossed'][1]:+.3f}]"
                 + (f", Holm p={v['p_holm']:.3f}" if "p_holm" in v else ""))
    L += ["", f"Owner ambition 0.68: {status} (connectome {conn_s}). Not a protocol gate; not evidence for anatomy."]
    if n_fail:
        L += ["", f"Execution invalid: {n_fail} episodes failed. Completeness is false; tables are descriptive under the zero-valued failure policy."]
    write_report_files(out, "\n".join(L) + "\n")
    print("\n".join(L))
    return 4 if n_fail else 0


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("freeze", cmd_freeze), ("collect", cmd_collect), ("search", cmd_search), ("train", cmd_train),
                     ("power", cmd_power), ("freeze-eval", cmd_freeze_eval), ("eval", cmd_eval),
                     ("report", cmd_report)):
        s = sub.add_parser(name)
        s.add_argument("--workers", type=int, default=ROLLOUT_WORKERS)
        s.add_argument("--threads", type=int, default=TRAIN_THREADS)
        s.add_argument("--max-minutes", type=V1._positive_minutes, default=None)
        s.set_defaults(fn=fn)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
