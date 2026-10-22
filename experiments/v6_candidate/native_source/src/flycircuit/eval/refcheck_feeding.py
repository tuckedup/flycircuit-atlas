"""Feeding-assay (sugar GRN -> MN9) reproduction on Shiu et al.'s own connectivity (v630 / v783).

Protocol and tolerances: ``configs/experiments/feeding_protocol.yaml`` (predeclared; its sha256 is
stored in every result). Model-vs-model comparison only; not a biological validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from flycircuit.config import project_root
from flycircuit.models.lif import LIFParams, poisson_events
from flycircuit.models.lif_stream import simulate_stream

SUGAR_RIGHT = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9 = 720575940660219265
RELEASES = {
    "630": ("2023_03_23_completeness_630_final.csv", "2023_03_23_connectivity_630_final.parquet"),
    "783": ("Completeness_783.csv", "Connectivity_783.parquet"),
}


def shiu_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "external" / "Drosophila_brain_model"


def protocol(root: Path | None = None) -> tuple[dict, str]:
    p = (root or project_root()) / "configs" / "experiments" / "feeding_protocol.yaml"
    raw = p.read_bytes()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


@dataclass
class ShiuNet:
    release: str
    ids: np.ndarray  # int64 FlyWire root IDs in Brian index order
    W: sp.csc_matrix  # [post, pre] in mV (signed contacts * w_syn)
    id2idx: dict[int, int]
    params: LIFParams
    nnz: int

    def idx(self, flyids) -> np.ndarray:
        return np.array([self.id2idx[int(i)] for i in flyids if int(i) in self.id2idx], np.int64)


def load_shiu(release: str, root: Path | None = None) -> ShiuNet:
    comp_f, con_f = RELEASES[release]
    d = shiu_dir(root)
    ids = pd.read_csv(d / comp_f, index_col=0).index.to_numpy(dtype=np.int64)
    con = pd.read_parquet(
        d / con_f, columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]
    )
    p = LIFParams.from_shiu_source()
    n = len(ids)
    W = sp.coo_matrix(
        (con["Excitatory x Connectivity"].to_numpy(np.float64) * p.w_syn,
         (con["Postsynaptic_Index"].to_numpy(np.int64), con["Presynaptic_Index"].to_numpy(np.int64))),
        shape=(n, n),
    ).tocsc()
    W.sum_duplicates()
    return ShiuNet(release, ids, W, {int(i): k for k, i in enumerate(ids.tolist())}, p, W.nnz)


def run_condition(net: ShiuNet, freq_hz: float, n_trials: int, seed: int = 20261015,
                  t_ms: float = 1000.0, backend: str = "torch", device: str = "cuda",
                  dtype: str = "float64", stim_ids=SUGAR_RIGHT, watch_ids=(MN9,)) -> dict[str, Any]:
    p = net.params.with_(r_poi=freq_hz)
    steps = int(round(t_ms / p.dt))
    tg = net.idx(stim_ids)
    watch = np.unique(np.concatenate([tg, net.idx(watch_ids)]))
    counts = np.zeros((n_trials, len(net.ids)), dtype=np.int32)
    metas = []
    for t in range(n_trials):
        es, et = poisson_events(tg, steps, freq_hz, p.dt, seed=seed + 1000 * t)
        s = simulate_stream(net.W, steps, p, es, et, no_refractory=tg, watch=watch,
                            backend=backend, device=device, dtype=dtype)
        counts[t] = s.counts
        metas.append(s.meta)
    return {"counts": counts, "meta": metas, "stim_idx": tg, "n_stim_found": int(len(tg)),
            "n_stim_requested": len(stim_ids), "t_ms": t_ms}


def published_counts(name: str, ids: np.ndarray, root: Path | None = None) -> np.ndarray:
    """Per-trial spike counts ``[30, n]`` from a fetched published raw parquet (sugarR_*Hz)."""
    path = ((root or project_root()) / "data/raw/shiu_published/results/figure_1/data" /
            f"{name}.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Figure 1 published output missing: {path}; run "
                                "python scripts/linux/fetch_shiu_published.py (preserved archive paths)")
    df = pd.read_parquet(path, columns=["trial", "flywire_id"])
    pos = pd.Series(np.arange(len(ids)), index=ids)
    g = df.groupby(["trial", "flywire_id"]).size().reset_index(name="c")
    out = np.zeros((int(df["trial"].max()) + 1, len(ids)), dtype=np.int32)
    out[g["trial"].to_numpy(), pos.loc[g["flywire_id"].to_numpy()].to_numpy()] = g["c"].to_numpy()
    return out


def _pearson(a, b):
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _jacc(a: set, b: set):
    u = len(a | b)
    return None if u == 0 else len(a & b) / u


def compare(ours: np.ndarray, ref: np.ndarray, ids: np.ndarray, stim_idx: np.ndarray, mn9_idx: int,
            tol: dict, t_s: float = 1.0) -> dict[str, Any]:
    """Compare per-trial count matrices ``[trials, n]`` (ours vs reference) on the predeclared metrics."""
    ro, rr = ours.mean(0) / t_s, ref.mean(0) / t_s
    mask = (ro > 0.5) | (rr > 0.5)
    thr = tol.get("responder_threshold_hz", 5.0)
    resp_o, resp_r = set(np.flatnonzero(ro >= thr).tolist()), set(np.flatnonzero(rr >= thr).tolist())
    nonstim = np.setdiff1d(np.arange(len(ids)), stim_idx)
    nonstim_set = set(nonstim.tolist())
    resp_o2 = resp_o & nonstim_set
    resp_r2 = resp_r & nonstim_set
    top_o = set(np.argsort(-ro, kind="stable")[:200].tolist())
    top_r = set(np.argsort(-rr, kind="stable")[:200].tolist())
    grn_o, grn_r = float(ro[stim_idx].mean()), float(rr[stim_idx].mean())
    mo, mr = ours[:, mn9_idx] / t_s, ref[:, mn9_idx] / t_s
    se = float(np.sqrt(mo.var(ddof=1) / len(mo) + mr.var(ddof=1) / len(mr))) if len(mo) > 1 and len(mr) > 1 else float("nan")
    mn9_bound = 3 * se + 0.05 * float(mr.mean())
    act_o, act_r = int((ours.sum(0) > 0).sum()), int((ref.sum(0) > 0).sum())
    out = {
        "rate_pearson_active": _pearson(ro[mask], rr[mask]), "n_compared_neurons": int(mask.sum()),
        "sugar_grn_mean_rate_ours": grn_o, "sugar_grn_mean_rate_ref": grn_r,
        "sugar_grn_mean_rate_rel_err": abs(grn_o - grn_r) / grn_r if grn_r else None,
        "responder_jaccard_all": _jacc(resp_o, resp_r),
        "responder_jaccard_nonstimulated": _jacc(resp_o2, resp_r2),
        "n_responders_ours_ref": [len(resp_o), len(resp_r)],
        "top200_jaccard": _jacc(top_o, top_r),
        "mn9_rate_ours_mean_sd": [float(mo.mean()), float(mo.std(ddof=1)) if len(mo) > 1 else None],
        "mn9_rate_ref_mean_sd": [float(mr.mean()), float(mr.std(ddof=1)) if len(mr) > 1 else None],
        "mn9_abs_diff": abs(float(mo.mean()) - float(mr.mean())), "mn9_allowed_diff": mn9_bound,
        "n_active_ours_ref": [act_o, act_r],
        "n_active_rel_err": abs(act_o - act_r) / act_r if act_r else None,
        "n_trials_ours_ref": [int(ours.shape[0]), int(ref.shape[0])],
    }
    crit = []

    def c(name, val, ok, tolv):
        crit.append({"criterion": name, "value": val, "tolerance": tolv, "pass": bool(ok) if val is not None else None})

    if "rate_pearson_min" in tol:
        c("rate_pearson_active", out["rate_pearson_active"],
          (out["rate_pearson_active"] or -1) >= tol["rate_pearson_min"], tol["rate_pearson_min"])
    if "sugar_grn_mean_rate_rel_err_max" in tol:
        c("sugar_grn_mean_rate_rel_err", out["sugar_grn_mean_rate_rel_err"],
          (out["sugar_grn_mean_rate_rel_err"] if out["sugar_grn_mean_rate_rel_err"] is not None else 9)
          <= tol["sugar_grn_mean_rate_rel_err_max"], tol["sugar_grn_mean_rate_rel_err_max"])
    if "responder_jaccard_min" in tol:
        c("responder_jaccard_all", out["responder_jaccard_all"],
          (out["responder_jaccard_all"] or 0) >= tol["responder_jaccard_min"], tol["responder_jaccard_min"])
    if "top200_jaccard_min" in tol:
        c("top200_jaccard", out["top200_jaccard"], (out["top200_jaccard"] or 0) >= tol["top200_jaccard_min"],
          tol["top200_jaccard_min"])
    if "mn9_mean_rate" in tol:
        c("mn9_mean_rate", out["mn9_abs_diff"], out["mn9_abs_diff"] <= mn9_bound, tol["mn9_mean_rate"])
    if "n_active_neurons_rel_err_max" in tol:
        c("n_active_rel_err", out["n_active_rel_err"],
          (out["n_active_rel_err"] if out["n_active_rel_err"] is not None else 9)
          <= tol["n_active_neurons_rel_err_max"], tol["n_active_neurons_rel_err_max"])
    out["criteria"] = crit
    out["all_pass"] = all(x["pass"] for x in crit) if crit else None
    return out


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=json_default))
