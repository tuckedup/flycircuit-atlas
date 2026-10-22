"""Build FlyGym rollout job files and aggregate their results (main env; no flygym import)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FLYGYM_PY = ROOT / "external/flygym_pilot/.venv/bin/python"
ROLLOUT = ROOT / "external/flygym_pilot/fly_rollout.py"
NP_POLICY = ROOT / "src/flycircuit/bodies/np_policy.py"


def episodes(split: str, idxs, policy: str, dropout: float = 0.0, action_noise: float = 0.0,
             record: bool = False, tag: str = ""):
    return [dict(key=f"{split}{i}_{tag or Path(policy).stem}_dr{dropout}_an{action_noise}", split=split, idx=int(i),
                 policy=policy, dropout=dropout, action_noise=action_noise, record=record) for i in idxs]


def write_jobs(path: Path, eps: list[dict], budget_ticks: int, radius: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(np_policy_py=str(NP_POLICY), budget_ticks=budget_ticks,
                                    success_radius_mm=radius, episodes=eps)))
    return path


def load_results(out_dir: Path) -> list[dict]:
    f = Path(out_dir) / "results.jsonl"
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()] if f.exists() else []


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    n = len(rows)
    return dict(n=n, n_failed_status=n - len(ok),
                success=float(np.mean([r["success"] for r in rows])) if n else None,
                spl=float(np.mean([r["spl"] for r in rows])) if n else None,
                final_dist_mm=float(np.mean([r["final_dist_mm"] for r in rows])) if n else None,
                time_to_success_ticks=float(np.mean([r["ticks_to_success"] for r in rows if r["success"]]))
                if any(r["success"] for r in rows) else None)
