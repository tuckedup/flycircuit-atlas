"""PROPOSAL: opt-in per-trial diagnostics written to a SIDECAR file, never into accepted records.

Intended hook in ``LearningAgent.trial`` (agent.py ~l.202-206), shown as a sketch only because frozen v3
source is not edited in place::

    gn = torch.nn.utils.clip_grad_norm_(self.params, 1.0)      # already returns the PRE-clip norm
    if self.diag is not None:
        self.diag.log(trial=self.n_trials, delta=float(delta), logp=float(logp[a]),
                      grad_norm=float(gn), clip_max=1.0, loss=float(-delta * logp[a]))

Nothing here touches scientific outputs; it only appends JSON lines.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


class TrialDiag:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, *, trial: int, delta: float, logp: float, grad_norm: float, clip_max: float,
            loss: float, extra: dict | None = None) -> None:
        row = {"trial": trial, "delta": delta, "logp": logp, "grad_norm_preclip": grad_norm,
               "clipped": bool(grad_norm > clip_max),
               "finite": all(math.isfinite(x) for x in (delta, logp, grad_norm, loss)),
               "loss": loss, **(extra or {})}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")

    @staticmethod
    def summarize(path: str | Path) -> dict:
        rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
        if not rows:
            return {"n": 0}
        n = len(rows)
        gn = sorted(r["grad_norm_preclip"] for r in rows if r["finite"])
        return {"n": n, "nonfinite": sum(not r["finite"] for r in rows),
                "clip_fraction": sum(r["clipped"] for r in rows) / n,
                "grad_norm_median": gn[len(gn) // 2] if gn else None,
                "grad_norm_p95": gn[int(0.95 * (len(gn) - 1))] if gn else None}
