"""End-to-end synthetic path: validated graph -> query -> dynamics -> manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from flycircuit.config import project_root
from flycircuit.graph import SYNTHETIC_BANNER, make_synthetic_graph
from flycircuit.sim import simulate_lif
from flycircuit.tracking import RunRecorder


def run_synthetic(seed: int = 0, steps: int = 500, use_mlflow: bool = True,
                  root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    cfg = {"graph": {"n": 40, "density": 0.08, "seed": 0}, "sim": {"steps": steps}, "seed": seed}
    g = make_synthetic_graph(**cfg["graph"])
    with RunRecorder(
        "synthetic", cfg, [seed], "discrete LIF (engineering toy, not the Shiu reference)",
        f"fixed {steps} steps", data_release=f"{g.dataset}:{g.version}",
        data_hashes={"edge_hash": g.edge_hash()}, mapping_hash=g.mapping_hash(),
        root=root, use_mlflow=use_mlflow, label=SYNTHETIC_BANNER,
    ) as rec:
        g.save(root / "data" / "processed" / "synthetic")
        stim = np.flatnonzero(g.node_masks["sensory_input"])
        out = simulate_lif(g, stim, steps=steps, seed=seed)
        res = {
            "banner": SYNTHETIC_BANNER, "graph": g.summary(),
            "mean_rate_hz": float(out["rates_hz"].mean()),
            "n_active": int((out["spike_counts"] > 0).sum()),
            "stim_mean_rate_hz": float(out["rates_hz"][stim].mean()),
            "params": out["params"],
        }
        rec.metrics.update({k: res[k] for k in ("mean_rate_hz", "n_active", "stim_mean_rate_hz")})
        art = rec.dir / "result.json"
        art.write_text(json.dumps(res, indent=2), encoding="utf-8")
        rec.add_artifact(art)
        np.save(rec.dir / "spike_counts.npy", out["spike_counts"])
        rec.add_artifact(rec.dir / "spike_counts.npy")
    return {"run_id": rec.run_id, "dir": str(rec.dir), "manifest": rec.manifest, "result": res}
