"""D2 rank-4 micro-profile of body training (CPU threads 2/4/8 and CUDA). NO physics. Writes only to
diag_out/d2/. Usage (guarded, via heavyjob.sh):
  FLYCIRCUIT_HOME=<CANONICAL_ROOT> PYTHONPATH=<copy>/src <venv>/python scripts_v4/profile_train.py
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ORIG = Path("<CANONICAL_ROOT>")
OUT = Path(__file__).resolve().parents[1] / "diag_out" / "d2"
OUT.mkdir(parents=True, exist_ok=True)

import flycircuit  # noqa: E402

assert "flycircuit_debug_copy" in flycircuit.__file__, flycircuit.__file__
from flycircuit.bodies import train_v4 as T  # noqa: E402
from flycircuit.bodies.export import export_policy  # noqa: E402
from flycircuit.bodies.stages_v3 import build_model_v3  # noqa: E402

tcfg = yaml.safe_load((ORIG / "configs/experiments/flygym_v3_protocol.yaml").read_text())["training"]
files = sorted(glob.glob(str(ORIG / "results/flygym_v3/bc_dart/rec_dart*.npz")))
if not files:
    files = sorted(glob.glob(str(ORIG / "results/flygym_v3/bc_dart/rec_*.npz")))
pick = [files[i] for i in np.linspace(0, len(files) - 1, 20).astype(int)]
T_MAX = 180
X = np.zeros((T_MAX, 20, 10), np.float32)
Y = np.zeros((T_MAX, 20, 2), np.float32)
M = np.zeros((T_MAX, 20), np.float32)
lens = []
for i, f in enumerate(pick):
    z = np.load(f)
    n = len(z["x"])
    X[:n, i], Y[:n, i], M[:n, i] = z["x"], z["y"], 1.0
    lens.append(n)
fix = {"files": [Path(f).name for f in pick], "mean_len": float(np.mean(lens)), "T": T_MAX,
       "fixture_padded_fraction": float(1 - np.mean(lens) / T_MAX)}
print("fixture", fix, flush=True)

import torch  # noqa: E402

res = {"fixture": fix, "tcfg": {k: tcfg[k] for k in ("batch_sequences", "tbptt_steps", "grad_clip",
                                                       "dropout_levels_in_training")},
       "runs": [], "cuda_available": bool(torch.cuda.is_available())}
LR = {"connectome_pilot_mb": 1e-3, "gru_param_matched": 3e-3}
configs = [("cpu", 2), ("cpu", 4), ("cpu", 8)] + ([("cuda", 2)] if torch.cuda.is_available() else [])
for cond in ("connectome_pilot_mb", "gru_param_matched"):
    for device, th in configs:
        model, _ = build_model_v3(cond, 7, 0)
        r = T.train_instrumented(model, (X, Y, M), LR[cond], epochs=4, seed=7, tcfg=tcfg, threads=th,
                                 device=device, warmup_epochs=1)
        r.update(condition=cond)
        # extrapolate: full body set is 600 sequences per epoch (300 teacher + 300 DART), batch 16
        r["extrapolated_full_epoch_s"] = r["wall_s_per_epoch"] * (600 / 20)
        r["extrapolated_40_epoch_model_s"] = r["extrapolated_full_epoch_s"] * 40
        if device == "cuda":
            r["peak_cuda_mib"] = torch.cuda.max_memory_allocated() / 2**20
            # export parity: torch(cuda) forward vs numpy policy on the first fixture sequence
            p = export_policy(model, OUT / f"parity_{cond}.npz")
            sys.path.insert(0, str(Path(flycircuit.__file__).parent / "bodies"))
            import np_policy
            pol = np_policy.NumpyPolicy.load(p)
            pol.reset()
            n0 = lens[0]
            xs = torch.as_tensor(X[:n0, 0:1]).to("cuda")
            with torch.no_grad():
                ref = torch.tanh(model(xs, None)[0])[:, 0].cpu().numpy()
            got = np.stack([pol.act(X[t, 0].astype(np.float64)) for t in range(n0)])
            r["export_parity_max_abs_diff"] = float(np.abs(ref - got).max())
        res["runs"].append(r)
        print(cond, device, th, {k: (round(v, 3) if isinstance(v, float) else v)
                                 for k, v in r.items() if k in ("wall_s_per_epoch", "extrapolated_40_epoch_model_s",
                                                                "padded_tick_fraction_realized",
                                                                "export_parity_max_abs_diff")}, flush=True)
        (OUT / "profile.json").write_text(json.dumps(res, indent=2))
        if device == "cuda":
            torch.cuda.empty_cache()
print("done", time.strftime("%H:%M:%S"))
