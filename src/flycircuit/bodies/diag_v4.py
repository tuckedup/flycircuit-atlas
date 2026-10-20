"""PROPOSAL: NumPy-only open-loop diagnostic for SAVED body policies (no physics, no training).

For each recorded validation episode (npz with ``x`` (T,10) observations and ``y`` (T,2) teacher actions)
and each policy bundle, report teacher-action MSE with (a) no dropout and (b) exactly one input channel
zeroed, so that a channel the policy depends on (e.g. channel 5 = sin bearing, per the P1 audit) shows up
as a large error. Mirrors rollout semantics: the policy sees ``np.where(mask, 0, obs)``, state is reset
per episode, action = tanh(raw output).

NOT RUN YET: awaiting the owner's approval for any new computation. It performs a few NumPy forward
passes per (episode, channel) and should take well under 5 minutes for 14 policies x 72 episodes.
"""
from __future__ import annotations

import glob
import importlib.util
from pathlib import Path

import numpy as np

N_CH = 10


def _load_numpy_policy_class():
    p = Path(__file__).with_name("np_policy.py")
    spec = importlib.util.spec_from_file_location("np_policy_v4diag", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NumpyPolicy


def episode_mse(policy, x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    policy.reset()
    err = 0.0
    for t in range(len(x)):
        a = policy.act(np.where(mask, 0.0, x[t].astype(np.float64)))
        err += float(((a - y[t]) ** 2).sum())
    return err / max(len(x), 1)


def channel_report(policy_path: str | Path, recordings_glob: str) -> dict:
    cls = _load_numpy_policy_class()
    pol = cls.load(policy_path)
    files = sorted(glob.glob(recordings_glob))
    if not files:
        raise FileNotFoundError(f"no recordings match {recordings_glob}")
    none = np.zeros(N_CH, bool)
    base, by_ch, lens = [], [[] for _ in range(N_CH)], []
    for f in files:
        z = np.load(f)
        x, y = z["x"], z["y"]
        lens.append(len(x))
        base.append(episode_mse(pol, x, y, none))
        for c in range(N_CH):
            m = none.copy()
            m[c] = True
            by_ch[c].append(episode_mse(pol, x, y, m))
    return {"n_episodes": len(files), "mean_len": float(np.mean(lens)),
            "mse_no_dropout": float(np.mean(base)),
            "mse_drop_channel": [float(np.mean(v)) for v in by_ch],
            "excess_over_no_dropout": [float(np.mean(v) - np.mean(base)) for v in by_ch]}
