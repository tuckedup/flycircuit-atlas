"""Swappable batch controllers for NavEnv plus the lockstep episode runner.

Interface (``BatchController``): ``reset(n)`` then ``act(obs[B,10], envs) -> actions[B,2]`` in
[-1,1]. Neural controllers must NOT use ``envs`` (privileged state); the runner passes ``None``
to anything with ``privileged = False``. Every controller records an action-provenance log
(source, kind, number of calls and a rolling hash of actions) so a run can show that actions came
from local model forward passes or local scripted code and never from an external service.
No network / LLM client is imported anywhere in this package.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import numpy as np

from flycircuit.envs.arena import (
    CONTROL_DT,
    N_ACT,
    N_OBS,
    V_MAX,
    W_MAX,
    EvalCondition,
    Layout,
    NavEnv,
    wrap,
)


class ActionProvenance:
    def __init__(self, kind: str, source: str):
        self.kind, self.source = kind, source
        self.calls = 0
        self.n_actions = 0
        self._h = hashlib.sha256()

    def record(self, actions: np.ndarray) -> None:
        self.calls += 1
        self.n_actions += int(actions.shape[0])
        self._h.update(np.ascontiguousarray(actions, dtype=np.float32).tobytes())

    def summary(self) -> dict:
        return {"controller": self.kind, "action_source": self.source, "calls": self.calls,
                "actions": self.n_actions, "action_hash": self._h.hexdigest()[:16],
                "external_service": False}


class BatchController:
    privileged = False
    kind = "base"
    source = "local"

    def reset(self, n: int) -> None:
        self.prov = ActionProvenance(self.kind, self.source)

    def _act(self, obs: np.ndarray, envs) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def act(self, obs: np.ndarray, envs=None) -> np.ndarray:
        a = np.clip(self._act(obs, envs if self.privileged else None), -1.0, 1.0)
        self.prov.record(a)
        return a


def _to_action(v: float, w: float) -> tuple[float, float]:
    return (2.0 * min(1.0, max(0.0, v / V_MAX)) - 1.0, max(-1.0, min(1.0, w / W_MAX)))


class PrivilegedTeacher(BatchController):
    """Scripted potential-field expert with ground-truth goal/obstacle access (imitation teacher).
    Teacher access is privileged by design and is never exposed to the students' observations."""

    privileged = True
    kind = "privileged_teacher"
    source = "scripted"

    def reset(self, n):
        super().reset(n)
        self._sides: list[dict] = [{} for _ in range(n)]  # chosen passing side per obstacle

    def _act(self, obs, envs):
        out = np.zeros((len(envs), N_ACT), dtype=np.float32)
        for i, e in enumerate(envs):
            if e.done:
                continue
            x, y, th = e.state
            gx, gy = e.goal[0] - x, e.goal[1] - y
            dg = math.hypot(gx, gy) + 1e-9
            vx, vy = gx / dg, gy / dg
            # steer around the nearest obstacle that blocks the straight line to the goal
            sides = self._sides[i]
            best = None
            for k, (cx, cy, r) in enumerate(e.obstacles):
                px, py = cx - x, cy - y
                tproj = (px * vx + py * vy)
                if tproj <= 0 or tproj > dg:
                    continue
                perp = px * -vy + py * vx  # signed lateral offset of centre (left +)
                if abs(perp) < r + 0.45 and (best is None or tproj < best[0]):
                    best = (tproj, k, cx, cy, r, perp)
            if best is not None:
                _, k, cx, cy, r, perp = best
                if k not in sides:
                    sides[k] = -1.0 if perp >= 0 else 1.0  # pass on the side opposite the centre
                off = r + 0.55
                ax = cx + sides[k] * -vy * off
                ay = cy + sides[k] * vx * off
                tx, ty = ax - x, ay - y
                tn = math.hypot(tx, ty) + 1e-9
                vx, vy = tx / tn, ty / tn
            for cx, cy, r in e.obstacles:
                ox, oy = x - cx, y - cy
                d = math.hypot(ox, oy)
                clear = d - r
                if clear < 0.4:
                    s = (1.0 - max(clear, 0.0) / 0.4) * 2.0
                    vx += s * ox / (d + 1e-9)
                    vy += s * oy / (d + 1e-9)
            m = 0.5
            if x < m:
                vx += (m - x) * 3
            if x > 8.0 - m:
                vx -= (x - (8.0 - m)) * 3
            if y < m:
                vy += (m - y) * 3
            if y > 8.0 - m:
                vy -= (y - (8.0 - m)) * 3
            err = wrap(math.atan2(vy, vx) - th)
            w = 4.0 * err
            v = V_MAX * max(0.0, math.cos(err)) ** 2
            out[i] = _to_action(v, w)
        return out


class SensorScripted(BatchController):
    """Scripted controller that uses ONLY the observation (odor L/R gradient steering + range
    finders). A hand-written non-learning reference; not a connectome result."""

    kind = "sensor_scripted"
    source = "scripted"

    def _act(self, obs, envs):
        out = np.zeros((obs.shape[0], N_ACT), dtype=np.float32)
        for i, o in enumerate(obs):
            ol, orr = o[0], o[1]
            rl, rc, rr = o[4], o[5], o[6]
            w = 6.0 * (ol - orr) * W_MAX / 4.0 * 2.0
            v = V_MAX * 0.9
            if rc > 0.35 or rl > 0.5 or rr > 0.5:
                w = W_MAX * (1.0 if rl < rr else -1.0)
                v = V_MAX * 0.3
            out[i] = _to_action(v, w)
        return out


class RandomWalk(BatchController):
    """Sanity-floor control: random bounded actions from a seeded numpy stream."""

    kind = "random_action"
    source = "scripted"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def reset(self, n):
        super().reset(n)
        self.rng = np.random.default_rng(self.seed)

    def _act(self, obs, envs):
        return self.rng.uniform(-1, 1, size=(obs.shape[0], N_ACT)).astype(np.float32)


class NeuralController(BatchController):
    """Wraps any model with ``forward(x_seq[T,B,n_in], state) -> (y[T,B,n_out], state)`` (sparse
    connectome RNN, random sparse RNN, GRU, encoder/decoder). Action = tanh(y)."""

    kind = "neural"
    source = "model_forward"

    def __init__(self, model, device: str = "cpu", name: str | None = None):
        self.model, self.device = model, device
        self.kind = name or getattr(model, "kind", "neural")

    def reset(self, n):
        super().reset(n)
        import torch

        self.torch = torch
        self.state = self.model.init_state(n, self.device)

    def _act(self, obs, envs):
        torch = self.torch
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            y, self.state = self.model(x, self.state)
            return torch.tanh(y[0]).cpu().numpy()


def run_episodes(controller: BatchController, layouts: Sequence[Layout], cond: EvalCondition,
                 noise_seed: int = 0, mask_seed: int = 0, student_for_dagger=None,
                 beta: float = 1.0, record: bool = False, teacher: PrivilegedTeacher | None = None,
                 rng_mix: np.random.Generator | None = None, env_cls=NavEnv):
    """Run all episodes in lockstep. Returns (metrics list, provenance[, recorded sequences]).

    DAgger-style collection (record=True): the *executed* action is the teacher's with
    probability ``beta`` per episode-step, else the controller's; labels are always the teacher's.
    Observations stored are exactly what the controller sees (after dropout).
    """
    envs = [env_cls(lay, cond, noise_seed, mask_seed) for lay in layouts]
    n = len(envs)
    controller.reset(n)
    obs = np.array([e.reset() for e in envs], dtype=np.float32)
    seqs_x, seqs_y, seqs_m = [], [], []
    if record:
        teacher = teacher or PrivilegedTeacher()
        teacher.reset(n)
        rng_mix = rng_mix or np.random.default_rng(0)
    t = 0
    while not all(e.done for e in envs):
        a = controller.act(obs, envs)
        if record:
            lab = teacher.act(obs, envs)
            use_t = rng_mix.random(n) < beta
            exec_a = np.where(use_t[:, None], lab, a)
            seqs_x.append(obs.copy())
            seqs_y.append(lab)
            seqs_m.append(np.array([not e.done for e in envs], dtype=np.float32))
        else:
            exec_a = a
        for i, e in enumerate(envs):
            if not e.done:
                o, *_ = e.step(exec_a[i])
                obs[i] = o
        t += 1
    metrics = [e.episode_metrics() for e in envs]
    prov = controller.prov.summary()
    if record:
        return metrics, prov, (np.stack(seqs_x), np.stack(seqs_y), np.stack(seqs_m))
    return metrics, prov


def aggregate(metrics: list[dict]) -> dict:
    """Episode-level aggregates (episodes of ONE model are not independent replicates)."""
    n = len(metrics)
    rec = [r for m in metrics for r in m["recovery_s"]]
    succ = [m for m in metrics if m["success"]]
    return {
        "episodes": n,
        "success_rate": sum(m["success"] for m in metrics) / n,
        "spl": float(np.mean([m["spl"] for m in metrics])),
        "path_efficiency_on_success": (
            float(np.mean([m["spl"] for m in succ])) if succ else None),
        "collision_rate_per_step": float(np.mean([m["collision_rate"] for m in metrics])),
        "episodes_with_collision": float(np.mean([m["collided_any"] for m in metrics])),
        "heading_error_deg": float(np.nanmean([m["heading_err_deg"] for m in metrics])),
        "recovery_time_s_mean": float(np.mean(rec)) if rec else None,
        "recovery_events": len(rec),
        "unrecovered_events": int(sum(m["unrecovered"] for m in metrics)),
        "mean_time_s": float(np.mean([m["time_s"] for m in metrics])),
    }


__all__ = [
    "BatchController", "NeuralController", "PrivilegedTeacher", "RandomWalk",
    "SensorScripted", "aggregate", "run_episodes", "N_OBS", "CONTROL_DT",
]
