"""Observation-only scripted reference controllers for navigation v2 (no learning, no privileged state).

These establish LEARNABILITY: ``ctx_memory`` is an oracle-from-observation upper bound that uses exactly
the 11 supplied channels (it must remember the ctx pulse). ``odor_only`` / ``vision_only`` /
``ctx_nomemory`` show that the benchmark is NOT trivially solved by a single cue or without memory.
All are local scripted code (action provenance records source='scripted').
"""

from __future__ import annotations

import math

import numpy as np

from flycircuit.envs.arena import N_ACT, V_MAX, W_MAX, wrap
from flycircuit.envs.controllers import BatchController, _to_action


def _avoid(o) -> tuple[bool, float]:
    rl, rc, rr = o[4], o[5], o[6]
    if rc > 0.35 or rl > 0.5 or rr > 0.5:
        return True, (W_MAX * (1.0 if rl < rr else -1.0))
    return False, 0.0


class _Scripted(BatchController):
    source = "scripted"

    def _follow_odor(self, o):
        return 6.0 * (o[0] - o[1]) * W_MAX / 4.0 * 2.0, V_MAX * 0.9  # same gain as v1 SensorScripted

    def _follow_vision(self, o):
        err = wrap(math.atan2(o[2], o[3]))  # bearing of the beacon; scale (contrast) invariant
        return 4.0 * err, V_MAX * max(0.0, math.cos(err)) ** 2 * 0.9 + 0.1 * V_MAX

    def _decide(self, i: int, o) -> tuple[float, float]:  # pragma: no cover - interface
        raise NotImplementedError

    def _act(self, obs, envs):
        out = np.zeros((obs.shape[0], N_ACT), dtype=np.float32)
        for i, o in enumerate(obs):
            w, v = self._decide(i, o)
            hit, wa = _avoid(o)
            if hit:
                w, v = wa, V_MAX * 0.3
            out[i] = _to_action(v, w)
        return out


class OdorOnly(_Scripted):
    kind = "odor_only"

    def _decide(self, i, o):
        return self._follow_odor(o)


class VisionOnly(_Scripted):
    kind = "vision_only"

    def _decide(self, i, o):
        return self._follow_vision(o)


class CtxNoMemory(_Scripted):
    """Uses the CURRENT ctx value only; falls back to odor when ctx is absent (after a pulse)."""

    kind = "ctx_nomemory"

    def _decide(self, i, o):
        ctx = o[10]
        return self._follow_vision(o) if ctx < -0.5 else self._follow_odor(o)


class CtxMemory(_Scripted):
    """Oracle-from-observation: latches the first non-zero ctx and follows the trusted cue forever."""

    kind = "ctx_memory"

    def reset(self, n):
        super().reset(n)
        self.latched = np.zeros(n)

    def _decide(self, i, o):
        if self.latched[i] == 0.0 and abs(o[10]) > 0.5:
            self.latched[i] = float(np.sign(o[10]))
        return self._follow_vision(o) if self.latched[i] < 0 else self._follow_odor(o)


SCRIPTED = {c.kind: c for c in (OdorOnly, VisionOnly, CtxNoMemory, CtxMemory)}
