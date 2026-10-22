"""Navigation arena v2 (``nav_tier1_v2``): a harder, *learnable* benchmark. ENGINEERED, not biomechanics.

Why v2 (limitations of v1, kept intact in ``arena.py``): the odor-only scripted controller solved 100 % of
nominal v1 episodes, conflicting cues never appeared in training, and nothing required memory.

What is new (physics, clocks, action space, reward and metrics are inherited unchanged from ``NavEnv``)
-----------------------------------------------------------------------------------------------------
* Two goal-related cues that can DISAGREE: the odor plume is centred on ``odor_src``, the visual beacon
  sits on ``beacon``. In a *conflict* episode one of them is on the goal G and the other on a decoy D.
* Which cue is truthful is a per-episode latent ``trusted`` in {odor, vision}, announced only by an
  11th observation channel ``ctx`` (+1 = odor trusted, -1 = vision trusted):
    - ``pulse``      ctx is shown for the first ``CTX_STEPS`` control steps, then 0   -> MEMORY demand
    - ``persistent`` ctx is shown for the whole episode                              -> no memory needed
    - ``hidden``     ctx is never shown                                              -> INTRINSICALLY
                     AMBIGUOUS in conflict episodes (the observation does not determine G; an
                     observation-only policy can do no better than ~chance between the two cues). These
                     conditions are reported separately and are never counted as model failures.
  Non-conflict episodes (both cues on G) show ctx = 0 and are solvable from either cue.
* The ctx channel is not subject to sensor dropout (it is a task instruction, not a sensor).
* Controlled reliability knobs: odor/vision noise multipliers, contrast, plume radius.
* Held-out perturbations (never in training): obstacle density 6-8 (train 3-5), plume radius 6 (train 10),
  contrast 0.5 (train {1, 0.75}), odor noise x8 / vision noise x6 (train {1, 2}).

Observation = 11 channels (``CHANNELS_V2``): the 10 v1 channels + ``ctx``.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

from flycircuit.envs.arena import (
    ANTENNA_FWD,
    ANTENNA_LAT,
    CHANNELS,
    ODOR_NOISE,
    PLUME_R,
    RAY_MAX,
    VISION_NOISE,
    DropoutSpec,
    Layout,
    NavEnv,
    _ray,
    make_layout,
    wrap,
)

CHANNELS_V2 = (*CHANNELS, "ctx")
N_OBS_V2 = len(CHANNELS_V2)
CTX_STEPS = 10  # control steps (= 1.0 s) during which the pulse is visible
TRAIN_OBSTACLES = (3, 5)
OOD_OBSTACLES = (6, 8)
OOD_PLUME_R = 6.0
SPLIT_V2 = ("train", "val", "test")


@dataclass(frozen=True)
class LayoutV2(Layout):
    trusted: str = "odor"  # which cue is truthful in a conflict episode (latent)

    def hash(self) -> str:  # geometry only (same definition as v1) so leakage checks are comparable
        return Layout.hash(self)


def make_layout_v2(index: int, split: str = "train",
                   n_obstacles: tuple[int, int] = TRAIN_OBSTACLES) -> LayoutV2:
    """Geometry from the v1 generator (disjoint seed ranges per split) + a balanced latent ``trusted``."""
    base = make_layout(index, split, n_obstacles=n_obstacles)
    trusted = "odor" if random.Random(f"trusted/{base.seed}").random() < 0.5 else "vision"
    return LayoutV2(base.seed, base.split, base.start, base.heading, base.goal, base.decoy,
                    base.obstacles, trusted)


@dataclass(frozen=True)
class EvalCondV2:
    """One evaluation/training condition. ``conflict``: cues disagree. ``ctx``: pulse|persistent|hidden."""

    name: str = "conflict_pulse"
    conflict: bool = True
    ctx: str = "pulse"
    contrast: float = 1.0
    odor_noise_mult: float = 1.0
    vision_noise_mult: float = 1.0
    plume_r: float = PLUME_R
    dropout: DropoutSpec = field(default_factory=DropoutSpec)

    def __post_init__(self):
        if self.ctx not in ("pulse", "persistent", "hidden"):
            raise ValueError("ctx must be pulse|persistent|hidden")


class NavEnvV2(NavEnv):
    """Single-episode v2 environment (subclass of the v1 env: same dynamics/metrics, new sensing)."""

    def __init__(self, layout: LayoutV2, condition: EvalCondV2 | None = None,
                 noise_seed: int = 0, mask_seed: int = 0, max_steps: int = 250):
        cond = condition or EvalCondV2()
        self.cond = cond
        trusted = layout.trusted
        if cond.conflict:
            self.odor_src = layout.goal if trusted == "odor" else layout.decoy
            beacon = layout.goal if trusted == "vision" else layout.decoy
        else:
            self.odor_src, beacon = layout.goal, layout.goal
        self._beacon_v2 = beacon
        super().__init__(layout, cond, noise_seed, mask_seed, max_steps)
        self.beacon = beacon  # NavEnv.__init__ set it from cond.conflict (v1 semantics); override

    def _odor(self, px, py) -> float:
        d = math.dist((px, py), self.odor_src)
        return max(0.0, 1.0 - d / self.cond.plume_r)

    def ctx_value(self) -> float:
        c = self.cond
        if not c.conflict or c.ctx == "hidden":
            return 0.0
        if c.ctx == "pulse" and self.t >= CTX_STEPS:
            return 0.0
        return 1.0 if self.layout.trusted == "odor" else -1.0

    def _observe(self) -> list[float]:
        c, s = math.cos(self.theta), math.sin(self.theta)
        n, cd = self._noise, self.cond
        on, vn = ODOR_NOISE * cd.odor_noise_mult, VISION_NOISE * cd.vision_noise_mult
        lx = self.x + ANTENNA_FWD * c - ANTENNA_LAT * s
        ly = self.y + ANTENNA_FWD * s + ANTENNA_LAT * c
        rx = self.x + ANTENNA_FWD * c + ANTENNA_LAT * s
        ry = self.y + ANTENNA_FWD * s - ANTENNA_LAT * c
        odor_l = 4 * (self._odor(lx, ly) - 0.5) + n.gauss(0, on)
        odor_r = 4 * (self._odor(rx, ry) - 0.5) + n.gauss(0, on)
        bx, by = self._beacon_v2
        bearing = wrap(math.atan2(by - self.y, bx - self.x) - self.theta)
        k = cd.contrast
        vis = [k * math.sin(bearing) + n.gauss(0, vn), k * math.cos(bearing) + n.gauss(0, vn)]
        for off in (math.pi / 4, 0.0, -math.pi / 4):
            d = _ray(self.x, self.y, self.theta + off, self.obstacles)
            vis.append(k * (1.0 - d / RAY_MAX) + n.gauss(0, vn))
        prop = [self.last_v / 1.5, self.last_w / 3.0, self.contact]
        obs = [odor_l, odor_r, *vis, *prop]
        if cd.dropout.per == "step":
            self._keep = cd.dropout.draw(self._maskrng)
        return [o if kp else 0.0 for o, kp in zip(obs, self._keep, strict=True)] + [self.ctx_value()]


def layout_set(split: str, lo: int, hi: int, n_obstacles=TRAIN_OBSTACLES) -> list[LayoutV2]:
    return [make_layout_v2(i, split, n_obstacles) for i in range(lo, hi)]


def geometry_hashes(lays) -> set[str]:
    return {lay.hash() for lay in lays}


def condition_hash(c: EvalCondV2) -> str:
    return hashlib.sha256(repr(c).encode()).hexdigest()[:12]
