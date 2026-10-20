"""Tier-1 navigation arena (lightweight 2-D, pure Python; ENGINEERED benchmark, not biomechanics).

Clocks
------
* physics clock : ``PHYSICS_DT`` = 0.05 s (unicycle kinematics + collision resolution)
* control/sensor/neural clock : ``CONTROL_DT`` = 0.1 s = 2 physics substeps. One neural-model
  step per control step; the action is held constant over the substeps.
* episode: at most ``MAX_STEPS`` control steps (25 s).

Observation (10 channels, fixed order = ``CHANNELS``)
-----------------------------------------------------
modality ``odor`` (2): left/right antenna intensity, linear plume ramp I = max(0, 1 - d/PLUME_R),
    reported as 4*(I - 0.5) plus Gaussian noise. The goal emits the odor.
modality ``vision`` (5): sin/cos of the bearing (relative to heading) of the visual BEACON, three
    range-finder rays (-45/0/+45 deg, near=1). All vision channels are scaled by ``contrast``.
    In the nominal condition the beacon sits on the goal; in the ``conflict`` condition it is
    moved to a decoy location, so vision and odor disagree (success still means reaching the goal).
modality ``proprio`` (3): previous forward speed / V_MAX, previous turn rate / W_MAX, contact flag.

Action: ``(a0, a1)`` in [-1, 1]. forward speed = V_MAX*(a0+1)/2, turn rate = W_MAX*a1 (CCW +).
Reward : 1.0*(d_prev - d_now) - 0.01 - 0.05*collision + 5*success.
Termination: success (within GOAL_R of the goal at any physics substep) => ``terminated``;
    MAX_STEPS reached => ``truncated``. There is no failure termination.
Reset: a new layout per episode, random start heading, neural state is reset by the caller.

Randomness uses independent streams (layout, sensor noise, dropout mask) so paired comparisons
across controllers see identical layouts, noise and masks.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

ARENA = 8.0
AGENT_R = 0.1
GOAL_R = 0.5
V_MAX = 1.5
W_MAX = 3.0
PHYSICS_DT = 0.05
SUBSTEPS = 2
CONTROL_DT = PHYSICS_DT * SUBSTEPS
MAX_STEPS = 250
PLUME_R = 10.0
ANTENNA_FWD = 0.3
ANTENNA_LAT = 0.5
RAY_MAX = 2.0
ODOR_NOISE = 0.02
VISION_NOISE = 0.05

CHANNELS = (
    "odor_L", "odor_R",
    "vis_beacon_sin", "vis_beacon_cos", "vis_ray_L", "vis_ray_C", "vis_ray_R",
    "prop_v", "prop_w", "prop_contact",
)
MODALITIES = {"odor": (0, 1), "vision": (2, 3, 4, 5, 6), "proprio": (7, 8, 9)}
N_OBS = len(CHANNELS)
N_ACT = 2

# Layout seed ranges for each split: disjoint by construction and re-checked via layout hashes.
SPLIT_OFFSET = {"train": 0, "val": 1_000_000, "test": 2_000_000}


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Layout:
    seed: int
    split: str
    start: tuple[float, float]
    heading: float
    goal: tuple[float, float]
    decoy: tuple[float, float]
    obstacles: tuple[tuple[float, float, float], ...]  # (x, y, r)

    def hash(self) -> str:
        """Geometry hash (excludes seed/split) so leakage across splits can be detected."""
        parts = [self.start, self.heading, self.goal, self.decoy, self.obstacles]
        return hashlib.sha256(repr([_round(p) for p in parts]).encode()).hexdigest()[:16]


def _round(x):
    if isinstance(x, float):
        return round(x, 6)
    if isinstance(x, (tuple, list)):
        return tuple(_round(i) for i in x)
    return x


def make_layout(index: int, split: str = "train", n_obstacles: tuple[int, int] = (3, 5)) -> Layout:
    if split not in SPLIT_OFFSET:
        raise ValueError(f"split must be one of {sorted(SPLIT_OFFSET)}")
    if not 0 <= index < 1_000_000:
        raise ValueError("layout index must be in [0, 1e6)")
    seed = SPLIT_OFFSET[split] + index
    rng = random.Random(f"layout/{seed}")
    m = 0.8
    for _ in range(1000):
        start = (rng.uniform(m, ARENA - m), rng.uniform(m, ARENA - m))
        goal = (rng.uniform(m, ARENA - m), rng.uniform(m, ARENA - m))
        if math.dist(start, goal) >= 4.0:
            break
    else:  # pragma: no cover - cannot happen in an 8x8 arena
        raise RuntimeError("could not place start/goal")
    obs: list[tuple[float, float, float]] = []
    k = rng.randint(*n_obstacles)
    tries = 0
    while len(obs) < k and tries < 2000:
        tries += 1
        r = rng.uniform(0.3, 0.7)
        c = (rng.uniform(r + 0.2, ARENA - r - 0.2), rng.uniform(r + 0.2, ARENA - r - 0.2))
        if math.dist(c, start) < r + 0.8 or math.dist(c, goal) < r + GOAL_R + 0.5:
            continue
        obs.append((c[0], c[1], r))
    for _ in range(1000):
        decoy = (rng.uniform(m, ARENA - m), rng.uniform(m, ARENA - m))
        if math.dist(decoy, goal) >= 3.0 and math.dist(decoy, start) >= 1.5:
            break
    heading = rng.uniform(-math.pi, math.pi)
    return Layout(seed, split, start, heading, goal, decoy, tuple(obs))


@dataclass(frozen=True)
class DropoutSpec:
    """Sensor dropout. ``kind='channel'``: every one of the 10 channels is dropped independently
    with probability ``level``; ``kind='modality'``: each of odor/vision/proprio is dropped as a
    unit with probability ``level``. ``per='episode'``: one mask drawn at reset and held; ``per=
    'step'``: redrawn every control step. Dropped channels are set to 0 (no mask indicator is
    given to the controller)."""

    level: float = 0.0
    kind: str = "channel"
    per: str = "episode"

    def __post_init__(self):
        if not 0.0 <= self.level <= 1.0:
            raise ValueError("dropout level must be in [0, 1]")
        if self.kind not in ("channel", "modality") or self.per not in ("episode", "step"):
            raise ValueError("kind in {channel, modality}; per in {episode, step}")

    def draw(self, rng: random.Random) -> list[bool]:
        """Return keep-mask (True = kept) over the 10 channels."""
        if self.kind == "channel":
            return [rng.random() >= self.level for _ in range(N_OBS)]
        keep = [True] * N_OBS
        for idx in MODALITIES.values():
            if rng.random() < self.level:
                for i in idx:
                    keep[i] = False
        return keep


@dataclass(frozen=True)
class EvalCondition:
    name: str = "nominal"
    contrast: float = 1.0
    conflict: bool = False
    dropout: DropoutSpec = field(default_factory=DropoutSpec)


def _ray(px, py, ang, obstacles) -> float:
    """Distance to the nearest obstacle/wall along a ray, capped at RAY_MAX."""
    dx, dy = math.cos(ang), math.sin(ang)
    best = RAY_MAX
    for cx, cy, r in obstacles:
        fx, fy = px - cx, py - cy
        b = fx * dx + fy * dy
        c = fx * fx + fy * fy - r * r
        disc = b * b - c
        if disc >= 0:
            t = -b - math.sqrt(disc)
            if 0 <= t < best:
                best = t
    for t in (
        (0 - px) / dx if dx < -1e-12 else None,
        (ARENA - px) / dx if dx > 1e-12 else None,
        (0 - py) / dy if dy < -1e-12 else None,
        (ARENA - py) / dy if dy > 1e-12 else None,
    ):
        if t is not None and 0 <= t < best:
            best = t
    return best


class NavEnv:
    """Single-episode environment. ``state`` (x, y, theta) is privileged: neural controllers must
    only see the returned observation."""

    def __init__(self, layout: Layout, condition: EvalCondition | None = None,
                 noise_seed: int = 0, mask_seed: int = 0, max_steps: int = MAX_STEPS):
        self.layout = layout
        self.cond = condition or EvalCondition()
        self.max_steps = max_steps
        self._noise = random.Random(f"noise/{noise_seed}/{layout.seed}")
        self._maskrng = random.Random(f"mask/{mask_seed}/{layout.seed}")
        self.obstacles = layout.obstacles
        self.goal = layout.goal
        self.beacon = layout.decoy if self.cond.conflict else layout.goal
        self.reset()

    # ---- episode ------------------------------------------------------
    def reset(self) -> list[float]:
        self.x, self.y = self.layout.start
        self.theta = self.layout.heading
        self.t = 0
        self.last_v = 0.0
        self.last_w = 0.0
        self.contact = 0.0
        self.path_len = 0.0
        self.collision_steps = 0
        self.success = False
        self.done = False
        self._keep = self.cond.dropout.draw(self._maskrng)
        self.d_start = math.dist((self.x, self.y), self.goal)
        self.dist_trace = [self.d_start]
        self.collision_trace = [0]
        self.heading_err_trace: list[float] = []
        return self._observe()

    # ---- sensing ------------------------------------------------------
    def _odor(self, px, py) -> float:
        d = math.dist((px, py), self.goal)
        return max(0.0, 1.0 - d / PLUME_R)

    def _observe(self) -> list[float]:
        c, s = math.cos(self.theta), math.sin(self.theta)
        n = self._noise
        lx = self.x + ANTENNA_FWD * c - ANTENNA_LAT * s
        ly = self.y + ANTENNA_FWD * s + ANTENNA_LAT * c
        rx = self.x + ANTENNA_FWD * c + ANTENNA_LAT * s
        ry = self.y + ANTENNA_FWD * s - ANTENNA_LAT * c
        odor_l = 4 * (self._odor(lx, ly) - 0.5) + n.gauss(0, ODOR_NOISE)
        odor_r = 4 * (self._odor(rx, ry) - 0.5) + n.gauss(0, ODOR_NOISE)
        bearing = wrap(math.atan2(self.beacon[1] - self.y, self.beacon[0] - self.x) - self.theta)
        k = self.cond.contrast
        vis = [
            k * math.sin(bearing) + n.gauss(0, VISION_NOISE),
            k * math.cos(bearing) + n.gauss(0, VISION_NOISE),
        ]
        for off in (math.pi / 4, 0.0, -math.pi / 4):  # left, centre, right (CCW positive)
            d = _ray(self.x, self.y, self.theta + off, self.obstacles)
            vis.append(k * (1.0 - d / RAY_MAX) + n.gauss(0, VISION_NOISE))
        prop = [self.last_v / V_MAX, self.last_w / W_MAX, self.contact]
        obs = [odor_l, odor_r, *vis, *prop]
        if self.cond.dropout.per == "step":
            self._keep = self.cond.dropout.draw(self._maskrng)
        return [o if kp else 0.0 for o, kp in zip(obs, self._keep, strict=True)]

    # ---- dynamics -----------------------------------------------------
    def step(self, action) -> tuple[list[float], float, bool, bool, dict]:
        if self.done:
            raise RuntimeError("step() after episode end; call reset()")
        a0 = min(1.0, max(-1.0, float(action[0])))
        a1 = min(1.0, max(-1.0, float(action[1])))
        v = V_MAX * (a0 + 1.0) / 2.0
        w = W_MAX * a1
        d_prev = math.dist((self.x, self.y), self.goal)
        collided = False
        for _ in range(SUBSTEPS):
            self.theta = wrap(self.theta + w * PHYSICS_DT)
            nx = self.x + v * math.cos(self.theta) * PHYSICS_DT
            ny = self.y + v * math.sin(self.theta) * PHYSICS_DT
            hit = False
            if not (AGENT_R <= nx <= ARENA - AGENT_R and AGENT_R <= ny <= ARENA - AGENT_R):
                nx = min(max(nx, AGENT_R), ARENA - AGENT_R)
                ny = min(max(ny, AGENT_R), ARENA - AGENT_R)
                hit = True
            for cx, cy, r in self.obstacles:
                if math.hypot(nx - cx, ny - cy) < r + AGENT_R:
                    hit = True
                    nx, ny = self.x, self.y  # reject the move
                    break
            if hit:
                collided = True
            self.path_len += math.dist((self.x, self.y), (nx, ny))
            self.x, self.y = nx, ny
            if math.dist((self.x, self.y), self.goal) < GOAL_R:
                self.success = True
                break
        self.t += 1
        self.last_v, self.last_w = v, w
        self.contact = 1.0 if collided else 0.0
        self.collision_steps += int(collided)
        d_now = math.dist((self.x, self.y), self.goal)
        self.dist_trace.append(d_now)
        self.collision_trace.append(int(collided))
        self.heading_err_trace.append(abs(wrap(
            math.atan2(self.goal[1] - self.y, self.goal[0] - self.x) - self.theta)))
        reward = (d_prev - d_now) - 0.01 - 0.05 * float(collided) + (5.0 if self.success else 0.0)
        terminated = self.success
        truncated = (not terminated) and self.t >= self.max_steps
        self.done = terminated or truncated
        info = {"success": self.success, "collision": collided, "dist": d_now}
        return self._observe(), reward, terminated, truncated, info

    @property
    def state(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.theta)

    # ---- metrics ------------------------------------------------------
    def episode_metrics(self) -> dict:
        """Per-episode metrics (units: steps/seconds/arena units).

        spl            success * L_opt / max(L_opt, path)   L_opt = start-goal Euclidean distance
                       minus GOAL_R (a lower bound; obstacle detours are not subtracted)
        heading_err_deg mean |heading - bearing to goal| over steps, degrees
        collision_rate fraction of control steps with contact
        recovery_s     per collision event (contiguous contact run): time from event start until
                       the goal distance is >= 0.2 smaller than at event start; unrecovered
                       events are counted in ``unrecovered`` and excluded from the mean.
        """
        l_opt = max(1e-9, self.d_start - GOAL_R)
        spl = (l_opt / max(l_opt, self.path_len)) if self.success else 0.0
        steps = max(1, self.t)
        rec, unrec = [], 0
        i, tr, ct = 1, self.dist_trace, self.collision_trace
        n = len(ct)
        while i < n:
            if ct[i] and not ct[i - 1]:
                d0, j = tr[i], i
                while j < n and tr[j] > d0 - 0.2:
                    j += 1
                if j < n:
                    rec.append((j - i) * CONTROL_DT)
                else:
                    unrec += 1
                while i < n and ct[i]:
                    i += 1
            else:
                i += 1
        he = self.heading_err_trace
        return {
            "success": bool(self.success), "steps": self.t, "time_s": self.t * CONTROL_DT,
            "path_len": self.path_len, "spl": spl,
            "collision_steps": self.collision_steps,
            "collision_rate": self.collision_steps / steps,
            "collided_any": self.collision_steps > 0,
            "heading_err_deg": math.degrees(sum(he) / len(he)) if he else float("nan"),
            "recovery_s": rec, "unrecovered": unrec,
        }
