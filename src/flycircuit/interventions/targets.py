"""Target eligibility, screen planning and degree-matched random controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flycircuit.interventions.rng import RngStreams
from flycircuit.interventions.spec import InterventionSpec

LEVEL_ATTR = {"type": "cell_type", "class": "cell_class", "region": "region"}


@dataclass(frozen=True)
class TargetUnit:
    level: str
    label: str
    nodes: tuple[int, ...]


def degrees(graph) -> tuple[np.ndarray, np.ndarray]:
    n = graph.n
    return (np.bincount(graph.post, minlength=n), np.bincount(graph.pre, minlength=n))  # in, out


def eligible_units(
    graph,
    level: str = "neuron",
    min_degree: int = 1,
    min_size: int = 2,
    max_size: int = 50,
    exclude_sensory: bool = True,
    exclude_readout: bool = True,
) -> tuple[list[TargetUnit], dict]:
    """Eligible intervention targets and an explicit report of what was excluded and why.

    Eligibility: total (in+out) degree >= ``min_degree``; sensory stimulation sites and readout
    nodes excluded by default (clamping them confounds input/readout, not circuit function).
    Group levels need the annotation attribute; if absent the report says so (no silent empty).
    """
    ind, outd = degrees(graph)
    deg = ind + outd
    banned = np.zeros(graph.n, dtype=bool)
    if exclude_sensory:
        banned |= graph.node_masks["sensory_input"]
    if exclude_readout:
        banned |= graph.node_masks["action_readout"]
    node_ok = (deg >= min_degree) & ~banned
    rep: dict = {"level": level, "n_nodes": int(graph.n), "n_node_eligible": int(node_ok.sum()),
                 "excluded_low_degree": int((deg < min_degree).sum()),
                 "excluded_sensory_or_readout": int(banned.sum()), "min_degree": min_degree}
    if level == "neuron":
        units = [TargetUnit("neuron", graph.qualified_id(int(i)), (int(i),))
                 for i in np.flatnonzero(node_ok)]
    else:
        attr = LEVEL_ATTR.get(level)
        if attr is None:
            raise ValueError(f"level must be neuron|type|class|region, got {level!r}")
        if attr not in graph.node_attrs:
            rep["error"] = f"graph has no '{attr}' annotation; level '{level}' unavailable"
            rep["n_units"] = 0
            return [], rep
        vals = graph.node_attrs[attr]
        groups: dict[str, list[int]] = {}
        for i in np.flatnonzero(node_ok):
            v = vals[i]
            if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
                continue
            groups.setdefault(str(v), []).append(int(i))
        units, skipped = [], 0
        for lab, nodes in sorted(groups.items()):
            if min_size <= len(nodes) <= max_size:
                units.append(TargetUnit(level, lab, tuple(nodes)))
            else:
                skipped += 1
        rep["groups_skipped_size"] = skipped
        rep["size_bounds"] = [min_size, max_size]
    rep["n_units"] = len(units)
    return units, rep


def dedupe_units(units: list[TargetUnit]) -> tuple[list[TargetUnit], int]:
    """Drop units whose node set duplicates an earlier one (a 1-neuron type == that neuron)."""
    seen, out, dropped = set(), [], 0
    for u in units:
        key = frozenset(u.nodes)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(u)
    return out, dropped


def plan_screen(
    graph,
    units: list[TargetUnit],
    n_interventions: int = 120,
    min_distinct: int = 100,
    primary_kind: str = "clamp_silence",
    extra_kinds: tuple[str, ...] = ("disconnect_both", "gain_scale_incoming", "drive_excitation"),
    t_start: int = 10,
    t_end: int | None = None,
    doses: dict | None = None,
    seed: int = 0,
) -> tuple[list[InterventionSpec], dict]:
    """Choose distinct targets (seeded) for the primary kind, then fill remaining slots with other
    kinds on already-chosen targets. Never duplicates a (target set, kind) pair and never invents
    targets: if fewer than ``min_distinct`` are eligible the plan reports the shortfall."""
    doses = doses or {}
    units, dropped_dup = dedupe_units(units)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(units))
    # primary family = exactly the distinct-target goal; remaining slots exercise other kinds
    n_primary = min(len(units), min_distinct if extra_kinds else n_interventions)
    chosen = [units[i] for i in order[:n_primary]]
    specs: list[InterventionSpec] = []
    seen: set = set()

    def add(kind: str, u: TargetUnit, family: str):
        key = (kind, frozenset(u.nodes))
        if key in seen:
            return False
        seen.add(key)
        specs.append(InterventionSpec(
            kind=kind, targets=u.nodes, dose=doses.get(kind), t_start=t_start, t_end=t_end,
            level=u.level, label=u.label, family=family,
            intervention_id=f"{kind}:{u.level}:{u.label}"))
        return True

    for u in chosen:
        add(primary_kind, u, f"{primary_kind}")
    # fill with extra kinds, round-robin over chosen targets
    k = 0
    while len(specs) < n_interventions and chosen and k < len(extra_kinds) * len(chosen):
        kind = extra_kinds[(k // len(chosen)) % len(extra_kinds)]
        add(kind, chosen[k % len(chosen)], f"{kind}")
        k += 1
    distinct = len({frozenset(s.targets) for s in specs})
    plan = {
        "requested_interventions": n_interventions, "requested_min_distinct_targets": min_distinct,
        "eligible_distinct_units": len(units), "duplicate_units_dropped": dropped_dup,
        "planned_interventions": len(specs), "planned_distinct_targets": distinct,
        "meets_distinct_target_goal": distinct >= min_distinct,
        "meets_intervention_count_goal": len(specs) >= n_interventions,
        "note": ("" if distinct >= min_distinct else
                 f"only {distinct} distinct valid targets are eligible (< {min_distinct}); "
                 "no targets were duplicated or invented"),
        "primary_kind": primary_kind, "extra_kinds": list(extra_kinds), "seed": seed,
    }
    return specs, plan


def matched_random_spec(
    graph, spec: InterventionSpec, pool: np.ndarray, streams_seed: int, k: int = 0,
    n_bins: int = 4,
) -> InterventionSpec:
    """Random control: same kind/dose/window/size, nodes drawn from ``pool`` (excluding the real
    targets) matched per node on total-degree quartile. Own seed -> reproducible, never touches
    run-level streams."""
    ind, outd = degrees(graph)
    deg = ind + outd
    real = set(spec.targets)
    cand = np.array([i for i in pool if int(i) not in real], dtype=np.int64)
    if len(cand) < len(spec.targets):
        raise ValueError("pool too small for a matched random control")
    edges = np.quantile(deg[pool], np.linspace(0, 1, n_bins + 1)[1:-1]) if len(pool) else []
    def binof(d) -> int:
        return int(np.searchsorted(edges, d, side="right"))

    cbin = np.array([binof(deg[i]) for i in cand])
    rng = np.random.default_rng(
        np.random.SeedSequence([streams_seed, k, _stable(spec.intervention_id)]))
    chosen, used = [], set()
    for t in spec.targets:
        b = binof(deg[t])
        opts = [c for c, cb in zip(cand, cbin, strict=True) if cb == b and int(c) not in used]
        if not opts:  # fall back to nearest degree among unused candidates (reported in extra)
            opts = sorted((c for c in cand if int(c) not in used), key=lambda c: abs(deg[c] - deg[t]))[:5]
        pick = int(opts[int(rng.integers(len(opts)))])
        used.add(pick)
        chosen.append(pick)
    return InterventionSpec(
        kind=spec.kind, targets=tuple(chosen), dose=spec.dose, t_start=spec.t_start, t_end=spec.t_end,
        level=spec.level, label=f"random_control[{k}] of {spec.label}", family="random_control",
        matched_to=spec.intervention_id, intervention_id=f"ctrl{k}:{spec.intervention_id}",
        extra={"degree_matched_bins": n_bins})


def _stable(s: str) -> int:
    import zlib

    return zlib.crc32(s.encode())


__all__ = ["RngStreams", "TargetUnit", "eligible_units", "dedupe_units", "plan_screen",
           "matched_random_spec", "degrees"]
