"""Intervention definitions. All causal statements are MODEL-RELATIVE.

Distinct names for distinct operations (node activity clamping != edge deletion):

=====================  =======================================================  ==================
kind                   operation                                                dose meaning
=====================  =======================================================  ==================
clamp_silence          node state h[S] clamped to ``dose`` (default 0) after    clamp level (a.u.)
                       every update in the window; edges untouched; bias,
                       input, plasticity and decoder cannot revive it
clamp_excite           same mechanism, clamped to a high level                  clamp level (a.u.)
delete_outgoing        w[e] *= (1-dose) for edges with pre in S                 fraction removed
delete_incoming        w[e] *= (1-dose) for edges with post in S                fraction removed
disconnect_both        both of the above                                        fraction removed
gain_scale_outgoing    w[e] *= dose for pre in S                                multiplicative
gain_scale_incoming    w[e] *= dose for post in S                               multiplicative
drive_excitation       additive current ``dose`` into S pre-activation          a.u. of drive
transient_disruption   additive N(0, dose^2) drive noise into S (own RNG        std of drive noise
                       stream 'disruption')
=====================  =======================================================  ==================

Window: active for steps ``t_start <= t < t_end`` (``t_end=None`` -> end of run).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

CLAMP_KINDS = ("clamp_silence", "clamp_excite")
EDGE_KINDS = (
    "delete_outgoing", "delete_incoming", "disconnect_both",
    "gain_scale_outgoing", "gain_scale_incoming",
)
DRIVE_KINDS = ("drive_excitation", "transient_disruption")
KINDS = CLAMP_KINDS + EDGE_KINDS + DRIVE_KINDS

DOSE_UNITS = {
    "clamp_silence": "clamp level (a.u. activity)", "clamp_excite": "clamp level (a.u. activity)",
    "delete_outgoing": "fraction of edges removed", "delete_incoming": "fraction of edges removed",
    "disconnect_both": "fraction of edges removed", "gain_scale_outgoing": "multiplicative factor",
    "gain_scale_incoming": "multiplicative factor", "drive_excitation": "additive drive (a.u.)",
    "transient_disruption": "std of drive noise (a.u.)",
}
DEFAULT_DOSE = {
    "clamp_silence": 0.0, "clamp_excite": 1.0, "delete_outgoing": 1.0, "delete_incoming": 1.0,
    "disconnect_both": 1.0, "gain_scale_outgoing": 2.0, "gain_scale_incoming": 2.0,
    "drive_excitation": 1.0, "transient_disruption": 1.0,
}


@dataclass(frozen=True)
class InterventionSpec:
    kind: str
    targets: tuple[int, ...]  # node indices (VersionedGraph order)
    dose: float | None = None
    t_start: int = 10
    t_end: int | None = None
    level: str = "neuron"  # neuron | type | class | region
    label: str = ""
    family: str = "primary"  # multiplicity family
    matched_to: str | None = None  # id of the real intervention a random control matches
    intervention_id: str = ""
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown intervention kind {self.kind!r}; allowed {KINDS}")
        if len(self.targets) == 0:
            raise ValueError("intervention needs >=1 target (run a branch with spec=None for the no-op baseline)")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("duplicate target nodes inside one intervention")
        if self.dose is None:
            object.__setattr__(self, "dose", DEFAULT_DOSE[self.kind])
        if self.t_end is not None and self.t_end <= self.t_start:
            raise ValueError("t_end must exceed t_start")

    def window(self, steps: int) -> tuple[int, int]:
        return self.t_start, steps if self.t_end is None else min(self.t_end, steps)

    def metadata(self) -> dict:
        d = asdict(self)
        d["targets"] = list(self.targets)
        d["dose_unit"] = DOSE_UNITS[self.kind]
        return d

    def to_json(self) -> str:
        return json.dumps(self.metadata(), sort_keys=True, default=str)
