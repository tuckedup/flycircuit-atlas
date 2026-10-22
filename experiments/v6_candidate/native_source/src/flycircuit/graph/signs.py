"""Versioned sign-policy layer (``SIGN_POLICY_VERSION``) for FlyWire annotation -> assumed edge sign.

Every number this module produces is an ASSUMED physiological sign for the *fast* (ionotropic-like)
edge of a presynaptic neuron, not a measured effect. Three evidence tiers are kept apart and reported
per neuron in ``SignAssignment.evidence``:

``known``      ``known_nt`` field of the FlyWire annotation (literature / experiment backed, with
               ``known_nt_source``). Parsed by :func:`parse_known_nt`.
``predicted``  ``top_nt`` with ``top_nt_conf`` (an image-based classifier prediction; for FlyWire v783
               it labels ~all Kenyon cells "dopamine" - present in the original TSV, not a pipeline
               bug; see docs/SIGN_AUDIT.md).
``prior``      an explicit literature prior supplied by the policy (e.g. histamine -> inhibitory,
               glutamate sign variant). Never measured neuron-specific data.
``unknown``    nothing usable; filled by ``unknown`` policy (zero | excitatory | inhibitory |
               random_fixed) and flagged.

Monoamines (dopamine, serotonin, octopamine) are MODULATORY: by default they get fast sign 0 and are
flagged in ``modulatory`` so a plasticity/modulation channel can use them. ``monoamine_mode=
"legacy_excitatory"`` reproduces the Shiu-style +1 simplification for comparison only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SIGN_POLICY_VERSION = "2.0"

CLASSICAL = ("acetylcholine", "gaba", "glutamate", "dopamine", "serotonin", "octopamine", "histamine")
MONOAMINES = frozenset({"dopamine", "serotonin", "octopamine"})
UNKNOWN_FILL = ("zero", "excitatory", "inhibitory", "random_fixed")
GLUTAMATE_MODES = {"inhibitory": -1, "excitatory": 1, "unknown": 0}
MONOAMINE_MODES = ("separate_channel", "legacy_excitatory")
SOURCES = ("known_then_predicted", "known_only", "predicted_only")


def parse_known_nt(raw: object) -> tuple[str | None, str]:
    """Reduce a FlyWire ``known_nt`` string to (single classical NT or None, status).

    Format observed in v783: ``;``-separated source groups, each a ``,``-separated list; tokens like
    ``gaba-negative`` mean *tested negative* (not evidence for that NT); non-classical entries
    (sNPF, tachykinin, nitric oxide ...) are co-transmitters and ignored. Status:
    ``single`` | ``ambiguous`` (>1 classical NT) | ``none_classical`` (only negatives/co-transmitters)
    | ``missing``.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None, "missing"
    found: list[str] = []
    for tok in raw.replace(";", ",").split(","):
        t = tok.strip().lower()
        if not t or t.endswith("-negative"):
            continue
        if t in CLASSICAL and t not in found:
            found.append(t)
    if len(found) == 1:
        return found[0], "single"
    if len(found) > 1:
        return None, "ambiguous"
    return None, "none_classical"


@dataclass(frozen=True)
class SignPolicy:
    name: str = "known_then_predicted"
    min_conf: float = 0.5
    glutamate: str = "inhibitory"  # inhibitory | excitatory | unknown
    histamine: str = "inhibitory"  # literature prior (Cl- channel); inhibitory | unknown
    monoamine_mode: str = "separate_channel"
    unknown_fill: str = "zero"
    fill_seed: int = 0
    kc_literature_prior: bool = False  # force Kenyon cells to ACh (prior; known_nt agrees in v783)

    def __post_init__(self):
        if self.name not in SOURCES:
            raise ValueError(f"source {self.name!r}; choose {SOURCES}")
        if self.glutamate not in GLUTAMATE_MODES:
            raise ValueError(f"glutamate {self.glutamate!r}; choose {sorted(GLUTAMATE_MODES)}")
        if self.histamine not in ("inhibitory", "unknown"):
            raise ValueError("histamine must be inhibitory|unknown")
        if self.monoamine_mode not in MONOAMINE_MODES:
            raise ValueError(f"monoamine_mode {self.monoamine_mode!r}; choose {MONOAMINE_MODES}")
        if self.unknown_fill not in UNKNOWN_FILL:
            raise ValueError(f"unknown_fill {self.unknown_fill!r}; choose {UNKNOWN_FILL}")

    def tag(self) -> str:
        return (f"v{SIGN_POLICY_VERSION}:{self.name}|conf>={self.min_conf}|glu={self.glutamate}"
                f"|his={self.histamine}|mono={self.monoamine_mode}|fill={self.unknown_fill}"
                f"{'#' + str(self.fill_seed) if self.unknown_fill == 'random_fixed' else ''}"
                f"|kc_prior={self.kc_literature_prior}")


@dataclass
class SignAssignment:
    sign: np.ndarray  # int8 in {-1,0,+1}: assumed fast-edge sign of the presynaptic neuron
    modulatory: np.ndarray  # bool: monoamine neuron (dopamine/serotonin/octopamine) by chosen evidence
    nt: np.ndarray  # object: resolved NT label or ""
    evidence: np.ndarray  # object: known | predicted | prior | unknown
    policy: SignPolicy
    summary: dict = field(default_factory=dict)

    def hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.policy.tag().encode())
        h.update(self.sign.tobytes())
        h.update(self.modulatory.tobytes())
        return h.hexdigest()


def _sign_of(nt: str, p: SignPolicy) -> int:
    if nt == "acetylcholine":
        return 1
    if nt == "gaba":
        return -1
    if nt == "glutamate":
        return GLUTAMATE_MODES[p.glutamate]
    if nt == "histamine":
        return -1 if p.histamine == "inhibitory" else 0
    if nt in MONOAMINES:
        return 1 if p.monoamine_mode == "legacy_excitatory" else 0
    return 0


def assign_signs(known_nt, top_nt, top_conf, cell_class, policy: SignPolicy | None = None
                 ) -> SignAssignment:
    """Vectorised-by-loop assignment (O(n); n is a circuit, not the whole brain)."""
    p = policy or SignPolicy()
    n = len(top_nt)
    sign = np.zeros(n, dtype=np.int8)
    modulatory = np.zeros(n, dtype=bool)
    nts = np.empty(n, dtype=object)
    ev = np.empty(n, dtype=object)
    rng = np.random.default_rng(p.fill_seed)
    fill_draw = rng.choice([-1, 1], size=n).astype(np.int8)
    parse_status: dict[str, int] = {}
    for i in range(n):
        nt, ev_i = "", "unknown"
        cc = cell_class[i] if cell_class is not None else None
        if p.kc_literature_prior and isinstance(cc, str) and "kenyon" in cc.lower():
            nt, ev_i = "acetylcholine", "prior"
        else:
            k, status = parse_known_nt(known_nt[i])
            parse_status[status] = parse_status.get(status, 0) + 1
            if p.name in ("known_then_predicted", "known_only") and k is not None:
                nt, ev_i = k, "known"
            elif p.name in ("known_then_predicted", "predicted_only"):
                t, c = top_nt[i], top_conf[i]
                if (isinstance(t, str) and t.lower() in CLASSICAL
                        and c is not None and not np.isnan(c) and c >= p.min_conf):
                    nt, ev_i = t.lower(), "predicted"
        nts[i], ev[i] = nt, ev_i
        if nt:
            modulatory[i] = nt in MONOAMINES
            s = _sign_of(nt, p)
            sign[i] = s
            if nt == "histamine" and ev_i != "prior" and s != 0:
                ev[i] = "prior"  # sign for histamine is a literature prior even with known label
            if nt == "glutamate" and p.glutamate != "inhibitory":
                ev[i] = "prior" if ev_i != "unknown" else ev_i
        else:
            if p.unknown_fill == "excitatory":
                sign[i] = 1
            elif p.unknown_fill == "inhibitory":
                sign[i] = -1
            elif p.unknown_fill == "random_fixed":
                sign[i] = fill_draw[i]
    summ = {
        "policy": p.tag(),
        "n": n,
        "evidence_counts": {k: int((ev == k).sum()) for k in ("known", "predicted", "prior", "unknown")},
        "sign_counts": {str(s): int((sign == s).sum()) for s in (-1, 0, 1)},
        "modulatory": int(modulatory.sum()),
        "nt_counts": {k: int((nts == k).sum()) for k in sorted(set(nts.tolist()))},
        "known_nt_parse_status": parse_status,
    }
    return SignAssignment(sign, modulatory, nts, ev, p, summ)


def load_assignment_for_ids(root_ids: np.ndarray, policy: SignPolicy | None = None,
                            processed: Path | None = None, dataset: str = "flywire",
                            version: str = "783") -> SignAssignment:
    """Look root ids up in ``neurons.parquet`` (exact int64 join) and assign signs."""
    import duckdb
    import pyarrow as pa

    from flycircuit.graph.bridge import processed_release_dir

    proc = Path(processed) if processed else processed_release_dir(dataset, version)
    ids = np.asarray(root_ids, dtype=np.int64)
    con = duckdb.connect()
    con.register("q", pa.table({"root_id": ids, "o": np.arange(len(ids), dtype=np.int64)}))
    rows = con.execute(
        f"SELECT q.o, n.known_nt, n.top_nt, n.top_nt_conf, n.cell_class FROM q "
        f"LEFT JOIN '{proc / 'neurons.parquet'}' n USING (root_id) ORDER BY q.o"
    ).fetchall()
    if len(rows) != len(ids):
        raise ValueError("duplicate root ids in neurons.parquet join")
    known = [r[1] for r in rows]
    top = [r[2] for r in rows]
    conf = [r[3] for r in rows]
    cls = [r[4] for r in rows]
    return assign_signs(known, top, conf, cls, policy)


def policy_grid() -> dict[str, SignPolicy]:
    """Named sensitivity grid (unknown-sign and glutamate-sign variants)."""
    g: dict[str, SignPolicy] = {"v2_default": SignPolicy()}
    for f in ("excitatory", "inhibitory", "random_fixed"):
        g[f"unknown_{f}"] = SignPolicy(unknown_fill=f)
    g["glu_excitatory"] = SignPolicy(glutamate="excitatory")
    g["glu_unknown"] = SignPolicy(glutamate="unknown")
    g["legacy_monoamine_excitatory"] = SignPolicy(monoamine_mode="legacy_excitatory")
    g["predicted_only"] = SignPolicy(name="predicted_only")
    g["known_only"] = SignPolicy(name="known_only")
    g["kc_prior_ach"] = SignPolicy(name="predicted_only", kc_literature_prior=True)
    return g


def policy_to_json(p: SignPolicy) -> str:
    return json.dumps({**asdict(p), "version": SIGN_POLICY_VERSION, "tag": p.tag()}, sort_keys=True)
