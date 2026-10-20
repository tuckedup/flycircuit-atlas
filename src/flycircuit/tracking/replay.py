"""Replay artifact: a bounded, provenance-carrying record of neural activity, plastic gains and the
stimulus/action/outcome timeline of ONE recorded run. ``replay.json`` sits next to ``manifest.json``.

The viewer plays these back as PRERECORDED data; nothing is simulated live and nothing is invented.
Rendering is downsampled by the viewer independently of what is stored here.

Format ``flycircuit.replay/1`` (all ids are release-qualified strings, e.g. ``flywire:783:72057...``)::

    format, label, kind, recorded=true, synthetic, structural_untrained, provenance{...},
    dt, time_unit, n_steps, neurons[K], neuron_roles[K], activity[n_steps][K], stimulus[n_steps],
    trials[{index, step_start, step_end, action, reward, delta, correct, label}],
    gains{edges[{pre, post}], trajectory[n_trials+1][n_edges]}  (optional),
    notes[...]
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

REPLAY_FORMAT = "flycircuit.replay/1"
MAX_VALUES = 400_000  # activity + gains values stored per replay (bounded by design)
REQUIRED = ("format", "label", "kind", "recorded", "synthetic", "structural_untrained",
            "provenance", "dt", "time_unit", "n_steps", "neurons", "activity", "stimulus")
PROVENANCE_REQUIRED = ("run_id", "release", "circuit", "seed", "sign_policy", "code_note")


def _finite(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def validate_replay(doc: dict[str, Any]) -> None:
    """Raise ValueError when the document is malformed or unbounded."""
    for k in REQUIRED:
        if k not in doc:
            raise ValueError(f"replay missing field {k!r}")
    if doc["format"] != REPLAY_FORMAT:
        raise ValueError(f"unsupported replay format {doc['format']!r}")
    for k in PROVENANCE_REQUIRED:
        if k not in doc["provenance"]:
            raise ValueError(f"replay provenance missing {k!r}")
    for nid in doc["neurons"]:
        if not isinstance(nid, str):
            raise ValueError("neuron ids must be strings (JS cannot hold 64-bit ints)")
    t, k = doc["n_steps"], len(doc["neurons"])
    if len(doc["activity"]) != t or len(doc["stimulus"]) != t:
        raise ValueError("activity/stimulus length must equal n_steps")
    if any(len(r) != k for r in doc["activity"]):
        raise ValueError("every activity row must have one value per tracked neuron")
    n_gain = 0
    g = doc.get("gains")
    if g:
        e = len(g["edges"])
        if any(len(r) != e for r in g["trajectory"]):
            raise ValueError("gain trajectory rows must match gains.edges")
        n_gain = e * len(g["trajectory"])
    if t * k + n_gain > MAX_VALUES:
        raise ValueError(f"replay too large: {t * k + n_gain} values > {MAX_VALUES}")


def write_replay(run_dir: Path, doc: dict[str, Any], digits: int = 4) -> Path:
    """Validate and write ``<run_dir>/replay.json`` (rounded floats, NaN -> null)."""
    doc = dict(doc)
    doc["recorded"] = True
    validate_replay(doc)
    doc["activity"] = [[_finite(round(float(x), digits)) for x in r] for r in doc["activity"]]
    doc["stimulus"] = [_finite(round(float(x), digits)) for x in doc["stimulus"]]
    if doc.get("gains"):
        doc["gains"]["trajectory"] = [[_finite(round(float(x), digits)) for x in r]
                                      for r in doc["gains"]["trajectory"]]
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / "replay.json"
    p.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    return p
