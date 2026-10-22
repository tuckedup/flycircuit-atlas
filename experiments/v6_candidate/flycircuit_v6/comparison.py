"""Reject cross-paper rankings when protocols are incompatible or unknown."""

from __future__ import annotations

import math

FIELDS = (
    "environment",
    "environment_version",
    "observation_contract",
    "action_contract",
    "success_rule",
    "episode_horizon",
    "control_period",
    "dropout_distribution",
    "train_data_protocol",
    "evaluation_data_protocol",
    "initialization",
    "parameter_budget",
    "replay_budget",
    "training_budget",
    "selection_budget",
    "metric_definition",
    "pretraining",
    "inference_information",
)


def comparability(candidate, reference):
    differences, unknown = {}, []
    for key in FIELDS:
        a, b = candidate.get(key), reference.get(key)
        if a is None or b is None or a == "unknown" or b == "unknown":
            unknown.append(key)
        elif a != b:
            differences[key] = {"candidate": a, "reference": b}
    return {
        "protocol_comparable": not unknown and not differences,
        "unknown_fields": unknown,
        "mismatches": differences,
        "sota_established": False,
        "note": "protocol compatibility is necessary, not sufficient for a SOTA claim",
    }


def resource_dominance(candidate, baseline):
    """Descriptive point-estimate Pareto check, never a statistical SOTA verdict."""
    needed = ("score", "trainable_parameters", "peak_rss_bytes", "optimizer_steps")
    for row in (candidate, baseline):
        for key in needed:
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"missing or nonnumeric {key}")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {key}")
    signs = [candidate["score"] >= baseline["score"]] + [
        candidate[k] <= baseline[k] for k in needed[1:]
    ]
    strict = candidate["score"] > baseline["score"] or any(
        candidate[k] < baseline[k] for k in needed[1:]
    )
    return {
        "point_estimate_dominates": all(signs) and strict,
        "sota_established": False,
        "uncertainty_accounted_for": False,
    }
