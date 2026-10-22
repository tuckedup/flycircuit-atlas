"""Fail-closed result schema, shared by append, resume and independent audit."""

from __future__ import annotations

import math
from pathlib import Path

from .metrics import continual


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("result count must be an integer within its declared bounds")
    return value


def finite(value, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError("result resource measurement must be finite and nonnegative")
    return value


def hash_value(value, length=64):
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(x not in "0123456789abcdef" for x in value)
    ):
        raise ValueError("result identity must be a lowercase hexadecimal digest")


def validate_result(result, coord, plan):
    required = {
        "method",
        "R_val",
        "metrics",
        "trial_counts",
        "optimizer_steps_expected",
        "current_schedule_sha256",
        "buffer",
        "split",
        "sota_established",
        "family",
        "seed",
        "native_n_contexts",
        "native_trials_per_phase",
        "native_protocol_file_sha256",
        "params",
        "model_info",
        "actual_adam_steps",
        "elapsed_s",
        "peak_rss_mib",
        "task_protocol_hash",
        "w76_sha256",
        "training_validation_identity_sha256",
        "optimizer_name",
        "torch_version",
        "graph_mapping_sha256",
        "graph_edges_sha256",
        "graph_signs_sha256",
    }
    if not isinstance(result, dict) or set(result) != required:
        raise ValueError("successful result has missing or unexpected fields")
    protocol = plan["protocol"]
    phases, n = protocol["native_n_contexts"], protocol["native_trials_per_phase"]
    if (
        result["method"] != coord["method"]
        or result["family"] != coord["family"]
        or type(result["seed"]) is not int
        or result["seed"] != coord["seed"]
        or result["split"] != "val"
        or result["sota_established"] is not False
        or integer(result["native_n_contexts"]) != phases
        or integer(result["native_trials_per_phase"]) != n
    ):
        raise ValueError(
            "result method/family/seed/split/task identity differs from plan"
        )
    matrix = result["R_val"]
    if (
        not isinstance(matrix, list)
        or len(matrix) != phases
        or any(not isinstance(row, list) or len(row) != phases for row in matrix)
    ):
        raise ValueError("missing or wrong-sized native task accuracy matrix")
    expected_metrics = continual(matrix)
    measured_metrics = result["metrics"]
    if not isinstance(measured_metrics, dict) or set(measured_metrics) != set(
        expected_metrics
    ):
        raise ValueError("incomplete continual metrics")
    for name, expected in expected_metrics.items():
        actual = measured_metrics[name]
        if type(expected) is float:
            if (
                type(actual) not in (int, float)
                or not math.isfinite(actual)
                or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise ValueError(
                    "metric does not reconstruct from the saved accuracy matrix"
                )
        elif actual != expected:
            raise ValueError("metric identity differs from the supported contract")
    counts = result["trial_counts"]
    names = {
        "current",
        "replay",
        "extra_current",
        "replay_old_context",
        "replay_current_context",
    }
    if not isinstance(counts, dict) or set(counts) != names:
        raise ValueError("trial counts are missing")
    for value in counts.values():
        integer(value)
    replaying = coord["method"] in ("uniform_replay", "balanced_replay")
    extras = (phases - 1) * n
    if (
        counts["current"] != phases * n
        or counts["replay"] != (extras if replaying else 0)
        or counts["extra_current"]
        != (extras if coord["method"] == "extra_current" else 0)
        or counts["replay_old_context"] + counts["replay_current_context"]
        != counts["replay"]
    ):
        raise ValueError("trial counts differ from the fixed update budget")
    expected_steps = phases * n + (extras if coord["method"] != "standard" else 0)
    if integer(result["optimizer_steps_expected"]) != expected_steps:
        raise ValueError("wrong expected optimizer steps")
    params, steps = result["params"], result["actual_adam_steps"]
    core, decoder = {"connectome": (7399, 152), "gru33": (7392, 68)}[coord["family"]]
    if (
        not isinstance(params, dict)
        or not isinstance(steps, dict)
        or set(steps) != {"min", "max", "n_tensors"}
    ):
        raise ValueError("missing actual optimizer metadata")
    if (
        params.get("optimizer") != "Adam"
        or result["optimizer_name"] != "Adam"
        or integer(params.get("owned_total")) != core + decoder
        or integer(params.get("core_owned")) != core
        or integer(params.get("decoder_owned")) != decoder
        or integer(params.get("requires_grad_total")) != core + decoder
        or integer(steps["min"]) != expected_steps
        or integer(steps["max"]) != expected_steps
    ):
        raise ValueError("actual optimizer/parameter/step identity differs from design")
    groups = params.get("groups")
    if not isinstance(groups, list) or len(groups) != 2:
        raise ValueError("expected the actual two-group Adam configuration")
    for index, (rate, size) in enumerate(
        ((coord["core_lr"], core), (coord["decoder_lr"], decoder))
    ):
        group = groups[index]
        if (
            not isinstance(group, dict)
            or integer(group.get("group")) != index
            or finite(group.get("lr")) != rate
            or integer(group.get("numel")) != size
        ):
            raise ValueError("actual optimizer group differs from coordinate")
    if integer(steps["n_tensors"], 1) != sum(
        integer(g.get("n_tensors"), 1) for g in groups
    ):
        raise ValueError("actual optimizer tensor counts differ")
    if (
        not isinstance(result["model_info"], dict)
        or not isinstance(result["torch_version"], str)
        or not result["torch_version"]
    ):
        raise ValueError("missing model/runtime identity")
    for name in (
        "current_schedule_sha256",
        "native_protocol_file_sha256",
        "w76_sha256",
        "training_validation_identity_sha256",
        "graph_mapping_sha256",
        "graph_edges_sha256",
        "graph_signs_sha256",
    ):
        hash_value(result[name])
    hash_value(result["task_protocol_hash"], 16)
    native_protocol = str(
        (
            Path(plan["source_root"]) / "configs/experiments/learning_v3_protocol.yaml"
        ).resolve()
    )
    if result["native_protocol_file_sha256"] != plan["source_sha256"][native_protocol]:
        raise ValueError("actual task protocol does not match the planned input")
    finite(result["elapsed_s"])
    if finite(result["peak_rss_mib"]) <= 0:
        raise ValueError("RSS measurement missing")
    buffer = result["buffer"]
    if (
        not isinstance(buffer, dict)
        or integer(buffer.get("capacity_events")) != coord["capacity"]
    ):
        raise ValueError("wrong buffer budget")
    stored = integer(buffer.get("stored_events"))
    if stored > coord["capacity"]:
        raise ValueError("buffer exceeded capacity")
    if replaying:
        if set(buffer) != {
            "capacity_events",
            "observed_events",
            "stored_events",
            "peak_stored_events",
            "peak_serialized_bytes",
            "memory_note",
            "contexts",
        }:
            raise ValueError("missing replay storage measurements")
        if (
            integer(buffer["observed_events"]) != phases * n
            or integer(buffer["peak_stored_events"]) != coord["capacity"]
            or stored != coord["capacity"]
        ):
            raise ValueError("wrong observed replay event/storage counts")
        integer(buffer["peak_serialized_bytes"], 1)
        contexts = buffer["contexts"]
        if (
            not isinstance(contexts, dict)
            or set(contexts) - {str(i) for i in range(phases)}
            or sum(integer(v) for v in contexts.values()) != stored
        ):
            raise ValueError("invalid buffer context counts")
    elif set(buffer) != {"capacity_events", "stored_events"} or stored != 0:
        raise ValueError("unexpected buffer in a no-replay control")


def validate_row(row, coord, plan):
    if not isinstance(row, dict):
        raise ValueError("JSONL result must be an object")
    base = {"key", "coordinate", "plan_sha256", "status", "acceptance_sha256"}
    hash_value(row.get("acceptance_sha256"))
    if row.get("status") == "ok":
        if set(row) != base | {"result"}:
            raise ValueError("ok row requires exactly one complete result payload")
        validate_result(row["result"], coord, plan)
    elif row.get("status") == "failed":
        if (
            set(row) != base | {"error"}
            or not isinstance(row["error"], str)
            or not row["error"].strip()
        ):
            raise ValueError(
                "failed row requires an error and cannot contain measurements"
            )
    else:
        raise ValueError("unknown or missing execution status")
