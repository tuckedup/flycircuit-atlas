"""Prospective source/config snapshots. No NumPy/Torch import or seed evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .runner import METHODS

NATIVE_SOURCES = (
    "src/flycircuit/eval/learning_v5.py",
    "src/flycircuit/eval/learning.py",
    "src/flycircuit/eval/learning_v2.py",
    "src/flycircuit/eval/learning_v3.py",
    "src/flycircuit/eval/learning_v4_select.py",
    "src/flycircuit/eval/stats.py",
    "src/flycircuit/models/baselines.py",
    "src/flycircuit/models/sparse_rnn.py",
    "src/flycircuit/plasticity/agent.py",
    "src/flycircuit/plasticity/three_factor.py",
    "src/flycircuit/tasks/association.py",
    "src/flycircuit/tasks/graph_loader.py",
    "configs/experiments/learning_v3_protocol.yaml",
    "results_v4/learning_readout_v4_1/W3_trained_w76_choices.json",
    "results_v4/learning_readout_v4_1/W0_width_probe_manifest.json",
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def snapshot(bundle, native):
    files = sorted((bundle / "flycircuit_v6").glob("*.py")) + [bundle / "protocol.json"]
    pairs = [(str(p.resolve()), digest(p)) for p in files]
    dependencies = set(native / name for name in NATIVE_SOURCES)
    dependencies.update((native / "src/flycircuit").rglob("*.py"))
    pairs += [(str(path.resolve()), digest(path)) for path in sorted(dependencies)]
    return dict(pairs)


def input_paths(native):
    root = Path(native).resolve().parent / "flycircuit"
    processed = root / "data/processed/flywire/783"
    circuit = processed / "circuits/mushroom_body_right_pilot"
    return (
        root / "configs/experiments/learning_v3_protocol.yaml",
        processed / "neurons.parquet",
        processed / "edges_pairs.parquet",
        circuit / "internal_ids.parquet",
        circuit / "selection.json",
    )


def prepare(bundle, native):
    bundle, native = Path(bundle).resolve(), Path(native).resolve()
    protocol = json.loads((bundle / "protocol.json").read_text())
    if protocol["benchmark_id"] != "flycircuit-native-v6" or protocol["split"] != "val":
        raise ValueError(
            "candidate runner only supports the prospective native validation pilot"
        )
    seeds = protocol["model_seeds"]
    if (
        len(seeds) < 2
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("model seeds must be unique nonnegative integers")
    # Existing reserved v5 heldout IDs remain untouched by this separate pilot.
    if set(seeds) & (
        set(range(100601, 100621)) | set(range(1601, 1613)) | set(range(1701, 1707))
    ):
        raise ValueError("candidate seeds overlap frozen v5 roles")
    if protocol["methods"] != list(METHODS) or protocol["families"] != [
        "connectome",
        "gru33",
    ]:
        raise ValueError("unexpected method/family design")
    if type(protocol["capacity_events"]) is not int or protocol["capacity_events"] < 3:
        raise ValueError("invalid memory budget")
    if (
        type(protocol.get("native_n_contexts")) is not int
        or protocol["native_n_contexts"] != 3
        or type(protocol.get("native_trials_per_phase")) is not int
        or protocol["native_trials_per_phase"] != 800
    ):
        raise ValueError("candidate supports only the frozen 3-context/800-trial task")
    if (
        type(protocol["wall_limit_s"]) is not int
        or not 1 <= protocol["wall_limit_s"] <= 1800
    ):
        raise ValueError("candidate measurement cap must not exceed 1800 seconds")
    choices = json.loads((native / NATIVE_SOURCES[-2]).read_text())
    coords = []
    for family in protocol["families"]:
        choice = choices["gru" if family == "gru33" else family]
        rates = [choice["core_lr"], choice["decoder_lr"]]
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            or x <= 0
            for x in rates
        ):
            raise ValueError("invalid fixed historical learning-rate anchor")
        for method in protocol["methods"]:
            for seed in seeds:
                coords.append(
                    {
                        "family": family,
                        "method": method,
                        "seed": seed,
                        "core_lr": rates[0],
                        "decoder_lr": rates[1],
                        "capacity": protocol["capacity_events"],
                    }
                )
    # Interleave methods/families per seed to reduce time-order confounding.
    coords.sort(key=lambda x: (x["seed"], x["family"], x["method"]))
    for coord in coords:
        coord["key"] = canonical(coord)
    return {
        "schema": 3,
        "protocol": protocol,
        "coordinates": coords,
        "source_root": str(native),
        "bundle_root": str(bundle),
        "source_sha256": snapshot(bundle, native),
        "input_sha256": {str(p.resolve()): digest(p) for p in input_paths(native)},
        "classification": "prospective exploratory native-task engineering pilot",
        "seed_status": "candidate identities only; independent exposure review required before acceptance",
        "input_status": "exact processed input bytes frozen; independent semantic/input review still required",
        "sota_established": False,
    }


def verify_plan(plan):
    for name, expected in plan["source_sha256"].items():
        if not Path(name).is_file() or digest(name) != expected:
            raise ValueError(f"source/config drift: {name}")
    # Reconstruct the design, not merely a self-consistent edited coordinate list.
    rebuilt = prepare(plan["bundle_root"], plan["source_root"])
    if canonical(rebuilt) != canonical(plan):
        raise ValueError(
            "plan differs from its prospective source/config reconstruction"
        )


def acceptance(plan_path, receipt_path, with_binding=False):
    plan = json.loads(Path(plan_path).read_text())
    verify_plan(plan)
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt.get("verdict") != "ship" or receipt.get("plan_sha256") != digest(
        plan_path
    ):
        raise ValueError("missing independent review bound to this exact plan")
    if not receipt.get("reviewer") or receipt.get("seed_exposure_review") != "passed":
        raise ValueError("reviewer identity and seed-exposure review are required")
    inventory = receipt.get("input_sha256")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("accepted graph/data inventory is missing")
    required_inputs = input_paths(plan["source_root"])
    if {str(p.resolve()) for p in required_inputs} != set(inventory):
        raise ValueError(
            "receipt must bind canonical protocol, three processed parquet inputs and selection.json"
        )
    if inventory != plan["input_sha256"]:
        raise ValueError(
            "accepted input inventory differs from the immutable prospective plan"
        )
    if receipt.get("native_smoke_test") != "passed":
        raise ValueError("real-model native adapter smoke test is still pending")
    for name, expected in inventory.items():
        if not Path(name).is_file() or digest(name) != expected:
            raise ValueError(f"accepted input drift: {name}")
    actual_protocol = required_inputs[0]
    planned_protocol = str(
        (
            Path(plan["source_root"]) / "configs/experiments/learning_v3_protocol.yaml"
        ).resolve()
    )
    if digest(actual_protocol) != plan["source_sha256"][planned_protocol]:
        raise ValueError("actual canonical protocol differs from the planned copy")
    return (plan, canonical(receipt)) if with_binding else plan
