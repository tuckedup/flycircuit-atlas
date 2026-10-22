"""Reward-only observed-stream replay experiment. Heavy imports are LAZY."""

from __future__ import annotations

import hashlib
import json
import random
import time

from .metrics import continual, probability
from .replay import BalancedReservoir, Item, Reservoir

METHODS = ("standard", "extra_current", "uniform_replay", "balanced_replay")


def run_continual(
    agent,
    task,
    proto,
    seed,
    method,
    streams,
    cue_factory,
    capacity=32,
    stop=lambda: None,
):
    """The live adapter supplies existing v5 streams/model; tests use fake agents.

    Replay is from observed training events (may include the current task).
    Sampling happens BEFORE insertion of the current event. Unlike historical
    v5 R1 this is not first-K storage populated from a completed task's dataset.
    These are new baselines, not a DER/A2ER reproduction or a v5 continuation.
    """
    if method not in METHODS:
        raise ValueError("unknown replay method")
    phases, n = task.spec.n_ctx, int(proto["training"]["trials_per_phase"])
    if phases < 2 or n < 1:
        raise ValueError("need >=2 tasks and positive trials per phase")
    if type(capacity) is not int or capacity < phases:
        raise ValueError("shared budget must allow >=1 stored event per task")
    buffer = None
    if method == "uniform_replay":
        buffer = Reservoir(capacity, seed + 6071)
    if method == "balanced_replay":
        buffer = BalancedReservoir(capacity, seed + 6071)
    draw = random.Random(seed + 6072)
    counts = {
        "current": 0,
        "replay": 0,
        "extra_current": 0,
        "replay_old_context": 0,
        "replay_current_context": 0,
    }
    schedule = hashlib.sha256()
    matrix = []
    for phase in range(phases):
        for trial in range(n):
            if trial % 25 == 0:
                stop()
            cue, label_array = task.sample_train(streams.A.order, phase, False)
            label = int(label_array[0])
            vector = cue.tolist()[0] if hasattr(cue, "tolist") else cue[0]
            item = Item(
                counts["current"], phase, tuple(float(x) for x in vector), label
            )
            schedule.update(
                json.dumps([phase, vector, label], separators=(",", ":")).encode()
            )
            agent.trial(cue, phase, label, streams.A.noise, streams.A.act)
            counts["current"] += 1
            if phase >= 1 and buffer is not None:
                old = buffer.sample(draw)
                agent.trial(
                    cue_factory(old.cue),
                    old.context,
                    old.label,
                    streams.B.noise,
                    streams.B.act,
                )
                counts["replay"] += 1
                which = (
                    "replay_old_context"
                    if old.context < phase
                    else "replay_current_context"
                )
                counts[which] += 1
            elif phase >= 1 and method == "extra_current":
                extra_cue, extra_label = task.sample_train(
                    streams.C.order, phase, False
                )
                agent.trial(
                    extra_cue,
                    phase,
                    int(extra_label[0]),
                    streams.C.noise,
                    streams.C.act,
                )
                counts["extra_current"] += 1
            if buffer is not None:
                buffer.observe(item)
        stop()
        matrix.append(
            [probability(agent.evaluate("val", context)) for context in range(phases)]
        )
    expected_current = phases * n
    expected_extra = (phases - 1) * n if method != "standard" else 0
    if (
        counts["current"] != expected_current
        or counts["replay"] + counts["extra_current"] != expected_extra
    ):
        raise RuntimeError("trial-count contract violated")
    return {
        "method": method,
        "R_val": matrix,
        "metrics": continual(matrix),
        "trial_counts": counts,
        "optimizer_steps_expected": expected_current + expected_extra,
        "current_schedule_sha256": schedule.hexdigest(),
        "buffer": buffer.report()
        if buffer
        else {"capacity_events": capacity, "stored_events": 0},
        "split": "val",
        "sota_established": False,
    }


def execute_native(coordinate, source_root, stop):
    """Adapter to the EXISTING copy's v5 model factories. No import-time training."""
    import sys
    from pathlib import Path

    root = Path(source_root).resolve()
    sys.path.insert(0, str(root / "src"))
    import numpy as np
    import torch
    from flycircuit.eval import learning_v5 as v5

    from .plan import digest

    torch.set_num_threads(2)
    if v5.COPY.resolve() != root:
        raise RuntimeError("model factory was imported from the wrong project")
    protocol_path = "configs/experiments/learning_v3_protocol.yaml"
    actual_protocol_sha = digest(v5.ORIG / protocol_path)
    if actual_protocol_sha != digest(root / protocol_path):
        raise RuntimeError("native canonical protocol differs from the planned copy")
    proto = v5.V3.load_protocol()
    if (
        proto["task"]["n_ctx"] != 3
        or proto["training"]["trials_per_phase"] != 800
        or proto["circuit"] != {"name": "mushroom_body_right_pilot", "min_contacts": 5}
    ):
        raise RuntimeError(
            "native task/circuit differs from the frozen supported contract"
        )
    env = v5.Env()
    family, seed = coordinate["family"], coordinate["seed"]
    core, decoder = coordinate["core_lr"], coordinate["decoder_lr"]
    task = env.make_task(seed)
    agent, model_info, params = env.make_agent(family, core, decoder, seed, task)
    v5.inspect_lr(agent, core, decoder)
    before = v5.adam_step_counts(agent)
    t0 = time.monotonic()
    result = run_continual(
        agent,
        task,
        env.proto,
        seed,
        coordinate["method"],
        v5.Streams(seed),
        lambda cue: np.asarray([cue], dtype=np.float32),
        coordinate["capacity"],
        stop,
    )
    after = v5.adam_step_counts(agent)
    expected = result["optimizer_steps_expected"]
    if before["max"] != 0 or after["min"] != expected or after["max"] != expected:
        raise RuntimeError("actual Adam state-step count differs from trial count")
    identity = hashlib.sha256(task.teachers.tobytes())
    for split in ("train", "val"):
        for context in range(task.spec.n_ctx):
            identity.update(task.cues[(split, context)].tobytes())
            identity.update(task.labels_for(split, context).tobytes())
    result.update(
        family=family,
        seed=seed,
        native_n_contexts=task.spec.n_ctx,
        native_trials_per_phase=int(env.proto["training"]["trials_per_phase"]),
        native_protocol_file_sha256=actual_protocol_sha,
        graph_mapping_sha256=env.base25.notes["mapping_hash"],
        graph_edges_sha256=env.base25.notes["edge_hash"],
        graph_signs_sha256=env.base25.notes["sign_assignment_hash"],
        params=params,
        model_info=model_info,
        actual_adam_steps=after,
        elapsed_s=time.monotonic() - t0,
        peak_rss_mib=v5.peak_rss_mib(),
        task_protocol_hash=env.protocol_hash,
        w76_sha256=env.w76_sha,
        training_validation_identity_sha256=identity.hexdigest(),
        optimizer_name=type(agent.opt).__name__,
        torch_version=torch.__version__,
    )
    return result
