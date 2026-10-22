"""prepare/status/audit are stdlib-only. 'run' is explicitly guarded and reviewed."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

from .guard import run, status
from .plan import acceptance, canonical, digest, prepare, verify_plan
from .records import validate_row


def strict_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
    return value


def reject_constant(value):
    raise ValueError(f"nonfinite JSON token: {value}")


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_records(path, plan, receipt_sha=None):
    expected = {x["key"]: x for x in plan["coordinates"]}
    seen, records, identities = set(), [], {}
    if not path.exists():
        return records
    data = path.read_bytes()
    if data and receipt_sha is None:
        raise ValueError(
            "saved rows require the exact accepted receipt for audit or resume"
        )
    if data and not data.endswith(b"\n"):
        raise ValueError(
            "torn JSONL tail: preserve and audit it; no automatic truncation"
        )
    for line in data.splitlines():
        row = json.loads(
            line, object_pairs_hook=strict_pairs, parse_constant=reject_constant
        )
        if not isinstance(row, dict):
            raise ValueError("JSONL result must be an object")
        key = row.get("key")
        if not isinstance(key, str):
            raise ValueError("result coordinate key must be a string")
        if key in seen or key not in expected:
            raise ValueError("duplicate or unexpected result coordinate")
        if (
            row.get("plan_sha256") != canonical(plan)
            or row.get("coordinate") != expected[key]
            or row.get("acceptance_sha256") != receipt_sha
        ):
            raise ValueError("result provenance differs from the prospective plan")
        validate_row(row, expected[key], plan)
        if row["status"] == "ok":
            result = row["result"]
            identity = tuple(
                result[x]
                for x in (
                    "current_schedule_sha256",
                    "training_validation_identity_sha256",
                )
            )
            seed_key = ("seed", expected[key]["seed"])
            global_key = ("global",)
            global_identity = tuple(
                result[x]
                for x in (
                    "task_protocol_hash",
                    "native_protocol_file_sha256",
                    "w76_sha256",
                    "torch_version",
                    "graph_mapping_sha256",
                    "graph_edges_sha256",
                    "graph_signs_sha256",
                )
            )
            for binding, value in ((seed_key, identity), (global_key, global_identity)):
                if binding in identities and identities[binding] != value:
                    raise ValueError(
                        "paired schedule/task/protocol/runtime identity differs across results"
                    )
                identities[binding] = value
        seen.add(key)
        records.append(row)
    return records


def child(plan_path, receipt_path):
    # Accidental direct invocation is rejected; this is not a security boundary.
    if os.environ.get("FC_V6_PLAN_SHA") != digest(plan_path):
        raise ValueError("invoke through the reviewed 'run' guard")
    fd = int(os.environ["FC_V6_LOCK_FD"])
    plan, receipt_sha = acceptance(plan_path, receipt_path, with_binding=True)
    if receipt_sha != os.environ.get("FC_V6_ACCEPTANCE_SHA"):
        raise ValueError("acceptance receipt changed after guarded launch")
    lock = Path(plan["source_root"]).parent / "flycircuit/runs/.heavyjob.lock"
    opened, expected = os.fstat(fd), lock.stat()
    if (opened.st_ino, opened.st_dev) != (expected.st_ino, expected.st_dev):
        raise ValueError("inherited descriptor is not the shared project lock")
    path = Path(plan_path).parent / "rows.jsonl"
    previous = read_records(path, plan, receipt_sha)
    done = {x["key"] for x in previous}
    if any(x.get("status") != "ok" for x in previous):
        raise ValueError(
            "previous failure requires audit, not a failed-coordinate retry"
        )
    from .runner import execute_native

    stopped = False

    def on_signal(_signum, _frame):
        nonlocal stopped
        stopped = True

    def stop():
        if stopped:
            raise InterruptedError("guard interrupted this unrecorded coordinate")

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    for coord in plan["coordinates"]:
        if coord["key"] in done:
            continue
        stop()
        verify_plan(plan)
        row = {
            "key": coord["key"],
            "coordinate": coord,
            "plan_sha256": canonical(plan),
            "acceptance_sha256": receipt_sha,
        }
        try:
            row.update(
                status="ok", result=execute_native(coord, plan["source_root"], stop)
            )
            validate_row(row, coord, plan)
            verify_plan(plan)  # Recheck processed inputs after this coordinate too.
        except InterruptedError:
            return (
                4  # completed rows persist; interrupted row has no fabricated outcome
            )
        except Exception as exc:
            row.pop(
                "result", None
            )  # A rejected payload is not a successful measurement.
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            append(path, row)
            return 2
        append(path, row)
        print(json.dumps({"completed_coordinate": coord, "status": "ok"}), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--native-root", required=True)
    p.add_argument("--output", required=True)
    commands.add_parser("status")
    for name in ("audit", "run", "_child"):
        p = commands.add_parser(name)
        p.add_argument("--plan", required=True)
        p.add_argument("--acceptance", required=name != "audit")
    args = parser.parse_args(argv)
    if args.command == "status":
        print(json.dumps(status(), indent=2))
    elif args.command == "prepare":
        bundle = Path(__file__).resolve().parents[1]
        path = Path(args.output).resolve()
        if path.exists():
            raise ValueError("refusing to replace a prospective plan")
        plan = prepare(bundle, Path(args.native_root))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(plan, stream, indent=2, allow_nan=False)
        print(
            json.dumps(
                {
                    "plan": str(path),
                    "plan_sha256": digest(path),
                    "coordinates": len(plan["coordinates"]),
                    "starts_training": False,
                }
            )
        )
    elif args.command == "audit":
        plan = json.loads(Path(args.plan).read_text())
        verify_plan(plan)
        receipt_sha = None
        if args.acceptance:
            plan, receipt_sha = acceptance(
                args.plan, args.acceptance, with_binding=True
            )
        rows = read_records(Path(args.plan).parent / "rows.jsonl", plan, receipt_sha)
        print(
            json.dumps(
                {
                    "planned": len(plan["coordinates"]),
                    "saved": len(rows),
                    "failed": sum(x.get("status") != "ok" for x in rows),
                    "complete": len(rows) == len(plan["coordinates"])
                    and all(x["status"] == "ok" for x in rows),
                    "sota_established": False,
                }
            )
        )
    elif args.command == "run":
        return run(args.plan, args.acceptance)
    else:
        return child(args.plan, args.acceptance)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(2)
