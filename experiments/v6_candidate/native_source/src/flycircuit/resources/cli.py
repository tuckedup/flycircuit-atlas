"""`flycircuit resources live|profile|adaptive|jobs` (mounted by cli.py onto the `resources` group)."""

from __future__ import annotations

import json
import os
import sys


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _cmd_live(args) -> int:
    from flycircuit.resources.live import snapshot

    _print(snapshot(windows=not args.no_windows))
    return 0


def _cmd_profile(args) -> int:
    if args.rederive:  # offline: no benchmarking, no lock needed
        from pathlib import Path

        from flycircuit.config import project_root
        from flycircuit.resources.suite import rederive

        rederive(Path(args.rederive), Path(args.carry_from) if args.carry_from else None,
                 Path(args.out) if args.out else project_root() / "configs/hardware/adaptive_profile.yaml")
        return 0
    if os.environ.get("FLYCIRCUIT_HEAVY_HELD") != "1":
        print("error: the benchmark suite must hold the project heavy-job lock. Run\n"
              "  bash scripts/linux/bench_profile.sh [--quick]", file=sys.stderr)
        return 2
    from pathlib import Path

    from flycircuit.resources.suite import run_suite

    run_suite(quick=args.quick, out=Path(args.out) if args.out else None)
    return 0


def _cmd_adaptive(args) -> int:
    from flycircuit.resources.adaptive import adaptive_preflight

    res = adaptive_preflight(
        {"neurons": args.neurons, "edges": args.edges, "batch": args.batch, "tbptt": args.tbptt},
        use_cuda=not args.cpu_only, allow_downsize=args.allow_downsize)
    if not args.verbose:
        res = {k: v for k, v in res.items() if k != "snapshot"} | {
            "live": {"windows_free_gib": (res["snapshot"].get("windows") or {}).get("free_gib"),
                     "wsl_available_gib": res["snapshot"]["wsl"]["available_gib"],
                     "gpu_free_gib": res["snapshot"]["gpu"].get("free_gib")}}
    _print(res)
    if not res["ok"]:
        print("ADAPTIVE PREFLIGHT REFUSED:\n  - " + "\n  - ".join(res["failures"]) +
              ("\n  " + res["advice"] if res.get("advice") else "") +
              ("" if args.allow_downsize else
               "\n  (retry with --allow-downsize to get a smaller batch/tbptt that fits)"),
              file=sys.stderr)
        return 2
    if res.get("downsized"):
        r = res["request"]
        print(f"DOWNSIZED to batch={r['batch']} tbptt={r['tbptt']}: {res['advice']}", file=sys.stderr)
    return 0


def _cmd_jobs(args) -> int:
    from flycircuit.resources import registry

    if args.jcmd == "list":
        rows = registry.list_jobs(active_only=args.active)
        if args.json:
            _print(rows)
        else:
            for r in rows:
                m = r.get("measurements") or {}
                print(f"{r['job_id']}  {r['status']:8s} pid={r['pid']:<7d} {r['label']}  "
                      f"elapsed={r.get('elapsed_s', 0):.0f}s  rss={m.get('tree_peak_mem_gib')}")
        return 0
    if args.jcmd == "reap":
        print(f"marked lost: {registry.reap()}")
        return 0
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else list(args.cmd)
    if not cmd:
        print("error: give a command after --", file=sys.stderr)
        return 2
    try:
        return registry.run_tracked(args.label, json.loads(args.config) if args.config else {"argv": cmd},
                                    cmd, outputs=args.output)
    except registry.DuplicateJobError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 5


def add_commands(rs) -> None:
    s = rs.add_parser("live", help="live Windows/WSL/GPU/CPU snapshot")
    s.add_argument("--no-windows", action="store_true", help="skip the powershell.exe query")
    s.set_defaults(func=_cmd_live)

    s = rs.add_parser("profile", help="run the measured benchmark suite (holds heavy lock; use "
                                      "scripts/linux/bench_profile.sh)")
    s.add_argument("--quick", action="store_true")
    s.add_argument("--out", default=None)
    s.add_argument("--rederive", default=None, metavar="RAW_JSON",
                   help="rebuild the YAML from a stored raw JSON (no benchmarking)")
    s.add_argument("--carry-from", default=None, metavar="OLD_RAW_JSON",
                   help="with --rederive: carry phases the new run skipped from this older raw JSON")
    s.set_defaults(func=_cmd_profile)

    s = rs.add_parser("adaptive", help="live-memory preflight; refuse or downsize with advice")
    s.add_argument("--neurons", type=int, required=True)
    s.add_argument("--edges", type=int, required=True)
    s.add_argument("--batch", type=int, default=1)
    s.add_argument("--tbptt", type=int, default=32)
    s.add_argument("--cpu-only", action="store_true")
    s.add_argument("--allow-downsize", action="store_true")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=_cmd_adaptive)

    j = rs.add_parser("jobs", help="job registry (runs/jobs/registry.json)")
    js = j.add_subparsers(dest="jcmd", required=True)
    jl = js.add_parser("list")
    jl.add_argument("--active", action="store_true")
    jl.add_argument("--json", action="store_true")
    js.add_parser("reap", help="mark dead running jobs as lost")
    jr = js.add_parser("run", help="run a command as a tracked job: jobs run --label X -- cmd ...")
    jr.add_argument("--label", required=True)
    jr.add_argument("--config", default=None, help="JSON identity of the job for duplicate detection")
    jr.add_argument("--output", action="append", default=[], help="output path (conflict-checked)")
    jr.add_argument("cmd", nargs="*")
    j.set_defaults(func=_cmd_jobs)
    for p in (jl, jr):
        p.set_defaults(func=_cmd_jobs)
    js.choices["reap"].set_defaults(func=_cmd_jobs)
