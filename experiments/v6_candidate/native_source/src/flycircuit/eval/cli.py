"""``flycircuit refcheck ...``: LIF reference validation (mounted by the optional-CLI hook)."""

from __future__ import annotations

import json


def _cmd_prepare(args) -> int:
    from flycircuit.eval.refcheck import prepare

    print(json.dumps(prepare(), indent=1))
    return 0


def _cmd_tolerances(args) -> int:
    from flycircuit.eval.refcheck import load_protocol

    proto, sha = load_protocol()
    print(json.dumps({"sha256": sha, "tiers": proto["tiers"], "amendments": proto["amendments"]},
                     indent=1, default=str))
    return 0


def _cmd_integrity(args) -> int:
    from flycircuit.eval.refcheck import integrity

    res = integrity()
    print(json.dumps(res, indent=1, default=str))
    return 1 if res["n_fail"] else 0


def _cmd_run(args) -> int:
    from flycircuit.eval.refcheck import render_markdown, run_all

    tiers = tuple(int(t) for t in args.tiers.split(",") if t)
    res = run_all(tiers, skip_brian=args.skip_brian, boundary=not args.no_boundary,
                  do_integrity=not args.no_integrity)
    if args.render:
        from pathlib import Path

        from flycircuit.config import project_root

        p = Path(project_root()) / "docs" / "refcheck_results.md"
        p.write_text(render_markdown(res))
        print(f"wrote {p}")
    statuses = {k: res[k]["status"] for k in ("tier2", "tier3") if k in res}
    if "tier1" in res:
        statuses["tier1"] = {k: v["status"] for k, v in res["tier1"].items()}
    print(json.dumps({"run_id": res["run_id"], "statuses": statuses,
                      "integrity_fail": res.get("integrity", {}).get("n_fail")}, indent=1))
    return 0


def register(sub) -> None:
    p = sub.add_parser("refcheck", help="LIF reference engine vs published Brian2 model")
    s = p.add_subparsers(dest="rcmd", required=True)
    s.add_parser("prepare", help="derive Shiu per-neuron sign table").set_defaults(func=_cmd_prepare)
    s.add_parser("tolerances", help="print predeclared protocol").set_defaults(func=_cmd_tolerances)
    s.add_parser("integrity", help="real-data integrity checks").set_defaults(func=_cmd_integrity)
    r = s.add_parser("run", help="run the comparison tiers (bounded)")
    r.add_argument("--tiers", default="1,2,3")
    r.add_argument("--skip-brian", action="store_true")
    r.add_argument("--no-boundary", action="store_true")
    r.add_argument("--no-integrity", action="store_true")
    r.add_argument("--render", action="store_true", help="write docs/refcheck_results.md")
    r.set_defaults(func=_cmd_run)
