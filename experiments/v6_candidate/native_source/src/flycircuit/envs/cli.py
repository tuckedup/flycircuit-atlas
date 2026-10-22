"""``flycircuit nav ...``: Tier-1 navigation benchmark commands."""

from __future__ import annotations

import json
import sys


def _print(o) -> None:
    print(json.dumps(o, indent=2, default=float))


def _cmd_check(args) -> int:
    """Scripted-controller sanity: transitions + success criterion + references (no training)."""
    from flycircuit.envs.controllers import PrivilegedTeacher, RandomWalk, SensorScripted
    from flycircuit.eval import navigation as nv

    proto = nv.load_protocol()
    out = {"protocol": nv.verify_protocol(), "leakage": nv.assert_no_layout_leakage(proto)}
    for name, c in (("privileged_teacher", PrivilegedTeacher()), ("sensor_scripted", SensorScripted()),
                    ("random_action", RandomWalk(0))):
        r = nv.evaluate_controller(c, proto)
        out[name] = {k: round(v["success_rate"], 3) for k, v in r.items()}
    _print(out)
    return 0


def _cmd_freeze(args) -> int:
    from flycircuit.eval import navigation as nv

    _print({"frozen_hash": nv.freeze_protocol()})
    return 0


def _cmd_verify(args) -> int:
    from flycircuit.eval import navigation as nv

    v = nv.verify_protocol()
    _print(v)
    return 0 if v["frozen"] and v["match"] else 2


def _cmd_graph(args) -> int:
    from flycircuit.eval import navigation as nv

    try:
        g, _, meta = nv.load_mb_graph()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    meta["reach_readout_fraction"] = nv.reach_fraction(g)
    _print(meta)
    return 0


def _cmd_run(args) -> int:
    from flycircuit.eval import navigation as nv

    proto = nv.load_protocol()
    seeds = args.seeds or (proto["seeds"]["confirmatory"] if args.confirmatory
                           else proto["seeds"]["pilot_exploratory"])
    try:
        res = nv.run_benchmark(seeds, args.conditions, label="confirmatory" if args.confirmatory
                               else "pilot", exploratory=not args.confirmatory)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(res["out_dir"])
    print(json.dumps(res["summary"]["primary"], indent=2, default=float))
    return 0


def register(sub) -> None:
    p = sub.add_parser("nav", help="Tier-1 navigation benchmark (engineered arena)")
    s = p.add_subparsers(dest="nav_cmd", required=True)
    s.add_parser("check", help="scripted reference controllers + leakage check").set_defaults(func=_cmd_check)
    s.add_parser("freeze", help="write protocol hash (freeze)").set_defaults(func=_cmd_freeze)
    s.add_parser("verify", help="verify the frozen protocol hash").set_defaults(func=_cmd_verify)
    s.add_parser("graph", help="load/describe the real MB pilot graph").set_defaults(func=_cmd_graph)
    r = s.add_parser("run", help="train + evaluate all conditions (sequential)")
    r.add_argument("--seeds", type=int, nargs="*")
    r.add_argument("--conditions", nargs="*")
    r.add_argument("--confirmatory", action="store_true",
                   help="use the preregistered confirmatory seed list")
    r.set_defaults(func=_cmd_run)
    from flycircuit.envs.cli_v2 import register as register_v2

    register_v2(sub)
