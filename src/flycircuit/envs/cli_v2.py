"""``flycircuit nav2 ...``: navigation v2 benchmark commands."""

from __future__ import annotations

import json
import sys


def _print(o) -> None:
    print(json.dumps(o, indent=2, default=float))


def _guard(fn):
    def run(args) -> int:
        try:
            return fn(args)
        except (RuntimeError, FileNotFoundError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    return run


@_guard
def _freeze(args) -> int:
    from flycircuit.eval import navigation_v2 as nv

    _print({"frozen_hash": nv.freeze_protocol()})
    return 0


@_guard
def _verify(args) -> int:
    from flycircuit.eval import navigation_v2 as nv

    v = nv.verify_protocol()
    _print(v)
    return 0 if v["frozen"] and v["match"] else 2


@_guard
def _check(args) -> int:
    """Learnability + leakage check on the VAL split (scripted controllers; no training)."""
    from flycircuit.eval import navigation_v2 as nv

    proto = nv.load_protocol()
    seeds = proto["seeds"]["search"] + proto["seeds"]["final"] + proto["seeds"]["dev_pilot"]
    out = {"protocol": nv.verify_protocol(), "leakage": nv.assert_no_leakage(proto, seeds),
           "scripted_val": {k: {c: round(v["success_rate"], 3) for c, v in row.items()}
                            for k, row in nv.scripted_references(proto, "val").items()}}
    _print(out)
    return 0


@_guard
def _search(args) -> int:
    from flycircuit.eval import navigation_v2 as nv

    p = nv.run_search(args.label, args.conditions or None)
    _print({k: v["lr"] for k, v in p["chosen"].items()})
    return 0


@_guard
def _run(args) -> int:
    from flycircuit.eval import navigation_v2 as nv

    p = nv.run_final(args.seeds or None, args.label, args.search_label, args.conditions or None)
    print(p["out_dir"])
    _print(p["summary"]["question_2_anatomical_advantage"])
    return 0


@_guard
def _sens(args) -> int:
    from flycircuit.eval import navigation_v2 as nv

    p = nv.run_v1_sensitivity(args.seeds or None, args.label, args.conditions or None)
    print(p["out_dir"])
    return 0


def register(sub) -> None:
    p = sub.add_parser("nav2", help="navigation benchmark v2 (conflict cues, memory, held-out perturbations)")
    s = p.add_subparsers(dest="nav2_cmd", required=True)
    s.add_parser("freeze", help="write the v2 protocol hash").set_defaults(func=_freeze)
    s.add_parser("verify", help="verify the frozen v2 protocol hash").set_defaults(func=_verify)
    s.add_parser("check", help="leakage + scripted learnability check (VAL split)").set_defaults(func=_check)
    r = s.add_parser("search", help="equal-budget lr search, validation split only")
    r.add_argument("--label", default="search")
    r.add_argument("--conditions", nargs="*")
    r.set_defaults(func=_search)
    r = s.add_parser("run", help="final matched comparison on the frozen protocol (test split)")
    r.add_argument("--label", default="final")
    r.add_argument("--search-label", default="search")
    r.add_argument("--seeds", type=int, nargs="*")
    r.add_argument("--conditions", nargs="*")
    r.set_defaults(func=_run)
    r = s.add_parser("sens-v1", help="v1 protocol re-run with annotation_v2 signs (exploratory)")
    r.add_argument("--label", default="v1_sens_annotation_v2")
    r.add_argument("--seeds", type=int, nargs="*")
    r.add_argument("--conditions", nargs="*")
    r.set_defaults(func=_sens)
