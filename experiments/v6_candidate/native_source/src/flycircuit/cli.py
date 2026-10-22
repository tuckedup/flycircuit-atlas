"""flycircuit command line. Other subpackages may add commands via ``<pkg>.cli.register``."""

from __future__ import annotations

import os

# Limit numerical-library threads BEFORE numpy/torch import (laptop profile: 4).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import importlib
import json
import sys

from flycircuit import __version__

OPTIONAL_CLI_PACKAGES = ("data", "models", "plasticity", "tasks", "interventions", "eval", "envs")


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _cmd_doctor(args) -> int:
    from flycircuit.resources.doctor import run_doctor

    info = run_doctor(write=not args.no_write)
    _print(info)
    return 1 if info["problems"] and args.strict else 0


def _estimate(args):
    from flycircuit.resources.estimate import estimate_sparse_rnn

    return estimate_sparse_rnn(args.neurons, args.edges, args.batch, args.tbptt)


def _hw(args):
    from flycircuit.config import load_hardware

    return load_hardware(args.profile)


def _cmd_estimate(args) -> int:
    _print(_estimate(args))
    return 0


def _preflight(args, walltime=None):
    from flycircuit.config import project_root
    from flycircuit.resources.estimate import preflight
    from flycircuit.resources.probe import probe

    return preflight(_estimate(args), _hw(args), probe(project_root()),
                     new_storage_gib=args.storage_gib, use_cuda=not args.cpu_only,
                     walltime_minutes=walltime)


def _cmd_preflight(args) -> int:
    res = _preflight(args)
    _print({"estimate": _estimate(args), "preflight": res.to_dict()})
    if not res.ok:
        print("PREFLIGHT FAILED:\n  - " + "\n  - ".join(res.failures), file=sys.stderr)
        return 2
    return 0


def _cmd_dryrun(args) -> int:
    from flycircuit.resources.estimate import dry_run_benchmark

    hw = _hw(args)
    bench = dry_run_benchmark(args.neurons, args.edges, args.tbptt, args.trials,
                              args.conditions * args.seeds, hw)
    res = _preflight(args, walltime=bench["est_total_minutes"])
    _print({"estimate": _estimate(args), "benchmark": bench, "preflight": res.to_dict()})
    if not res.ok:
        print("DRY-RUN FAILED:\n  - " + "\n  - ".join(res.failures), file=sys.stderr)
        return 2
    return 0


def _cmd_synthetic(args) -> int:
    from flycircuit.synthetic_run import run_synthetic

    out = run_synthetic(seed=args.seed, steps=args.steps, use_mlflow=not args.no_mlflow)
    _print({"run_id": out["run_id"], "dir": out["dir"], "exit_status": out["manifest"]["exit_status"],
            "mlflow": out["manifest"]["mlflow"], "result": out["result"]})
    return 0 if out["manifest"]["exit_status"] == "ok" else 1


def _cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("flycircuit.api:app_factory", factory=True, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flycircuit", description="FlyCircuit Atlas (Atul Pandey)")
    p.add_argument("--version", action="version", version=f"flycircuit {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="inspect environment, write results/environment.json")
    d.add_argument("--no-write", action="store_true")
    d.add_argument("--strict", action="store_true", help="exit 1 if problems were found")
    d.set_defaults(func=_cmd_doctor)

    r = sub.add_parser("resources", help="resource estimate / preflight / dry-run")
    rs = r.add_subparsers(dest="rcmd", required=True)
    for name, fn in (("estimate", _cmd_estimate), ("preflight", _cmd_preflight),
                     ("dry-run", _cmd_dryrun)):
        s = rs.add_parser(name)
        s.add_argument("--neurons", type=int, required=True)
        s.add_argument("--edges", type=int, required=True)
        s.add_argument("--batch", type=int, default=1)
        s.add_argument("--tbptt", type=int, default=32)
        s.add_argument("--profile", default="atul_laptop")
        s.add_argument("--storage-gib", type=float, default=0.0)
        s.add_argument("--cpu-only", action="store_true")
        if name == "dry-run":
            s.add_argument("--trials", type=int, default=100, help="TBPTT windows per run")
            s.add_argument("--conditions", type=int, default=4)
            s.add_argument("--seeds", type=int, default=2)
        s.set_defaults(func=fn)
    from flycircuit.resources.cli import add_commands as _add_resource_commands

    _add_resource_commands(rs)  # live | profile | adaptive | jobs (measured adaptive profile)

    sy = sub.add_parser("synthetic", help="synthetic end-to-end pipeline (not biology)")
    sys_sub = sy.add_subparsers(dest="scmd", required=True)
    sr = sys_sub.add_parser("run")
    sr.add_argument("--seed", type=int, default=0)
    sr.add_argument("--steps", type=int, default=500)
    sr.add_argument("--no-mlflow", action="store_true")
    sr.set_defaults(func=_cmd_synthetic)

    a = sub.add_parser("api", help="local API")
    a_sub = a.add_subparsers(dest="acmd", required=True)
    sv = a_sub.add_parser("serve")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=_cmd_serve)

    for pkg in OPTIONAL_CLI_PACKAGES:
        try:
            mod = importlib.import_module(f"flycircuit.{pkg}.cli")
        except ModuleNotFoundError as exc:
            if exc.name != f"flycircuit.{pkg}.cli" and exc.name != f"flycircuit.{pkg}":
                raise
            continue
        mod.register(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except FileNotFoundError as exc:
        print(f"error: missing required file/artifact: {exc}", file=sys.stderr)
        return 3
    except NotImplementedError as exc:
        print(f"error: unsupported/not implemented: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
