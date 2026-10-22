"""`flycircuit data ...` subcommands. Mount with: cli_data.register(subparsers)."""

from __future__ import annotations

import argparse
import json
import sys


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", default="flywire")
    p.add_argument("--version", default="783")


def _sources(a) -> int:
    from . import sources as S

    rows = S.sources_for(a.dataset, a.version)
    man = S.load_manifest(a.dataset, a.version)
    for k, s in rows.items():
        print(f"{k:28s} tier={s.tier:15s} bytes={s.size} md5={s.md5} "
              f"downloaded={man.get(k, {}).get('success', False)}\n    {s.url}\n"
              f"    {s.record} | license: {s.license}")
    if a.verify:
        probs = S.verify_live_metadata()
        print("live Zenodo check:", "OK" if not probs else probs)
        return 1 if probs else 0
    return 0


def _fetch(a) -> int:
    from . import sources as S

    if a.dry_run:
        print(json.dumps(S.dry_run(a.dataset, a.version, a.tier), indent=2))
        return 0
    rc = 0
    for s in S.tier_sources(a.dataset, a.version, a.tier):
        e = S.download(a.dataset, a.version, s, min_free_gib=a.min_free_gib)
        print(f"{e.key}: success={e.success} size={e.actual_size} sha256={e.sha256} "
              f"err={e.error}")
        rc |= 0 if e.success else 1
    return rc


def _ingest(a) -> int:
    from .ingest import run_ingest

    rep = run_ingest(a.dataset, a.version, threshold=a.threshold, mem_gib=a.mem_gib)
    print(json.dumps(rep, indent=2))
    return 0


def _validate(a) -> int:
    from .validate import validate

    res = validate(a.dataset, a.version, recheck_hashes=a.recheck_hashes)
    for c in res:
        print(("PASS " if c.ok else "FAIL ") + c.name + (f" - {c.detail}" if c.detail else ""))
    return 0 if all(c.ok for c in res) else 1


def _select(a) -> int:
    from .selectors import SelectionError, select_circuit

    try:
        s = select_circuit(a.circuit, dataset=a.dataset, version=a.version, side=a.side,
                           pilot_max=a.pilot_max, seed=a.seed, threshold=a.threshold,
                           out_name=a.name)
    except SelectionError as e:
        print(f"selection failed: {e}", file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in s.items() if k != "groups"}, indent=2))
    return 0


def register(subparsers) -> None:
    d = subparsers.add_parser("data", help="connectome data: sources, fetch, ingest, validate")
    sub = d.add_subparsers(dest="data_cmd", required=True)

    p = sub.add_parser("sources", help="list verified sources")
    _common(p)
    p.add_argument("--verify", action="store_true", help="re-check live Zenodo metadata")
    p.set_defaults(func=_sources)

    p = sub.add_parser("fetch", help="download a tier (resumable, atomic)")
    _common(p)
    p.add_argument("--tier", default="proofread-pairs",
                   choices=["proofread-pairs", "neuropil-counts", "synapses"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--min-free-gib", type=float, default=40.0)
    p.set_defaults(func=_fetch)

    p = sub.add_parser("ingest", help="raw -> neurons/edges Parquet (external-memory)")
    _common(p)
    p.add_argument("--threshold", type=int, default=5)
    p.add_argument("--mem-gib", type=float, default=2.0)
    p.set_defaults(func=_ingest)

    p = sub.add_parser("validate", help="integrity checks on raw manifest + outputs")
    _common(p)
    p.add_argument("--recheck-hashes", action="store_true")
    p.set_defaults(func=_validate)

    p = sub.add_parser("select", help="annotation-aware circuit selection + boundary report")
    _common(p)
    p.add_argument("circuit", choices=["mushroom_body", "central_complex", "visual_motion",
                                       "feeding_reference", "whole_brain"])
    p.add_argument("--side", choices=["left", "right", "center"])
    p.add_argument("--pilot-max", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=int, default=5)
    p.add_argument("--name")
    p.set_defaults(func=_select)
