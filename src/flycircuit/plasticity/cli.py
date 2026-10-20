"""``flycircuit learning ...``: search / pilot / freeze / confirm / summary for the learning benchmark."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime

from flycircuit.config import project_root


def _load_bundle(circuit: str, proto: dict):
    from flycircuit.tasks.graph_loader import load_flywire_circuit, make_learning_synthetic

    if circuit == "synthetic":
        return make_learning_synthetic(seed=0)
    return load_flywire_circuit(proto["circuit"]["name"], min_contacts=proto["circuit"]["min_contacts"])


def _stage_name(circuit: str, stage: str) -> str:
    return f"{circuit}_{stage}"


def _cmd_stage(args) -> int:
    from flycircuit.eval import learning as L
    from flycircuit.tracking.manifest import RunRecorder

    if args.stage == "confirm":
        pre = (project_root() / "configs" / "prereg.yaml").read_text()
        if not re.search(r"^frozen: true", pre, re.M):
            print("error: prereg.yaml is not frozen; run `flycircuit learning freeze` first", flush=True)
            return 2
    proto = L.load_protocol()
    base = _load_bundle(args.circuit, proto)
    nulls = L.make_nulls(base, proto)
    stage = _stage_name(args.circuit, args.stage)
    seeds = proto["seeds"]["confirmatory"] if args.stage == "confirm" else proto["seeds"]["pilot"]
    with RunRecorder(f"learning-{stage}", proto, seeds, f"learning benchmark stage={stage}",
                     "fixed trial budget per phase; resumable records.jsonl",
                     data_release=base.notes.get("release"), mapping_hash=base.notes.get("mapping_hash"),
                     use_mlflow=not args.no_mlflow) as rec:
        sel_path = L.stage_dir(_stage_name(args.circuit, "search")) / "lr_selected.json"
        if args.stage == "search":
            sel = L.run_search(proto, base, nulls, stage=_stage_name(args.circuit, "search"))
        else:
            if not sel_path.exists():
                print(f"error: missing {sel_path}; run `flycircuit learning run --stage search` first")
                return 3
            sel = json.loads(sel_path.read_text())
            L.run_stage(stage, proto, base, nulls, sel, seeds, proto["seeds"]["null_graphs"])
            summ = L.write_summary(stage, proto, {"circuit_notes": base.notes,
                                                  "null_stats": [n.stats for n in nulls]})
            rec.metrics["n_records"] = summ["n_records"]
            rec.metrics["n_failed"] = summ["n_failed"]
            rec.add_artifact(L.stage_dir(stage) / "summary.json")
        rec.add_artifact(L.stage_dir(stage) / "records.jsonl")
    print(f"manifest: {rec.dir / 'manifest.json'}")
    return 0


def _cmd_summary(args) -> int:
    from flycircuit.eval import learning as L

    proto = L.load_protocol()
    stage = _stage_name(args.circuit, args.stage)
    s = L.write_summary(stage, proto)
    print(json.dumps({"stage": stage, "n_records": s["n_records"], "n_failed": s["n_failed"],
                      "targets": s["targets"]}, indent=2, default=str))
    return 0


def _cmd_freeze(args) -> int:
    """Freeze prereg.yaml: record date + protocol hash. Refuses to re-freeze (use amendments)."""
    from flycircuit.eval import learning as L

    p = project_root() / "configs" / "prereg.yaml"
    txt = p.read_text()
    if re.search(r"^frozen: true", txt, re.M):
        print("error: already frozen; append to `amendments` instead of re-freezing")
        return 2
    h = L.protocol_hash(L.load_protocol())
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    txt = txt.replace("status: draft_not_frozen", "status: frozen", 1)
    txt = re.sub(r"^frozen: false", "frozen: true", txt, count=1, flags=re.M)
    txt = re.sub(r"^frozen_date: null", f'frozen_date: "{today}"', txt, count=1, flags=re.M)
    txt = txt.replace("evaluation_protocol_hash: null   # filled at freeze",
                      f'evaluation_protocol_hash: "{h}"', 1)
    txt = txt.replace("    frozen: false\n    frozen_date: null",
                      f'    frozen: true\n    frozen_date: "{today}"\n    protocol_hash: "{h}"', 1)
    p.write_text(txt)
    print(f"prereg frozen {today}, protocol hash {h}")
    return 0


# ---- learning v2 --------------------------------------------------------------------------------
def _v2_ctx(args):
    from flycircuit.eval import learning_v2 as V

    proto = V.load_protocol()
    base = V.load_base(proto, args.circuit)
    packs = V.load_packs(base, proto, args.circuit)
    return V, proto, base, packs


def _v2_stage_name(circuit: str, stage: str) -> str:
    return stage if circuit == "flywire" else f"{circuit}_{stage}"


def _cmd2_stage(args) -> int:
    from flycircuit.tracking.manifest import RunRecorder

    V, proto, base, packs = _v2_ctx(args)
    V.set_time_budget(args.max_minutes)
    stage = _v2_stage_name(args.circuit, args.stage)
    if args.stage == "confirm":
        pre = (project_root() / "configs" / "prereg_v2.yaml").read_text()
        if not re.search(r"^frozen: true", pre, re.M):
            print("error: prereg_v2.yaml is not frozen; run `flycircuit learning2 freeze` first")
            return 2
        if V.protocol_hash(proto) not in pre:
            print("error: protocol changed since the freeze (hash mismatch); amend prereg_v2.yaml")
            return 2
    seeds = proto["seeds"]["confirmatory"] if args.stage == "confirm" else proto["seeds"]["pilot"]
    try:
        return _cmd2_body(args, V, proto, base, packs, stage, seeds, RunRecorder)
    except V.TimeBudget:
        print(f"time slice used up; stage {stage} is resumable (partial). exit 75")
        return 75


def _cmd2_body(args, V, proto, base, packs, stage, seeds, RunRecorder) -> int:
    with RunRecorder(f"learning2-{stage}", proto, seeds, f"learning v2 stage={stage}",
                     "fixed trial budgets; resumable records.jsonl",
                     data_release=base.notes.get("release"), mapping_hash=base.notes.get("mapping_hash"),
                     use_mlflow=not args.no_mlflow) as rec:
        sel_path = V.stage_dir(_v2_stage_name(args.circuit, "search")) / "lr_selected.json"
        try:
            partial = _cmd2_work(args, V, proto, base, packs, stage, seeds, rec, sel_path)
        except V.TimeBudget:
            partial = True
        rec.metrics["partial_time_slice"] = int(partial)
        rec.add_artifact(V.stage_dir(stage) / "records.jsonl")
    print(f"manifest: {rec.dir / 'manifest.json'}")
    if partial:
        print(f"time slice used up; stage {stage} is resumable (partial). exit 75")
        return 75
    return 0


def _cmd2_work(args, V, proto, base, packs, stage, seeds, rec, sel_path) -> bool:
    """Runs the stage; returns True if a time slice cut it short."""
    if args.stage == "search":
        V.run_search(proto, base, packs, stage=stage)
        return False
    if not sel_path.exists():
        raise SystemExit(f"error: missing {sel_path}; run ")
    sel = json.loads(sel_path.read_text())
    V.run_stage(stage, proto, base, packs, sel, seeds, small=args.stage == "pilot",
                save_ckpt=args.stage in ("confirm", "pilot"))
    summ = V.write_summary(stage, proto, {"circuit_notes": base.notes})
    rec.metrics["n_records"] = summ["n_records"]
    rec.add_artifact(V.stage_dir(stage) / "summary.json")
    return False


def _cmd2_summary(args) -> int:
    from flycircuit.eval import learning_v2 as V

    proto = V.load_protocol()
    s = V.write_summary(_v2_stage_name(args.circuit, args.stage), proto)
    print(json.dumps({"stage": s["stage"], "n_ok": s["n_ok"], "n_failed": s["n_failed"],
                      "targets": s["targets"]}, indent=2, default=str))
    return 0


def _cmd2_freeze(args) -> int:
    from flycircuit.eval import learning_v2 as V

    p = project_root() / "configs" / "prereg_v2.yaml"
    txt = p.read_text()
    if re.search(r"^frozen: true", txt, re.M):
        print("error: already frozen; append to `amendments` instead of re-freezing")
        return 2
    if V.read_records("confirm"):
        print("error: confirm records already exist; cannot freeze after seeing confirmatory data")
        return 2
    h = V.protocol_hash(V.load_protocol())
    today = datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ")
    txt = re.sub(r"^frozen: false", "frozen: true", txt, count=1, flags=re.M)
    txt = re.sub(r"^frozen_utc: null", f'frozen_utc: "{today}"', txt, count=1, flags=re.M)
    txt = re.sub(r"^protocol_hash: null", f'protocol_hash: "{h}"', txt, count=1, flags=re.M)
    p.write_text(txt)
    print(f"prereg_v2 frozen {today}, protocol hash {h}")
    return 0


def _cmd2_sensitivity(args) -> int:
    from flycircuit.eval import learning_v2 as V

    proto = V.load_protocol()
    V.set_time_budget(args.max_minutes)
    sel = json.loads((V.stage_dir("search") / "lr_selected.json").read_text())
    try:
        V.run_sensitivity(proto, sel)
    except V.TimeBudget:
        print("time slice used up; sensitivity stage is resumable. exit 75")
        return 75
    return 0


def register(sub) -> None:
    q = sub.add_parser("learning2", help="learning study v2 (protocol learning-v2.0)")
    qs = q.add_subparsers(dest="l2cmd", required=True)
    r2 = qs.add_parser("run", help="run a stage (search | pilot | confirm); resumable")
    r2.add_argument("--stage", choices=["search", "pilot", "confirm"], required=True)
    r2.add_argument("--circuit", choices=["synthetic", "flywire"], default="flywire")
    r2.add_argument("--no-mlflow", action="store_true")
    r2.add_argument("--max-minutes", type=float, default=None,
                    help="stop (resumably, exit 75) after this wall-clock slice")
    r2.set_defaults(func=_cmd2_stage)
    sn = qs.add_parser("sensitivity", help="sign-policy sensitivity (validation only, exploratory)")
    sn.add_argument("--max-minutes", type=float, default=None)
    sn.set_defaults(func=_cmd2_sensitivity)
    s2 = qs.add_parser("summary", help="(re)write summary.json for a stage")
    s2.add_argument("--stage", choices=["search", "pilot", "confirm"], required=True)
    s2.add_argument("--circuit", choices=["synthetic", "flywire"], default="flywire")
    s2.set_defaults(func=_cmd2_summary)
    f2 = qs.add_parser("freeze", help="freeze configs/prereg_v2.yaml (once, before confirm)")
    f2.set_defaults(func=_cmd2_freeze)

    p = sub.add_parser("learning", help="association/reversal/continual learning benchmark")
    ls = p.add_subparsers(dest="lcmd", required=True)
    r = ls.add_parser("run", help="run a stage (search | pilot | confirm); resumable")
    r.add_argument("--stage", choices=["search", "pilot", "confirm"], required=True)
    r.add_argument("--circuit", choices=["synthetic", "flywire"], default="synthetic")
    r.add_argument("--no-mlflow", action="store_true")
    r.set_defaults(func=_cmd_stage)
    s = ls.add_parser("summary", help="(re)write summary.json for a stage and print target status")
    s.add_argument("--stage", choices=["pilot", "confirm"], required=True)
    s.add_argument("--circuit", choices=["synthetic", "flywire"], default="synthetic")
    s.set_defaults(func=_cmd_summary)
    f = ls.add_parser("freeze", help="freeze configs/prereg.yaml (once)")
    f.set_defaults(func=_cmd_freeze)
