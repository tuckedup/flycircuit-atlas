"""Portable, stdlib-only verification of the archived v6 validation study."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import fmean, stdev

PLAN_SHA = "29d859643e32b508058cdabcba279010fb6af7e43aac3ea7fd82921d48498f5b"
METHODS = ("standard", "extra_current", "uniform_replay", "balanced_replay")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def paired(a, b):
    if set(a) != set(b) or len(a) != 8:
        raise ValueError("paired comparison requires all eight identical model seed IDs")
    delta = [a[s] - b[s] for s in sorted(a)]
    mean = fmean(delta)
    half = 2.364624251 * stdev(delta) / math.sqrt(8)
    return {"mean": mean, "ci95": [mean - half, mean + half], "n": 8,
            "interval": "paired t, df=7, descriptive, no multiplicity correction"}


def recompute(root):
    root = Path(root).resolve()
    evidence = root / "results_v6/native_validation"
    plan_path = evidence / "plan.json"
    if digest(plan_path) != PLAN_SHA:
        raise ValueError("original prospective plan bytes changed")
    plan = json.loads(plan_path.read_text())
    mapping = json.loads((root / "provenance/V6_SOURCE_MAP.json").read_text())
    if set(mapping) != set(plan["source_sha256"]):
        raise ValueError("archived source closure differs from the frozen plan")
    for original, relative in mapping.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or digest(path) != plan["source_sha256"][original]:
            raise ValueError(f"archived source mismatch: {relative}")
    receipt = json.loads((evidence / "acceptance.json").read_text())
    if receipt.get("verdict") != "ship" or receipt.get("plan_sha256") != PLAN_SHA or receipt.get("input_sha256") != plan["input_sha256"]:
        raise ValueError("receipt does not bind the frozen input inventory and plan")
    if receipt.get("native_smoke_test") != "passed" or receipt.get("seed_exposure_review") != "passed":
        raise ValueError("missing recorded runtime acceptance gates")
    for field, filename in (("smoke_receipt_sha256", "smoke_receipt.json"), ("exposure_check_sha256", "exposure_check.json")):
        if receipt["evidence"][field] != digest(evidence / "smoke" / filename):
            raise ValueError(f"acceptance evidence changed: {filename}")
    review_sha = digest(evidence / "smoke/independent_review.md")
    if review_sha not in receipt["reviewer"]:
        raise ValueError("accepted review bytes do not match recorded identity")
    sys.path.insert(0, str(root / "experiments/v6_candidate"))
    from flycircuit_v6.__main__ import read_records

    rows = read_records(evidence / "rows.jsonl", plan, canonical(receipt))
    if len(rows) != 64 or {r["key"] for r in rows} != {c["key"] for c in plan["coordinates"]} or any(r["status"] != "ok" for r in rows):
        raise ValueError("v6 matrix is incomplete, duplicated, unexpected or failed")
    ledger = json.loads((evidence / "resource_ledger.json").read_text())
    if ledger["plan_sha256"] != PLAN_SHA or ledger["acceptance_sha256"] != canonical(receipt) or ledger["attempts"] != 1 or ledger["inflight"] is not None or ledger["last_result"] != {"reason": None, "child_rc": 0}:
        raise ValueError("execution ledger differs from the single completed accepted run")
    if not 0 < ledger["charged_wall_s"] <= plan["protocol"]["wall_limit_s"]:
        raise ValueError("measured validation run exceeded its prospective wall cap")
    groups, metrics = {}, ("final_accuracy", "learned_accuracy", "forgetting")
    for row in rows:
        coordinate, result = row["coordinate"], row["result"]
        matrix = result["R_val"]
        # Independent arithmetic from raw matrices, not the stored summaries.
        n = len(matrix)
        values = {"final_accuracy": fmean(matrix[-1]),
                  "learned_accuracy": fmean(matrix[i][i] for i in range(n)),
                  "forgetting": fmean(max(matrix[i][j] for i in range(j, n - 1)) - matrix[-1][j] for j in range(n - 1))}
        groups.setdefault((coordinate["family"], coordinate["method"]), {})[coordinate["seed"]] = values
    if set(groups) != {(f, m) for f in ("connectome", "gru33") for m in METHODS} or any(set(v) != set(range(26001, 26009)) for v in groups.values()):
        raise ValueError("family/method/seed design differs from the declared eight-seed study")
    summary = {"scope": "exploratory validation-only native cue task", "n_rows": 64, "failures": 0,
               "n_seeds": 8, "plan_sha256": PLAN_SHA, "families": {}, "paired": {},
               "peak_row_rss_mib": max(r["result"]["peak_rss_mib"] for r in rows),
               "charged_wall_s": ledger["charged_wall_s"], "sota_established": False,
               "body_robustness_established": False, "heldout_evaluation": False}
    for family in ("connectome", "gru33"):
        summary["families"][family] = {m: {k: {"mean": fmean(v[k] for v in groups[family, m].values()), "sd": stdev(v[k] for v in groups[family, m].values())} for k in metrics} for m in METHODS}
        for method in METHODS[1:]:
            for baseline in ("standard", "extra_current"):
                if method == baseline:
                    continue
                key = f"{family}:{method}-{baseline}"
                summary["paired"][key] = {k: paired({s: v[k] for s, v in groups[family, method].items()}, {s: v[k] for s, v in groups[family, baseline].items()}) for k in metrics}
    for method in METHODS:
        summary["paired"][f"connectome-gru33:{method}"] = {k: paired({s: v[k] for s, v in groups["connectome", method].items()}, {s: v[k] for s, v in groups["gru33", method].items()}) for k in metrics}
    return summary


def table(summary):
    lines = ["| Family | Standard | Extra current steps | Uniform replay | Balanced replay |", "|---|---:|---:|---:|---:|"]
    for family, label in (("connectome", "Connectome W76"), ("gru33", "GRU33")):
        lines.append("| " + label + " | " + " | ".join(f"{summary['families'][family][m]['final_accuracy']['mean'] * 100:.1f}%" for m in METHODS) + " |")
    return "\n".join(lines)


def verify(root):
    root = Path(root).resolve()
    summary = recompute(root)
    saved = json.loads((root / "results_v6/native_validation/public_summary.json").read_text())
    if canonical(saved) != canonical(summary):
        raise ValueError("public summary differs from the raw matrices")
    for relative in ("README.md", "docs/V6_VALIDATION.md"):
        if table(summary) not in (root / relative).read_text():
            raise ValueError(f"v6 table drift: {relative}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--write-summary", action="store_true", help="maintainer: generate summary from immutable records")
    args = parser.parse_args(argv)
    try:
        if args.write_summary:
            root = Path(args.root).resolve()
            summary = recompute(root)
            (root / "results_v6/native_validation/public_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            print(table(summary))
        else:
            summary = verify(args.root)
            print(table(summary))
            print("v6 verification PASSED: 64/64 accepted validation rows, no SOTA or body claim")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"v6 verification FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
