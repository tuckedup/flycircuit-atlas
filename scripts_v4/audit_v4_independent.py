"""Stdlib-only audit of the v4 pilot (Part A), W0 width probe (Part B) and provenance (Part C).

Read-only on all inputs. Writes ABSOLUTE-path outputs atomically to results_v4/audit_v4/{audit.json,audit.md}.
Every check status is computed (pass / fail / incomplete / not_logged). NOTE: this is the author's own
re-implementation of the arithmetic; it is a consistency audit, NOT an independent verification.
"""
import ast
import datetime
import hashlib
import json
import math
import os
import re
import sys

ROOT = "<PROJECT_ROOT>"
CANON = "<CANONICAL_ROOT>"
P = ROOT + "/results_v4/learning_v4"
W = ROOT + "/results_v4/learning_readout_v4_1"
OUTD = ROOT + "/results_v4/audit_v4"
TOL = 1e-9
CHECKS = []


def add(cid, part, desc, status, required=True, details=None):
    assert status in ("pass", "fail", "incomplete", "not_logged", "informational"), status
    CHECKS.append({"id": cid, "part": part, "description": desc, "status": status,
                   "required": required, "details": details if details is not None else {}})


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mtime(path):
    return datetime.datetime.fromtimestamp(os.stat(path).st_mtime, datetime.timezone.utc).isoformat()


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def mean(v):
    return math.fsum(v) / len(v)


def se(v):
    n = len(v)
    if n < 2:
        return float("nan")
    m = mean(v)
    return math.sqrt(math.fsum((x - m) ** 2 for x in v) / (n - 1)) / math.sqrt(n)


def close(a, b, tol=TOL):
    return a is not None and b is not None and abs(a - b) <= tol


def fk(x):
    return format(float(x), ".9g")


class Cmp:
    """Accumulates comparisons of recomputed values against logged fields."""
    def __init__(self):
        self.n = 0
        self.mismatch = []
        self.absent = []

    def num(self, label, mine, logged, tol=TOL):
        if logged is None:
            self.absent.append(label)
            return
        self.n += 1
        if not close(mine, logged, tol):
            self.mismatch.append({"field": label, "recomputed": mine, "logged": logged})

    def eq(self, label, mine, logged):
        if logged is None:
            self.absent.append(label)
            return
        self.n += 1
        if mine != logged:
            self.mismatch.append({"field": label, "recomputed": mine, "logged": logged})

    def status(self):
        if self.mismatch:
            return "fail"
        if self.absent:
            return "incomplete"
        return "pass" if self.n else "not_logged"

    def details(self):
        return {"n_compared": self.n, "mismatches": self.mismatch, "absent_fields": self.absent}


def scan_keys(obj, path="", out=None):
    """Recursive scan for test-split keys. Config words 'test' in values (other than splits lists) are ignored."""
    out = [] if out is None else out
    pat = re.compile(r"(^|[^a-z])(r_?test|test)([^a-z]|$)", re.I)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if pat.search(str(k)):
                out.append(path + "/" + str(k))
            if k == "splits" and isinstance(v, list) and any(str(s).lower().startswith("test") for s in v):
                out.append(path + "/splits=" + json.dumps(v))
            if str(k).lower() in ("split", "splits"):
                vals = v if isinstance(v, list) else [v]
                if any(isinstance(x, str) and x.strip().lower() == "test" for x in vals):
                    out.append(path + "/" + str(k) + "=value 'test'")
            scan_keys(v, path + "/" + str(k), out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5000]):
            scan_keys(v, path + f"[{i}]", out)
    return out


# ======================================================================= PART A
def part_a():
    sr = jl(P + "/search_records.jsonl")
    cr = jl(P + "/check_records.jsonl")
    choices = json.load(open(P + "/choices.json"))
    pman = json.load(open(P + "/pilot_manifest.json"))
    rep = json.load(open(P + "/pilot_report.json"))
    ARMS = ["connectome", "gru", "encdec"]
    RATIOS = [1 / 9, 1 / 3, 1.0, 3.0, 9.0]
    anchors = rep.get("v3_lrs")
    if not anchors:
        add("A.anchors", "A", "v3 anchors present in pilot_report (Part A aborted early if missing)", "incomplete")
        return None
    grid = {f: [float(f"{anchors[f] * r:.6g}") for r in RATIOS] for f in ARMS}
    exp_search = {(f, fk(c), fk(d), s) for f in ARMS for c in grid[f] for d in grid[f] for s in range(501, 505)}
    got = [(r["family"], fk(r["core_lr"]), fk(r["decoder_lr"]), r["seed"]) for r in sr]
    d = {"expected": len(exp_search), "records": len(got), "unique_records": len(set(got)),
         "missing": len(exp_search - set(got)), "unexpected": len(set(got) - exp_search),
         "bad_status": sum(r.get("status") != "ok" for r in sr),
         "arm_not_bptt": sum(r.get("arm") != "bptt" for r in sr)}
    man_c = pman.get("identity", {}).get("expected_search_coordinates")
    if man_c is None:
        d["manifest_coords"] = "not_logged"
    else:
        ms = {(c["family"], fk(c["core_lr"]), fk(c["decoder_lr"]), c["seed"]) for c in man_c}
        d["manifest_coords_equal_expected"] = ms == exp_search
    d["report_grid_equals_formula"] = all([fk(x) for x in rep["arms"][f]["grid"]] == [fk(x) for x in grid[f]]
                                          for f in ARMS)
    ok = (d["expected"] == 300 and d["records"] == 300 and d["unique_records"] == 300 and d["missing"] == 0
          and d["unexpected"] == 0 and d["bad_status"] == 0 and d["arm_not_bptt"] == 0
          and d.get("manifest_coords_equal_expected", True) is True and d["report_grid_equals_formula"])
    add("A1.search_coverage", "A", "300 expected search coords, all ok, no duplicates", "pass" if ok else "fail",
        details=d)
    splits_bad = [r for r in sr + cr if r.get("splits") != ["val"]]
    add("A2.val_only_splits", "A", "all search+check records have splits == ['val']",
        "pass" if not splits_bad else "fail", details={"bad": len(splits_bad)})

    # rules
    def table(fam, diag, comp):
        t = {}
        for r in sr:
            if r["family"] != fam or r.get("status") != "ok":
                continue
            if diag and fk(r["core_lr"]) != fk(r["decoder_lr"]):
                continue
            a, c = r["association"]["val_acc_acq"], r["continual"]["final_mean_acc_val"]
            t.setdefault((r["core_lr"], r["decoder_lr"]), []).append(((a + c) / 2) if comp else c)
        return t

    mine = {}
    rule_details = {}
    for fam in ARMS:
        base = sorted({r["core_lr"] for r in sr if r["family"] == fam})[2]   # middle grid value
        lo, hi = min(r["core_lr"] for r in sr if r["family"] == fam), max(r["core_lr"] for r in sr if r["family"] == fam)
        dlo, dhi = min(r["decoder_lr"] for r in sr if r["family"] == fam), max(r["decoder_lr"] for r in sr if r["family"] == fam)

        def key(k):
            rc, rd = k[0] / base, k[1] / base
            return (max(rc, rd), math.sqrt(rc * rd), k[0], k[1])

        mine[fam] = {}
        for rule in "ABCD":
            t = table(fam, rule in "AB", rule in "BC")
            means = {k: mean(v) for k, v in t.items()}
            ses = {k: se(v) for k, v in t.items()}
            top = max(means.values())
            tied = [k for k in means if abs(means[k] - top) <= 1e-12]
            best = min(tied, key=key)
            info = {"n_settings": len(means), "n_seeds_min": min(len(v) for v in t.values()),
                    "n_top_ties(1e-12)": len(tied), "best": list(best), "best_mean": means[best]}
            if rule in "AD":
                chosen = best
            else:
                thr = means[best] - ses[best]
                cands = [k for k in means if means[k] >= thr]
                chosen = min(cands, key=key)
                margins = sorted((abs(means[k] - thr), list(k), means[k]) for k in means)
                info.update(threshold=thr, best_se=ses[best], n_indistinguishable=len(cands),
                            chosen_equals_best=chosen == best,
                            min_boundary_margin=margins[0][0],
                            boundary_margin_flag_lt_1e9=margins[0][0] < 1e-9,
                            nearest_boundary_settings=[{"setting": m[1], "mean": m[2], "margin": m[0]} for m in margins[:3]])
            info.update(chosen=list(chosen), chosen_mean=means[chosen],
                        edge_flags={"chosen_core_at_grid_edge": chosen[0] in (lo, hi),
                                    "chosen_decoder_at_grid_edge": chosen[1] in (dlo, dhi),
                                    "best_core_at_grid_edge": best[0] in (lo, hi),
                                    "best_decoder_at_grid_edge": best[1] in (dlo, dhi)},
                        base_lr_middle_grid=base)
            mine[fam][rule] = info
        rule_details[fam] = mine[fam]
    # compare to choices.json and report choices
    c1 = Cmp()
    for fam in ARMS:
        for rule in "ABCD":
            m = mine[fam][rule]
            ch = choices.get(fam, {}).get(rule, {})
            rc = rep["arms"][fam]["choices"].get(rule, {})
            for src, lg in (("choices.json", ch), ("pilot_report.choices", rc)):
                p = f"{fam}.{rule}.{src}."
                c1.num(p + "core_lr", m["chosen"][0], lg.get("core_lr"), 1e-12)
                c1.num(p + "decoder_lr", m["chosen"][1], lg.get("decoder_lr"), 1e-12)
                c1.num(p + "mean_score", m["chosen_mean"], lg.get("mean_score"))
                c1.eq(p + "n_settings", m["n_settings"], lg.get("n_settings"))
                if rule in "BC":
                    c1.eq(p + "n_indistinguishable", m["n_indistinguishable"], lg.get("n_indistinguishable"))
                    c1.eq(p + "chosen_equals_best", m["chosen_equals_best"], lg.get("chosen_equals_best"))
    add("A3.rules_ABCD_reproduce", "A", "rules A-D recomputed from search records match choices.json and pilot_report.choices",
        c1.status(), details=dict(c1.details(), rules=rule_details))
    flags = [f"{f}.{r}" for f in ARMS for r in "BC" if mine[f][r]["boundary_margin_flag_lt_1e9"]]
    add("A4.one_se_boundary_margin", "A", "no one-SE threshold boundary within 1e-9 of any setting mean",
        "pass" if not flags else "fail", details={"flagged": flags,
        "min_margins": {f"{f}.{r}": mine[f][r]["min_boundary_margin"] for f in ARMS for r in "BC"}})
    edge = {f"{f}.{r}": mine[f][r]["edge_flags"] for f in ARMS for r in "ABCD"}
    add("A5.grid_edge_flags", "A", "INFORMATIONAL (not required): grid-edge flags computed from actual searched coordinates (descriptive; "
        "computed, edge hits are listed not failed)", "informational",
        required=False, details={"edge_flags": edge,
                                 "any_chosen_at_edge": sorted(k for k, v in edge.items()
                                                              if v["chosen_core_at_grid_edge"] or v["chosen_decoder_at_grid_edge"])})

    # check coords
    exp_check = set()
    for f in ARMS:
        uniq = {tuple(mine[f][r]["chosen"]) for r in "ABCD"}
        for c, dd in uniq:
            for s in range(601, 605):
                exp_check.add((f, fk(c), fk(dd), s))
    gotc = [(r["family"], fk(r["core_lr"]), fk(r["decoder_lr"]), r["seed"]) for r in cr]
    json_uniq = set()
    for f in ARMS:
        for r in "ABCD":
            json_uniq.add((f, fk(choices[f][r]["core_lr"]), fk(choices[f][r]["decoder_lr"])))
    dd = {"expected": len(exp_check), "records": len(gotc), "unique": len(set(gotc)),
          "missing": len(exp_check - set(gotc)), "unexpected": len(set(gotc) - exp_check),
          "bad_status": sum(r.get("status") != "ok" for r in cr),
          "unique_settings_from_choices_json_equal": {(a, b, c) for a, b, c, _ in exp_check} == json_uniq}
    ok = (dd["expected"] == 32 and dd["records"] == 32 and dd["unique"] == 32 and dd["missing"] == 0
          and dd["unexpected"] == 0 and dd["bad_status"] == 0 and dd["unique_settings_from_choices_json_equal"])
    add("A6.check_coverage", "A", "32 deduplicated fresh-check coords (601-604 x unique selected setting)",
        "pass" if ok else "fail", details=dd)

    # leakage scan
    skip = {"source_files", "canonical_input_inventory", "config_files"}
    pm2 = {"identity": {k: v for k, v in pman.get("identity", {}).items() if k not in skip}}
    hits = {"search_records": scan_keys(sr), "check_records": scan_keys(cr), "choices": scan_keys(choices),
            "pilot_report": scan_keys(rep), "pilot_manifest(excl path inventories)": scan_keys(pm2)}
    n_hit = sum(len(v) for v in hits.values())
    add("A7.no_test_split_leakage", "A", "recursive scan: no test-split keys / R_test / splits containing test",
        "pass" if n_hit == 0 else "fail", details={"hits": hits, "note": "values equal to the word 'test' (e.g. config text) are not treated as leakage; only key names and 'splits' lists"})

    # check results & report
    ck = {}
    for fam in ARMS:
        ck[fam] = {}
        for rule in "ABCD":
            c, dc = mine[fam][rule]["chosen"]
            rows = sorted((r for r in cr if r["family"] == fam and fk(r["core_lr"]) == fk(c) and fk(r["decoder_lr"]) == fk(dc)),
                          key=lambda r: r["seed"])
            ck[fam][rule] = {
                "seeds": [r["seed"] for r in rows],
                "acq": {r["seed"]: r["association"]["val_acc_acq"] for r in rows},
                "cont": {r["seed"]: r["continual"]["final_mean_acc_val"] for r in rows},
                "forget": {r["seed"]: r["continual"]["avg_forgetting_keep_val"] for r in rows},
                "rows": rows}
    c2 = Cmp()
    for fam in ARMS:
        for rule in "ABCD":
            e = ck[fam][rule]
            lg = rep["arms"][fam]["check"].get(rule, {})
            p = f"{fam}.{rule}."
            c2.eq(p + "n", len(e["seeds"]), lg.get("n"))
            for nm, lk in (("acq", "acq"), ("cont", "cont"), ("forget", "forget_keep")):
                vals = [e[nm][s] for s in e["seeds"]]
                if lg.get(lk) is None:
                    c2.absent.append(p + lk)
                else:
                    c2.num(p + lk + ".mean", mean(vals), lg[lk][0])
                    c2.num(p + lk + ".se", se(vals), lg[lk][1])
                ps = lg.get("per_seed_" + nm)
                if ps is None:
                    c2.absent.append(p + "per_seed_" + nm)
                else:
                    for s in e["seeds"]:
                        c2.num(p + f"per_seed_{nm}.{s}", e[nm][s], ps.get(str(s), ps.get(s)))
            rows = e["rows"]
            c2.num(p + "diag_clip_fraction", mean([r["diag"].get("clip_fraction") or 0 for r in rows]), lg.get("diag_clip_fraction"))
            med = sorted(r["diag"].get("grad_norm_median") or 0 for r in rows)
            medv = (med[len(med) // 2 - 1] + med[len(med) // 2]) / 2 if len(med) % 2 == 0 else med[len(med) // 2]
            c2.num(p + "diag_grad_norm_median", medv, lg.get("diag_grad_norm_median"))
            c2.eq(p + "diag_nonfinite", int(sum(r["diag"].get("nonfinite", 0) for r in rows)), lg.get("diag_nonfinite"))
            acq_ge = mean([e["acq"][s] for s in e["seeds"]]) >= 0.82
            c2.eq(p + "descriptive_acquisition_ge_0_82", acq_ge, lg.get("descriptive_acquisition_ge_0_82"))
        # search-grid noise stats
        t = table(fam, False, True)
        means = {k: mean(v) for k, v in t.items()}
        ses = {k: se(v) for k, v in t.items()}
        best = max(means, key=means.get)
        lg = rep["arms"][fam]
        c2.num(f"{fam}.best_composite.mean", means[best], lg.get("best_composite_setting", {}).get("mean"))
        c2.num(f"{fam}.best_composite.se", ses[best], lg.get("best_composite_setting", {}).get("se"))
        c2.eq(f"{fam}.n_within_1se_of_best_full_grid", sum(m >= means[best] - ses[best] for m in means.values()),
              lg.get("n_within_1se_of_best_full_grid"))
        c2.num(f"{fam}.mean_seed_se_over_grid", mean(list(ses.values())), lg.get("mean_seed_se_over_grid"))
    add("A8.check_stats_vs_report", "A", "fresh-seed means/SEs/per-seed maps/diag fields recomputed vs pilot_report",
        c2.status(), details=c2.details())

    # paired forgetting gap & acquisition
    gaps = {}
    c3 = Cmp()
    for rule in "ABCD":
        cs, gs = ck["connectome"][rule], ck["gru"][rule]
        seeds = [s for s in cs["seeds"] if s in gs["seeds"]]
        fd = [gs["forget"][s] - cs["forget"][s] for s in seeds]
        ad = [cs["acq"][s] - gs["acq"][s] for s in seeds]
        od = [cs["cont"][s] - gs["cont"][s] for s in seeds]
        g = {"paired_seeds": seeds, "per_seed_gru_minus_conn_forgetting": dict(zip(seeds, fd)),
             "mean_gru_minus_conn_forgetting": mean(fd), "se": se(fd),
             "signed_ge_0_07": mean(fd) >= 0.07,
             "conn_acq_mean": mean([cs["acq"][s] for s in cs["seeds"]]), "gru_acq_mean": mean([gs["acq"][s] for s in gs["seeds"]]),
             "conn_acq_ge_0_82": mean([cs["acq"][s] for s in cs["seeds"]]) >= 0.82,
             "gru_acq_ge_0_82": mean([gs["acq"][s] for s in gs["seeds"]]) >= 0.82,
             "per_seed_conn_minus_gru_acq": dict(zip(seeds, ad)), "per_seed_conn_minus_gru_cont": dict(zip(seeds, od))}
        gaps[rule] = g
        lg = rep.get("gru_minus_connectome_forgetting_fresh_val_seeds", {}).get(rule, {})
        c3.num(f"{rule}.gap_mean", g["mean_gru_minus_conn_forgetting"], lg.get("mean_diff_gru_minus_conn"))
        c3.num(f"{rule}.gap_se", g["se"], lg.get("se"))
        c3.eq(f"{rule}.gap_n", len(seeds), lg.get("n"))
        c3.eq(f"{rule}.gap_ge_0_07", g["signed_ge_0_07"], lg.get("descriptive_gru_minus_conn_improvement_ge_0_07"))
        lg2 = rep.get("connectome_minus_gru_fresh_val_seeds", {}).get(rule, {})
        c3.num(f"{rule}.acq_diff_mean", mean(ad), lg2.get("per_seed_acq", {}).get("mean_diff_conn_minus_gru"))
        c3.num(f"{rule}.acq_diff_se", se(ad), lg2.get("per_seed_acq", {}).get("se"))
        c3.num(f"{rule}.cont_diff_mean", mean(od), lg2.get("per_seed_cont", {}).get("mean_diff_conn_minus_gru"))
        c3.num(f"{rule}.cont_diff_se", se(od), lg2.get("per_seed_cont", {}).get("se"))
    add("A9.paired_forgetting_gap_vs_report", "A", "signed paired GRU-minus-connectome forgetting (seeds 601-604) vs report",
        c3.status(), details=dict(c3.details(), gaps=gaps))
    # expected known values (from task statement)
    exp = {"A": (75.00, 87.89, -11.72), "B": (75.00, 87.11, -2.54), "C": (73.83, 87.11, 1.76), "D": (73.83, 87.11, 1.76)}
    mm = []
    for r, (ca, ga, gp) in exp.items():
        g = gaps[r]
        got3 = (round(g["conn_acq_mean"] * 100, 2), round(g["gru_acq_mean"] * 100, 2), round(g["mean_gru_minus_conn_forgetting"] * 100, 2))
        if any(abs(x - y) > 0.006 for x, y in zip(got3, (ca, ga, gp))):
            mm.append({"rule": r, "got": got3, "expected": (ca, ga, gp)})
    se_c = round(gaps["C"]["se"] * 100, 3)
    if abs(se_c - 0.667) > 0.0006:
        mm.append({"rule": "C", "se_pp": se_c, "expected": 0.667})
    add("A10.known_expected_values", "A", "regression pin to author-supplied expected values, not verification (recomputed acq/gap vs values supplied in task, pp, 2 dp)",
        "informational" if not mm else "fail", required=False, details={"mismatches": mm, "recomputed_pp": {r: [round(gaps[r]["conn_acq_mean"] * 100, 2), round(gaps[r]["gru_acq_mean"] * 100, 2), round(gaps[r]["mean_gru_minus_conn_forgetting"] * 100, 2)] for r in "ABCD"}, "C_se_pp": se_c})
    same = {f: {"C_equals_D": mine[f]["C"]["chosen"] == mine[f]["D"]["chosen"],
                "A_equals_B": mine[f]["A"]["chosen"] == mine[f]["B"]["chosen"]} for f in ARMS}
    add("A11.rule_independence_note", "A", "INFORMATIONAL (not required): C and D chose the same setting for connectome and GRU (so their fresh-check rows are the same records, not independent evidence)",
        "informational", required=False, details={"same_setting": same,
        "connectome_and_gru_C_equals_D": same["connectome"]["C_equals_D"] and same["gru"]["C_equals_D"],
        "note": "identical settings => identical check records; 'agreement' of C and D is not an independent replication"})
    return anchors


# ======================================================================= PART B
def part_b():
    src = open(ROOT + "/scripts_v4/width_probe.py", encoding="utf-8").read()
    rows = jl(W + "/W0_width_probe_rows.jsonl")
    man = json.load(open(W + "/W0_width_probe_manifest.json"))
    md = open(W + "/W0_RESULT.md", encoding="utf-8").read()
    # cell list from source text
    try:
        sets = re.findall(r'SETS = \{(.*?)\}', src, re.S)[0]
        sset = re.findall(r'"(W\d+)"', sets)
        fams = re.findall(r'for fam in \[(.*?)\]:', src, re.S)[0]
        fams = re.findall(r'"([\w]+)"', fams)
        ks = [int(x) for x in re.findall(r'for K in \(([\d, ]+)\)', src)[0].replace(" ", "").split(",") if x]
        nat = re.findall(r'"((?:kc_static|gru|encdec)\|native\d+)"', src)
        sa, sb = re.findall(r'SEEDS = list\(range\((\d+), (\d+)\)\)', src)[0]
        seeds = list(range(int(sa), int(sb)))
        cells = [f"{f}|{s}" for s in sset for f in fams] + [f"{x}_sub|K{k}" for k in ks for x in ("kc_static", "encdec")] + nat
    except Exception as e:  # noqa
        add("B0.parse_cells", "B", "parse cell list from width_probe.py text", "incomplete", details={"error": repr(e)})
        return
    keys = [(r["family"], r["seed"]) for r in rows]
    exp_keys = {(c, s) for c in cells for s in seeds}
    d = {"n_cells_from_source": len(cells), "seeds": [seeds[0], seeds[-1]], "n_rows": len(rows),
         "n_unique": len(set(keys)), "missing": len(exp_keys - set(keys)), "unexpected": len(set(keys) - exp_keys),
         "cell_labels_unique_in_source": len(set(cells)) == len(cells)}
    ok = d["n_cells_from_source"] == 37 and d["n_rows"] == 296 and d["n_unique"] == 296 and not d["missing"] and not d["unexpected"] and d["seeds"] == [801, 808]
    add("B1.row_coverage", "B", "296 unique (family,seed) = 37 cells x seeds 801-808, no duplicates", "pass" if ok else "fail", details=d)
    nonfin = []
    for r in rows:
        for k, v in r.items():
            if isinstance(v, float) and not math.isfinite(v):
                nonfin.append((r["family"], r["seed"], k))
            if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isfinite(float(v)):
                nonfin.append((r["family"], r["seed"], k))
    metrics = ["ctx0_val", "ctx0_train", "eff_rank", "silent_frac", "clamp_frac", "n_feat", "percontext_val", "percontext_train", "oracle_val"]
    miss = [(r["family"], r["seed"], m) for r in rows for m in metrics if not isinstance(r.get(m), (int, float))]
    add("B2.metrics_finite", "B", "all metrics present and finite", "pass" if not nonfin and not miss else "fail",
        details={"nonfinite": nonfin[:20], "missing_metric": miss[:20]})
    try:
        js = json.load(open(W + "/W0_width_probe_rows.json"))
        diffs = [i for i, (a, b) in enumerate(zip(rows, js)) if a != b]
        add("B3.jsonl_vs_json", "B", "append-only jsonl equals rows.json representation exactly",
            "pass" if len(js) == len(rows) and not diffs else "fail",
            details={"n_jsonl": len(rows), "n_json": len(js), "n_differing_rows": len(diffs), "first_diffs": diffs[:5]})
    except Exception as e:  # noqa
        add("B3.jsonl_vs_json", "B", "jsonl vs json", "incomplete", details={"error": repr(e)})
    # cell means
    by = {}
    for r in rows:
        by.setdefault(r["family"], []).append(r)
    cm = {}
    for c in cells:
        v = sorted(by.get(c, []), key=lambda r: r["seed"])
        cm[c] = {"n": len(v)}
        for m in ("ctx0_val", "percontext_val", "ctx0_train", "eff_rank", "clamp_frac", "silent_frac"):
            cm[c][m] = mean([r[m] for r in v]) if v else None
        cm[c]["ctx0_val_sd"] = math.sqrt(math.fsum((r["ctx0_val"] - cm[c]["ctx0_val"]) ** 2 for r in v) / (len(v) - 1)) if len(v) > 1 else None
    # md table
    cmp = Cmp()
    w115_vs_76 = {}
    trows = {}
    for line in md.splitlines():
        m = re.match(r"\|\s*(25|64|76|115)\b[^|]*\|(.*)", line)
        if m:
            trows[m.group(1)] = [x.strip() for x in m.group(2).split("|")[:-1]] if line.rstrip().endswith("|") else [x.strip() for x in m.group(2).split("|")]
    for w, cells_md in trows.items():
        # columns: connectome | nulls range | random_reach | kc matched | encdec matched
        def fl(s):
            mm = re.match(r"([\d.]+)", s)
            return float(mm.group(1)) if mm else None
        nulls = [cm[f"null{i}|W{w}"]["ctx0_val"] for i in range(5)]
        cmp.num(f"W{w}.connectome", cm[f"connectome|W{w}"]["ctx0_val"], fl(cells_md[0]), 0.0005 + 1e-9)
        rng = re.findall(r"[\d.]+", cells_md[1])
        if len(rng) == 2:
            cmp.num(f"W{w}.null_min", min(nulls), float(rng[0]), 0.0005 + 1e-9)
            cmp.num(f"W{w}.null_max", max(nulls), float(rng[1]), 0.0005 + 1e-9)
        else:
            if w == "115" and cells_md[1].lower().startswith("same as w76"):
                for fam in [f"null{i}" for i in range(5)]:
                    w115_vs_76[fam] = abs(cm[f"{fam}|W115"]["ctx0_val"] - cm[f"{fam}|W76"]["ctx0_val"])
                cmp.num("W115.nulls_same_as_W76_max_abs_diff", max(w115_vs_76.values()), 0.0, 1e-9)
            else:
                cmp.absent.append(f"W{w}.null_range(md says '{cells_md[1]}')")
        cmp.num(f"W{w}.random_reach", cm[f"random_reach|W{w}"]["ctx0_val"], fl(cells_md[2]), 0.0005 + 1e-9)
        if w != "115":
            cmp.num(f"W{w}.kc_static_K", cm[f"kc_static_sub|K{w}"]["ctx0_val"], fl(cells_md[3]), 0.0005 + 1e-9)
            cmp.num(f"W{w}.encdec_K", cm[f"encdec_sub|K{w}"]["ctx0_val"], fl(cells_md[4]), 0.0005 + 1e-9)
    nm = re.search(r"GRU\(37\) (\d\.\d+), encdec\(407\) (\d\.\d+), kc_static\(1385\) (\d\.\d+)", md)
    if nm:
        cmp.num("native.gru37", cm["gru|native37"]["ctx0_val"], float(nm.group(1)), 0.0005 + 1e-9)
        cmp.num("native.encdec407", cm["encdec|native407"]["ctx0_val"], float(nm.group(2)), 0.0005 + 1e-9)
        cmp.num("native.kc_static1385", cm["kc_static|native1385"]["ctx0_val"], float(nm.group(3)), 0.0005 + 1e-9)
    else:
        cmp.absent.append("native row")
    if len(trows) != 4:
        cmp.mismatch.append({"field": "table_rows_parsed", "recomputed": len(trows), "logged": 4})
    add("B4.means_vs_W0_RESULT_md", "B", "per-cell ctx0_val means recomputed vs W0_RESULT.md table (3 dp)", cmp.status(),
        details=dict(cmp.details(), cell_means=cm, w115_vs_w76_per_family_max_abs_diff=dict(
            w115_vs_76, random_reach=abs(cm["random_reach|W115"]["ctx0_val"] - cm["random_reach|W76"]["ctx0_val"])),
            w115_vs_w76_note="nulls claimed identical (tol 1e-9, gates this check); random_reach expected to differ "
            "(md: 0.871 at W76 vs 0.896 at W115, verified by the table comparison above at 3 dp)"))
    # md prose claims
    cl = {}
    cl["W115_connectome_equals_W76"] = close(cm["connectome|W115"]["ctx0_val"], cm["connectome|W76"]["ctx0_val"], 1e-12)
    cl["W115_nulls_equal_W76"] = all(close(cm[f"null{i}|W115"]["ctx0_val"], cm[f"null{i}|W76"]["ctx0_val"], 1e-12) for i in range(5))
    cl["connectome_gain_W25_to_W64"] = cm["connectome|W64"]["ctx0_val"] - cm["connectome|W25"]["ctx0_val"]
    cl["connectome_W76_minus_W64"] = cm["connectome|W76"]["ctx0_val"] - cm["connectome|W64"]["ctx0_val"]
    cl["connectome_le_every_null_mean"] = {w: all(cm[f"connectome|W{w}"]["ctx0_val"] <= cm[f"null{i}|W{w}"]["ctx0_val"] for i in range(5)) for w in ("64", "76")}
    best = max(cm["connectome|W64"]["ctx0_val"], cm["connectome|W76"]["ctx0_val"])
    cl["gonogo_best_primary_connectome"] = best
    cl["gonogo_met_ge_0_85"] = best >= 0.85
    cl["train_ctx0_W64_W76"] = [cm["connectome|W64"]["ctx0_train"], cm["connectome|W76"]["ctx0_train"]]
    ok = (cl["W115_connectome_equals_W76"] and cl["W115_nulls_equal_W76"] and abs(cl["connectome_gain_W25_to_W64"] - 0.07) < 0.01
          and abs(cl["connectome_W76_minus_W64"] - 0.002) < 0.0015 and all(cl["connectome_le_every_null_mean"].values())
          and not cl["gonogo_met_ge_0_85"] and all(abs(x - 0.95) < 0.02 for x in cl["train_ctx0_W64_W76"]))
    add("B5.md_prose_claims", "B", "recomputed W0_RESULT.md prose claims (W115==W76, +7pt, +0.002, conn<=null, go/no-go not met, train ~0.95)",
        "pass" if ok else "fail", details=cl)
    # n_feat
    nf = {}
    bad = []
    for c in cells:
        vals = {r["n_feat"] for r in by.get(c, [])}
        nf[c] = sorted(vals)
        if "|W" in c:
            e = len(man["sets"][c.split("|")[1]])
        elif "_sub|K" in c:
            e = int(c.split("K")[-1])
        else:
            e = int(re.search(r"native(\d+)", c).group(1))
        if vals != {e}:
            bad.append({"cell": c, "expected": e, "got": sorted(vals)})
    wk = {"W25": 25, "W64": 64, "W76": 76, "W115": 115}
    bad_man = [k for k, v in wk.items() if len(man["sets"].get(k, [])) != v]
    add("B6.n_feat_vs_manifest", "B", "n_feat per cell vs manifest set sizes / K / native widths", "pass" if not bad and not bad_man else "fail",
        details={"bad": bad, "bad_manifest_sizes": bad_man, "n_feat": nf, "manifest_n_elig": man.get("n_elig")})
    # oracle
    ov = {}
    for r in rows:
        ov.setdefault(r["seed"], set()).add(r["oracle_val"])
    add("B7.oracle_paired_task", "B", "oracle_val identical across all cells for a given seed", "pass" if all(len(v) == 1 for v in ov.values()) and len(ov) == 8 else "fail",
        details={"distinct_per_seed": {s: len(v) for s, v in sorted(ov.items())}, "oracle_by_seed": {s: sorted(v)[0] for s, v in sorted(ov.items())}})
    # source text checks
    tree = ast.parse(src)
    seg = {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    pr = seg.get("probe", "")
    mu_line = re.search(r"mu, sd = (.*)", pr)
    fit_line = re.search(r"_fit_logreg\((.*)\)\s*$", pr, re.M)
    s1 = {"mu_sd_from_train_only": bool(mu_line) and "xt" in mu_line.group(1) and "xv" not in mu_line.group(1),
          "fit_uses_train_only": bool(fit_line) and "xt" in fit_line.group(1) and "yt" in fit_line.group(1) and "xv" not in fit_line.group(1) and "yv" not in fit_line.group(1),
          "val_score_uses_xv_yv": bool(re.search(r"va = .*xv.*yv", pr)),
          "stats_computed_on_xt_only": bool(re.search(r"\*\*stats\(xt\)", seg.get("run_cell", ""))),
          "val_features_not_used_in_fit": "xv" not in (fit_line.group(1) if fit_line else "xv"),
          "no_test_split_in_source": not re.search(r"""["']test["']""", src),
          "oracle_val_only": 'oracle_ceiling(task, ("val",))' in src,
          "features_requested_only_train_val": sorted(set(re.findall(r'feats\(agent, task, "(\w+)"', src))) == ["train", "val"]}
    add("B8.source_train_fit_val_score", "B", "SOURCE text (parsed, not imported): standardize/fit on train only, score on val only",
        "pass" if all(s1.values()) else "fail", details=dict(s1, caveat="textual pattern check of width_probe.py; does not prove the executed process matched this text (no source hash bound before run)"))
    # documented design facts
    rngm = re.search(r"np\.random\.default_rng\(\[seed, 99, sub\]\)\.choice\(xt\.shape\[1\], sub, replace=False\)", src)
    train200 = None
    cfgs = {}
    for nm in ("learning_v3_protocol.yaml", "learning_v2_protocol.yaml"):
        p = ROOT + "/configs/experiments/" + nm
        if os.path.exists(p):
            mm = re.search(r"n_train:\s*(\d+)", open(p).read())
            cfgs[nm] = int(mm.group(1)) if mm else None
    v3src = open(ROOT + "/src/flycircuit/eval/learning_v3.py").read()
    uses_v3 = "learning_v3_protocol.yaml" in v3src and "V3.load_protocol()" in src
    tr = [r["ctx0_train"] * 200 for r in rows]
    va = [r["ctx0_val"] * 64 for r in rows]
    int200 = all(abs(x - round(x)) < 1e-3 for x in tr)
    int64 = all(abs(x - round(x)) < 1e-3 for x in va)
    st = "pass" if (cfgs.get("learning_v3_protocol.yaml") == 200 and uses_v3 and int200 and int64) else "incomplete"
    add("B9.train_examples_per_head_200", "B", "each context head has 200 train examples (config + row granularity)", st,
        details={"config_n_train": cfgs, "script_loads_V3_protocol": uses_v3, "ctx0_train_x200_all_integer": int200,
                 "ctx0_val_x64_all_integer": int64,
                 "caveat": "row granularity is consistent with 200/64 but would also be consistent with multiples; value is not itself logged in rows"})
    add("B10.design_notes", "B", "documented design facts (derived from source/prereg text)", "pass" if rngm else "not_logged", required=False, details={
        "matched_K_subsets": "K-dependent RNG draws default_rng([seed,99,K]).choice(n_feat,K,replace=False), sorted; separate draw per K, NOT nested across K=25/64/76" if rngm else "pattern not found",
        "matched_K_subsets_pattern_found": bool(rngm),
        "native_width_controls": "descriptive only (prereg: 'reported descriptively only'); widths 37/407/1385 differ from matched widths and are not width-matched comparisons",
        "native_descriptive_in_prereg": "descriptively only" in open(W + "/PREREG_WIDTH_PROBE.md").read(),
        "sub_subset_same_for_every_context_within_cell": "fresh default_rng with same seed each context => same index subset per context"})
    # paired diffs
    out = {}
    for w in ("25", "64", "76"):
        cs = {r["seed"]: r["ctx0_val"] for r in by[f"connectome|W{w}"]}
        o = {"vs_each_null": {}, "vs_mean_of_5_nulls": None, "vs_random_reach": None}
        mean_null_d = []
        for s in sorted(cs):
            mean_null_d.append(cs[s] - mean([next(r["ctx0_val"] for r in by[f"null{i}|W{w}"] if r["seed"] == s) for i in range(5)]))
        for i in range(5):
            nv = {r["seed"]: r["ctx0_val"] for r in by[f"null{i}|W{w}"]}
            dl = [cs[s] - nv[s] for s in sorted(cs)]
            o["vs_each_null"][f"null{i}"] = {"per_seed": dict(zip(sorted(cs), dl)), "mean": mean(dl), "se": se(dl), "n": len(dl)}
        o["vs_mean_of_5_nulls"] = {"per_seed": dict(zip(sorted(cs), mean_null_d)), "mean": mean(mean_null_d), "se": se(mean_null_d), "n": len(mean_null_d)}
        rr = {r["seed"]: r["ctx0_val"] for r in by[f"random_reach|W{w}"]}
        dl = [cs[s] - rr[s] for s in sorted(cs)]
        o["vs_random_reach"] = {"per_seed": dict(zip(sorted(cs), dl)), "mean": mean(dl), "se": se(dl), "n": len(dl)}
        out[f"W{w}"] = o
    add("B11.connectome_minus_null_paired", "B", "paired per-seed connectome-minus-null ctx0_val differences (DESCRIPTIVE ONLY; SE ddof=1/sqrt(n); 5 nulls are not iid samples of a pre-specified test; no inference)",
        "informational", required=False, details=out)


# ======================================================================= PART C
def part_c(anchors):
    files = {
        "learning_v4.py": ROOT + "/src/flycircuit/eval/learning_v4.py",
        "learning_v4_select.py": ROOT + "/src/flycircuit/eval/learning_v4_select.py",
        "learning_v3.py(debug_copy)": ROOT + "/src/flycircuit/eval/learning_v3.py",
        "width_probe.py": ROOT + "/scripts_v4/width_probe.py",
        "run_pilot.py": ROOT + "/scripts_v4/run_pilot.py",
        "readout_ceiling.py": ROOT + "/scripts_v4/readout_ceiling.py",
        "summarize_w0.py": ROOT + "/scripts_v4/summarize_w0.py",
        "count_nodes.py": ROOT + "/scripts_v4/count_nodes.py",
        "PREREG_WIDTH_PROBE.md": W + "/PREREG_WIDTH_PROBE.md",
        "W0_RESULT.md": W + "/W0_RESULT.md",
        "W0_run.log": W + "/W0_run.log",
        "W0_width_probe_manifest.json": W + "/W0_width_probe_manifest.json",
        "W0_width_probe_rows.jsonl": W + "/W0_width_probe_rows.jsonl",
        "W0_width_probe_rows.json": W + "/W0_width_probe_rows.json",
        "search_records.jsonl": P + "/search_records.jsonl",
        "check_records.jsonl": P + "/check_records.jsonl",
        "choices.json": P + "/choices.json",
        "pilot_manifest.json": P + "/pilot_manifest.json",
        "pilot_report.json": P + "/pilot_report.json",
        "learning_v3_protocol.yaml(debug_copy)": ROOT + "/configs/experiments/learning_v3_protocol.yaml",
        "learning_v2_protocol.yaml(debug_copy)": ROOT + "/configs/experiments/learning_v2_protocol.yaml",
    }
    rec = {}
    for k, p in files.items():
        if os.path.exists(p):
            rec[k] = {"path": p, "sha256": sha(p), "mtime_utc": mtime(p), "bytes": os.path.getsize(p)}
        else:
            rec[k] = {"path": p, "sha256": None, "mtime_utc": None}
    add("C1.file_hashes_recorded", "C", "sha256 + mtime recorded for source/config/script/raw files",
        "pass" if all(v["sha256"] for v in rec.values()) else "fail", details={"files": rec})
    pman = json.load(open(P + "/pilot_manifest.json"))["identity"]
    sf = pman.get("source_files", {})
    cmpd = {}
    for rel in ("src/flycircuit/eval/learning_v4.py", "src/flycircuit/eval/learning_v4_select.py"):
        cmpd[rel] = {"manifest_sha": sf.get(rel), "current_sha": sha(ROOT + "/" + rel)}
        cmpd[rel]["equal"] = cmpd[rel]["manifest_sha"] == cmpd[rel]["current_sha"] if cmpd[rel]["manifest_sha"] else None
    st = "not_logged" if any(v["equal"] is None for v in cmpd.values()) else ("pass" if all(v["equal"] for v in cmpd.values()) else "fail")
    # source tree hash re-derivation
    tree_ok = None
    if sf:
        try:
            ents = {rel: sha(ROOT + "/" + rel) for rel in sf}
            tree_ok = {"files_equal_to_manifest": sum(ents[r] == sf[r] for r in sf), "n": len(sf),
                       "changed": sorted(r for r in sf if ents[r] != sf[r])}
        except Exception as e:  # noqa
            tree_ok = {"error": repr(e)}
    if st == "pass":  # gate on ALL manifest source files, not only the two learning_v4 files
        if not isinstance(tree_ok, dict) or "error" in tree_ok:
            st = "incomplete"
        elif tree_ok["changed"] or tree_ok["files_equal_to_manifest"] != tree_ok["n"]:
            st = "fail"
    add("C2.pilot_source_unchanged_since_manifest", "C", "current sha of ALL pilot_manifest source files (incl. learning_v4*.py) equals recorded sha (manifest created before search; see mtimes)",
        st, details={"compare": cmpd, "all_manifest_source_files": tree_ok,
                     "pilot_manifest_created_utc": json.load(open(P + "/pilot_manifest.json")).get("created_utc"),
                     "note": "pilot_manifest.json is the only pre-run binding that exists for the pilot"})
    # config files vs manifest (canonical read-only)
    cfd = {}
    for p, h in pman.get("config_files", {}).items():
        cfd[p] = {"manifest_sha": h, "current_sha": sha(p) if os.path.exists(p) else None}
        cfd[p]["equal"] = cfd[p]["current_sha"] == h
    add("C3.pilot_config_hashes", "C", "protocol yaml sha256 (canonical path, read-only) equals pilot_manifest",
        "pass" if cfd and all(v["equal"] for v in cfd.values()) else ("not_logged" if not cfd else "fail"), details=cfd)
    # W0 missing bindings (derived)
    wman_text = open(W + "/W0_width_probe_manifest.json").read()
    hashes = {"PREREG": rec["PREREG_WIDTH_PROBE.md"]["sha256"], "width_probe.py": rec["width_probe.py"]["sha256"],
              "learning_v3.py": rec["learning_v3.py(debug_copy)"]["sha256"]}
    wm = json.loads(wman_text)
    present = {k: (h in wman_text) for k, h in hashes.items()}
    present["any_source_manifest_key"] = any(k in wm for k in ("source_files", "source_tree_sha256", "prereg_sha256", "script_sha256"))
    miss = []
    if not present["PREREG"]:
        miss.append("W0 PREREG_WIDTH_PROBE.md sha256 not recorded in any artifact before/at the run (W0 manifest has none)")
    if not present["width_probe.py"] or not present["any_source_manifest_key"]:
        miss.append("W0 launched WITHOUT a frozen source manifest (no source_files/source_tree sha in W0 manifest; unlike the pilot)")
    if not present["width_probe.py"]:
        miss.append("width_probe.py hash not bound pre-run; 'script not patched after the run except readout_ceiling.py' is a statement by the operator, not verifiable from a recorded hash (only mtimes)")
    wp, rw = os.stat(files["width_probe.py"]).st_mtime, os.stat(files["W0_width_probe_rows.jsonl"]).st_mtime
    wm_t = os.stat(files["W0_width_probe_manifest.json"]).st_mtime
    after = sorted(k for k in ("width_probe.py", "PREREG_WIDTH_PROBE.md", "readout_ceiling.py", "summarize_w0.py", "count_nodes.py")
                   if os.stat(files[k]).st_mtime > wm_t)
    if not present["width_probe.py"]:
        miss.append("W0 launched from the debug copy with no immutable copy of the executed script; current file is the only evidence")
    if "summarize_w0.py" in after:
        miss.append("summarize_w0.py was modified after the W0 manifest was written (mtime-based)")
    if after:
        miss.append(f"files with mtime after W0 manifest: {after}")
    c4_status = "pass" if (all(present.values()) and not after) else "not_logged"
    add("C4.w0_pre_run_bindings", "C", "W0 pre-run bindings: derived presence of prereg/source hashes in W0 manifest",
        c4_status, required=False, details={
            "missing_bindings": miss, "hash_present_in_W0_manifest": present,
            "mtime_facts": {"width_probe_mtime_le_W0_manifest_mtime": wp <= wm_t,
                            "width_probe_mtime_minus_prereg_mtime_s": wp - os.stat(files["PREREG_WIDTH_PROBE.md"]).st_mtime,
                            "files_modified_after_W0_manifest_written": after,
                            "W0_manifest_to_rows_end_s": rw - wm_t},
            "WARNING": "these are NOT historical immutable freezes; file hashes here are taken at audit time"})
    # canonical protected state (read-only)
    c1, c2 = CANON + "/src/flycircuit/eval/learning_v3.py", CANON + "/results/flygym_v3/train_log.json"
    e = "62a6fe4e9262277f34d6699a140cf7df4b059d90875434af62e3f0e6cbca7e5a"
    h1 = sha(c1) if os.path.exists(c1) else None
    h2 = sha(c2) if os.path.exists(c2) else None
    add("C5.canonical_train_log_sha", "C", "canonical flygym_v3 train_log.json sha256 equals expected (read-only)",
        "pass" if h2 == e else ("incomplete" if h2 is None else "fail"), details={"path": c2, "sha256": h2, "expected": e})
    add("C6.canonical_learning_v3_sha", "C", "canonical learning_v3.py sha256 listed (no expected value supplied; compared to debug copy only)",
        "informational" if h1 else "incomplete", required=False,
        details={"path": c1, "sha256": h1, "debug_copy_sha256": hashes["learning_v3.py"], "equal_to_debug_copy": h1 == hashes["learning_v3.py"]})


def main():
    anchors = part_a()
    add("A0.part_a_completed", "A", "Part A ran to completion (anchors/inputs present)",
        "pass" if anchors else "incomplete")
    part_b()
    part_c(anchors)
    req = [c for c in CHECKS if c["required"]]
    overall = "pass" if req and all(c["status"] == "pass" for c in req) else (
        "fail" if any(c["status"] == "fail" for c in req) else "incomplete")
    nonreq_open = sorted(f"{c['id'].split('.')[0]} {c['status']}" for c in CHECKS
                         if not c["required"] and c["status"] in ("not_logged", "fail", "incomplete"))
    if overall == "pass" and nonreq_open:
        overall = "pass (required checks only; non-required open: " + ", ".join(nonreq_open) + ")"
    scope = [
        "Author's own re-implementation in stdlib python; shares conceptual assumptions with the audited code. It is a consistency audit, NOT an independent verification of correctness.",
        "Pilot is VALIDATION-ONLY, 4 seeds per cell (search 501-504, fresh 601-604); descriptive thresholds (acq>=0.82, signed forgetting gap>=0.07) are not confirmatory.",
        "C and D select the same setting for connectome and GRU, so they are not independent rules/evidence there.",
        "W0 is exploratory: untrained frozen features, validation split, 8 seeds, descriptive statistics only; no inference.",
        "Training/model code was not re-executed; raw metrics are taken as logged. No check here verifies that features/records were produced by the stated code.",
        "Source-text checks (B8, B10) are textual patterns on the current width_probe.py; no hash of the executed script was bound before the W0 run.",
        "Provenance hashes/mtimes are measured at audit time; they are not historical immutable freezes. Missing pre-run bindings are listed in C4.",
        "Records carry no per-record checksum, so values produced by in-memory training are taken on trust.",
        "B8 is a source-text regex check, not proof of what executed.",
        "A7 scans key names, plus string values equal to 'test' under keys named split/splits; other values are not scanned.",
        "Canonical <CANONICAL_ROOT> was only hashed read-only; learning_v3.py has no expected hash supplied.",
    ]
    result = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "overall_status": overall,
              "required_checks": [c["id"] for c in req], "summary": {c["id"]: c["status"] for c in CHECKS},
              "scope_limits": scope, "checks": CHECKS}
    os.makedirs(OUTD, exist_ok=True)

    def atomic(path, text):
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    atomic(OUTD + "/audit.json", json.dumps(result, indent=2, sort_keys=False, default=str) + "\n")
    L = [f"# v4 audit (overall: {overall})", "", f"Generated {result['generated_utc']}. Author's own arithmetic; not independent.", "",
         "| check | req | status | description |", "|---|---|---|---|"]
    for c in CHECKS:
        L.append(f"| {c['id']} | {'Y' if c['required'] else 'n'} | {c['status']} | {c['description']} |")
    L += ["", "## Scope limits"] + [f"- {s}" for s in scope]
    for c in CHECKS:
        if c["status"] not in ("pass", "informational") or c["id"] in ("A9.paired_forgetting_gap_vs_report", "C4.w0_pre_run_bindings"):
            L += ["", f"## {c['id']} ({c['status']})"]
            d = c["details"]
            if c["id"].startswith("A9"):
                for r, g in d["gaps"].items():
                    L.append(f"- rule {r}: GRU-minus-conn forgetting {g['mean_gru_minus_conn_forgetting']*100:.2f}pp (SE {g['se']*100:.3f}pp), signed>=0.07: {g['signed_ge_0_07']}; conn acq {g['conn_acq_mean']*100:.2f}, GRU acq {g['gru_acq_mean']*100:.2f}; per-seed {g['per_seed_gru_minus_conn_forgetting']}")
            elif c["id"].startswith("C4"):
                L += [f"- {m}" for m in d["missing_bindings"]] + [f"- {d['WARNING']}"]
            else:
                L.append("```\n" + json.dumps(d, indent=1, default=str)[:3000] + "\n```")
    atomic(OUTD + "/audit.md", "\n".join(L) + "\n")
    print(overall)
    for c in CHECKS:
        print(f"{c['id']:42s} {c['status']:11s} {'req' if c['required'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
