"""Audit of stage W3 (trained W76 pilot). OWN RE-IMPLEMENTATION by the same project team: a consistency audit,
NOT an independent review of the author's own work. Read-only on all inputs; no training; no graph generation.
Writes NEW files results_v4/audit_v4/W3_audit.json and W3_audit.md.
Run: PYTHONPATH=<root>/src FLYCIRCUIT_HOME=$FLYCIRCUIT_HOME <venv python> scripts_v4/audit_w3.py
"""
from _paths import CANON as _CANON  # noqa: F401,E402
from _paths import ROOT as _ROOT  # noqa: F401,E402
import datetime
import glob
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = _ROOT
CANON = _CANON
W = ROOT + "/results_v4/learning_readout_v4_1"
OUTD = ROOT + "/results_v4/audit_v4"
STAGE = "W3_trained_w76"
FAMS = ["connectome", "gru", "encdec", "null0", "null1", "null2", "random_reach"]
SPARSE = {"connectome", "null0", "null1", "null2", "random_reach"}
SS, CS = [1401, 1402, 1403, 1404], [1501, 1502, 1503, 1504]
T3 = 3.182446305284263  # t_{0.975, df=3}
CHECKS = []


def add(cid, desc, status, details=None):
    assert status in ("pass", "fail", "incomplete", "not_logged", "informational")
    CHECKS.append({"id": cid, "description": desc, "status": status, "details": details or {}})


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def iso(p):
    return datetime.datetime.fromtimestamp(os.stat(p).st_mtime).isoformat()


def ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else float("nan"))


def ci(v):
    v = np.asarray(v, float)
    m, se = ms(v)
    return {"n": int(v.size), "mean": m, "sd": float(v.std(ddof=1)), "se": se,
            "t3_95ci": [m - T3 * se, m + T3 * se], "t_stat_vs0": m / se if se else None}


def simp(core, dec, base):  # re-implemented: (max multiplier, geo-mean multiplier, core, dec)
    rc, rd = core / base, dec / base
    return (max(rc, rd), math.sqrt(rc * rd), core, dec)


if os.path.exists(f"{OUTD}/W3_audit.json") and "--overwrite" not in sys.argv:
    raise SystemExit("W3_audit.json exists")

rows = [json.loads(l) for l in open(f"{W}/{STAGE}_rows.jsonl") if l.strip()]
man = json.load(open(f"{W}/{STAGE}_manifest.json"))
choices = json.load(open(f"{W}/{STAGE}_choices.json"))
final = json.load(open(f"{W}/{STAGE}_final.json"))
reg = json.load(open(f"{W}/seed_registry.json"))
anc = man["anchors"]
out = {"scope": "OWN re-implementation by the project team; a consistency audit, not an independent review.",
       "n_rows": len(rows)}

# ---------------- 1 coverage ----------------
keys = [(r["family"], r["core_lr"], r["decoder_lr"], r["seed"]) for r in rows]
dups = [k for k, c in Counter(keys).items() if c > 1]
sr = [r for r in rows if r["seed"] in SS]
cr = [r for r in rows if r["seed"] in CS]
other = [r for r in rows if r["seed"] not in SS + CS]
st = Counter(r["status"] for r in rows)


def finite_ok(r):
    if r["status"] != "ok":
        return True
    vals = [r["association"]["val_acc_acq"], r["continual"]["final_mean_acc_val"], r["continual"]["avg_forgetting_keep_val"]]
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in vals)


nonfin = [k for r, k in zip(rows, keys) if not finite_ok(r)]
per_fam_search = {f: len({(r["core_lr"], r["decoder_lr"]) for r in sr if r["family"] == f}) for f in FAMS}
cnt_cell = Counter((r["family"], r["core_lr"], r["decoder_lr"]) for r in sr)
cov = {"unique_keys": len(set(keys)), "dups": len(dups), "search_rows": len(sr), "check_rows": len(cr), "other_seed_rows": len(other),
       "status_counts": dict(st), "n_not_ok": sum(v for k, v in st.items() if k != "ok"), "nonfinite_metric_rows": len(nonfin),
       "settings_per_family_search": per_fam_search,
       "search_rows_per_cell_min_max": [min(cnt_cell.values()), max(cnt_cell.values())],
       "phase_tag_consistent_with_seed": all((r["w3_phase"] == "search") == (r["seed"] in SS) for r in rows),
       "final_json": {"n_rows": final.get("n_rows"), "n_failed": final.get("n_failed"), "aborted": final.get("aborted")},
       "elapsed_s": final.get("elapsed_s")}
add("1_coverage", "280 unique keys = 252 search + 28 check; no dups; all ok; metrics finite",
    "pass" if (len(set(keys)) == 280 and not dups and len(sr) == 252 and len(cr) == 28 and not other and st.get("ok") == 280
               and not nonfin and all(v == 9 for v in per_fam_search.values())) else "fail", cov)

# ---------------- 2 seed identity ----------------
used = reg["used"]
all_prior = {s for k, v in used.items() if not k.startswith("W3") for s in v}
sd = {"search_seeds_in_rows": sorted({r["seed"] for r in sr}), "check_seeds_in_rows": sorted({r["seed"] for r in cr}),
      "search_check_disjoint": not (set(SS) & set(CS)),
      "overlap_with_other_registry_entries": sorted((set(SS) | set(CS)) & all_prior),
      "registry_W3": {"search": used["W3_search_committed"], "check": used["W3_check_committed"]},
      "registry_reserved_unseen": reg.get("reserved_unseen"),
      "check_seeds_in_search_rows": sorted({r["seed"] for r in cr} & {r["seed"] for r in sr}),
      "registry_note": reg.get("note"),
      "overlap_with_reserved_validation": sorted((set(SS) | set(CS)) & set(reg["reserved_unseen"]["validation_checks"]))}
seed_ok = (sd["search_seeds_in_rows"] == SS and sd["check_seeds_in_rows"] == CS and sd["search_check_disjoint"]
           and not sd["overlap_with_other_registry_entries"] and not sd["check_seeds_in_search_rows"]
           and sd["registry_W3"] == {"search": SS, "check": CS})
add("2a_seed_identity_registry", "search rows use only 1401-1404, check 1501-1504, disjoint, unused elsewhere",
    "pass" if seed_ok else "fail", sd)

data_cmp = {}
proto = None
gen_ok = None
try:
    from flycircuit.eval import learning as L
    from flycircuit.eval import learning_v3 as V3
    proto = V3.load_protocol()
    seeds_all = SS + CS
    tasks = {s: L.make_task(proto, s) for s in seeds_all}
    spec = proto["task"]
    nctx = tasks[1401].spec.n_ctx
    keyset = {s: {sp: {x.tobytes() for ctx in range(nctx) for x in tasks[s].cues[(sp, ctx)]} for sp in ("train", "val", "test")}
              for s in seeds_all}
    n_val = sum(len(tasks[1401].cues[("val", c)]) for c in range(nctx))
    identical_arrays, pat_overlap = [], {}
    for a in SS:
        for b in CS:
            for sp in ("train", "val"):
                if all(np.array_equal(tasks[a].cues[(sp, c)], tasks[b].cues[(sp, c)]) for c in range(nctx)):
                    identical_arrays.append((a, b, sp))
                pat_overlap[f"{a}v{b}:{sp}"] = [len(keyset[a][sp] & keyset[b][sp]), len(keyset[a][sp])]
    within = {f"{a}v{b}:val": [len(keyset[a]["val"] & keyset[b]["val"]), len(keyset[a]["val"])]
              for i, a in enumerate(SS) for b in SS[i + 1:]}
    teach_equal = [(a, b) for a in SS for b in CS if np.allclose(tasks[a].teachers, tasks[b].teachers)]
    agree = {}
    for a, b in [(1401, 1501), (1402, 1502), (1403, 1503), (1404, 1504)]:
        va = {x.tobytes(): int(y) for c in range(nctx) for x, y in zip(tasks[a].cues[("val", c)], tasks[a].labels[("val", c)])}
        vb = {x.tobytes(): int(y) for c in range(nctx) for x, y in zip(tasks[b].cues[("val", c)], tasks[b].labels[("val", c)])}
        sh = set(va) & set(vb)
        agree[f"{a}v{b}"] = {"shared_val_patterns(ignoring ctx)": len(sh), "same_label": sum(va[k] == vb[k] for k in sh)}
    data_cmp = {"task_spec": spec, "n_val_cues_total_per_seed": n_val,
                "identical_cue_arrays_search_vs_check": identical_arrays,
                "teachers_identical_search_vs_check": teach_equal,
                "val_pattern_overlap_search_vs_check[shared,total]": {k: v for k, v in pat_overlap.items() if k.endswith("val")},
                "train_pattern_overlap_search_vs_check[shared,total]": {k: v for k, v in pat_overlap.items() if k.endswith("train")},
                "val_pattern_overlap_within_search[shared,total]": within,
                "label_agreement_on_shared_val_patterns_paired_seeds": agree}
    gen_ok = (not identical_arrays) and (not teach_equal)
except Exception as e:  # noqa
    data_cmp = {"error": repr(e)}
code_path = {
    "seed_to_task": "learning_v2.run_one_v2: task = L.make_task(proto_run, seed) -> ContingencyTask(spec, seed); teachers from "
                    "SeedSequence([seed,99]); each (split,ctx) set from SeedSequence([seed, split_id, ctx]) so train/val/test all depend on seed",
    "seed_to_model": "build_model_v2(.., seed): model init seed and fixed encoder SeedSequence([seed,3]); random_reach graph drawn from seed; "
                     "null0..2 graphs are FIXED cached packs (family name picks the file), NOT seed dependent",
    "seed_to_agent": "LearningAgent(seed=seed) (explore noise); run_association_v2 / run_continual_v2 receive seed",
    "splits_evaluated": sorted({tuple(r["splits"]) for r in rows}),
    "search_and_check_both_validation": all(r["splits"] == ["val"] for r in rows),
    "interpretation": "Check seeds 1501-1504 generate different task instances (new teachers, new cues) than 1401-1404, "
                      "BUT both are the VALIDATION split of the same task family/protocol and the same fixed graphs and hyper-parameter "
                      "grid. The check is a fresh-seed re-evaluation on validation; it is NOT held-out/test, so it is exploratory only."}
add("2b_check_data_differs", "check seeds produce data different from search seeds; both are validation split",
    "pass" if gen_ok else ("incomplete" if gen_ok is None else "fail"), {"code_path": code_path, "data": data_cmp})

# ---------------- 3 selection ----------------
sel, sel_ok = {}, True
for f in FAMS:
    tab = defaultdict(list)
    for r in sr:
        if r["family"] == f and r["status"] == "ok":
            tab[(r["core_lr"], r["decoder_lr"])].append((r["association"]["val_acc_acq"] + r["continual"]["final_mean_acc_val"]) / 2)
    grid_c, grid_d = sorted({k[0] for k in tab}), sorted({k[1] for k in tab})
    stats = {k: ms(v) for k, v in tab.items()}
    bm = max(m for m, _ in stats.values())
    best = min([k for k, (m, _) in stats.items() if m == bm], key=lambda k: simp(k[0], k[1], anc[f]))
    thr = stats[best][0] - stats[best][1]
    cand = [k for k, (m, _) in stats.items() if m >= thr]
    chosen = min(cand, key=lambda k: simp(k[0], k[1], anc[f]))
    ch = choices[f]
    match = (abs(ch["core_lr"] - chosen[0]) < 1e-15 and abs(ch["decoder_lr"] - chosen[1]) < 1e-15 and
             abs(ch["best"][0] - best[0]) < 1e-15 and abs(ch["best"][1] - best[1]) < 1e-15 and
             abs(ch["best_mean"] - bm) < 1e-12 and abs(ch["threshold"] - thr) < 1e-12 and ch["n_indistinguishable"] == len(cand))
    sel_ok &= match

    def edge(k):
        return {"core_on_edge": k[0] in (grid_c[0], grid_c[-1]), "decoder_on_edge": k[1] in (grid_d[0], grid_d[-1]),
                "core_side": "low" if k[0] == grid_c[0] else "high" if k[0] == grid_c[-1] else "mid",
                "decoder_side": "low" if k[1] == grid_d[0] else "high" if k[1] == grid_d[-1] else "mid"}
    eb, ec = edge(best), edge(chosen)
    sel[f] = {"n_settings": len(tab), "n_seeds_each": sorted({len(v) for v in tab.values()}), "best": list(best), "best_mean": bm,
              "best_se": stats[best][1], "threshold": thr, "n_within_1se": len(cand), "chosen": list(chosen), "chosen_mean": stats[chosen][0],
              "matches_choices_json": match, "best_edge": eb, "chosen_edge": ec,
              "best_on_any_edge": eb["core_on_edge"] or eb["decoder_on_edge"], "chosen_on_any_edge": ec["core_on_edge"] or ec["decoder_on_edge"],
              "grid_core": grid_c, "grid_decoder": grid_d,
              "cell_means[core|dec]": {f"{k[0]:.6g}|{k[1]:.6g}": round(v[0], 5) for k, v in sorted(stats.items())}}
add("3_selection", "independent recompute of composite/one-SE/two_group_simplicity equals choices.json",
    "pass" if sel_ok else "fail", sel)

# ---------------- 4 check-row arithmetic ----------------
C = {f: {r["seed"]: r for r in cr if r["family"] == f} for f in FAMS}
arith = {"check_setting_equals_selected": {f: sorted({(r["core_lr"], r["decoder_lr"]) for r in C[f].values()}) ==
                                           [(choices[f]["core_lr"], choices[f]["decoder_lr"])] for f in FAMS}}
per = {}
for f in FAMS:
    a = [C[f][s]["association"]["val_acc_acq"] for s in CS]
    c = [C[f][s]["continual"]["final_mean_acc_val"] for s in CS]
    fk = [C[f][s]["continual"]["avg_forgetting_keep_val"] for s in CS]
    per[f] = {"acq": a, "acq_mean_se": ms(a), "cont": c, "cont_mean_se": ms(c), "forget_keep": fk, "forget_mean_se": ms(fk)}
arith["per_family"] = per
gap = [C["gru"][s]["continual"]["avg_forgetting_keep_val"] - C["connectome"][s]["continual"]["avg_forgetting_keep_val"] for s in CS]
arith["gru_minus_connectome_forgetting_signed"] = {"per_seed": gap, **ci(gap), "mean_ge_0.07": ci(gap)["mean"] >= 0.07,
                                                    "t3_ci_lower_above_0": ci(gap)["t3_95ci"][0] > 0}
ca = per["connectome"]["acq"]
cci = ci(ca)
arith["connectome_acquisition"] = {**cci, "scores": ca, "distance_above_0.82": cci["mean"] - 0.82,
                                   "t_vs_0.82": (cci["mean"] - 0.82) / cci["se"], "ci_lower_minus_0.82": cci["t3_95ci"][0] - 0.82,
                                   "claimed_mean": 0.82421875, "claimed_se": 0.034572703,
                                   "matches_claim": abs(cci["mean"] - 0.82421875) < 1e-9 and abs(cci["se"] - 0.034572703) < 1e-8}
nm = [float(np.mean([per[n]["acq"][i] for n in ("null0", "null1", "null2")])) for i in range(4)]
arith["connectome_minus_mean_null012_acq_paired"] = {"null_mean_per_seed": nm, "per_seed": [ca[i] - nm[i] for i in range(4)],
                                                      **ci([ca[i] - nm[i] for i in range(4)])}
arith["connectome_minus_random_reach_acq_paired"] = {"per_seed": [ca[i] - per["random_reach"]["acq"][i] for i in range(4)],
                                                      **ci([ca[i] - per["random_reach"]["acq"][i] for i in range(4)])}
arith["connectome_minus_each_null_acq_means"] = {n: per["connectome"]["acq_mean_se"][0] - per[n]["acq_mean_se"][0] for n in ("null0", "null1", "null2")}
arith["connectome_minus_gru_acq_paired"] = ci([ca[i] - per["gru"]["acq"][i] for i in range(4)])
arith["connectome_minus_gru_cont_paired"] = ci([per["connectome"]["cont"][i] - per["gru"]["cont"][i] for i in range(4)])
add("4_check_arithmetic", "check-row means/SEs/gaps; claimed connectome acq 0.82421875 (SE 0.034572703)",
    "pass" if arith["connectome_acquisition"]["matches_claim"] and all(arith["check_setting_equals_selected"].values()) else "fail", arith)

# ---------------- 5 parameter counts ----------------
pc = {}
for f in FAMS:
    rr = [r for r in rows if r["family"] == f and r["status"] == "ok"]
    pc[f] = {"n_params_trainable_values": sorted({r.get("n_params_trainable") for r in rr}),
             "core_n": sorted({r["param_groups"]["core_n"] for r in rr}), "decoder_n": sorted({r["param_groups"]["decoder_n"] for r in rr}),
             "core_plus_decoder_equals_total": all(r["param_groups"]["core_n"] + r["param_groups"]["decoder_n"] == r["n_params_trainable"] for r in rr),
             "n_edges": sorted({r.get("n_edges") for r in rr}), "n_nodes": sorted({r.get("n_nodes") for r in rr}),
             "distinct_edge_hashes": len({r.get("edge_hash") for r in rr}) if f in SPARSE else None}
ref = pc["connectome"]["n_params_trainable_values"][0]
pc["_summary"] = {"connectome_total": ref, "connectome_core_edge_gains": pc["connectome"]["core_n"],
                  "connectome_decoder": pc["connectome"]["decoder_n"],
                  "totals": {f: pc[f]["n_params_trainable_values"] for f in FAMS},
                  "gru_over_connectome": [t / ref for t in pc["gru"]["n_params_trainable_values"]],
                  "encdec_over_connectome": [t / ref for t in pc["encdec"]["n_params_trainable_values"]],
                  "sparse_families_same_total": len({tuple(pc[f]["n_params_trainable_values"]) for f in SPARSE}) == 1,
                  "gru_within_1pct": all(abs(t - ref) / ref < 0.01 for t in pc["gru"]["n_params_trainable_values"]),
                  "encdec_within_1pct": all(abs(t - ref) / ref < 0.01 for t in pc["encdec"]["n_params_trainable_values"]),
                  "caveat": "Reported as logged; matching is NOT asserted. Prereg says gru/encdec were deliberately built from the W25 base "
                            "(hidden 37 / 407) to equal the v4 pilot architectures, i.e. matched to the W25 sparse count, not the W76 count."}
constructed = {}
try:
    sys.path.insert(0, ROOT + "/scripts_v4")
    import w3_pilot as WP
    from flycircuit.eval import learning as L5
    from flycircuit.eval import learning_v2 as V25
    from flycircuit.eval import learning_v4 as P5
    proto5, base25, packs5 = P5._setup()
    _r25, r76_, _h = WP.build_w76(base25)
    base76 = WP.with_ro(base25, r76_)
    task5 = L5.make_task(proto5, 1401)
    for f in FAMS:
        ag, meta5 = P5._orig_make(proto5, WP.base_for(f, base25, base76), packs5, "bptt", f, 1e-3, 1401, task5, 1e-3)
        dec_ids = {id(p) for p in ag.decoder_params()}
        c_n = int(sum(p.numel() for p in ag.params if id(p) not in dec_ids and p.requires_grad))
        d_n = int(sum(p.numel() for p in ag.params if id(p) in dec_ids and p.requires_grad))
        _, _, _, info5 = V25.build_model_v2(proto5, WP.base_for(f, base25, base76), packs5, f, 1401, task5.spec.n_in)
        logged = pc[f]
        constructed[f] = {"core_n": c_n, "decoder_n": d_n, "total": c_n + d_n, "info": info5,
                          "base_used": "W76 bundle" if f in SPARSE else "W25 base",
                          "logged_total": logged["n_params_trainable_values"], "logged_core": logged["core_n"], "logged_decoder": logged["decoder_n"],
                          "match_logged": [c_n + d_n] == logged["n_params_trainable_values"] and [c_n] == logged["core_n"] and [d_n] == logged["decoder_n"]}
    constructed["_gru_encdec_hidden_check"] = {"gru_hidden": constructed["gru"]["info"].get("hidden"), "encdec_hidden": constructed["encdec"]["info"].get("hidden"),
                                               "expected": {"gru": 37, "encdec": 407}}
    pc5_ok = all(constructed[f]["match_logged"] for f in FAMS) and constructed["gru"]["info"].get("hidden") == 37         and constructed["encdec"]["info"].get("hidden") == 407
except Exception as e:  # noqa
    constructed["error"] = repr(e)
    pc5_ok = None
pc["_constructed_models_no_training"] = constructed
pc["_summary"]["sizing_statement"] = ("GRU/encdec were sized from the W25 base per prereg (hidden 37 / 407, verified from the constructed models' info), "
                                      "so they match the W25 sparse count, not the W76 count; matching to W76 is not claimed.")
add("5_param_counts", "trainable parameter counts per family: models CONSTRUCTED here without training (W76 bundle for sparse, W25 base for gru/encdec) "
    "and requires_grad params compared with logged row values",
    "pass" if (pc5_ok and all(pc[f]["core_plus_decoder_equals_total"] for f in FAMS)) else ("incomplete" if pc5_ok is None else "fail"), pc)

# ---------------- 6 provenance ----------------
prov = {"prelaunch_hashes": [], "manifest_sha256": []}
for l in [l.strip() for l in open(f"{W}/W3_prelaunch_hashes.txt") if l.strip()]:
    if len(l.split()[0]) == 64:
        h, p = l.split(None, 1)
        pp = p if p.startswith("/") else f"{ROOT}/{p}"
        cur = sha(pp)
        prov["prelaunch_hashes"].append({"path": p, "recorded": h, "current": cur, "match": h == cur, "mtime": iso(pp)})
for p, h in sorted(man["sha256"].items()):
    pp = f"{ROOT}/{p}"
    cur = sha(pp)
    prov["manifest_sha256"].append({"path": p, "recorded": h, "current": cur, "match": h == cur, "mtime": iso(pp)})
core = {k: v for k, v in man.items() if k != "written"}
canon = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
row_core = {r["w3_manifest_core_sha256"] for r in rows}
t_end = os.stat(f"{W}/{STAGE}_rows.jsonl").st_mtime
t_start_est = t_end - final["elapsed_s"]
prov["manifest_core_hash_recomputed"] = canon
prov["row_manifest_core_hashes"] = sorted(row_core)
prov["rows_all_reference_this_manifest"] = (row_core == {canon})
prov["times"] = {"prereg_mtime": iso(f"{W}/W3_PREREG.md"), "prelaunch_hashes_mtime": iso(f"{W}/W3_prelaunch_hashes.txt"),
                 "manifest_mtime": iso(f"{W}/{STAGE}_manifest.json"), "manifest_written_field": man["written"],
                 "w3_pilot_py_mtime": iso(f"{ROOT}/scripts_v4/w3_pilot.py"), "seed_registry_mtime": iso(f"{W}/seed_registry.json"),
                 "est_process_start (rows_last_mtime - elapsed_s)": datetime.datetime.fromtimestamp(t_start_est).isoformat(),
                 "rows_last_mtime": iso(f"{W}/{STAGE}_rows.jsonl"), "choices_mtime": iso(f"{W}/{STAGE}_choices.json"),
                 "final_mtime": iso(f"{W}/{STAGE}_final.json"), "summary_mtime": iso(f"{W}/W3_summary.json")}
prov["manifest_mtime_not_after_first_training_row_estimate"] = os.stat(f"{W}/{STAGE}_manifest.json").st_mtime <= t_start_est + 60
prov["classification"] = {
    "prospective (existed before any W3 training row)": [
        "W3_PREREG.md (mtime 08:11:16 = launch instant; script/author state it was written before any run)",
        "W3_trained_w76_manifest.json (08:11:22; written by main() before first run_cell; its core hash is stamped in every row)",
        "w3_pilot.py / summarize_w3.py (mtime 08:09:31)", "seed_registry.json W3 entries (hash is in manifest)"],
    "retrospective or launch-time": [
        "W3_prelaunch_hashes.txt: created by launch_w3.sh AT LAUNCH TIME (08:11:16), seconds before compute; it proves 'unchanged since launch', "
        "not 'frozen well before launch'; it is also a strict subset of the manifest sha256 list.",
        "W3_trained_w76_choices.json, W3_summary.json: derived after the data"],
    "caveat": "Rows carry no per-row wall-clock timestamp. No git history in this copy, so prereg-before-run rests on mtimes and the author's "
              "statement, not on an external timestamp. Hashes can only show that files are unchanged, not when they were authored."}
allm = all(x["match"] for x in prov["prelaunch_hashes"]) and all(x["match"] for x in prov["manifest_sha256"])
add("6_provenance", "current sha256 vs recorded (prelaunch + manifest); manifest/rows linkage",
    "pass" if allm and prov["rows_all_reference_this_manifest"] else "fail", prov)

# ---------------- 7 null graph inventory ----------------
inv, inv_status = {}, "incomplete"
try:
    nd = f"{CANON}/results/nulls_v2/{proto['circuit']['name']}"
    inv["dir"] = nd
    inv["kind_used_by_W3"] = ("plain: build_model_v2 maps family 'nullI' -> packs['plain'][I]; (group_constrained is used only by 'gnullI', "
                              "which W3 does not run)")
    for kind in ("plain", "group_constrained"):
        fl = sorted(glob.glob(f"{nd}/{kind}/null_*.npz"))
        rep = json.load(open(f"{nd}/{kind}/report.json"))
        inv[kind] = {"count": len(fl), "files": [{"path": p, "sha256": sha(p), "mtime": iso(p)} for p in fl],
                     "report_sha256": sha(f"{nd}/{kind}/report.json"), "report_meta": rep["meta"],
                     "verdict": rep["report"]["verdict"], "adequate": rep["report"]["adequate"],
                     "chain_seeds": [p["seed"] for p in rep["report"]["per_null"]],
                     "generation": "nulls_v2.make_ensemble(base_seed=2000; chain seed = 2000 + 7919*i): degree-preserving edge-swap chains "
                                   "(plain) / edge-class-constrained (group_constrained)"}
    inv["debug_copy_has_own_nulls_dir"] = os.path.isdir(f"{ROOT}/results/nulls_v2")
    inv["search_only_nulls_null_10_null_11"] = "UNAVAILABLE: only null_00..null_09 per kind exist on disk"
    inv["load_packs_slice"] = f"plain[:{proto['seeds']['null_graphs']}] -> null0..null{proto['seeds']['null_graphs'] - 1} are loaded; W3 uses null0..2"
    from flycircuit.eval import learning_v2 as V2
    from flycircuit.graph import nulls_v2 as N2
    base = V2.load_base(proto, "flywire")
    g = base.graph
    cset = set((g.pre.astype(np.int64) * g.n + g.post.astype(np.int64)).tolist())
    match, overlap = {}, {}
    for i in range(10):
        z = np.load(f"{nd}/plain/null_{i:02d}.npz")
        pre, post = z["pre"].astype(np.int64), z["post"].astype(np.int64)
        overlap[f"null_{i:02d}"] = len(set((pre * g.n + post).tolist()) & cset) / len(cset)
        if i < 3:
            h = N2._rebuild(g, pre, post, g.count.copy(), "x").edge_hash()
            rh = sorted({r["edge_hash"] for r in rows if r["family"] == f"null{i}"})
            match[f"null{i}"] = {"file": f"plain/null_{i:02d}.npz", "rebuilt_edge_hash": h, "W3_row_edge_hashes": rh, "match": rh == [h]}
    inv["W3_null_to_file"] = match
    inv["edge_overlap_fraction_with_connectome_per_plain_null"] = overlap
    inv["fixed_across_seeds"] = "yes: each null family has exactly one edge_hash across all its rows (see match)"
    inv_status = "pass" if all(m["match"] for m in match.values()) else "fail"
except Exception as e:  # noqa
    inv["error"] = repr(e)
exp = {}
try:
    for lab, bd in (("canonical_results", f"{CANON}/results"), ("debug_copy_results_v4", f"{ROOT}/results_v4")):
        cnt = defaultdict(lambda: defaultdict(Counter))
        for p in glob.glob(f"{bd}/**/*.jsonl", recursive=True):
            if os.path.getsize(p) > 80e6:
                continue
            for line in open(p, errors="ignore"):
                if '"family": "null' not in line and '"family": "gnull' not in line and '"family":"null' not in line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                fam = r.get("family", "")
                if fam.startswith(("null", "gnull")):
                    cnt[os.path.relpath(p, bd)][fam][r.get("status", "?")] += 1
        exp[lab] = {p: {f: dict(s) for f, s in d.items()} for p, d in cnt.items()}
except Exception as e:  # noqa
    exp["error"] = repr(e)
inv["prior_exposure_records_with_null_families"] = exp
def _null_ids_in(lab, rel_filter=None):
    ids = defaultdict(set)
    for p, d in exp.get(lab, {}).items():
        if isinstance(d, dict):
            for fam in d:
                if fam.startswith("null") and fam[4:].isdigit():
                    ids[int(fam[4:])].add(p)
    return ids


_all = defaultdict(set)
for _lab in ("canonical_results", "debug_copy_results_v4"):
    for _k, _v in _null_ids_in(_lab).items():
        _all[_k] |= {f"{_lab}:{x}" for x in _v}
_w3 = sorted(_k for _k in _all if any("W3_trained_w76" in x for x in _all[_k]))
_pm = f"{ROOT}/results_v4/learning_v4/pilot_manifest.json"
_pm_txt = open(_pm).read() if os.path.exists(_pm) else ""
_in_manifest = {f"null_{i:02d}": (f"null_{i:02d}" in _pm_txt) for i in range(12)}
_on_disk = {f"null_{i:02d}": os.path.exists(f"{CANON}/results/nulls_v2/{proto['circuit']['name']}/plain/null_{i:02d}.npz") for i in range(12)}
_trained = sorted(_all)
_untrained_on_disk = [i for i in range(10) if i not in _all]
inv["exposure_computed"] = {"null_index_to_training_records": {f"null{k}": sorted(v) for k, v in sorted(_all.items())},
                            "null_indices_with_training_rows": _trained, "null_indices_in_W3_rows": _w3,
                            "null_indices_on_disk_but_in_no_training_record": _untrained_on_disk,
                            "null_files_named_in_learning_v4_pilot_manifest.json": _in_manifest, "null_files_on_disk": _on_disk}
inv["exposure_reading"] = (f"Derived from the scan above: null indices with any training row in the scanned records = {_trained}; "
                           f"those in W3 rows = {_w3} (W3 tuned null0..2 over 9 settings x 4 seeds, then checked). "
                           f"null_00..null_09 exist on disk for the plain kind; indices on disk with NO training record in either results tree = "
                           f"{_untrained_on_disk}. Of these, the files named in results_v4/learning_v4/pilot_manifest.json (as file hashes only) = "
                           f"{[k for k, v in _in_manifest.items() if v and int(k[5:]) in _untrained_on_disk]}. "
                           "null_10/null_11 do not exist on disk. So null0..2 are NOT fresh (same fixed graphs across stages, tuning-exposed); "
                           "null3 and null4 were trained in earlier (v2/v3-era, flywire) records but not in W3; null_05..09 appear in no training record (only as file hashes in the pilot manifest).")
add("7_null_inventory", "null graph files used by W3 (null0..2), hashes, exposure, availability of search-only nulls", inv_status, inv)

# ---------------- 8 other ----------------
oth = {"nulls_anchor_over_connectome_anchor": anc["null0"] / anc["connectome"],
       "families_with_best_or_chosen_on_grid_edge": [f for f in FAMS if sel[f]["best_on_any_edge"] or sel[f]["chosen_on_any_edge"]],
       "families_chosen_not_best": [f for f in FAMS if sel[f]["chosen"] != sel[f]["best"]],
       "connectome_best_vs_chosen": {"best[core,dec]": sel["connectome"]["best"], "chosen[core,dec]": sel["connectome"]["chosen"]},
       "search_best_composite_by_family": {f: sel[f]["best_mean"] for f in FAMS},
       "chosen_search_composite_by_family": {f: sel[f]["chosen_mean"] for f in FAMS},
       "val_resolution": "acquisition val accuracy is a multiple of 1/64 (see scores); n=4 seeds, SE ~0.035 for connectome acq",
       "null_families_identical_selected_lr": {f: [choices[f]["core_lr"], choices[f]["decoder_lr"]] for f in ("null0", "null1", "null2")},
       "peak_rss_max_mib": max(r.get("peak_rss_mib", 0) for r in rows)}
add("8_other", "other observations (informational; excluded from pass table/overall)", "informational", oth)

out["checks"] = CHECKS
os.makedirs(OUTD, exist_ok=True)
json.dump(out, open(f"{OUTD}/W3_audit.json", "w"), indent=1, default=str)
md = ["# W3 audit (own re-implementation; consistency audit, not an independent review)", "", "| check | status |", "|---|---|"]
md += [f"| {c['id']} | {c['status']} |" for c in CHECKS if c["status"] != "informational"] + [""]
md += ["Informational (not counted): " + ", ".join(c["id"] for c in CHECKS if c["status"] == "informational"), ""]
md += ["Full machine-readable detail: `W3_audit.json`. Details below are truncated to 7000 chars each.", ""]
for c in CHECKS:
    md += [f"## {c['id']} ({c['status']})", c["description"], "", "```json", json.dumps(c["details"], indent=1, default=str)[:7000], "```", ""]
open(f"{OUTD}/W3_audit.md", "w").write("\n".join(md))
print(json.dumps({c["id"]: c["status"] for c in CHECKS}))
