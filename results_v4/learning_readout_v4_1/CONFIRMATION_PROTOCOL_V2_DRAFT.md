# Confirmation protocol V2 DRAFT (repairs CONFIRMATION_PROTOCOL_DRAFT.md)
Status: DESIGN ONLY. Nothing frozen, run, authorized or launched. Seeds 100601-100620 are untouched. The 210-minute proposal in the first draft is NOT authorized and is withdrawn here. Power numbers come from `power_calc_v2.py` (output `power_calc_v2_output.json`, numpy, ~4 s).

## 0. Status table
| item | known | unknown / not established |
|---|---|---|
| W3 run | 280/280 rows ok, 0 failures; search seeds 1401-1404, check seeds 1501-1504, disjoint, per audit | both are VALIDATION split: exploratory, not held out |
| Connectome acquisition (4 check seeds) | mean 0.82421875, SD 0.0691, SE 0.0346, t(3) 95% CI [0.714, 0.934]; 0.0042 above 0.82 | whether population mean >= 0.82 (undetermined; see section 2) |
| GRU-minus-connectome forgetting gap | -0.0332, SE 0.0273; per seed -0.1016, +0.0313, -0.0234, -0.0391 | no evidence the +0.07 target is reachable; wrong sign |
| Tuning | every family's W3 optimum on the 3x3 grid edge | whether better settings exist (limited searched support only) |
| Capacity | trainable: sparse W76 7551, GRU 9252, encdec 8956 (not matched) | effect of matching on the gap |
| Anatomy | nulls 0.83-0.86 acquisition, 3 nulls | nothing graph-level; descriptive only |
| Fresh data | seeds 100601-100620 unused as numbers | whether the generator yields novel test data (section 6) |

## 1. Corrections to the old draft (verified against files/code)
1. Nulls are `plain/null_00..09.npz`, not `group_constrained` (W3 `build_model_v2` uses `packs["plain"]` for `nullK`). The "two search-only nulls null_10..11" do not exist.
2. "GRU/encdec matched to the W25 sparse count" is wrong in effect: `build_model_v2` matches to `ref_param_report(...)["trainable_params"]` of the reference sparse model. Arithmetic from the code (n_in = 19, derived: 22*407+2 = 8956): that target lies in (8934, 8956], consistent with 7399 edges + 1500 biases + 50 W25-decoder params = 8949 (inferred, not re-run). The agent actually trains 7551 (7399 core + 152 decoder).
3. The first draft's stage arithmetic is self-consistent (5880 s search, 2477 s R0, 835 s R1) but labels "75 cells x 3 seeds" for what is 75 runs (25 cells x 3 seeds).
4. "n=12 or 20 can distinguish a true 0.07 gap from zero" tests against zero, not against 0.07; it does not support a 0.07 claim.
5. The detectable-effect table used two-sided alpha and a point-estimate pass rule; the pass rule was undefined as a statistical claim (section 2).

## 2. Descriptive thresholds versus statistical support
Historical descriptive targets stay as they are: mean acquisition >= 0.82; signed mean gap (GRU minus connectome forgetting_keep) >= +0.07, abs() never allowed. A point estimate meeting a target is a description of the sample. Population claims need the rules below, fixed before data.

Let theta1 = population mean acquisition, theta2 = population mean paired gap (n seeds, paired by seed, primary matching of section 4). One-sided 95% t bounds (alpha = .05 each).
- E1 (threshold claim): SUPPORTED if lower 95% bound >= 0.82; BELOW if upper 95% bound < 0.82; otherwise INCONCLUSIVE.
- E2a (weaker, different claim, "gap > 0"): SUPPORTED if lower bound > 0.
- E2b (magnitude claim, "gap >= 0.07"): SUPPORTED if lower bound >= 0.07; BELOW if upper bound < 0.07; else INCONCLUSIVE.
CI exclusion of zero (E2a) never implies E2b.
Hierarchy (fixed sequence, no alpha split): E1 -> E2a -> E2b; a step is tested only if the previous one is SUPPORTED, otherwise later steps are reported descriptively as "not tested under the hierarchy". E2a precedes E2b because the same lower bound decides both (one bound, no multiplicity cost). E3 (matched-accuracy), anatomy, random_reach and replay are secondary/descriptive and carry no pass/fail claim. A search-adequacy failure (section 4) makes that family's comparison UNRESOLVED and excludes it from the hierarchy.

### Power and precision (real output, alpha = .05 one-sided, P(lower bound clears threshold))
Acquisition vs 0.82, pilot SD 0.0691 (analytic):
| true mean | n=8 | 12 | 16 | 20 | 24 | 32 | 40 |
|---|---|---|---|---|---|---|---|
| 0.80 | .008 | .005 | .003 | .002 | .001 | .001 | .000 |
| 0.82 | .050 | .050 | .050 | .050 | .050 | .050 | .050 |
| 0.824 | .067 | .073 | .077 | .0815 | .085 | .093 | .099 |
| 0.85 | .296 | .408 | .505 | .590 | .662 | .775 | .854 |
| 0.88 | .714 | .879 | .952 | .982 | .993 | .999 | 1.000 |

Same at SD uncertainty (truth 0.85): SD 0.043 (lower 90% end) .556/.735/.848/.915/.954/.987/.997; SD 0.202 (upper 90% end) .103/.123/.141/.158/.174/.205/.235; SD 0.10 sensitivity .190/.252/.309/.363/.413/.506/.587. Averaged over pilot-SD uncertainty (scaled inverse chi-square, an assumption): truth 0.85 gives .281/.376/.453/.515/.574/.653/.714; truth 0.824 gives .067 to .099.

Gap vs 0.07, pilot SD 0.0546 (per-seed SD = SE 0.0273 x 2):
| true gap | n=8 | 12 | 16 | 20 | 24 | 32 | 40 |
|---|---|---|---|---|---|---|---|
| -0.03, 0.00, 0.03 | .000 | .000 | .000 | .000 | .000 | .000 | .000 |
| 0.07 | .050 | .050 | .050 | .050 | .050 | .050 | .050 |
| 0.10 | .405 | .556 | .674 | .765 | .833 | .918 | .962 |
At SD 0.0338 truth 0.10 gives .730/.891/.959/.985/.995/.999/1.0; at SD 0.159 .122/.151/.177/.202/.227/.273/.317; at SD 0.10 .190/.252/.309/.363/.413/.506/.587. At truth = threshold power is alpha, by construction.

Required n for 80% power (analytic, binary search to n = 5000):
| claim | SD 0.043 | SD 0.069 | SD 0.10 | SD 0.202 |
|---|---|---|---|---|
| acquisition, truth 0.85 | 15 | 35 | 71 | 282 |
| acquisition, truth 0.88 | 5 | 10 | 19 | 72 |
| acquisition, truth 0.824 | 710 | 1847 | 3866 | >5000 |
Gap >= 0.07 (SD 0.034/0.055/0.10/0.159): truth 0.10 needs 10/22/71/176; truth <= 0.07 is never powered (infeasible). Gap > 0 (weaker): truth 0.03 needs 10/22/71/176 (same SD columns), truth 0.07 needs 4/6/15/34.
So at truth 0.824 (the pilot estimate) establishing mean >= 0.82 is infeasible (about 1850 seeds at the pilot SD); at truth 0.85 it needs ~35 seeds at the pilot SD, 71 at SD 0.10, up to 282 at the upper SD bound. Only 20 reserved seeds exist.
Precision (two-sided 95% CI half-width): pilot SD n=12 0.044, n=16 0.037, n=20 0.032, n=40 0.022; at SD 0.10 n=20 0.047; at SD 0.202 n=20 0.094.

Power assumptions: per-seed values are treated as normal and independent across seeds with known true SD; acquisition is bounded at 1 and takes discrete values (ceiling and discreteness effects ignored), and the gap SD is taken as 2 x its pilot SE. Heavy tails or ceiling would change the numbers; treat them as approximate.

### Decision rule for n (not chosen by which passes)
(a) Fixed-n estimation study: declare a CI half-width target before data (for example <= 0.035 at the pilot SD, which needs n >= 18, so n = 20 using all reserved seeds); report E1 as SUPPORTED, BELOW or INCONCLUSIVE and make no power claim; expect INCONCLUSIVE unless the truth is well above 0.82 (at truth 0.824 P(E1 supported) is about 8%). Under the fixed sequence E2a/E2b are tested only if E1 is SUPPORTED; otherwise the gap is reported descriptively (estimate, interval, per-seed values) with the label "not tested under the hierarchy" and no category. (b) Powered threshold study: n justified by this table under a stated plausible effect; E1 at truth 0.85 needs ~35+ seeds, beyond the reserved 20, so (b) for E1 requires new seeds and a new reservation. For the gap, (b) is not justified: the pilot gap is -0.033 and every truth <= 0.07 has zero power. Sample size is a function of the plausible effect and SD only.

## 3. Anatomy inference
The 10 sign-flip minimum p-value (2/2^10 = .002) is only an algebraic floor; it does not validate the test.
- All contrasts d_i = c - n_i share ONE connectome estimate c. With var(c) = var(n), corr(d_i, d_j) = var(c)/(var(c)+var(n)) = 0.5. They are positively dependent, not independent units.
- Sign flipping d_i needs d_i symmetric about 0 and independent. Under the null "the connectome is exchangeable with the nulls", the exchangeable objects are the graphs, not the differences; flipping one d_i = c - n_i to n_i - c swaps c and n_i for that i only, which a shared c does not allow. Nulls also share the sign vector and tuning setting with the connectome (W1 addendum), so exchangeability is conditional on those.
- The estimand is graph-level: where the connectome's per-graph mean (over the same seeds) falls in the distribution of null-graph means. Seeds are repeated measures per graph, not units for this question.
Valid alternative: rank/permutation. On each metric compute per-graph means over the same seeds for the connectome and K nulls under identical settings; the connectome's rank among K+1 gives exact one-sided p = rank/(K+1). Resolution 1/(K+1).
- K = 10: minimum p = 1/11 = .091, cannot reach .05. K = 19 is needed for .05.
- Only `plain/null_00..09` exist (10 graphs). null_00..02 are known to be exposed (W3 ran null0-2; earlier-stage records also use them); null_03/04 were exposed in earlier records too (W0 reports nulls0-4 and W2_PREREG lists null0-null4 as families; W3 audit check 7 shows only that null_00..09 exist with hashes and that null_10/11 are unavailable, it does not itself document exposure). Only null_05..09 appear in no record I found, so K = 5 and minimum p = 1/6 = .167.
- Generating more graphs is a data change needing its own freeze and review; do not generate silently; no confirmation graph may be used for tuning.
Rank-test assumptions, all required and none verified: (i) the real graph is exchangeable with the edge-swap null graphs under the null (a modelling claim about the null generator, not proven); (ii) the null chain mixing was judged `adequate_with_limitations` in the null-generation report/audit, so independence between nulls is partial; (iii) tuning only on the connectome favors the connectome; (iv) the rank is conditional on the shared task seeds, which are the same for every graph, so it does not average over task draws.
Verdict: anatomy is DESCRIPTIVE (rank of connectome among null_05..09, per-graph means and the 5 null values reported, no p-value as a claim). Independent statistical review is required before any anatomy inference. Primary nulls run at the connectome's selected setting (one setting for all sparse graphs, no null tuning, so null_05..09 are never used for selection); this favours the connectome and is a stated limitation.

## 4. Capacity and tuning honesty
Actual trainable parameters (W3 logged; n_in = 19):
| family | trainable | ratio to sparse |
|---|---|---|
| sparse W76 (7399 core + 152 decoder) | 7551 | 1.000 |
| GRU h=37 (6h^2+28h+2) | 9252 | 1.225 |
| encdec h=407 (22h+2) | 8956 | 1.186 |
Primary matching policy: GRU and encdec resized so trainable count is as close as possible to 7551: GRU h=33 gives 7460 (0.988x; h=34 gives 7890, 1.045x), encdec h=343 gives 7548 (1.000x). Historical GRU37/encdec407 are labelled SECONDARY controls (capacity not matched, larger). Equal parameter count is not equal capacity (the sparse encoder is fixed; recurrent structure differs); no exact-capacity claim is made.
Code change: yes. `build_model_v2` takes the target only from `ref_param_report`, so resizing needs a new construction path (explicit hidden, or a different target) in a new script; it must not edit `learning_v2.py`, and it requires a new source freeze plus selftest asserting 7460/7548 counts.
Tuning exposure: connectome/GRU/encdec anchors come from v3 selections, nulls and random_reach borrow other v3 anchors, and W3 tried only ×{1/3,1,3}. A common grid gives equal new coordinates, not equal prior exposure; this is documented, not repaired. Prospective procedure (identical for every family including the resized ones): 5x5 grid, multipliers {1/9,1/3,1,3,9} on each family's v3 anchor (all W3 winners are then interior of the grid), S fresh validation seeds, one rule (composite, one-SE with the existing simplicity order). The resized GRU/encdec have no anchor of their own; they reuse the historical GRU/encdec anchors, which is stated.
Search-adequacy gate (predeclared): the best cell and the chosen cell must both be interior (not on any grid edge) and have all S seeds ok. An edge optimum means limited searched support, not proof a better setting exists. A family failing the gate is UNRESOLVED: no held-out seeds are spent on its comparison, no grid extension, no rescue.

## 5. Runtime as a coordinate manifest (parametrized, S = 3)
A search is 5 x 5 settings x S seeds = 25S runs per graph/family. No auto extension of grids, replay launches, or runtime. One heavy job at a time, only via `scripts_v4/run_guarded.sh`. Row identity = (stage, family, core_lr, decoder_lr, seed); the manifest lists all coordinates and its hash is frozen before compute.
Search seeds: none assigned now; S fresh validation seeds are to be assigned from an unused range and grep-verified against `seed_registry.json` and the generator check before the freeze (disjoint from 1401-1404, 1501-1504, 901-908 and 100601-100620).
Timing (measured in v4 pilot/W3): connectome ~16 s, GRU ~10 s, encdec ~4.4 s per run; nulls and random_reach assumed 16 s (no measurement); resized GRU33/encdec343 assumed at the same cost as their historical versions (unmeasured); replay UNMEASURED, range 1.0x to 3.0x.
Stage S (search, validation): 
| search graph/family | settings | seeds | runs | s/run | seconds |
|---|---|---|---|---|---|
| connectome | 25 | 3 | 75 | 16 | 1200 |
| random_reach | 25 | 3 | 75 | 16 (assumed) | 1200 |
| GRU h=33 (primary) | 25 | 3 | 75 | 10 | 750 |
| encdec h=343 (primary) | 25 | 3 | 75 | 4.4 | 330 |
| GRU h=37 (secondary) | 25 | 3 | 75 | 10 | 750 |
| encdec h=407 (secondary) | 25 | 3 | 75 | 4.4 | 330 |
| nulls null_05..09 | none (connectome setting reused) | | 0 | | 0 |
| total (6 families x 75) | | | 450 | | 4560 (76 min) |
Stage C (confirmation, TEST split, R0 only), n seeds: connectome n, random_reach n, null_05..09 5n, GRU33 n, encdec343 n, GRU37 n, encdec407 n = 11n runs; seconds per seed = 16+16+80+10+4.4+10+4.4 = 140.8. n=12: 132 runs, 1690 s (28 min). n=20: 220 runs, 2816 s (47 min).
Replay (R1) is excluded from this protocol: it is a separate factor needing its own matched design and freeze. For information only, connectome+random_reach+GRU33+encdec343 would cost n x 46.4 s x (1.0 to 3.0): n=12 gives 557 to 1670 s, outside the frozen total.
Guard overhead: data load 1-2 min per process start; 3 starts (search, pre-freeze data check, confirmation) = 3-6 min.
| scenario | runs | compute | + overhead | frozen TOTAL wall cap (compute + overhead, +25% margin on assumed timings, rounded) |
|---|---|---|---|---|
| S=3, n=12 | 450 + 132 = 582 | 6250 s = 104 min | 107-110 min | 140 min |
| S=3, n=20 | 450 + 220 = 670 | 7376 s = 123 min | 126-129 min | 165 min |
One TOTAL cap covers all stages together (no per-stage resets). Exceeding it aborts, keeps the JSONL and reports partial. Failed or non-finite rows are recorded, never dropped or rerun; abort above 5% failures. These caps are proposals only: the old 210 min is not authorized and neither is any number here.

### Appendix: coordinate manifest and anchors
Each search family f has anchor a_f and the grid core_lr = a_f x m_i, decoder_lr = a_f x m_j for m in {1/9, 1/3, 1, 3, 9} (i, j = 1..5; values rounded as float(f"{x:.6g}")) and seed s_k, k = 1..S. Rows are (stage S, f, core_lr, decoder_lr, s_k): 25 x S per family, 6 families = 150 S rows (450 at S = 3).
| family | v3 anchor a_f | W3 chosen (core, decoder) |
|---|---|---|
| connectome | 0.0017172484 | 0.00171725, 0.00515175 |
| GRU (h=33 and h=37, reuse historical anchor) | 0.00015098037 | 0.000452941, 5.03268e-05 (h=37) |
| encdec (h=343 and h=407, reuse historical anchor) | 0.00014588493 | 0.000437655, 0.000437655 (h=407) |
| random_reach | 0.001 | 0.001, 0.001 |
(null anchor 0.0054259466 is unused: nulls are not searched. W3 chosen values are from `W3_trained_w76_choices.json`.) Confirmation rows are (stage C, family, chosen setting from stage S, seed c_k), k = 1..n, for the 11 families listed above.

## 6. Fresh-data verification and freeze order
Unused numeric seeds do not prove fresh data. Before freezing (requires owner authorization to touch the range; the check builds task objects only, trains nothing, reveals no results): through `learning_v2.make_task` / `ContingencyTask(spec, seed)` generate the TEST split for each confirmation seed; hash the arrays (teachers from `SeedSequence([seed,99])`, split sets from `SeedSequence([seed, split_id, ctx])`) and compare against the hashes of train/val/test of every seed in the registry (v2, v3, v4 pilot, W0-W3). Record the full inventory. Whole-task identity with any prior seed disqualifies that seed; this is decided and logged BEFORE the freeze, when the seed list is still open (it is the only point where a seed may be swapped; the replacement comes from the reserved range in order and is logged). After the freeze there are no replacements: if fewer than the frozen n seeds complete (failures, aborts, wall cap), the analysis uses the completed seeds as they are, any endpoint with fewer seeds than a frozen minimum (set at freeze, e.g. 0.75 n) is INCONCLUSIVE, and nothing is rerun or substituted; partial cue-pattern overlap is expected by chance (the audit found validation cue patterns overlapping across seeds) and is reported, not a disqualifier. W3's "fresh check" was validation split, therefore exploratory.
Freeze order: (0) fresh-data check above, with its inventory and any pre-freeze seed swaps logged (it requires explicit owner authorization because it runs the task generators for seeds 100601-100620; no model is trained and no result is seen); (1) validation decisions final (S search done, choices.json, search-adequacy verdicts); (2) source hashes; (3) config and this document (final); (4) selection files; (5) input inventory (circuit parquet, null npz for null_05..09, W76 manifest, coordinate manifest); (6) analysis script, written against the rules in sections 2-3; (7) seed list and n written to `seed_registry.json`; (8) fresh reviewer (not the author, not a W3 result reviewer) gives a written ship/no-ship; the author does not launch. After the freeze: no retuning, no replacement seeds, no hidden or rerun failed attempts, no optional stopping. Any change voids the freeze.

## 7. Recommendation
Do not launch anything now. The first-draft confirmation is not justified: the observed signed gap is -0.033 (per seed three of four negative); at every true gap <= 0.07 the chance of establishing "gap >= 0.07" is at most alpha, and a powered study of 0.07 would need a plausible truth near 0.10, which no evidence supports. Acquisition is 0.0042 over its threshold with SE 0.035, so a threshold study at the truth the data suggest is infeasible.
What a confirmation (design (a), n = 20, primary matching, adequacy gates passed) could establish: a TEST-split estimate of acquisition with half-width ~0.03 if SD is near 0.07 (up to 0.09 if not), and the E1 category. The gap is reported descriptively (estimate, interval, per-seed) unless E1 is SUPPORTED, which is unlikely if the truth is near 0.824; only then are E2a/E2b tested. It also gives descriptive ranks versus 5 unexposed nulls.
What it could not establish: population acquisition >= 0.82 if truth is near 0.824; a 0.07 advantage unless the true gap is already about 0.10; any wiring-specific claim (K = 5, min rank p .167); fair comparison for any family failing the adequacy gate; replay or memory-redesign effects; body-level (68% body) claims.
Any next step belongs to a prospective memory-focused design, validation only, with its cost stated before a compute decision.
