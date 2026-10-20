# Stage W1 (encoder bundle E8) result + next-stage ledger -- 2026-10-19
Run: 144/144 rows ok, 0 failed, 95 s, 2 threads, seeds 1101-1108, reviewed script (fresh reviewer: ship), pre-launch hashes in
W1_prelaunch_hashes.txt (hashes collected at launch time, not a historical freeze beyond that). Raw: W1_encoder_E8_rows.jsonl/.json, W1_run.log.
Untrained frozen features, validation split, context-0 probe. Exploratory. Dense models excluded (no sparse encoder).

## Gate G-E1 (declared before the run in ENCODER_DIAGNOSTIC_PLAN.md Amendment 1): FAIL on all three criteria -> STOP
- (i) connectome paired gain E8-baseline on ctx0_val: -0.027 (SE 0.019, 95% CI -0.071..+0.017) -- needed >= +0.03 with CI lower > 0.
- (ii) primary contrast P (connectome gain minus mean control gain): -0.015 (SE 0.018, CI -0.057..+0.028) -- needed > 0.
- (iii) KC saturation fraction |input drive|>2: 0.073 -> 0.156 (+0.082) -- needed <= +0.05. Holding per-unit L2 norm did NOT hold saturation fixed
  (this is descriptive of input drive only; context share also doubled, 0.63 -> 1.26 per KC).
E8 did not help: all graph cells have negative mean gains (kc_static native 1385: -0.037, CI excludes 0), i.e. no support that this bundle fixes the W0 shortfall.
No trained pilot is justified by this stage. W0 remains a failed gate (0.834 < 0.85); neither stage changes the other.

## Descriptive (uncorrected, measurement only)
- Reachability: 100% of W76 readout nodes are within 5 sensory hops (<=7 too) in connectome, nulls and random_reach: propagation depth is not
  limiting in this time budget (descriptive; path-strength not assessed).
- Hidden-state saturation (|h|>0.95) is 0 at KC and readout nodes; the pre-activation (input drive) is what saturates.
- Silent features: connectome W76 has 10.5% silent features (8 of 76) vs 0% in all nulls and 0.2% random_reach (baseline). Not explained; unknown-sign
  (emit 0) nodes in the connectome readout set are a candidate hypothesis, UNTESTED. This is the only connectome-specific signature found so far and
  deserves a label-free check (sign/silence of the 8 nodes) before any further claim.
- Effective rank moves little in graph cells (+-1) but rises sharply in kc_static (+3 to +6), without an accuracy gain.

## Ledger (do not repeat)
| Phase | Status |
|---|---|
| Pilot audit (300+32) | reproduced by two separate scripts (mine, audit_v4_independent.py); fresh code review: ship-with-fixes; fixes applied; overall pass; fixes themselves NOT re-reviewed. Own re-implementation is not independent verification of historical in-memory training |
| W0 audit (296 rows) | audit pass (B-checks); W0 pre-run bindings not_logged (no hash before launch) |
| W0 | FAILED engineering gate (0.834 < 0.85) |
| Encoder source inspection + plan + plan review | done; blocking findings fixed in Amendment 1 |
| W1 E8 probe | executed; G-E1 FAIL; STOP |
| Trained pilot / full study / body physics | NOT run, not justified; body protocol still to be written as a separate workstream |
Seed registry: results_v4/learning_readout_v4_1/seed_registry.json (1101-1108 now used; 901-908 and 100601-100620 still reserved/unseen).
No job is running. Canonical v3, 14 body models, v4 pilot records untouched.

## Addendum: silent-node structural check (scripts_v4/silent_nodes.py, forward-only, seed 1201, W76, untrained connectome)
15 of 76 W76 features have window-std < 1e-6 (effectively dead), including 3 of the ORIGINAL 25 readout nodes (exactly zero). Pattern (measured):
5 silent nodes have in-degree > 0 but ZERO non-zero-sign presynaptic edges (their inputs are all unknown/modulatory-sign nodes, which emit 0 under
unknown_sign_policy=zero); most others have a single input edge from a non-zero-sign node at weight ~0.009, giving sd ~1e-7, which the fixed calibration
floor (sd+1e-3) leaves at ~0. Sign vector: 12 inhibitory, 1397 excitatory, 91 unknown of 1500. Nulls keep the same per-node sign vector but rewire, so
their readout nodes receive active inputs (0% silent in W0/W1). So the connectome-specific ~10-20% dead-feature signature is linked to the sign policy +
weak init coupling. HYPOTHESIS (untested): the unknown-sign policy (and the older navigation workflow resolves unknown->excitatory, the 266-edge
difference found earlier) costs the connectome features that the nulls keep. This is a graph/sign-policy factor, i.e. a declared scientific change.

## W2 result (unknown-sign policy zero -> excitatory; seeds 1301-1308; 240/240 rows ok, 106 s; prereg W2_PREREG.md; reviewer: ship)
Gate G-E2: FAIL / STOP. (i) connectome W76 gain -0.002 (CI -0.007..+0.003) needed >= +0.03: FAIL; (ii) P = +0.0007 (CI -0.006..+0.008) > 0: PASS but ~0;
(iii) connectome silent fraction 0.105 -> 0.040 (fall 0.066): PASS. Removing most dead features changed accuracy by ~0: the silent features are NOT what limits the probe.
kc_static sanity gain exactly 0 (OK). Sign policy is not engineering-neutral; no anatomy claim (nulls share the sign vector).
KEY CAUTION (measured): the IDENTICAL baseline connectome|W76 configuration scored ctx0_val 0.883 on seeds 1301-1308 versus 0.834 on W0 seeds 801-808
(W25: 0.803 vs 0.762); nulls 0.865-0.887 vs 0.840-0.879. An 8-seed mean therefore moves ~0.04-0.05 with the seed set, larger than every effect tested
(E8, sign policy). W0's 0.85 gate stays FAILED on its declared seeds (not retroactively passed); pooled over both seed sets the connectome W76 mean is ~0.859, and the
connectome remains statistically indistinguishable from the degree-preserving nulls in every probe. Frozen-feature probes cannot resolve <~0.05 effects with 8 seeds.

## W3 result (TRAINED validation-only pilot, W76 readout; 280/280 runs ok, 3822 s; seeds search 1401-1404, check 1501-1504; reviewer: ship)
Exploratory; 4 seeds; each family at its own selected setting; ALL selected settings sit on the 3x3 grid edge (not converged); 3 nulls are not graph-level inference.
| family | acquisition (SE) | continual final | forgetting keep |
|---|---|---|---|
| connectome W76 | 0.824 (0.035) | 0.751 | 0.084 (0.021) |
| null0 / null1 / null2 | 0.863 / 0.832 / 0.832 | 0.749 / 0.738 / 0.753 | 0.135 / 0.109 / 0.113 |
| random_reach | 0.785 | 0.763 | 0.061 |
| GRU | 0.867 (0.029) | 0.784 | 0.051 |
| encdec | 0.844 (0.011) | 0.780 | 0.076 |
- Acquisition >= 0.82: connectome 0.824 on this 4-seed validation check = DESCRIPTIVELY above 0.82 (SE 3.5 pp; per-seed 0.75-0.89; two of four seeds below 0.82). NOT a confirmation: validation data, seen seeds, edge-of-grid settings.
- Signed GRU-minus-connectome forgetting gap: -0.033 (SE 0.027) -> target >= +0.07 MISSED (GRU forgets LESS than the connectome in this pilot, opposite sign).
- Connectome minus mean(null0-2) acquisition -0.018 (SE 0.023): no anatomy advantage; nulls match or exceed it. Connectome minus random_reach +0.039 (SE 0.024).
- Net: widening the readout lifted the trained connectome from the pilot's ~74% to ~82% (different seeds; not a controlled paired comparison with the W25 pilot), but the same
  widened interface brings nulls/GRU/encdec to 83-87%, so the gain is an engineering/interface gain, not evidence for connectome anatomy; forgetting target remains missed.
Status of goals: acquisition >=82% -> met descriptively (unconfirmed); forgetting benefit >=7pp -> missed; body 68% -> not run.
