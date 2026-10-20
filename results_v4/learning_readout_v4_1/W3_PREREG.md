# W3 preregistration: trained validation-only exploratory pilot of the W76 widened readout
Written 2026-10-19 BEFORE any W3 run. Namespace results_v4/learning_readout_v4_1, stage tag `W3_trained_w76`.
Exploratory, validation split only, NOT a study result, NOT a confirmation.

## 0. Honest justification
- W0 gate (0.85 on seeds 801-808) FAILED (connectome W76 0.834) and STAYS failed. W1 (E8) gate FAILED, W2 (sign policy) gate FAILED.
- The identical connectome W76 configuration scored 0.883 on seeds 1301-1308 (W1_RESULT_AND_LEDGER.md, W2 section); pooled over both seed sets
  ~0.859. The seed-set swing (~0.04-0.05) exceeds every effect tested; frozen-feature probes cannot resolve it.
- This pilot is justified by the owner's standing directive to keep iterating plus that pooled estimate, NOT by passing a gate. It cannot change
  any earlier gate status (W0/W1/W2 verdicts are fixed). No W76 result here, positive or negative, is retroactively a W0 pass.

## 1. Single scientific change
Readout node set W25 (base.readout_idx) -> W76 for every sparse family, applied BY INDEX. Construction identical to scripts_v4/width_probe.py:
W76 = W25 + eligible nodes (not sensory, not in W25, in-degree>0) sorted by (-contacts, index); asserted equal to W0_width_probe_manifest.json
`sets.W76` (first 25 = original). Nothing else changes: same protocol, encoder, task, learner, optimizer, z-scoring, budgets.
Consequences of the single change, stated: the decoder grows 25x2 -> 76x2 (50 -> 152 decoder params); random_reach resamples until every READOUT node
is reachable, so its graph is re-drawn against the W76 set (same procedure, different target set). Null graphs are the same packs, readout by index.
gru/encdec have no readout set. Their hidden sizes are matched to the sparse parameter count, which WOULD change with W76, so they are built from the
ORIGINAL W25 base so they are exactly the v4-pilot architectures (gru hidden 37, encdec 407); the selftest asserts this (measured: a W76 base would
leave gru at 37 but move encdec to 412, which would be a second, unintended change).

## 2. Design
Families: connectome, null0, null1, null2, random_reach, gru, encdec. Arm bptt, mode std, splits ['val'], protocols association+continual, exactly the
v4 learner (learning_v4.run_setting: core/decoder Adam param groups, make_agent_v4). No replay. No test split.
Anchors (learning_v4.v3_lrs() = results/learning_v3/search/selection.json bptt|<arm>, the source of the v4 pilot anchors):
connectome 0.0017172484, gru 0.00015098037, encdec 0.00014588493. DEVIATION from the brief's expectation: the v4 pilot ran ONLY connectome/gru/encdec
(its records contain no null or random_reach), so there is no pilot null anchor. null0/null1/null2 use the v3 `bptt|null` selection (0.0054259466) and
random_reach uses v3 `bptt|random_reach` (0.001), from the same selection.json that supplies the pilot anchors.
Search: 3x3 grid (core_lr, decoder_lr), each axis = anchor x {1/3, 1, 3} (values float(f"{c*r:.6g}")), search seeds 1401-1404.
Search rows: 7 x 9 x 4 = 252. Check: single chosen setting per family on fresh seeds 1501-1504: 28 rows. Total 280.
Selection rule (ONE, predeclared, deterministic, per family): per-seed composite=(val_acc_acq + continual final_mean_acc_val)/2; per setting mean and
SE (sd/sqrt(4)) over the 4 seeds; best = max mean (ties -> two_group_simplicity); chosen = simplest setting (learning_v4_select.two_group_simplicity,
base_lr = family anchor) among settings with mean >= best_mean - SE_best (learning_v4_select.one_se_choice_2d). A setting with fewer than 4 ok
seeds is ineligible; a family with no eligible setting gets no check run (reported as such). Edge-of-grid optima are flagged and NOT extended.
Reported for check seeds, per seed and mean(SE): acquisition (association.val_acc_acq), continual final accuracy (continual.final_mean_acc_val), forgetting
(continual.avg_forgetting_keep_val); signed paired (by check seed) GRU-minus-connectome forgetting gap with SE; connectome-minus-null acquisition
(paired by seed; against the mean of null0-2, and against random_reach separately). Targets (acquisition >= 0.82; signed gap gru-minus-connectome
>= 0.07, abs() never allowed, negative never passes) are descriptive pass/fail only.
Statement: three nulls are NOT graph-level inference; the connectome-vs-null contrast is descriptive on 3 fixed null graphs + 1 random_reach draw per
seed, each at its own selected setting, n=4 seeds. Not powered, not a claim about wiring.

## 3. Budget and failure handling
Runtime estimate from v4 pilot elapsed_s per run (search/check means: connectome 15.8/16.0 s, gru 10.2/10.0 s, encdec 4.3/4.9 s; nulls and random_reach
have no pilot timing and are assumed = connectome, W76 adds only a 76-wide decoder): 40 runs per family (36 search + 4 check).
connectome 40x16=640 s; 4 more sparse families 4x640=2560 s; gru 40x10.2=410 s; encdec 40x4.4=175 s. Total ~3785 s = ~63 min (+ data load, ~1-2 min).
This is < 100 min, so all three nulls are kept (no reduction to null0/null1). Hard cap 120 min wall from process start (checked between runs):
abort, write partial, preserve JSONL. torch threads 2. Run only via scripts_v4/run_guarded.sh. Failure rule: failed/non-finite rows are recorded, never
dropped, and the run aborts if failures exceed 5% of 280 (>14). No reruns of failed or completed cells (resume skips every recorded key, including failed).
Manifest (sha256 of this file, scripts, sources, anchors, W76 set, plan hash) is written BEFORE compute; append-only JSONL keyed by
(family, core_lr, decoder_lr, seed); atomic final JSON. The measurement run is NOT launched by the author of this prereg.

## 4. Seeds
`W3_search_committed` [1401-1404] and `W3_check_committed` [1501-1504] added to seed_registry.json after grep: they occur in the repo only as node
indices inside W0_width_probe_manifest.json (sets lists, 1404 in W76) and inside hash strings, never as seeds.

## 5. Limits stated in advance
4 search seeds with a 9-cell grid will often leave many settings within 1 SE; the simplicity order then picks the lowest-multiplier setting. 4 check
seeds cannot resolve differences < ~0.05 (seed-set swing seen in W0/W2). The 0.883 vs 0.834 swing means a good or bad number here is weak evidence.
Passing the descriptive targets would not retro-pass W0 and would not be a confirmation; failing them changes nothing already decided.

## Pre-launch clarifications (reviewer nits, 2026-10-19)
- random_reach: the code stops at the connectome's own reachable fraction (f >= ref), otherwise takes the best of up to 200 tries with a warning; it does NOT guarantee every readout node is reachable.
- random_reach anchor 0.001 and the null anchor (~0.0054, ~3x the connectome anchor) are borrowed from v3 selections; families are tuned at different quality, so the signed gap mixes tuning effects.
- Seeds 1402/1403 also appear as table labels in docs/interventions_v2.md (not seeds). A non-finite composite score would raise in selection after all search rows are saved (rows preserved).
