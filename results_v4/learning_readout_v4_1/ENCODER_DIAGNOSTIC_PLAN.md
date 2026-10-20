# Encoder / sensory-interface causal diagnostic plan (PLAN ONLY, nothing executed)
Status: proposal, needs owner nod. Does NOT alter W0 (gate connectome >= 0.85 failed, best 0.834; stays failed).

## 1. Source facts (line refs, project copy)
- learning.py:69-80 `_fixed_encoder`: enc[n_in, n_sens] zeros; per sensory unit j, 4 input rows drawn without replacement, set to 1.0 (binary, +1, unnormalised); frozen (requires_grad False). `n_cue` unused (signature l.69, passed l.91; also learning_v2.py:237). Rows span ALL n_in=19 channels (16 cue + 3 context), so context channels are sampled like cues.
- configs/experiments/learning_v3_protocol.yaml: encoder_fan_in 4 (also v2, v1), n_cue 16, k_active 5, n_ctx 3, T 8, cue_steps 6, resp_steps 3, noise 0.05, tau 5, dt 1 (alpha 0.2), init_gain 0.005, prior log1p, unknown_sign_policy zero. 1385 sensory/KC, 25 readout, W76 = 25 + 51 in-degree>0 non-sensory (PREREG l.5).
- sparse_rnn.py:215-228 step: drive = W@h + bias(0) ; inp = x @ encoder added at sensory idx (l.218-219, gain 1, no scaling); h' = 0.8 h + 0.2 tanh(drive) (l.226; nonlinearity tanh l.80). Edge w = sign[pre]*softplus(raw)*log1p(count) (l.194); recurrent weights ~0.005*log1p(>=5) ~0.009 each, so recurrent drive is tiny vs encoder drive 0..4 (tanh saturates for sums >=2). Unknown-sign nodes emit 0 (policy zero). z_dim=0 so M unused.
- association.py:120-129 make_input: cue on steps [0,6) in cue channels (5 of 16 =1), context one-hot constant for all 8 steps, gaussian noise 0.05; response_window = steps 5..7 (l.131-132). width_probe.py:43-56 reads h after step t for t in window, z-scored per feature, mean over window; probe = logreg (l.65-70), ctx0 val = `ctx0_val`.
- Per-KC expected context channels = 4*3/19 = 0.63 (about 47% of KCs see >=1 context channel; constant, cue-independent offset, shifts tanh operating point).

## 2. Plausible but UNTESTED
(a) cue/context mixing (context channel sharing KC with cue lines); (b) drive scaling/saturation (unnormalised sums up to 4, tanh); (c) fan-in density vs scale confound (changing fan-in changes both); (d) propagation depth: sensory->readout hops vs usable steps. Reachability was NOT computed (no cached result found; do not run heavy code now). Method: from edge arrays (pre,post) build scipy.sparse CSR, multi-source BFS from sensory_idx limited to depth 7 (state after step t holds hop <= t; window t=5..7; cue off after step 5, so cue information needs hop <= ~7 and survives leak 0.8^k), record per readout node min hop, number of paths by hop (A^k @ 1 counts), and fraction of readout nodes reachable by hop <= 5/7. Edge-only, seconds.

## 3. Measurements (UNTRAINED, readout set W76 fixed, frozen features, no training)
Families: connectome, null0-4, random_reach (W76), kc_static (1385, plus K76 sub-sample as in W0). Per seed and condition:
1. input x: mean/var per channel; per-KC count of context vs cue inputs.
2. KC preactivation (x@enc + recurrent): mean, sd, fraction |pre|>2 (saturation); post-activation tanh: sd, fraction |h|>0.95; same for W76 readout nodes at window steps.
3. variance across cues vs across contexts of KC and readout activity (variance decomposition: cue-driven / context-driven / noise); effective rank and mean |corr| (reuse stats() in width_probe.py).
4. Reachability: sensory->readout min-hop and path-count distribution vs steps available (window start 5, end 7); fraction of W76 nodes with hop<=7; per-node silent fraction.
5. Cue/context separability: ctx0 probe accuracy from KC layer alone (kc_static) vs W76; linear decodability of context identity from features; cross-context cue-probe transfer.

## 4. Single intervention
Baseline: fan_in 4, binary weight 1 (row L2 norm 2).
Intervention E8: fan_in 8, weight 2/sqrt(8) = 0.7071 (per-unit L2 norm held at 2, i.e. identical to baseline preactivation variance for independent inputs; 1/sqrt(fan_in) normalisation held explicit). Same SeedSequence([seed,3]) stream, same freeze. Only density/mixing changes; scale is held fixed, so any effect is attributable to density, and measurements in section 3 (saturation, context share) verify the scale claim empirically rather than assume it.
Why: source shows (i) unnormalised sums so any fan-in change alters scale, which is the confound to remove; (ii) fan-in 4 with 19 channels means 4-subsets of 16 cues give limited distinct codes per KC and kc_static/encdec matched-K already beat the connectome, pointing at the interface rather than anatomy. Caveat: higher fan-in also raises context share (1.26/KC), recorded as a co-varying mechanism, not separated (declared limitation).
Applied identically to connectome, null0-4, random_reach (W76), and kc_static (K1385 and K76 sub). Implement as a NEW script/wrapper outside existing source (monkeypatch/subclass of _fixed_encoder); existing files untouched.

## 5. Seeds
Primary: unused seeds 1101-1108 (not in v2/v3/v4/probe registries, not 901-908 or 100601-100620; owner to confirm against registry). Optional free exploratory replication on 801-808 only as paired diagnostic, declared reuse, not confirmatory evidence. 901-908 and 100601-100620 are NOT consumed.
Paired design: same seed (task, decoder init, encoder stream) across baseline and E8.

## 6. Primary metric and analysis
ctx0 mean val probe accuracy (width_probe.py probe/feats, unchanged). Per seed d = acc(E8) - acc(baseline); report mean, SE = sd(d)/sqrt(8), t(7) 95% CI; also train-val gap and eff_rank. Same for each family; contrast: connectome gain minus mean control gain (paired by seed).

## 7. Budget and runtime
9 cells (7 graph families at W76 + kc_static K1385 + K76) x 2 conditions x 8 seeds = 144 rows. W0 ran 296 rows in 128 s (0.43 s/row) so ~65 s plus ~1-2 min for untrained stats and BFS; est. < 5 min, 2 threads (torch.set_num_threads(2)), one heavy job, well under 15 min. Controls: nulls/random_reach/kc_static are the control budget; no extra tuning, no LR search, no training.
Failure handling: append-only jsonl, flush per row, resume skipping done (family,cond,seed); any NaN/exception recorded and counted, never dropped silently; if > 5% rows fail, stop and report without inferring; no reruns of selected seeds; abort at 12 min wall.

## 8. Prospective stop/proceed (NEW separate gate G-E1, justified: W0 gate unchanged and unmet)
Proceed to a prereg for a trained pilot only if ALL: connectome paired gain >= +0.03 with 95% CI lower bound > 0; connectome gain exceeds mean null/random_reach gain by >= 0.01 (otherwise gain is generic readout richness, no anatomy claim); saturation (fraction |pre|>2) in E8 not larger than baseline by > 5 pts (confirms scale held). Otherwise stop: report that encoder density at fixed scale does not explain the W0 shortfall and move to option (ii) (report negative result). G-E1 does not retroactively pass or fail W0.

---
# AMENDMENT 1 (2026-10-19, written after independent plan review, BEFORE any measurement). Supersedes conflicting text above.
Review finding source: fresh reviewer; its blocking items are resolved as follows. W0's gate stays FAILED (0.834 < 0.85).

1. Nested wiring. Baseline encoder stays byte-identical (stream SeedSequence([seed,3]), fan-in 4, weight 1). E8 = baseline 4-sets
   (same stream, same draws) PLUS 4 extra distinct channels per unit drawn from the other 15 channels with a SEPARATE stream
   SeedSequence([seed,5]); all 8 weights = 2/sqrt(8) (per-unit L2 norm 2 held at baseline). The baseline stream is therefore not shifted.
2. Attribution wording. The intervention is a BUNDLE "E8": more channels per unit + expected context share 0.63 -> 1.26 per KC +
   normalised weight. A positive effect is attributed to "E8 versus baseline", NOT to density alone. No context-excluded control is run (declared limitation).
3. Dense models. gru/encdec have no fixed sparse encoder; they are EXCLUDED from E8 (W0 native/matched-K rows stand as descriptive reference).
   Cells: connectome, null0-4, random_reach at W76 (7) + kc_static native 1385 + kc_static K76 = 9 cells x {baseline, E8} x 8 seeds = 144 rows.
4. Registry. Seeds 1101-1108 are committed here as used by this stage (see seed_registry.json). 801-808 NOT replicated. 901-908 and 100601-100620 stay reserved.
5. One primary contrast: P = (connectome ctx0_val gain E8-minus-baseline) minus (mean gain over null0-4 + random_reach), paired by seed, with mean/SE/t(7) CI.
   Everything in section 3 (saturation, reachability, rank, separability) is DESCRIPTIVE, uncorrected, not used for any claim beyond measurement.
   Untrained note: the probe reads frozen initial-gain features; it tests the interface's representation, not learning.
6. Gate G-E1 (replaces section 8; thresholds justified from W0): W0 per-seed ctx0 sd of connectome is 0.04-0.07, so a paired mean gain of +0.03 is
   about half a seed sd and about the size of one W0 width step's within-noise differences; below it the effect is not distinguishable from noise with 8 seeds.
   Proceed to a PREREGISTRATION (not a run) of a trained pilot only if ALL: (i) connectome paired gain >= +0.03 AND 95% t(7) CI lower bound > 0;
   (ii) primary contrast P point estimate > 0 (reported with CI; the design is underpowered for margins below ~0.05, so a nonsignificant P means any gain is not
   shown to be connectome-specific and no anatomy claim is allowed); (iii) fraction |pre|>2 in E8 within +5 points of baseline (scale held).
   If any fails: STOP; report that this encoder bundle does not explain the W0 shortfall; no trained study; next decision goes to the owner
   (candidate follow-ups: reachability/propagation investigation, or finalize the negative result).
7. Mechanics: new script scripts_v4/encoder_probe.py wraps width_probe functions; absolute paths; append-only JSONL resumable by (family,cond,seed);
   atomic final JSON; explicit stage id W1_encoder_E8; timeout 12 min; run via run_guarded.sh (2 threads, shared heavy lock, Windows free RAM >=1.5 GiB);
   source/prereg/script sha256 recorded to a manifest BEFORE launch. Must pass a fresh code review before launch.
