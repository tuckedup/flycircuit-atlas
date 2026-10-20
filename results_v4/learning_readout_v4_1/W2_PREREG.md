# W2 preregistration: unknown-sign policy probe (UNTRAINED, frozen features)
Written 2026-10-19 BEFORE any W2 measurement. Namespace results_v4/learning_readout_v4_1, stage tag `W2_sign_policy`.
Does NOT alter W0 (gate 0.834 < 0.85, FAILED) or W1 (G-E1 FAIL, STOP). Those verdicts stand regardless of W2.

## 1. Motivation
Addendum at the end of W1_RESULT_AND_LEDGER.md (scripts_v4/silent_nodes.py, seed 1201, untrained connectome, W76): 15 of 76 W76 features are
effectively dead (window std < 1e-6). 5 silent nodes have in-degree > 0 but zero non-zero-sign presynaptic edges (inputs all from unknown-sign
nodes, which emit 0 under unknown_sign_policy=zero); most others have one weak input edge (~0.009) from a non-zero-sign node, so the
calibration floor (sd+1e-3) leaves them ~0. Sign vector: 12 inhibitory, 1397 excitatory, 91 unknown of 1500. Nulls keep the same per-node
sign vector but rewire, so their readout nodes receive active inputs (0% silent in W0/W1). The older navigation workflow resolves
unknown -> excitatory. The hypothesis (UNTESTED) is that the zero policy costs the connectome features that nulls keep.

## 2. Single intervention
Model `unknown_sign_policy`: `zero` (baseline; protocol value in learning_v3_protocol.yaml) vs `excitatory` (unknown-sign nodes get +1).
`excitatory` is an EXISTING supported value (src/flycircuit/models/sparse_rnn.py: SIGN_POLICIES = excitatory|inhibitory|zero|random_fixed;
resolve_signs sets s[unk]=+1). No bundle sign-vector workaround and no edit to existing source are needed: the new script deep-copies the
loaded protocol and sets proto["model"]["unknown_sign_policy"]="excitatory" for the policy condition only (build_sparse_model reads it from proto).
Nothing else changes (same task, seed, encoder stream, probe). Nulls/random_reach keep the node-sign vector by index (bundle.sign), as in W0.
kc_static native 1385 hardcodes sign=ones and policy "zero" in learning_v2._kc_static (no unknown nodes), so it is unaffected by construction:
expected paired gain exactly 0 (sanity control; any nonzero value indicates a pipeline bug).
Scientific labeling: this is a GRAPH / SIGN-POLICY scientific change, NOT engineering-neutral. It changes the biological model.

## 3. Design
Families: connectome, null0-null4, random_reach. Readout widths W25 (original readout_idx) and W76 (W25 + eligible nodes sorted by
(-contacts, index), identical construction to scripts_v4/width_probe.py, cross-checked against W0 manifest). Control: kc_static native 1385.
Cells: 7 families x 2 widths = 14, + kc_static native = 15. Conditions: baseline, policy. Seeds 1301-1308 (8), paired by seed.
Rows = 15 x 2 x 8 = 240 (exact). Registered in seed_registry.json under `W2_sign_policy_committed`. Grep (word match) of results*/configs/src/scripts
shows 1301-1308 occur only inside W0_width_probe_manifest.json, as node indices in the sets lists, not as seeds; unused as seeds.
Metric: ctx0 mean val probe accuracy via feats/probe/stats logic copied (not imported) from width_probe.py.
Paired effect per cell: d_seed = acc(policy) - acc(baseline); mean, SE=sd/sqrt(8), 95% CI mean +/- 2.365*SE (t(7)).
Primary contrast P = (connectome W76 gain) - mean gain over {null0..null4, random_reach} at W76, paired by seed.
Descriptive (no inference): silent fraction, eff_rank, clamp_frac, W25 results, train-val gap, percontext_val.

## 4. Gate G-E2 (proceed only to a PREREGISTRATION of a trained pilot, never directly to a run)
ALL of: (i) connectome W76 paired gain >= +0.03 AND 95% CI lower bound > 0; (ii) P point estimate > 0;
(iii) connectome silent fraction at W76 falls by >= 0.05 absolute (baseline minus policy, mean over seeds); plus completeness (240/240 ok rows).
Any failure => STOP and report.

## 5. Limits stated in advance
- Cannot support an anatomical claim: null graphs share the sign vector, so a gain is a sign-policy effect, not evidence about wiring.
- 8 seeds: underpowered for effects below ~0.05. Untrained frozen features only; no learning claim.
- Passing G-E2 changes neither W0's nor W1's failed gates.

## 6. Budget and failure handling (as W1)
<= 15 min, torch 2 threads; expected ~2-4 min (240 rows at ~0.4 s). Abort at 12 min wall (measured from start of compute loop; data load is
before it, so total incl. load must stay < 12 min; the script also checks the clock from process start). Manifest of sha256 hashes written
BEFORE compute. Append-only JSONL, flush per row, resume skips done (family,condition,seed). Exceptions/non-finite recorded as failed rows, never
dropped; abort if failures > 5% of 240 (12). Final JSON written atomically (tmp + os.replace). The measurement run is NOT launched by this prereg's author.

## Reviewer note (added before launch, 2026-10-19)
The z-score calibration statistics (f_mu, f_sd) are computed through edge_values and therefore shift with the sign policy; this is a downstream
consequence of the one declared change, not a second intervention. silent_nodes.py used seed 1201 (diagnostic, not a registered measurement seed).
