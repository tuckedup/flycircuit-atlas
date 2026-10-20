# W3 decision status and corrected ledger (2026-10-19)

Decision from owner: HOLD the 210-minute confirmation; keep seeds 100601-100620 unused; one audit/protocol repair round; no new search, confirmation, physics or training. No job is running. Nothing in this round consumed a seed, wrote the seed registry, or changed an earlier result.

## Corrected status table
| Item | Status | Evidence |
|---|---|---|
| W3 execution | 280/280 rows ok, 0 failures | W3_audit.md check 1 |
| W3 search vs check data | Search seeds 1401-1404 and check seeds 1501-1504 are disjoint, different task data (no identical arrays; 3-14 of 192 val cue patterns coincide by chance). BOTH are the validation split: "fresh check" is exploratory, not held-out. Earlier statement that selection used the same check data was wrong. | W3_audit checks 2a/2b |
| W3 selection | Recomputed from raw rows by two scripts and a reviewer: matches choices.json for all 7 families. Every family's best or chosen setting is on the 3x3 grid edge (not converged). For connectome best != chosen (core/decoder multipliers transposed). | check 3 |
| Connectome acquisition | 0.82421875, SD 0.0691, SE 0.0346, t(3) 95% CI [0.714, 0.934]; 0.0042 above 0.82. Descriptive point-estimate hit only; no statistical support for >= 0.82. | check 4 |
| Forgetting gap (GRU minus connectome, signed) | -0.0332 (SE 0.0273, CI [-0.120, +0.054]); per seed -0.102, +0.031, -0.023, -0.039. Target >= +0.07 MISSED, wrong sign. | check 4 |
| Anatomy | connectome minus mean(null0-2) acquisition -0.018 (SE 0.023); no advantage; 3 exposed nulls, descriptive only. | check 4 |
| Capacity | Trainable params: sparse W76 7551 (7399 + 152), GRU 9252 (1.225x), encdec 8956 (1.186x). NOT matched (constructed without training and compared to logged values). | check 5 |
| Provenance | Manifest written before the first row (mtime ordering); W3_prelaunch_hashes.txt collected at launch time (proves unchanged since launch, not an earlier freeze). Prereg-before-run rests on mtimes (no git history). | check 6 |
| Null inventory | The nulls are plain null_00..09 (the earlier draft's "group_constrained" was wrong). null_00..04 exposed to earlier training; null_00..02 in W0/W1/W3; null_05..09 appear in no training record; null_10/11 do not exist (search-only nulls unavailable). | check 7 |
| Earlier audit script repairs | Fresh reviewer: ship-with-fixes (low); fixes applied; second fresh reviewer re-checked: ship-with-fixes (minor wording only, no wrong numbers). Residual minor: some hardcoded prose in exposure text and A11 description; B10 weak text check. | reviewer reports |
| W0 / W1 / W2 gates | Remain FAILED as originally declared. | unchanged |
| Body 68% | Not run. | - |

## New design artifacts (documents only; each reviewed once by a fresh reviewer, fixes applied, fixes themselves NOT re-reviewed)
- CONFIRMATION_PROTOCOL_V2_DRAFT.md (+ power_calc_v2.py/output): threshold vs support separation, gatekeeping, power; graph-level rank analysis replaces sign-flip; capacity policy; executable manifest (450 search runs for S=3; 11n confirmation runs). Conclusion: a powered threshold study of acquisition needs ~35 seeds even if truth is 0.85 (only 20 reserved); the 0.07 gap is not powered because the pilot gap is negative; recommendation is to launch nothing.
- MEMORY_REDESIGN_PROPOSAL.md: replay as a separate factor, matched for connectome and GRU, plus step-matched control.
- body_v4/BODY_PROTOCOL_V2_DRAFT.md: 90 paired masks; minimal 1,992 episodes (7.14 h at the unsourced 12.9 s planning rate; D3 measured 10.2 s), larger 5,784 (20.7 h); not approved.
