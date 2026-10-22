# Roadmap and outstanding work (2026-10-22)

Status: experimental and in progress. Nothing below is guaranteed to reach its target.

## Open targets (all unverified)
| Target | Status |
|---|---|
| Connectome acquisition ≥ 82 % | Point estimate 82.4 % on 4 validation seeds; no held-out or statistical support yet |
| GRU-minus-connectome forgetting ≥ +7 pp | Measured −3.3 pp (validation, 4 seeds); not established |
| Body navigation ≥ 68 % at 25 % channel dropout | Not verified; an earlier, different pipeline fell below this and is not part of this release |
| Anatomical advantage over degree-preserving nulls | Not demonstrated |

## Current measured and interrupted follow-up
- **v6 complete:** 64/64 validation-only rows, no clear final-accuracy replay gain;
  see [the full report](docs/V6_VALIDATION.md). It does not pass the historical
  acquisition/forgetting/body targets by substituting a different metric.
- **v5 incomplete:** 295/616 rows, including an orphaned continuation whose
  accounting and protocol acceptance remain unresolved. See
  [the incident note](docs/V5_INTERRUPTION.md). Do not resume it automatically.
- Reserved-seed data were generated for hash-only exposure checks; no performance
  evaluation was reported. Future held-out work requires exposure-policy review.

## Prior plan (superseded; retained for context)
1. **Replay pilot** — connectome W76 vs GRU (hidden 33; 7,551 vs 7,460 trainable parameters), standard training vs episodic replay vs an extra-trial control,
   one shared learning-rate pair per family chosen on validation-search data only. A 150-minute budget has been set; the run has not completed.
2. **Complete the body model matrix** — train the 11 missing models to reach 5 families × 5 seeds.
3. **Body evaluation** — 100 fixed paired episodes per model with balanced 25 % dropout masks, plus heuristic references and a reset-state ablation.
4. **One held-out learning assessment** on 20 reserved unseen seeds, run once, only if the pilot is technically adequate; thresholds tested against 0.82 and 0.07, with "inconclusive" allowed.

## Known gaps
- No licence file yet (owner decision). Several upstream citations are marked UNVERIFIED in `CITATIONS.md`.
- Training re-runs depend on data and caches that are not redistributed (see README).
- Parameter counts across families are not matched in the published W3 run (GRU +22 %, encoder-decoder +19 % over the connectome).
  The planned replay pilot uses a closer GRU (hidden 33; 7,460 vs 7,551, about 1.2 % apart) and reports the residual mismatch.
