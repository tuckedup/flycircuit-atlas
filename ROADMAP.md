# Roadmap and outstanding work (2026-10-20)

Status: experimental and in progress. Nothing below is guaranteed to reach its target.

## Open targets (all unverified)
| Target | Status |
|---|---|
| Connectome acquisition ≥ 82 % | Point estimate 82.4 % on 4 validation seeds; no held-out or statistical support yet |
| GRU-minus-connectome forgetting ≥ +7 pp | Measured −3.3 pp (validation, 4 seeds); not established |
| Body navigation ≥ 68 % at 25 % channel dropout | Not verified; an earlier, different pipeline fell below this and is not part of this release |
| Anatomical advantage over matched nulls | Not demonstrated |

## Planned, in this order
1. **Replay pilot** — connectome W76 vs GRU (hidden 33; 7,551 vs 7,460 trainable parameters), standard training vs episodic replay vs an extra-trial control,
   one shared learning-rate pair per family chosen on validation-search data only. A 150-minute budget has been set; the run has not completed.
2. **Complete the body model matrix** — train the 11 missing models to reach 5 families × 5 seeds.
3. **Body evaluation** — 100 fixed paired episodes per model with balanced 25 % dropout masks, plus heuristic references and a reset-state ablation.
4. **One held-out learning assessment** on 20 reserved unseen seeds, run once, only if the pilot is technically adequate; thresholds tested against 0.82 and 0.07, with "inconclusive" allowed.

## Known gaps
- No licence file yet (owner decision). Several upstream citations are marked UNVERIFIED in `CITATIONS.md`.
- Training re-runs depend on data and caches that are not redistributed (see README).
- Parameter counts across families are only approximately matched.
