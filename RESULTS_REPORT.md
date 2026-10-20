# Results report — snapshot 2026-10-20 (v0.1.0-preview)

Complete record of the connectome-readout experiments in this release, including weaker outcomes and failed gates.
All numbers come from the raw files under `results_v4/`; recompute W3 with `python scripts_release/verify_w3_headline.py`.
Everything here is **exploratory, validation-split, small-n**. Nothing is a held-out confirmation.

## 1. Metrics are not interchangeable
| Metric | Where | What it is |
|---|---|---|
| Frozen-feature probe accuracy | W0, W1, W2 | Untrained network features + a linear head, validation split |
| Trained acquisition | W3 | Accuracy on validation cues after 800 online REINFORCE trials (`val_acc_acq`) |
| Continual final accuracy / forgetting | W3 | Three-task sequence, 2,400 trials, validation split |
| Body navigation success | not in this release | FlyGym physics, 25 % channel dropout, target 68 % — pending |

## 2. W3 — trained, validation-only pilot (headline stage)
280 runs (7 families × 9 learning-rate settings × 4 search seeds 1401–1404 = 252, plus 28 check runs, seeds 1501–1504), 0 failures, 3,822 s.
Each family is shown at its own selected setting. **Every family's best or chosen setting sits on the edge of the 3×3 grid**, so the search did not converge.

| Family | Acquisition mean (SE) | Per-seed 1501 / 1502 / 1503 / 1504 | Continual final | Forgetting (keep) |
|---|---|---|---|---|
| **Connectome W76** | **0.824 (0.035)** | 0.875 / **0.891** / 0.781 / 0.750 | 0.751 | 0.084 (0.021) |
| Null 0 | 0.863 (0.026) | 0.922 / 0.859 / 0.797 / 0.875 | 0.749 | 0.135 |
| Null 1 | 0.832 (0.023) | 0.875 / 0.844 / 0.766 / 0.844 | 0.738 | 0.109 |
| Null 2 | 0.832 (0.024) | 0.859 / 0.875 / 0.828 / 0.766 | 0.753 | 0.113 |
| Random-reach graph | 0.785 (0.030) | 0.766 / 0.875 / 0.750 / 0.750 | 0.763 | 0.061 |
| GRU (hidden 37) | 0.867 (0.029) | 0.938 / 0.859 / 0.797 / 0.875 | 0.784 | 0.051 |
| Encoder-decoder | 0.844 (0.011) | 0.875 / 0.844 / 0.828 / 0.828 | 0.780 | 0.076 |

Reading it:
- **Acquisition ≥ 82 %:** the connectome mean is 0.8242, 0.4 pp above 0.82 — a descriptive point-estimate hit only. The t(3) 95 % interval is [0.714, 0.934];
  two of four seeds are below 0.82. No statistical support for ≥ 0.82. Seed 1502 (0.8906) is the best of four seeds, selected post hoc.
- **Forgetting advantage ≥ +7 pp (GRU minus connectome):** measured **−3.3 pp** (SE 2.7; per seed −10.2, +3.1, −2.3, −3.9). Target **missed**, wrong sign:
  the GRU forgot less than the connectome.
- **Anatomy:** connectome minus the mean of the three nulls is −1.8 pp (SE 2.3). No advantage. Three fixed null graphs are not graph-level inference.
- **Capacity is not matched:** trainable parameters connectome 7,551, GRU 9,252, encoder-decoder 8,956.
- **Interpretation:** widening the readout from 25 to 76 nodes lifted the connectome from about 74 % (an earlier pilot, different seeds, not paired) to about 82 %,
  but the same interface lifted the controls to 83–87 %. This is an engineering/interface gain, not evidence for connectome anatomy.
- **Selection leakage check:** search seeds (1401–1404) and check seeds (1501–1504) are disjoint, but both draw from the validation split;
  the "fresh check" is therefore exploratory. Audit: `results_v4/audit_v4/W3_audit.md`. Status ledger: `W3_DECISION_STATUS.md`.

## 3. Earlier gates — all failed, kept as declared
| Stage | Declared gate | Result |
|---|---|---|
| W0 width probe (296 rows, seeds 801–808) | connectome frozen-feature accuracy ≥ 0.85 | **Failed**: 0.834 (W76) |
| W1 encoder bundle E8 (144 rows, seeds 1101–1108) | paired gain ≥ +0.03, contrast > 0, saturation rise ≤ 0.05 | **Failed** all three: gain −0.027, contrast −0.015, saturation +0.082 |
| W2 unknown-sign policy (240 rows, seeds 1301–1308) | gain ≥ +0.03, etc. | **Failed**: gain −0.002 (silent features did fall 10.5 % → 4.0 %, with no accuracy change) |

Caution recorded at W2: the identical baseline configuration scored 0.883 on seeds 1301–1308 and 0.834 on seeds 801–808, so an 8-seed frozen-probe mean moves
about 0.04–0.05 with the seed set — larger than any effect tested. The W0 gate stays failed on its declared seeds. Details: `W0_RESULT.md`, `W1_RESULT_AND_LEDGER.md`.

## 4. Statistical limits
- 4 seeds per family. About 35 seeds would be needed for 80 % power to show acquisition ≥ 0.82 if the true value were 0.85, and about 1,850 if it were 0.824.
  The +7 pp forgetting gap is not powered while the pilot gap is negative (`CONFIRMATION_PROTOCOL_V2_DRAFT.md`, `power_calc_v2_output.json`).
- Selection used the same validation split as the check; there is no held-out test result.
- Provenance: the W3 manifest was written before the first row (file-time ordering, no version-control history); `W3_prelaunch_hashes.txt` proves files unchanged since launch only.

## 5. Not in this release
Body-navigation results, the held-out learning assessment, replay experiments, and intermediate/partial runs of the follow-on program
(see [ROADMAP.md](ROADMAP.md)). The full local experiment history is preserved by the author.
