# Stage W0 result (executed 2026-10-19, 296 rows = 37 cells x 8 seeds 801-808, 128 s, 2 threads, zero failures)
Raw: W0_width_probe_rows.jsonl / .json, W0_width_probe_manifest.json, W0_run.log. Summary: scripts_v4/summarize_w0.py.
Exploratory, validation split, frozen untrained features, shared-nothing per-context heads. Not a trained-learner result.

Probe (a) association context 0, mean val accuracy (sd across seeds ~0.02-0.07):
| width | connectome | nulls0-4 (range) | random_reach | kc_static matched-K | encdec matched-K |
|---|---|---|---|---|---|
| 25  | 0.762 | 0.777-0.797 | 0.828 | 0.797 | 0.889 |
| 64  | 0.832 | 0.840-0.869 | 0.869 | 0.842 | 0.908 |
| 76  | 0.834 | 0.840-0.879 | 0.871 | 0.877 | 0.914 |
| 115 (adds 39 silent nodes) | 0.834 (identical to W76) | same as W76 | 0.896 | - | - |
Native, descriptive: GRU(37) 0.816, encdec(407) 0.912, kc_static(1385) 0.906. Per-context averages are 1-3 pts higher.

Findings (measured):
- Widening helps the connectome: +7 pts at W64 (0.762 -> 0.832), then saturates (W76 = W64 + 0.002; the 39 extra nodes in W115 have no input and add nothing).
- The pre-declared go/no-go (connectome >= 0.85 at a primary width) is NOT met (best 0.834). Per the preregistration, no trained
  pilot follows from this stage.
- Widening helps degree-preserving nulls and random_reach at least as much: the connectome is at or below every null mean at W64/W76
  (differences within seed sd; 5 nulls, 8 seeds, no inference claimed). No anatomical advantage is indicated; the shared gain looks
  like added readout dimensionality. Matched-K static KC and encdec features equal or exceed the connectome.
- Train accuracy rises to 0.95 while val is 0.83: widening also overfits a 600-ish-sample head (generalization gap ~0.12).
Not established: effect on trained acquisition/forgetting; whether a different encoder or readout from the recurrent core would reach 85%.
Next evidence-based options (each a scientific change, needing its own prereg and owner nod on compute): (i) encoder/sensory-to-core
interface investigation (the 25-70 features limit is rank ~7-12 of 25-76); (ii) accept and report the negative result for the
connectome-versus-controls claim, with the engineered readout as a system-level improvement only.
