# learning_readout_v4_1 -- Stage W0: label-free readout-width probe (EXPLORATORY, validation split only)
Written 2026-10-19 BEFORE any new measurement. Relationship to the failed pilot: results_v4/learning_v4 is
preserved and untouched; this namespace tests ONE factor (readout width) on frozen, untrained features.

Facts (scripts_v4/count_nodes.py): 1500 nodes; 1385 sensory/KC; 25 original readout (MBON) nodes;
90 other non-sensory nodes, of which 51 have in-degree > 0 (39 receive no input: constant/silent).
No 128-wide non-sensory set exists.

Readout sets (node-index slots, identical for connectome, 5 degree-preserving nulls, random_reach):
- W25  = original 25 readout nodes (baseline).
- W64  = W25 + top 39 eligible.  W76 = W25 + all 51 eligible (largest justified primary width).
- Eligible = non-sensory, not original readout, connectome in-degree > 0.
  Rank = total incoming contact count (descending), ties by node index (ascending). Label-free; computed once
  from the connectome graph.
- Diagnostic only: W115 = W25 + all 90 non-sensory nodes (includes silent nodes).
Primary sets never expose sensory/KC nodes. Engineered diagnostic: all-KC (1385) readout on zero-edge kc_static.
Capacity controls: kc_static and encdec features sub-sampled label-free (fixed seed) to K in {25,64,76};
native GRU(37)/encdec(407)/kc_static(1385) are reported descriptively only.

Measurement: untrained model, same z-score calibration as the deployed learner; head = L2 logistic regression
(1e-3); standardization and fit on TRAIN split only; accuracy on VAL. Two probes reported separately:
(a) association context 0 (the actual association endpoint's context); (b) per-context heads averaged
(the earlier diagnostic). Also: train accuracy, effective rank (participation ratio of train feature
covariance), clamp-saturated fraction (|z|>=5), silent fraction (train sd<1e-6).
Common seeds 801-808 (task seed = model seed).

Seed registry: used v2 {11,12,21,22,100201-100205}; v3 {13,14,23,24,100401-100408}; v4 pilot {501-504,601-604};
probe 701-708; this probe 801-808. RESERVED, unseen: validation checks 901-908; confirmation 100601-100620.
The registry is reconstructed from saved records and may be incomplete for v1.

Go/no-go (heuristic, NOT the 82% target, no statistical guarantee): proceed to a trained pilot only if the
connectome at some primary width (W64/W76) reaches mean val accuracy >= 0.85 on probe (a), and the result is
reported next to nulls/random_reach and matched-K kc_static so a direct-input bypass is not credited to anatomy.
If it fails: report a representation/encoder/topology investigation; no trained study.
Budget: <= 15 min wall, 2 CPU threads, one heavy job, Windows free RAM >= 1.5 GiB at start.
