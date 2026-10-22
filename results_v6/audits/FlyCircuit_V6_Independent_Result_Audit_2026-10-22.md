# FlyCircuit v6 independent result audit — 2026-10-22

## Disposition

**Conditional pass for publication as an exploratory, validation-only negative result, after the mandatory edits below.** The measured result itself is complete and arithmetically reproducible. It does not support replay benefit in final average accuracy at this budget. The current Claude report must not ship verbatim because two connectome means were double-rounded, peak memory uses the wrong binary unit, and the sentence that the reserved v5 seeds were “untouched” is contradicted by the exposure evidence.

This audit was read-only except for this report and its JSON companion. I used Python standard-library parsing, hashing, arithmetic, and statistics. I did not import Torch or NumPy, execute a model, run a heavy test, alter study artifacts, restart a worker, change settings, or message Claude.

## Evidence and provenance

The frozen plan file hashes to `29d859643e32b508058cdabcba279010fb6af7e43aac3ea7fd82921d48498f5b`, as specified. Its canonical JSON digest is `5fb39e28b30ea8e90203a20f205f9477ee6607b0076e2d538e3dc2c41edfb23f`. This distinction explains why rows contain `5fb39...` while the acceptance receipt and resource ledger name `29d859...`; the former is the canonical object digest and the latter is the file-byte digest.

The acceptance file’s canonical digest is `341d614f6136aebdbe986724b16607472a9627d00c1b59468081cfa719182aec`. Every row contains that digest. The ledger contains the same acceptance digest and the plan file digest. The acceptance receipt binds the same five input hashes as the plan. Its smoke, exposure, and independent-review hashes all reproduce exactly.

I recomputed all 117 planned source hashes and all five planned input hashes; there were no mismatches. I also ran the candidate’s standard-library-only audit path with bytecode disabled. It returned:

```json
{"planned": 64, "saved": 64, "failed": 0, "complete": true, "sota_established": false}
```

Independent row checks found:

- exactly 64 unique rows covering 2 families × 4 methods × 8 seeds;
- 64 `ok` rows and no failed or missing coordinate;
- every coordinate, recomputed coordinate key, plan binding, and acceptance binding matched;
- every saved continual-learning metric reconstructed from its saved 3×3 `R_val` matrix within `1e-12`;
- the current-task schedule hash was identical across all four methods and both families within each model seed;
- each row’s training/validation data identity matched the exposure receipt for its model seed;
- actual Adam steps, parameter ownership, trial counts, and buffer counts passed the saved schema checks.

The ledger records one attempt, 933.721 s charged, no inflight job, child return code 0, and no termination reason. The maximum row RSS is 1028.895 MiB, which is **1.005 GiB**, not approximately 1.03 GiB.

All 64 rows contain the same mapping, edge, and sign hashes. That proves internal identity consistency. It does not prove the graph hashes were correctly derived from the raw graph: neither this audit nor the earlier Claude reviewer recomputed those three identities from raw graph data.

## Independent metric recomputation

Values are mean ± sample SD over the eight model seeds. “Acquisition” is the saved `learned_task_accuracy`, the mean diagonal of `R_val`. The table uses direct rounding from the unrounded row means.

| Family | Method | Final average accuracy | Forgetting keep | Acquisition |
|---|---|---:|---:|---:|
| connectome | standard | 0.7461 ± 0.0503 | 0.1104 ± 0.0885 | 0.8171 ± 0.0346 |
| connectome | extra_current | 0.7461 ± 0.0406 | 0.1592 ± 0.0911 | 0.8516 ± 0.0314 |
| connectome | uniform_replay | 0.7435 ± 0.0310 | 0.0918 ± 0.0625 | 0.8014 ± 0.0247 |
| connectome | balanced_replay | 0.7585 ± 0.0331 | 0.0713 ± 0.0516 | 0.8060 ± 0.0271 |
| GRU33 | standard | 0.7741 ± 0.0408 | 0.0723 ± 0.0516 | 0.8197 ± 0.0223 |
| GRU33 | extra_current | 0.7891 ± 0.0420 | 0.1211 ± 0.0522 | 0.8685 ± 0.0322 |
| GRU33 | uniform_replay | 0.7773 ± 0.0306 | 0.0566 ± 0.0270 | 0.8118 ± 0.0377 |
| GRU33 | balanced_replay | 0.7780 ± 0.0356 | 0.0762 ± 0.0487 | 0.8255 ± 0.0363 |

At three decimals, the connectome uniform and balanced final means are **0.743** and **0.758**. The Claude report gives 0.744 and 0.759 because it first rounded the means to four decimals in `analysis.json` and then rounded those already-rounded values to three decimals. The other six three-decimal final means are unaffected.

## Paired contrasts

I recomputed paired differences from the rows, with model seed as the independent unit. Intervals are two-sided 95% Student-t intervals with `df=7` (`t*=2.364624251`). They reproduce all values in `analysis.json` at its reported precision.

Final accuracy, candidate minus comparator, percentage points:

| Family | Contrast | Mean | Descriptive 95% t interval |
|---|---|---:|---:|
| connectome | uniform − standard | −0.26 | [−3.49, 2.97] |
| connectome | balanced − standard | +1.24 | [−2.34, 4.81] |
| connectome | uniform − extra_current | −0.26 | [−3.06, 2.54] |
| connectome | balanced − extra_current | +1.24 | [−1.84, 4.32] |
| GRU33 | uniform − standard | +0.33 | [−1.32, 1.97] |
| GRU33 | balanced − standard | +0.39 | [−3.99, 4.77] |
| GRU33 | uniform − extra_current | −1.17 | [−3.44, 1.09] |
| GRU33 | balanced − extra_current | −1.11 | [−3.97, 1.76] |

Every replay-versus-standard and replay-versus-extra_current final-accuracy interval includes zero. This supports the bounded statement that the run did not detect a final-accuracy improvement. It does not establish equivalence or prove the true effect is zero.

Replay versus the step-matched `extra_current` control shows a retention/acquisition tradeoff:

| Family | Contrast | Forgetting mean [95% interval], pp | Acquisition mean [95% interval], pp |
|---|---|---:|---:|
| connectome | uniform − extra_current | −6.74 [−11.90, −1.57] | −5.01 [−6.22, −3.81] |
| connectome | balanced − extra_current | −8.79 [−13.53, −4.04] | −4.56 [−6.91, −2.21] |
| GRU33 | uniform − extra_current | −6.45 [−10.53, −2.36] | −5.66 [−7.97, −3.35] |
| GRU33 | balanced − extra_current | −4.49 [−7.59, −1.39] | −4.30 [−5.73, −2.87] |

`extra_current − standard` acquisition is +3.45 pp [1.60, 5.30] for connectome and +4.88 pp [2.27, 7.49] for GRU33. Forgetting is +4.88 pp in both families, with intervals excluding zero. Final accuracy is 0.00 pp [−1.68, 1.68] for connectome and +1.50 pp [−1.52, 4.52] for GRU33. Thus the lower replay forgetting versus `extra_current` co-occurs with lower acquisition and no clear final-accuracy improvement. This is a descriptive metric decomposition, not evidence for a causal replay mechanism.

The complete machine-readable audit includes all within-family contrasts among standard, extra-current, uniform replay, and balanced replay for final accuracy, forgetting, and acquisition. It also includes seed-paired connectome-minus-GRU33 values. Those family comparisons must remain architecture-family comparisons. They do not identify an anatomical effect.

## Exposure and smoke caveats

The smoke receipt used only seed 99001, only `balanced_replay`, and both families. It passed. The standard, extra-current, and uniform-replay paths were not covered by the pre-run real-model smoke, although the completed measurement subsequently exercised all methods and all rows passed the strict result schema.

The exposure receipt says `n_hashed=278`, `n_hist=270`, eight distinct candidate hashes, no candidate collision with the historical/reserved set, and `reserved_included=true`. The exposure operation generated and hashed deterministic data identities for reserved v5 seeds 100601–100620. It did not execute models or produce v5 outcome metrics. Nevertheless, **“reserved seeds untouched” is false under a strict no-processing/no-exposure meaning**. Any later claim that those v5 seeds remained prospectively hidden needs separate resolution; this v6 audit cannot restore that status.

## Claim disposition

Accepted for the bounded v6 publication:

- Exploratory validation-only, native 3-task cue-task result; eight paired model seeds; one run attempt; no tuning during this run.
- Complete 64-row evidence with a valid plan → acceptance → row provenance chain.
- No replay final-accuracy contrast versus standard or step-matched extra-current has a descriptive 95% paired t interval excluding zero.
- Replay shows lower forgetting than extra-current together with lower acquisition; this is a tradeoff, not a final-accuracy gain.
- SOTA, test/held-out generalization, body robustness, and external-benchmark performance were not established.

Accepted only with caveats:

- The t intervals are descriptive: `n=8`, many contrasts, no multiplicity correction. An interval crossing zero does not prove equivalence or absence of effect.
- The acceptance reviewer was a Claude subagent and functions as an operational review gate, not formal cryptographic or organizational independence.
- The three graph hashes are consistent across rows but were not independently reconstructed from raw graph data.
- Family-level performance differences are confounded architecture comparisons and cannot be interpreted anatomically.

Blocked from publication:

- “v5 (55 saved rows, reserved seeds 100601–100620) remains paused and untouched.” The paused/55-row status belongs to the separate v5 accounting, and “untouched” is contradicted by the documented reserved-seed generation/hash operation.
- Any claim of equivalence, causal mechanism, SOTA, held-out/test generalization, body robustness, anatomical advantage, or independently verified graph identity.

## Mandatory edits to the Claude result report

1. Report connectome uniform and balanced final accuracy as 0.743 and 0.758 at three decimals, or use the four-decimal table above consistently.
2. Delete “reserved seeds ... untouched.” If exposure must be mentioned, state that the reserved seeds were included in a data-identity collision scan, with no model outcomes computed.
3. Replace “peak RSS ≈ 1.03 GiB” with “peak RSS 1028.9 MiB (1.005 GiB)” or approximately 1.00 GiB.
4. Keep all intervals explicitly descriptive and omit anatomy, body, held-out, and SOTA language except clear limitations saying those outcomes were not measured or established.

With those edits, the v6 result is suitable to ship as a measured validation-only negative result: no detected final-accuracy gain from replay at the tested budget, plus a descriptive lower-forgetting/lower-acquisition tradeoff against the step-matched control.
