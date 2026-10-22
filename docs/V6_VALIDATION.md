# v6: completed exploratory replay validation

This is a validation-only engineering comparison, not confirmation or a public
benchmark leaderboard result. Eight model seeds (26001–26008), two model families,
four methods; all 64 planned coordinates completed with zero execution failures.

| Family | Standard | Extra current steps | Uniform replay | Balanced replay |
|---|---:|---:|---:|---:|
| Connectome W76 | 74.6% | 74.6% | 74.3% | 75.8% |
| GRU33 | 77.4% | 78.9% | 77.7% | 77.8% |

Values are direct means of the saved raw accuracy matrices. The original Claude
analysis rounded to four decimals, and its prose rounded again; uniform/balanced
connectome values are therefore corrected here to 74.3%/75.8%. Original analysis
bytes are retained; the new public_summary.json contains full-precision recomputation.

## Interpretation

- Connectome balanced replay minus standard final accuracy: +1.24 pp (descriptive 95% paired-t interval -2.34 to +4.81).
- Connectome balanced replay minus extra-current forgetting: -8.79 pp (descriptive 95% paired-t interval -13.53 to -4.04).
- The same contrast's learned-task accuracy: -4.56 pp (descriptive 95% paired-t interval -6.91 to -2.21).

Lower measured forgetting is not sufficient to establish improved retention at
equal acquisition. Replay's lower forgetting coincides with lower learned-task
accuracy relative to the step-matched extra-current control. Final accuracy does
not show a clear replay improvement. Connectome mean final accuracy is lower
than GRU33 under all four methods; no anatomical advantage is demonstrated.

The confidence intervals are descriptive paired-t intervals over eight MODEL
seeds, df=7, with no multiplicity correction. Episodes/tasks are not independent
replicates. This small study cannot establish equivalence or prove zero effect.

## Evidence and execution

- Immutable plan: SHA256 29d859643e32b508058cdabcba279010fb6af7e43aac3ea7fd82921d48498f5b.
- One attempt, 933.721 seconds charged of the 1,800-second cap; no inflight record,
  child exit zero. Per-row peak RSS is 1028.9 MiB (1.005 GiB); this is not total laptop RAM.
- Actual Adam steps and parameters, paired current schedules and recorded graph
  identities are checked for every successful row by the archived strict reader.
- Smoke evidence covered balanced replay on seed 99001 in both families; the
  other methods were subsequently exercised by the completed pilot itself.
- The accepted input inventory contains the canonical protocol, three processed
  graph parquet files and selection.json. The acceptance receipt binds source,
  smoke, exposure and operational review artifacts. Review is not cryptographic
  proof of independence or a replacement for recomputation.
- The exposure check generated and hashed data for 278 seeds, including the
  twenty v5-reserved IDs. No hash collisions were reported. That was data-level
  inspection, not reserved-seed performance evaluation; do not call those data
  untouched. A future confirmation requires an explicitly reviewed exposure policy.

## How to verify, and what is absent

Run `python scripts_release/verify_v6_results.py` from the repository root. It
checks all 117 frozen source files, original plan/receipt/evidence identities,
exact 64-coordinate coverage, strict result schemas, single-attempt accounting,
raw-matrix summaries, and the tables in this report and README. It performs no
training, imports no scientific dependencies, and downloads no data. New release
checks and the 31 candidate contract tests run in CPU CI.

The original rows, plan, receipt, ledger and evidence are under
results_v6/native_validation/. The original native dependencies are archived
under experiments/v6_candidate/native_source/; provenance/V6_SOURCE_MAP.json
maps their original paths without editing the measured source bytes. Absolute
paths in original freeze records identify the measured host, not portable launch
paths. This is an archival source/evidence release, not a verified portable
training distribution. Processed FlyWire data/caches and environments are absent.

No held-out performance, SOTA, biological advantage, multi-rate body robustness
or completion of the v5 pilot is claimed. The body curve functions exist, but a
multi-rate body-evaluation driver and an external benchmark adapter remain absent.

## Next decision

Publish this mixed/negative validation result, preserve the raw records, and
stop expanding replay claims. Before any new heavy run, audit the separate v5
orphan incident and its cumulative budget; do not resume it automatically. Then
choose one prospectively specified study: a reviewed body-robustness evaluator,
or a retention-at-equal-acquisition question with fair memory/update budgets and
fresh confirmation data. Neither is guaranteed to improve performance.

Independent artifact/arithmetic audit:
[report](../results_v6/audits/FlyCircuit_V6_Independent_Result_Audit_2026-10-22.md)
and [machine-readable results](../results_v6/audits/FlyCircuit_V6_Independent_Result_Audit_2026-10-22.json).
The audit independently checked source/input file hashes and raw-matrix statistics;
the recorded graph mapping/edge/sign hashes were checked for consistency, not
independently reconstructed from raw graph arrays.

Archived launch code is not guaranteed to survive abrupt supervisor termination.
The separate v5 incident demonstrates that an inherited lock alone does not keep
memory/wall monitoring alive after supervisor death. Public training/resume is
unsupported; do not run archived launchers as a recovery action.
