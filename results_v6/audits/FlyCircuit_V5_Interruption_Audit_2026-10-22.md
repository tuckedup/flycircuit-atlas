# FlyCircuit v5 interruption and accounting audit — 2026-10-22

## Release verdict

**Ship v5 only as `INTERRUPTED / INCOMPLETE / SCIENTIFICALLY QUARANTINED`.** Do not describe the v5 pilot as complete, do not claim 616 rows, and do not use the partial v5 selection or metrics for a release conclusion, freeze gate, or confirmatory comparison.

The result file contains 295 validly encoded `ok` rows, not 616. The last runner wrote `complete: false`, `aborted: "row interrupted (sigterm); not recorded"`, and `n_rows: 295`. The seventh supervised attempt did not reach the supervisor's settlement path: `attempts.jsonl` has only six records; `budget.json` still has a stale `inflight` entry for supervisor PID 670; `sup5.out` has no `RESULT`; and `jobs.log` has `START` but no `END` for `p1_pilot_r5`.

Preserve the 240 rows appended by PID 673 as raw interrupted-run evidence, but quarantine them from scientific reporting. They pass row-level integrity and coordinate checks, yet the attempt that produced them lacks authoritative guard closure, final drift comparison, and ledger settlement. The existing 55 rows can remain identified as earlier supervised partial rows, but the combined 295-row prefix is incomplete and execution-order imbalanced, so no aggregate v5 scientific conclusion should ship from it.

## Scope and method

This was a read-only audit of `/home/atulp/projects/flycircuit_debug_copy`. The only files written are this report and its JSON companion in the Windows workspace. No supervisor was launched, no ledger was reconciled, no process was killed, and no source, result, setting, or memory configuration was changed.

All data checks used Python's standard library only. The audit did not import NumPy, Torch, or any FlyCircuit module and did not run training, tests, or benchmarks.

## Row and provenance findings

The current `rows.jsonl` SHA-256 is `0514e1ee3690f3998fa230197d373360b636649e8c924a6cc5785081cfec5531`. It is newline-terminated and parses as 295 JSON objects.

- 295/295 rows have `status: ok`; there are no duplicate keys and no non-finite numeric values.
- The rows comprise 176 `connectome` and 119 `gru33` coordinates: 263 search rows plus 32 connectome check rows; 221 continual and 74 association rows.
- Every row key exactly recomputes from `(kind, family, mode, core_lr, decoder_lr, seed)`.
- Every row matches the manifest protocol hash, W76 hash, RNG version and entropy, stage label, validation split, optimizer-owned parameter counts, and recorded trial/optimizer-step counts.
- All five pilot/resume manifests have the same core. Its canonical SHA-256 is `657f2162b8d856cdbfaaf65faeccb9c1e02f3530c92d86dffbc8e21af8b1b1db`, and all 295 rows carry that value.
- The 504-coordinate search-key list reconstructed from the frozen manifest centers and the v5 source has SHA-256 `857f6a6c3310901bf11cbb3ed64fdbff1b2b70c59925a63b6ba66ae11e324d09`, exactly matching the manifest.
- The connectome check-coordinate hash recomputes to `7dd2fffc2fb946e9a8024ed1b45b836217c4362457da8bc06a659f9b4b267f68`, and the stored connectome selection hash recomputes to `f35eeb5b413a00a1874fbf2a242a86601d4781501f74b094ea0a29c449a9d137`; both match `check_manifest_connectome.json`.
- Current SHA-256 values for every file in `train_source_sha256` match the pilot manifest. This proves current byte identity to the recorded inputs; it does not replace the missing supervisor finalization.
- The 295 keys are the exact execution prefix: all connectome search coordinates, all 32 connectome check coordinates, then the first 119 GRU33 search coordinates. The next planned key is `["cont","gru33","C",0.00015098,0.00015098,1604]`, exactly the key recorded by the final `row_interrupted` event.

The historical/new boundary is unambiguous. Attempt `p1_pilot_r5` started with `n_prior_rows: 55`. Row 55 finished at `2026-10-21T01:27:13Z`; row 56 finished at `2026-10-22T10:25:24Z`; row 295 finished at `2026-10-22T11:05:09Z`. Therefore PID 673 appended exactly 240 complete rows before its interrupted, unrecorded next row.

There is also a pre-compute manifest-process gap. The nominally completed `p1_manifest` attempt logged `DRY (no timing_summary.json yet, nothing written)`, and `results_v5/learning/pilot/coordinate_manifest.json` is absent. The later pilot manifest did hash the 504 search keys before compute, and the connectome check manifest was written before its check rows, so the existing rows remain reconstructible. The intended explicit coordinate-manifest artifact was nevertheless never produced because the manifest phase ran before timing.

## Causal and accounting findings

The artifact evidence supports an abrupt supervisor death, but it does not identify the actor that killed it. A normal SIGTERM/SIGINT path in `supervisor.py` would set `external_sigterm`, terminate and reap the child group, call `settle`, append an attempt, and log `END`. None of that occurred for PID 670. The reported memory-reaper attribution is therefore plausible but not independently proven by the v5 artifacts.

The child outlived the supervisor's durable accounting horizon. The last persisted supervisor heartbeat is `1792667057.6293573`; the child's `final.json` mtime is 57.909 seconds later. Six completed rows (290–295) have finish timestamps after that heartbeat. This does not prove the exact instant PID 670 died, but it proves those seconds and rows are beyond the ledger's recorded heartbeat horizon. PID 673 then recorded SIGTERM at `2026-10-22T11:05:15Z` and wrote the incomplete final record. At audit time, PIDs 670 and 673 were absent and no FlyCircuit Python job was running.

The v5 recovery logic has two concrete accounting defects for this incident:

1. `recover_inflight` charges execution only through `heartbeat_mono_wall`. It cannot charge the orphan child interval after the supervisor's last heartbeat, despite the inherited lock allowing the child to continue.
2. `recover_inflight` appends an `orphaned_supervisor_died` attempt but never increments `phases[P1].attempts`. A future automatic recovery would therefore leave the attempt count at six rather than seven.

The README claim that orphan recovery has at most a five-second error does not hold when the child continues after the supervisor dies. Here, the observable final-write tail alone is 57.909 seconds after the last heartbeat. The tests cover ordinary child failure, memory abort, budget persistence, duplicate supervisors, and stray grandchildren, but contain no test that kills the supervisor, lets the inherited-lock child continue, and verifies tail charging plus attempt-count recovery.

The frozen ledger currently reports P1 `attempts=6`, `exec_s=1067.366`, and `wait_s=992.378`. Attempt 7's persisted timing fields establish:

- wait before execution: 0.625392 s;
- execution through last heartbeat: 2356.215665 s;
- execution from start through the child's final-file write: at least 2414.124876 s;
- unaccounted heartbeat-to-final-write tail: at least 57.909211 s.

Thus the evidence-backed minimum corrected P1 totals would be seven attempts, `exec_s >= 3481.490876`, and `wait_s = 993.003392`. These remain under the 9000-second P1 execution budget and the 79200-second overall budget. Because the authoritative supervisor end timestamp is missing, the safest conservative incident charge would be the full 7900-second grant: P1 `exec_s = 8967.366`, leaving only 32.634 execution seconds in P1, with seven attempts. These figures are audit calculations only; the frozen ledger was not edited or accepted.

## Guard implication

While PID 670 was alive, its monitor enforced the 1.5 GiB Windows free-memory rule and refreshed the ledger. After abrupt supervisor death, PID 673 retained the canonical lock but had no Windows RAM monitor, supervisor wall deadline, process-group cleanup, or outer post-run hash comparison. The child still checked its own wall/SIGTERM flag and source file stat signatures between rows, and current source hashes match the manifest, which supports row integrity. It does not recreate the missing resource-guard evidence.

## Safe decision

Freeze v5 exactly as found and publish the interruption warning with the hashes and counts above. Do not invoke recovery merely to rewrite the ledger, do not resume the 616-row v5 pilot, and do not treat `selection.json` or the connectome check results as accepted release evidence. Any future scientific work should use a separately accepted execution path whose supervisor-death handling records the child PID, preserves guard coverage or terminates the child, charges the complete child lifetime, and increments the attempt count atomically. v6 review and publication remain a separate decision.
