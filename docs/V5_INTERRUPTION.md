# v5 interrupted pilot: not accepted scientific evidence

Status on 2026-10-22: **295/616 rows saved; interrupted, incomplete and quarantined**.
Of these, 55 predate the final attempt and 240 were appended by a child whose
supervisor did not settle its guard and resource ledger. Offline structural and
provenance checks pass, but `status: ok` is not scientific acceptance. Combined
rows are an execution-order prefix, not a balanced completed comparison. The
selection and partial check outputs are not accepted as release conclusions.

The frozen ledger still records six attempts and 1,067.366 seconds, with stale
inflight supervisor PID 670. The seventh attempt's child outlived that accounting
horizon. The independently observable execution from start to final-file write
is at least 2,414.125 seconds, including at least 57.909 seconds beyond its final
recorded heartbeat. The precise actor/time of supervisor death is not proven
by the artifacts; the memory-reaper explanation remains a reported explanation.

No v5 row, source, ledger, input or budget was changed by this publication audit.
Full raw records remain preserved locally. This preview publishes the incident
audit, not unaccepted v5 metrics. The supervisor recovery code would undercharge
the orphan tail and omit the attempt increment. Do not invoke recovery or resume
training merely to make the ledger look settled. No new training was started.

Before any future continuation, a separately reviewed recovery decision must
charge the full execution conservatively and address supervisor-death handling,
child lifetime/resource coverage, attempt increments and atomic accounting. The
audit's bounds are calculations, not an accepted ledger reconciliation or a new
compute allowance. No remaining budget is promised. Simply closing apps does
not repair this accounting or make the execution path safe.

Reserved v5 data identities were generated/hashed in the separate v6 exposure
scan; no performance evaluation of those reserved IDs was reported. They are
not accurately described as untouched data. Any later held-out design needs
an explicit exposure-policy decision and fresh prospective review.

See [the detailed read-only audit](../results_v6/audits/FlyCircuit_V5_Interruption_Audit_2026-10-22.md)
and [its machine-readable evidence](../results_v6/audits/FlyCircuit_V5_Interruption_Audit_2026-10-22.json).
This incident does not make the separately completed 64-row v6 study incomplete.
