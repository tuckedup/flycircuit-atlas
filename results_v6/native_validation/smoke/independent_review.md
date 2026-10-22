# Independent review: FlyCircuit v6 candidate, plan native_v6_validation_r3

Method: read-only. Commands run via WSL (sha256, small pyarrow reads of the three parquet inputs, stdlib python over JSON, `verify_plan` from the bundle with `-B` so no bytecode was written). No torch import, no training. Only this report file was written.

## 1. Hashes (all recomputed)

- sha256(plan.json) = 29d859643e32b508058cdabcba279010fb6af7e43aac3ea7fd82921d48498f5b. This matches the required value.
- Every entry of `plan['source_sha256']` (about 120 files: the 9 bundle `.py` files, `protocol.json`, the native `src/flycircuit/**/*.py` tree, the protocol yaml, `W0_width_probe_manifest.json` and `W3_trained_w76_choices.json`) was recomputed. All match.
- All 5 `plan['input_sha256']` entries match (JSON block at the end).
- `flycircuit_v6.plan.verify_plan(plan)` run by me returned OK. The prepare() reconstruction is byte-identical to the plan (canonical hash equal).
- Plan structure: 64 coordinates, 64 unique keys. Seeds are exactly 26001-26008. Each family/method pair has 8 seeds. Learning rates are one fixed pair per family (connectome 0.00171725/0.00515175; gru33 0.000452941/5.03268e-05). Capacity is 32 for all.
- The r1 (`native_v6_validation`) and r2 plans contain the same seeds, but neither directory has a `rows.jsonl` or any result. These seeds have never been executed.

## 2. Protocol and input semantics

- `learning_v3_protocol.yaml` has `task.n_ctx: 3`, `training.trials_per_phase: 800` and `circuit: {name: mushroom_body_right_pilot, min_contacts: 5}`. `runner.execute_native` enforces exactly these (runner.py:148-155). The smoke trial counts (2400 current, 800 per phase) confirm they take effect.
- `selection.json` names circuit `mushroom_body_right_pilot`, dataset flywire 783, side right, `threshold: 5`, `n_neurons: 1500`, `internal_pairs_retained: 7399` and `internal_contacts_retained: 105326`.
- The parquet inputs agree with `selection.json`:
  - `internal_ids.parquet` has 1500 rows with unique `root_id`.
  - `edges_pairs.parquet` has 15,091,983 rows with columns pre, post, count, n_neuropils. I filtered it to pre and post both in the 1500 ids. That gives 63,553 pairs (= `internal_pairs_all`) and 181,421 contacts (= `internal_contacts_all`). With count >= 5 it gives 7,399 pairs and 105,326 contacts, which equals the retained figures.
  - The connectome `raw_gain` numel in the receipt (7399) equals these retained pairs.
  - `neurons.parquet` has 139,255 rows. I did not cross-check neurons against the 1500 ids or recompute `ids_hash_sha256` / `edges_hash_sha256` (light-read scope). This is a minor gap.
- Smoke receipt graph hashes (mapping, edges, signs) are identical for connectome and gru33: 634d0acc..., 1ac1457b..., d5f4ad95.... They are not the same hash definition as `selection.json`'s `edges_hash_sha256`, so I could not equate them directly. They are consistent with each other and across families.

## 3. Smoke receipt integrity (seed 99001, balanced_replay only)

- `passed: true`; both runs `ok`.
- Trial counts for both families: current 2400, replay 1600, extra_current 0, replay_old_context 955, replay_current_context 645. Old plus current replay = 1600.
- Adam steps min = max = 4000 (2400 + 1600). Tensors: 2 (connectome), 8 (gru33).
- Buffer: capacity 32, stored 32, peak 32, observed 2400. Contexts are balanced 11/11/10.
- Params:
  - connectome: owned 7551 = 7399 core + 152 decoder. Encoder 26315 and bias 1500 are frozen. The 20 critic parameters are not optimizer-owned, as disclosed.
  - gru33: owned 7460 = 7392 core + 68 decoder.
  - lr per group equals the plan anchors.
- R_val matrices are 3x3, all values in [0,1] and finite. `continual()` metrics are consistent with them.
- Seed 99001 is not in 26001-26008 and not in 100601-100620.
- I ran `records.validate_result` on a payload built from the real smoke fields (placeholder hashes), for both families, and on a standard-shaped variant. All passed. The real validator was never run on a real result by the smoke script, so this is the closest available evidence.
- Gaps:
  - The smoke covered only `balanced_replay`. `standard`, `extra_current` (stream C and `sample_train`) and `uniform_replay` have not been run on a real model. Their code paths are small and share the same `agent.trial` call, so the risk is low. The first failure would append a `failed` row and stop the run (return 2). That would not produce silent corruption.
  - Timing: connectome took 17.8 s and gru33 10.9 s, so 64 runs take roughly 15 min plus the 30 s guard sleep. That fits the 1800 s cumulative cap with about 2x margin.
  - Memory: the smoke launch waited because Windows free memory was 2.42-2.52 GiB, below the 2.6 GiB launch gate, until it rose to 3.2-3.7 GiB. Peak RSS was 986 MiB. The launch will likely refuse to start at 2.4 GiB. This is an operational note, not a correctness defect.

## 4. Seed exposure

- `/home/atulp/projects/v6_used_seeds.json` (430 ints) does contain 26001-26008, plus 99001 and 100601-100620. The candidate and smoke seeds got there only through v6 files. A text search over `flycircuit`, `flycircuit_debug_copy` and `flycircuit_release` found 26001-26008 only in `v6_candidate/{protocol.json, tests/test_candidate.py, runs/*/plan.json, runs/smoke_r3/exposure_check.json}`. A grep with the `.venv`, `.git`, `data/raw` and `data/processed` paths excluded showed no hits in any non-v6 json, jsonl, yaml, md, py, txt, csv, log or sh file. A first raw `grep -l` matched several `results/interventions_v2/trained/*.parquet` files, but those are byte coincidences in binary data (see the next bullet).
- Gap in the original scan: `v6_seed_scan.py` reads only json, jsonl and yaml with seed-named keys. It did not read parquet. I scanned all 63 parquet result files for any column with "seed" in its name: 14 distinct seeds, none of them in 26001-26008 or 100601-100620, and all already in the used list.
- Reserved 100601-100620 are disjoint from the candidate set (also enforced in plan.py:84-87).
- `exposure_check.json`: n_hashed 278 (270 historical plus 8 candidates), 8 distinct candidate hashes, and an empty collision list for every candidate. The check is nearly tautological, because task data derives from `SeedSequence([seed, ...])`. It is still harmless and supportive.
- Derived-seed arithmetic. I enumerated candidate-derived values (s, s+17, s+31, s+6071, s+6072, s*1000+k for k<200) against the same transforms of all 421 non-v6 historical seeds, in every pairwise combination, and excluded the v6 seeds themselves. There is no overlap.
  - The nearest historical seeds are 25757 and 26757 (and values near 10919/17838...). Their offsets from the candidates are about 244 and 756, which is not 17, 31, 6071 or 6072.
  - The numpy and torch seeds used are `seed` (torch.manual_seed in baselines.py:24 and 58; default_rng in sparse_rnn.py:60 and graph_loader.py:47), `seed+17` (agent.py:51) and `seed+31` (agent.py:71). Everything else uses `SeedSequence([seed, tag, ...])`, which cannot collide across different seeds. `seed*1000+k` (learning_v2.py:159) is only the null-graph path and does not apply to the connectome and gru33 families here.
  - The v6-internal offsets (`+6071` for the buffer, `+6072` for the draw) are Python `random.Random` seeds that exist only in v6. Within the candidate set, `s+6072 == (s+1)+6071` (for example 32073). That is only the replay-buffer RNG versus the draw RNG of adjacent seeds, so the streams are correlated across seeds in a trivial way. It is not a concern between different model seeds, because the buffer and draw streams belong to different `random.Random` instances.
- Smoke seed 99001 and its derived values do not overlap with the candidate seeds or their derived values.

## 5. Code review

I found no concrete defect that would corrupt the measurement run. Items, minor or informational:

1. runner.py:57-100 (`run_continual`). The order of operations is: current trial, sample from buffer (before observe), replay trial, observe. `phase >= 1` gating, the `expected_extra` contract (runner.py:105-111), and the schedule hash of the stream A order, which is independent of method, are all correct. Streams A, B and C are separate, so the current-task schedule is identical across methods and families for a seed. This is what `read_records` requires (`__main__.py:78-97`).
2. replay.py:90-132 (`BalancedReservoir`). Quotas shrink in arrival order via `rng.sample`, then per-context reservoir. This is the documented approximation. The final stored_events is 32, and context counts are 11/11/10 in the smoke.
3. guard.py:179. The mandatory 30 s sleep is charged to the cap (disclosed in the comment). guard.py:215-239 re-verifies the plan every 10 s. Hashing cost is negligible (sha256 of the 57 MB edges file plus ~120 source files takes under 0.1 s).
4. guard.py:240-260. A `return 4` inside `try` with `finally` is fine. If the child finishes normally, `child.returncode` is returned.
5. __main__.py:153-166. An interrupted coordinate leaves no row (documented), and a failed coordinate appends a `failed` row and halts (conservative). On resume, any prior failed row forces an audit.
6. records.py:131-167. The validator hard-codes the family parameter counts (7399/152 and 7392/68), which equal the observed smoke values. It does not check `current_schedule_sha256` content or the graph hashes beyond format. Consistency across seeds and rows is enforced in `__main__.read_records`.
7. Unexercised on real models: standard, extra_current, uniform_replay (see section 3).
8. No rows exist for any plan, so no accidental pre-exposure. Tests exist (`tests/test_candidate.py`); I did not run them (the bundle imports are stdlib, but pytest may write caches and I was told not to modify anything).

## Acceptance receipt reminder

`plan.acceptance()` requires: `verdict == "ship"`, `plan_sha256` equal to the plan digest, a reviewer identity, `seed_exposure_review == "passed"`, `input_sha256` equal to the plan's 5 inputs (JSON below), and `native_smoke_test == "passed"`.

## Verdict

The plan, source, and input hashes verify. The input semantics agree with `selection.json` and the parquet contents. The smoke receipt honors the contract. The candidate seeds are unexposed outside v6 files, and derived-seed arithmetic gives no collisions. I found no defect that would corrupt the measurement. Residual risks are operational (memory gate at 2.6 GiB, 4 of the 4 methods not smoked except balanced_replay), not correctness issues.

VERDICT: ship

```json
{
  "/home/atulp/projects/flycircuit/configs/experiments/learning_v3_protocol.yaml": "6b492862db6e00154e25a917b0205c78f934d0b82383006b0af30812d033894d",
  "/home/atulp/projects/flycircuit/data/processed/flywire/783/neurons.parquet": "836eacd95c2c74cd5908578141ab6f21db512caef712fb01e381a8ede3a29edf",
  "/home/atulp/projects/flycircuit/data/processed/flywire/783/edges_pairs.parquet": "1fd24463041811eaa517c113908ccdd27f84d26bec5f9196a9ed90b4df96595d",
  "/home/atulp/projects/flycircuit/data/processed/flywire/783/circuits/mushroom_body_right_pilot/internal_ids.parquet": "5826d00a60870062499258f950cc333ef2f5f11887331792ee4a33638fac3d52",
  "/home/atulp/projects/flycircuit/data/processed/flywire/783/circuits/mushroom_body_right_pilot/selection.json": "3d6c7f7b9d7b4926e7d9a839345fd844caf03424c7dfc9a8cabb55fa7bba62e5"
}
```
