# Changelog

## v0.1.1-preview (maintenance, 2026-10-20)
The `v0.1.0-preview` tag is unchanged. No result, row or number changed.
- Wording: the GRU control is **not** parameter-matched (9,252 vs 7,551 trainable parameters); "matched GRU / matched controls" removed.
- `scripts_release/verify_w3_headline.py` now also detects duplicate or missing rows, wrong design coverage, check rows that do not use the chosen
  setting, disagreement with `W3_summary.json`, wrong parameter counts, and drift between the documents and the recomputed values.
  `tests/unit/test_release_verifier.py` feeds it corrupted copies to prove each detector fires.
- CI now runs the verifier and `flycircuit synthetic run`.
- Training paths are configurable (`FLYCIRCUIT_COPY`, `FLYCIRCUIT_HOME`, `FLYCIRCUIT_PYTHON`) instead of redaction placeholders; see
  `docs/PREPARE_TRAINING_DATA.md`. Added the small historical input `results/learning_v3/search/selection.json` and `scripts/linux/make_nulls_v2.py`.
- `scripts_v4/*` and `learning_v4.py` therefore differ from the files hashed in `W*_prelaunch_hashes.txt`; those hashes describe the original
  unredacted files (see `provenance/PATH_REDACTION_ORIGINAL_SHA256.txt`). Unsupported features and the missing licence remain stated in the README.

## v0.1.0-preview (2026-10-20)
First public snapshot.
