# DATA (FlyWire v783) - verified sources, schemas, measured counts

Everything below marked **measured** was computed locally on 2026-10-15 from the files in
`data/raw/flywire/783/SOURCE_MANIFEST.json` (sizes + md5 verified against Zenodo; sha256 stored).
Rerun: `flycircuit data fetch`, `flycircuit data ingest`, `flycircuit data validate --recheck-hashes`.

## Sources (verified at original source)
| file | bytes | md5 (Zenodo) | status |
|---|---|---|---|
| proofread_connections_783.feather | 852,022,274 | f48f972d262323a102aed49af1396b8a | downloaded, verified |
| proofread_root_ids_783.npy | 1,114,168 | e0e6c19732fd8c7a4e39a2d170105421 | downloaded, verified |
| Supplemental_file1_neuron_annotations.tsv (flywire_annotations @ a83b277) | 31,720,298 | none published | downloaded |
| per_neuron_neuropil_count_{pre,post}_783.feather | 16.9 MB / 233.8 MB | see sources | not downloaded |
| flywire_synapses_783.feather | 9,492,998,242 | f8f1b97c... | downloaded, md5 verified (2026-10-16); full scan done, see "Full synapse export" |

Zenodo record 10676866 "FlyWire Whole-brain Connectome Connectivity Data" v783.0, CC-BY-4.0.
`flycircuit data sources --verify` re-checks the live API (passed 2026-10-15).

## Schemas
`proofread_connections_783.feather` (Feather V2, 258 record batches of <=65,536 rows):
`pre_pt_root_id:int64, post_pt_root_id:int64, neuropil:string, syn_count:int64,
gaba_avg, ach_avg, glut_avg, oct_avg, ser_avg, da_avg: double`. Rows are per
(pre, post, neuropil). Transmitter-average columns are NOT carried into edges parquet (deferred).
Annotation TSV: 31 columns (root_id, supervoxel_id, pos_*/soma_* in 4x4x40 nm voxel space,
flow, super_class, cell_class, cell_sub_class, cell_type, hemibrain_type, top_nt, top_nt_conf,
known_nt, side, nerve, ... ); coordinates are nullable.

## Outputs (`data/processed/flywire/783/`, git-ignored)
neurons.parquet (release-qualified `uid`, exact int64 `root_id` + `root_id_str`),
edges_by_neuropil.parquet (pre, post, neuropil, count, n_source_rows),
edges_pairs.parquet (pre, post, count summed over neuropils, n_neuropils), counts.json.
Orientation: pre -> post (graph layer converts to W[post, pre]).

## Measured counts (counts.json)
| quantity | value |
|---|---|
| proofread root IDs (unique) | 139,255 |
| annotation rows / unique roots | 139,248 / 139,248 (no duplicates) |
| annotated proofread roots | 139,241 (14 proofread roots lack annotation; 7 annotation roots not in proofread array) |
| source rows (pair x neuropil) | 16,847,997 |
| total contacts (syn_count sum) | 54,492,922 |
| unique directed pairs | 15,091,983 |
| duplicate (pre,post,neuropil) keys merged | 0 |
| self-loop pairs | 0 |
| roots appearing in edges | 138,639 |
| edge endpoints outside proofread set | 0 (all pairs internal) |
| pairs retained at >=5 contacts (model choice, not applied to files) | 2,700,513 pairs / 34,153,566 contacts |

The 54.5 M figure is the proofread-pair contact total; it is NOT the ~130 M broad synapse export
count, and neither is "the proofread-brain contact count" automatically.
Ingestion cost (measured): 21 s wall, peak RSS 2.56 GiB (budget 4.5 GiB), DuckDB cap 2 GiB.
Cross-batch aggregation scale test (`scripts/linux/scale_test_ingest.py`): 8 M synthetic rows,
8 batches, 0.5 GiB cap, exact vs Python oracle; peak RSS 1.0 GiB incl. the oracle dict.

## Circuit selections (`flycircuit data select`, threshold >=5 contacts/pair)
| circuit | neurons | internal pairs (all / retained) | boundary partners in / out (retained) |
|---|---|---|---|
| mushroom_body (KC 5177, MBON 96, DAN 331 [PAM/PPL], APL 2, DPM 2) | 5,608 | 523,784 / 44,024 | 3,596 / 3,194 |
| central_complex (cell_class CX) | 2,875 | 300,234 / 71,415 | 2,780 / 843 |
| visual_motion (heuristic type list) | 30,863 | 822,731 / 173,134 | 30,032 / 33,675 |
| feeding_reference (gustatory 410 + brain motor 105; APPROXIMATE vs Shiu assay) | 515 | 4,129 / 497 | 2,056 / 835 |
| mushroom_body_right_pilot (PILOT SUBSET, seed 0, cap 1500; KC 1385, DAN 88, MBON 25, APL 1, DPM 1) | 1,500 | retained 7,399 (all-pairs/boundary figures: see its selection.json/BOUNDARY.md) | see BOUNDARY.md |

Note: the 4 cell_class MBIN neurons are APL (2) and DPM (2); each selection.json holds group counts.
The 5,608-neuron full MB exceeds the 1-5 k engineering-pilot sizing guide; the pilot is a
separately named, proportionally subsampled right-hemisphere subset (not a biological claim).
Each circuit dir holds internal_ids.parquet, boundary_ids.parquet, selection.json, BOUNDARY.md
(hashes of ids and retained edges).

## Not verified / unavailable
- Annotation repo license (GitHub API reports none): inspect before redistributing derived tables.
- MaleCNS, BANC, hemibrain, larval: adapters raise NotImplementedError (access/release unverified).
- Full synapse coordinates: acquired in the export (see below); units are source-documented nanometres,
  only partially cross-checked (one root); no viewer-space transform is defined.

## Full synapse export (`flywire_synapses_783.feather`) — added 2026-10-16

* **Downloaded and verified:** 9,492,998,242 bytes = expected size; local md5
  `f8f1b97c9d4b0ea9b4c8b287f6b99091` = Zenodo md5 (`SOURCE_MANIFEST.json`). The first attempt stopped at
  3.48 GB (server closed the stream; the old downloader hashed the partial and reported a size mismatch);
  `data/sources.py::download` now resumes automatically (up to 30 attempts), the partial was resumed, not restarted.
* **Schema (measured, `results/synapses/schema.json`):** 18 columns, 1,985 record batches, identical to the
  source description (no missing/extra columns). Coordinates are `int64`; the *source* documents them as
  nanometres — carried as `source_documented`, **not independently verified**; no transform to viewer space is defined.
* **Code:** `src/flycircuit/data/synapses.py` (record batches + column projection, DuckDB cap 1 GiB, spill dir
  under `data/processed/_tmp`, no full pandas load). Keeps four counts distinct: broad-export rows, rows with
  both endpoints proofread (= contacts), directed pairs among them, and the proofread-connections file counts.
  Tested on synthetic Feather (`tests/unit/test_data_synapses.py`: cross-batch aggregation vs a Python oracle, ids > 2^53, orientation).
* **Full-pass counts (measured, `results/synapses/scan.json`, complete, 1,985 batches, 259 s, peak RSS 5.5 GiB):**

  | Quantity | Count |
  |---|---|
  | broad export rows (synapses) | 130,054,535 |
  | both endpoints proofread (= contacts) | 54,492,970 |
  | only pre proofread | 67,421,243 |
  | only post proofread | 3,608,754 |
  | neither proofread | 4,531,568 |
  | self-loop contacts (both proofread) | 0 |
  | directed pairs (both proofread) | 15,092,004 |
  | distinct root IDs anywhere in export | 43,464,703 |

  Null/zero pre or post root IDs: 0 rows. These four classes are different things from each other and from
  the proofread-connections file: the 130 M figure is the broad export, not the proofread-brain contact count.
* **Comparison with the ingested proofread file (`results/synapses/compare.json`, `diff_pairs.json`).**
  An earlier `compare.json` showed `null` for the proofread side because the reader looked for the counts at the
  top level of `counts.json`, where they are nested under `edges` (fixed in `compare_with_pairs`, regression test
  added). Corrected comparison: export has **21 more directed pairs and 48 more contacts** than the proofread
  file (15,092,004 vs 15,091,983 pairs; 54,492,970 vs 54,492,922 contacts).
  * Pair-level anti-join (bounded DuckDB, `scripts/linux/x1_synapse_diff.py`): all 21 differing pairs are
    *only in the export*; **0 pairs have a differing count and 0 pairs are only in the proofread file**. The other
    15,091,983 pairs match exactly in count. No duplicate (pre, post) keys on either side.
  * All 21 pairs involve one root, `720575940623940963` (17 as presynaptic, 4 as postsynaptic partner).
    It is in `proofread_root_ids_783.npy` and in the annotation table (cell_type R7, super_class sensory,
    side right) but has **no rows in `proofread_connections_783.feather`** (0 pairs in `edges_pairs.parquet` and
    `edges_by_neuropil.parquet`). Its raw synapse rows (`results/synapses/root_rows.json`): 69 rows, all in
    `ME_R`, of which 48 have both endpoints proofread (= the 48-contact gap exactly); `cleft_score` 52–179,
    `connection_score` 6.7–629 (not unusually low-confidence).
  * **Explained to the root level; the cause is not determined.** Why the connections file omits this one root
    is unverified (the record metadata we inspected does not say). Hypotheses (e.g. a build-time root change or
    omission) are untested. Practical consequence: the ingested graph misses 21 pairs / 48 contacts of one
    peripheral neuron, 0.00009% of contacts; no model result depends on it.
* **Coordinates:** `pre_pt_position_*` are finite for all 69 rows checked. For that root the synapse positions
  (x 745k–770k, y 351k–365k, z 127k–147k) sit within about 0.1 mm of the annotation `pos_*` (208067, 91871, 3045
  voxels at 4×4×40 nm = 832k, 367k, 122k nm), which is consistent with nanometres. This is a one-root
  consistency check, not verification of the unit or of any viewer transform; spatial use needs a transform to be
  defined and checked first.
* **Mushroom-body synapse partition: not written.** `extract_pathway` exists and is tested, but a pass over the
  export took 259 s with a 5.5–5.8 GiB peak RSS (Arrow batches, not DuckDB), which is unsafe while Windows has
  1–2.5 GiB free. Contacts in the eight `MB_*` neuropils total 1,535,281 (sum of
  `contacts_by_neuropil_both_proofread`), so the partition would be modest, but it is deferred until a spatial
  drill-down needs it. No CLI command is wired for it yet; the building block is
  `flycircuit.data.synapses.extract_pathway(path, pre_ids, post_ids, out)`, to be run under `heavyjob.sh`.
