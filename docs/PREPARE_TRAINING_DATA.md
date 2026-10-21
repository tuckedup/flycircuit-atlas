# Preparing the inputs for the W3 training scripts

**Status: unverified from a clean checkout.** These are the steps the code requires, taken from the loaders' own error messages and the
original workflow. Nobody has yet re-run `scripts_v4/w3_pilot.py` from this published repository. Some provenance checks inside the
scripts may need further artifacts from the original workspace; if one is missing the script stops with a message rather than guessing.

## Layout
Everything is configurable with two environment variables:
- `FLYCIRCUIT_COPY` — this repository (default: the checkout containing the script).
- `FLYCIRCUIT_HOME` — the directory holding the prepared inputs below (default: the same checkout, i.e. a single-root layout).

## Required inputs (none are shipped except where noted)
| Input | Where it must be | How to produce it |
|---|---|---|
| FlyWire v783 proofread connections and annotations (~1 GB, CC-BY-4.0) | `$FLYCIRCUIT_HOME/data/processed/flywire/783/` | `flycircuit data fetch --dataset flywire --version 783 --tier proofread-pairs` then `flycircuit data ingest --dataset flywire --version 783` |
| Circuit selection `mushroom_body_right_pilot` | `data/processed/flywire/783/circuits/mushroom_body_right_pilot/` | `flycircuit data select mushroom_body --side right --pilot-max 1500` |
| Cached degree-preserving null graphs (10 plain, 10 group-constrained) | `results/nulls_v2/mushroom_body_right_pilot/{plain,group_constrained}/` | `python scripts/linux/make_nulls_v2.py mushroom_body_right_pilot 10 plain,group_constrained` (deterministic base seed 2000; shipped script, not re-verified here) |
| Learning-rate anchors from the earlier v3 search | `results/learning_v3/search/selection.json` | **Shipped** in this repository (read-only historical input) |
| Protocol and config files | `configs/` | **Shipped** |

## Run
```bash
export FLYCIRCUIT_COPY=$PWD FLYCIRCUIT_HOME=$PWD      # or point FLYCIRCUIT_HOME at a separate prepared directory
export FLYCIRCUIT_PYTHON=$(which python)
python scripts_v4/w3_pilot.py --selftest               # builds the models only; no training
bash scripts_v4/run_guarded.sh W3 scripts_v4/w3_pilot.py
```
The W3 pilot took about 64 minutes on the original two-thread laptop run. It appends to
`results_v4/learning_readout_v4_1/W3_trained_w76_rows.jsonl`; **move or rename the shipped rows first** if you do not want a resume
(the script skips every recorded key) — a re-run writes new rows and is not expected to match the published ones exactly.

## Not supported
`heavyjob.sh` (single-heavy-job lock wrapper) is not shipped; without it `run_guarded.sh` runs the script directly. The Windows free-RAM guard
works only under WSL2 and is skipped with a warning elsewhere. FlyGym body evaluation and the replay program are not part of this release.
Licence: none chosen yet (all rights reserved by default).
