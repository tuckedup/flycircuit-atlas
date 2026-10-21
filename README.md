# FlyCircuit Atlas — v0.1.0-preview

**Experimental, work-in-progress research code** (author: Atul Pandey). Connectome-constrained learning
experiments on a sparse recurrent network built from the FlyWire v783 *Drosophila* connectome, with
control models (GRU, encoder-decoder, rewired graphs) that are *not* parameter-matched. Development and evaluation are ongoing; this is a dated snapshot (2026-10-20), not a finished study.

## Best verified result: connectome acquisition on a validation split

| | |
|---|---|
| **Metric** | Association *acquisition* accuracy (`association.val_acc_acq`): accuracy on the validation cues after 800 online training trials, fresh agent |
| **Model** | Sparse connectome RNN (1,500 nodes, 7,399 edges), frozen random encoder, 76-node readout ("W76"), 7,551 trainable parameters |
| **Split** | **Validation** (exploratory). *Not* a held-out test set |
| **Selected single run** | **89.06 %** — seed 1502 (the best of four check seeds, chosen after the fact) |
| **Aggregate, same four seeds** | **82.42 %** mean (SE 3.5 pp; seeds 1501–1504: 87.5, 89.1, 78.1, 75.0 %) |
| **Configuration** | REINFORCE + TD baseline, Adam, core lr 0.00171725 / decoder lr 0.00515175, 2,400-trial continual protocol; `results_v4/learning_readout_v4_1/W3_PREREG.md` |
| **Reproducible artifact** | Raw rows: `results_v4/learning_readout_v4_1/W3_trained_w76_rows.jsonl` (280 runs, 0 failures); recompute with `python scripts_release/verify_w3_headline.py` |

The 89.06 % is one run picked as the maximum of four; the 82.42 % mean is the fair summary and has wide uncertainty
(95 % CI roughly 71–93 %). These are historical exploratory validation results, **not** confirmed general performance.

### Read this with it
- The seven-percentage-point forgetting advantage over the GRU control has **not** been established (measured, validation: −3.3 pp). That GRU is larger (9,252 vs 7,551 trainable parameters), so it is not a parameter-matched comparison.
- The 68 % body-navigation target (FlyGym, 25 % channel dropout) is **unverified** in this release; a 25-model evaluation is pending.
- Anatomical superiority has **not** been demonstrated: the GRU, the encoder-decoder and all three degree-preserving nulls scored equal or higher on acquisition (only the random-graph control was lower).
- Frozen-feature probes, synthetic 2D navigation, FlyGym navigation and trained acquisition are different metrics and are not interchangeable.

Complete per-seed results, weaker outcomes, failed gates and limitations: **[RESULTS_REPORT.md](RESULTS_REPORT.md)**.
Roadmap and outstanding work: **[ROADMAP.md](ROADMAP.md)**.

## What it is
- **Graph:** a 1,500-node sub-circuit of the FlyWire v783 proofread connectome (signs from neurotransmitter annotations).
- **Model:** sparse rate RNN whose edge weights follow the connectome; sensory input through a fixed random binary encoder
  (fan-in 4 over 19 channels); a small trainable decoder reads 76 nodes. Only edge gains and the decoder train.
- **Learner:** online one-trial REINFORCE with a TD baseline, Adam with separate core/decoder learning rates, gradient clip 1.0.
- **Task:** context-conditioned cue→action association (acquisition), then a three-task continual sequence measuring forgetting.
- **Controls:** three degree-preserving null graphs, a reachability-matched random graph, a GRU, and an encoder-decoder MLP.
  Parameter counts are *not* exactly matched (connectome 7,551; GRU 9,252; encdec 8,956 in the W3 run).

## Install and cheap example (CPU, no data download)
```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch
uv pip install -e ".[dev]"
python scripts_release/verify_w3_headline.py     # <1 s: recomputes the headline numbers from raw rows
pytest -q -m "not gpu and not real_data" tests   # offline unit tests (synthetic fixtures)
flycircuit synthetic run                         # tiny synthetic-graph demo
```
Checked on a fresh CPU-only install at release time: `ruff` clean, 326 tests passed / 7 skipped (GPU and real-data tests are deselected).
Not included, so not supported from this repository: the FlyGym body-navigation rollout environment (`external/flygym_pilot`, needs its own Python 3.12
environment; one body-protocol test that needs it was removed), and the in-progress replay/body-evaluation modules.

(`pyproject.toml` points `torch` at a CUDA 12.8 index for GPU installs; the CPU line above avoids that.)

## Reproducing the reported numbers
- **Recompute from raw rows:** `scripts_release/verify_w3_headline.py` (above). This checks the arithmetic of the published run records.
- **Re-running the training** (`scripts_v4/w3_pilot.py`, launcher `scripts_v4/run_guarded.sh`) needs inputs that are **not** shipped: the ingested
  FlyWire v783 data (~1 GB download, CC-BY-4.0) and the cached degree-preserving null graphs. Follow
  [docs/PREPARE_TRAINING_DATA.md](docs/PREPARE_TRAINING_DATA.md) to build them. Paths are configurable through `FLYCIRCUIT_COPY` and
  `FLYCIRCUIT_HOME` (default: this repository). A re-run produces new rows; it is not expected to match the published rows bit for bit,
  and no re-run has been verified from the published repository yet.
- Unsupported here: the single-heavy-job lock wrapper `heavyjob.sh` and the Windows memory guard (WSL2 only; skipped with a warning elsewhere),
  FlyGym body evaluation, and the replay/body-evaluation program (see ROADMAP).
- The original file hashes from before the 0.1.0 path redaction are in `provenance/`; the patch release changed further script lines
  (see `CHANGELOG.md`), so launch-time hashes in `results_v4/*/W*_prelaunch_hashes.txt` describe the original unredacted files only.

## Provenance
`provenance/SHA256SUMS.txt` lists every published file. `provenance/PATH_REDACTION_ORIGINAL_SHA256.txt` records the hashes of files
*before* local paths were replaced, so the pre-launch hashes in `W*_prelaunch_hashes.txt` can still be checked against the originals.
Hashes collected at launch time show files unchanged since launch; they are not an earlier freeze.

## Licence and attribution
No licence file has been chosen yet, so all rights are reserved by default until the author adds one. Data and methods are third-party
(FlyWire v783, CC-BY-4.0; Shiu et al. LIF model, MIT; FlyGym, Apache-2.0); see [CITATIONS.md](CITATIONS.md) and `docs/citations_data.md`.
Items marked UNVERIFIED there have not been re-checked.
