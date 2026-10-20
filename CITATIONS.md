# CITATIONS

Source-verification table. "Verified" means checked against the original source on the date given; anything else is marked.
Licenses must be re-inspected before redistributing derived assets.

| Source | Identifier / URL | Access date | Relevant claim | Implementation consequence | License | Verified |
|---|---|---|---|---|---|---|
| FlyWire connectivity release v783.0 | Zenodo 10676866, DOI 10.5281/zenodo.10676866; https://zenodo.org/records/10676866 | 2026-10-15 | proofread_connections_783.feather ~852 MB (md5 f48f972d...), synapse export ~9.5 GB | default ingest = proofread pairs only; synapse tier dry-run | CC-BY-4.0 | yes (Zenodo API, `flycircuit data sources --verify`) |
| FlyWire annotations | https://github.com/flyconnectome/flywire_annotations @ a83b2776d60d5764cef36b927f5f9679c16c47a2 | 2026-10-15 | cell class/type, top_nt, known_nt | selectors use these field names; missing labels raise | none declared by GitHub API - inspect before redistributing | yes (commit pinned) |
| Shiu et al. connectome LIF model code | https://github.com/philshiu/Drosophila_brain_model @ 91bdd1e7dcf193f3e7ca5a8933497fcef63b7960 | 2026-10-15 | LIF model; v630 defaults, v783 file option | independent LIF engine + Brian2 reference check (docs/refcheck.md) | MIT | yes (cloned, commit pinned). Paper title/DOI not re-verified here: UNVERIFIED |
| Dorkenwald et al., Nature 2024 (FlyWire connectome) | cited via Zenodo record/annotation README | - | whole-brain connectome and proofreading | citation only | - | UNVERIFIED (DOI not re-fetched) |
| Schlegel et al., Nature 2024 (cell types) | cited via annotation README | - | cell-type annotation | citation only | - | UNVERIFIED (DOI not re-fetched) |
| Apache Arrow Feather V2 | https://arrow.apache.org/docs/python/feather.html | listed, not re-fetched | Feather V2 = Arrow IPC | batch-wise reads with column projection | Apache-2.0 | docs link only |
| uv PyTorch guide | https://docs.astral.sh/uv/guides/integration/pytorch/ | listed | explicit index for CUDA wheels | `pytorch-cu128` explicit index | - | docs link; behavior verified by running |
| NVIDIA CUDA on WSL | https://docs.nvidia.com/cuda/wsl-user-guide/index.html | listed | no Linux display driver in WSL | Windows driver only | - | docs link; GPU use verified by running |
| NeuroMechFly / FlyGym | https://neuromechfly.org/installation/ ; PyPI `flygym` 2.1.0 | 2026-10-15 | flygym 2.1.0, Python >= 3.12 | isolated env `external/flygym_pilot`; tutorial 1a smoke only | Apache-2.0 (per agent check; LICENSE.flygym copied) | partially (PyPI/GitHub checked; paper not cited) |
| MaleCNS, BANC, hemibrain, larval releases | - | - | other connectome releases | lazy adapters raising NotImplementedError | - | UNVERIFIED (release names/access not checked) |
| Mushroom-body dopamine/learning, central-complex, flyvis, null-model literature | - | - | topic guidance only | none cited as evidence | - | UNVERIFIED (no papers fetched) |

Any 2026 paper or repository from the earlier draft that cannot be verified is UNVERIFIED and not cited as evidence.
Data-layer detail: `docs/citations_data.md`.
