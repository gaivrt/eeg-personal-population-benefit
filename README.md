# Personal and population gains in EEG foundation models

Code and saved analysis results for **Separating personal from population gains when calibrating EEG foundation models for new users**.

Authors: Xilin Tao and Kani Chen, The Hong Kong University of Science and Technology. Corresponding author: Kani Chen (makchen@ust.hk). Version: 1.0.0.

The study compares personal adaptation of CBraMod, REVE and LaBraM against named population models and exchanged adapters. It covers 235 held-out subjects in three core motor-imagery datasets and additional CBraMod controls. This release includes negative results, unavailable outcomes and query-informed prior diagnostics; those diagnostics do not establish a leakage-free personalization benefit.

## Contents

- `src/subject_context/`: preprocessing, population/personal adaptation, context and meta-learning implementations.
- `scripts/`: original analysis and execution entry points, including the three-model and population-training-budget extensions. These are supplied for inspection and reproduction with separately obtained data and weights.
- `configs/`: fixed hyperparameters, split definitions and pinned model sources.
- `reports/`: the saved subject-level scores, comparison tables, convergence summaries and run records used by the manuscript reporting scripts.
- `paper/nature_portfolio/figures/`: scripts that produce figures, tables and the numerical ledger from saved results.
- `paper/nature_portfolio/numbers.csv`: the 37,265-record numerical provenance ledger.
- `paper/nature_portfolio/supplementary_data/`: Supplementary Data 1 (904 comparisons), CSV, XLSX and column documentation.
- `RELEASE_MANIFEST.json`: checksums for the distributed files.

## Verify the saved release without training

With Python 3.12, run `python verify_release.py`. This checks file hashes, provenance paths, stable comparison IDs and the saved numeric ledger. It does not load models, fit parameters, perform inference or run bootstrap.

## Recreate figures and summaries from saved results

Install the reporting environment with `python -m pip install -r paper/nature_portfolio/requirements-reporting.txt`, then run `python paper/nature_portfolio/figures/build_all.py` from this repository. This recreates descriptive tables and plots and repeats the documented, deterministic subject bootstrap (20,000 resamples). It does not run training or inference. The script writes reporting outputs; use a separate checkout to preserve the deposited files. Numeric source hashes may differ from the original local workspace because operational path strings were replaced, but all scientific numeric values are retained.

## Data and model access

Original EEG recordings and pretrained weights are not redistributed. Obtain recordings from the original repositories listed in `DATA_SOURCES.md`, and obtain model assets under the upstream terms and pinned revisions in `configs/assets.json` and `configs/cross_model_x.yaml`. The REVE/LaBraM source-file URLs and checksums are in `docs/cross_model_audit_2026-09-27/source_manifest.json`; `python fetch_upstream_sources.py` downloads only those pinned source/metadata files, never EEG or weights. CBraMod setup uses its pinned upstream repository as documented in `configs/assets.json` and the setup script in the original study. End-to-end training was not rerun to prepare this release.

## Interpretation

Seeds are averaged within subject before the subject-bootstrap comparisons. Intervals do not model dependence induced by shared population models and reused donors. Source-list pretraining overlap is recorded by model and dataset. The original query-informed prior controls are retained for traceability and are explicitly distinguished from the ordinary-continuation meta-learning comparison. Missing values are not zero.

## License and citation

Original analysis software is provided under the MIT license. Authors' derived tables and documentation are provided under CC BY 4.0; upstream data, model code and weights retain their own terms (see `THIRD_PARTY_NOTICES.md`). Cite this version using `CITATION.cff`. The Zenodo DOI will be recorded after a public archive is successfully created; no DOI is claimed by this initial release.

Repository: https://github.com/gaivrt/eeg-personal-population-benefit
