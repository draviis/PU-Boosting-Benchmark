# PU Boosting Benchmark

Source code and processed results for the revised positive–unlabeled (PU)
boosting benchmark. The repository separates experiment implementations
from the processed data used for analysis and publication displays.

## Start here

| Location | Contents |
| --- | --- |
| [`code/`](code/) | Frozen simulation, PU sampling, fitting, analysis, and plotting implementations |
| [`results/`](results/) | Repetition-level metrics, summaries, figure inputs, tables, and PDFs |
| [`reproduce.py`](reproduce.py) | Verified no-fitting analysis and figure reproduction entry point |
| [`docs/RESULTS_GUIDE.md`](docs/RESULTS_GUIDE.md) | Which results to use and their coverage limits |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Experiment code map, environments, and limitations |
| [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) | Real-data sources and preparation recipes |
| [`MANIFEST.json`](MANIFEST.json) | File sizes, SHA-256 checksums, and original workspace-relative paths |

The core contains 25 PU configurations and two separately reported
supervised references, 40 settings, two sampling designs, and 100 repetitions.
The nonlinear extension has six separately calibrated DGPs, both designs,
and 100 repetitions. The main real-data benchmark contains eight tasks,
two designs, two label budgets, and 20 repeated splits per task.

## Reproduce analyses without fitting models

Python 3.10 or newer is required. Install the analysis dependencies:

```bash
python -m pip install -r requirements-analysis.txt
python reproduce.py --verify-only
python reproduce.py --output ../pu-reproduction --figures
```

On Windows, clone into a short directory and enable Git's long-path support:
`git -c core.longpaths=true clone https://github.com/draviis/PU-Boosting-Benchmark.git`.

Choose a **new output directory outside this repository**. The command
verifies input hashes, decompresses the data into an isolated workspace,
rebuilds core family-weighted ranking summaries and paired comparisons,
re-executes the frozen nonlinear analyses, checks the results against the
archive, and regenerates publication figures. It does not fit or refit models.
`VALIDATION.json`, CSV analyses, and PDFs are written beneath the chosen output.

To compile the released final supplementary table sources as well, install
a LaTeX distribution with `pdflatex`, then add `--tables`. For example:

```bash
python reproduce.py --output ../pu-reproduction-with-tables --figures --tables
```

Large CSV/JSON inputs use standard gzip compression. Pandas can read
`.csv.gz` directly. `--materialize-only --output ../pu-workspace` restores
the original relative directory layout without running analysis or fitting.
The distributed repetition data use CSV, not Python pickle.

## Interpretation

- AUROC and AP are primary. Supervised references are excluded from practical
  PU rankings. Fixed methods and paired repetition-level differences replace
  best-of-group maxima.
- The preferred core overall summary averages settings within each of ten
  families, then weights the families equally. The ordinary 40-setting mean
  remains a sensitivity summary.
- Nonlinear DGPs are reported separately. Six-DGP and equal-five-mechanism
  summaries are supplementary; NL01 and NL05 share a mechanism in the latter.
- Nonlinear AUROC/AP use final train+validation refits. BA/F1/MCC use the selected
  train-only model for both validation threshold estimation and test scoring.
  Six RN final refits are unavailable; they are not imputed.
- Historical Task-2 BA tables use the earlier refit-transfer rule. They are
  preserved for provenance and must **not** be relabeled as complete results
  under the revised same-model protocol. See the coverage guide.
- Real-data repeated splits are not independent datasets. Overall real-data
  summaries give equal weight to datasets.

## Data and provenance

The repository provides metric-level processed data and numerical display
inputs. It does not redistribute original UCI observation-level datasets,
every per-observation prediction array, fitted model objects, or cluster
runtime binaries. The supplied recipes explain how the real tasks were
acquired, cleaned, encoded, split, and converted into PU samples.
Reproducing an entire expensive fitting study is distinct from reproducing
its analyses; the latter is the tested default workflow.

Experiment code and saved outputs are retained in their historical relative
paths so method versions and analysis stages remain traceable. The public
wrapper adds CSV portability and independent consistency checks. Three
non-scientific path substitutions are documented in the manifest. Original
benchmark outputs were not changed in preparing this release.

## License and citation

Original code is released under the [MIT license](LICENSE). Third-party code
and source datasets retain their own terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The original AdaPU MIT notice
is retained. The separate upstream KM2 source is not redistributed; a
checksum-verified download helper is provided.

For a reproducible reference, cite this repository and the exact commit used.
[`CITATION.cff`](CITATION.cff) supplies repository citation metadata. No article
DOI or acceptance status is asserted by this release.
