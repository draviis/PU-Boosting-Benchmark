# Results guide

All paths below are relative to `results/`. Compression adds `.gz` where
needed. The manifest maps every file to its restored workspace path.

| Analysis | Main location | Recommended entry |
| --- | --- | --- |
| Original 40 settings | `mlj_revision/02_fixed_method_and_family_analysis/` | `family_balanced_results.csv`, `fixed_method_scenario_results.csv`, `repetition_data.csv.gz` |
| Nonlinear extension | `mlj_revision/07_nonlinear_simulations/` | `nonlinear_results.csv.gz`, `nonlinear_dgp_calibration.csv`, `nonlinear_scenario_summary.csv` |
| Final nonlinear analysis | `mlj_revision/07b_nonlinear_final_analysis/` | learner/strategy decomposition, unique-score ranks, original-versus-nonlinear comparisons, unavailable RN refits |
| Real-data rankings | `publication_preparation/eta_scoring_20261013/collected/real_raw.csv.gz` | 17,280 method records across 640 cases; `AUPRC` denotes average precision (AP) |
| Final supplementary displays | `mlj_revision/final_supplement_revision/` | `revised_tables/`, `revised_figures/`, `source_data/` |
| Main simulation displays | `mlj_revision/display_package/main/` | core fixed-method figure and tables; use final supplementary revision for updated nonlinear labels |
| Main real-data displays | `mlj_revision/real_data/main/` | fixed-method tables and learner/strategy decomposition |
| RN mechanism / selector | `mlj_revision/03_rn_mechanism/`, `mlj_revision/04_rn_selector_sensitivity/` | selection quality, performance, paired uncertainty, fixed downstream learner diagnostics |
| Supplied-prior sensitivity | `publication_preparation/eta_scoring_20261013/collected/prior_*` and `phase7_*/results/`, `phase9_*/results/` | raw metrics and paired changes |
| Estimated-prior sensitivity | `mlj_revision/06_estimated_prior/` | results, coverage, estimation accuracy, and failure tables |
| Linear–neural extensions | `publication_preparation/epoch_check_analysis/`, `publication_preparation/real_complete_analysis/` | paired simulation summaries and real-data repetition metrics |
| Standardized timing | `mlj_revision/final_supplement_revision/timing/` | 540 timing records, including separately measured correction overhead |

## IDs and fields

`M001`/`TM001` are supervised logistic regression and `M002`/`TM002` are
supervised CatBoost. Practical PU methods are 003–027. IDs 003–005 are
corrected logistic, linear uPU, and linear nnPU. Within the five-method ranges
006–010 (naive), 011–015 (corrected), 016–020 (RN), and 021–025 (CS), engines
are AdaBoost, GBM, XGBoost, LightGBM, CatBoost. IDs 026 and 027 are AdaPU and
PU gradient boosting. `TM` denotes historical two-sample IDs; newer outputs
use `M` and a separate design field.

AUROC and AP/AUPRC are on the 0–1 scale unless the column ends in `_pp`.
Paired Monte Carlo uncertainty is computed from within-repetition differences;
the denominator is the actual number of available pairs, normally 100.
`MC_lower` and `MC_upper` are mean ± 1.96 MCSE, not tests treating designed
scenarios as randomly sampled observations. Missing metric fields remain missing.

## Coverage and decision rules

The 216,000 core method records supply 432,000 AUROC/AP observations.
Core repetition data additionally preserve historical BA results and a small
PU-estimated-BA pilot. The revised same-model hard-metric pilot covers eight
original simulation settings per design and one repetition (400 practical
PU method/case results), not all 40 settings × 100 repetitions. The verified
real-data same-model hard-metric coverage is also partial. No full-core
same-model BA/F1/MCC aggregate is claimed.

The nonlinear extension supplies 32,400 method records. Six RN final refits
have no selected negatives and therefore missing AUROC/AP. Selected-model
hard metrics remain available. See `unavailable_RN_refits.csv`. Do not replace
these entries with chance performance or borrow selected-model ranking scores.

Naive/corrected nonlinear same-engine scores are exact aliases and collapse
to 20 unique practical PU ranking entries. In the original two-sample archive,
four non-roundoff AdaBoost metric discrepancies across two repetitions prevent
collapsing that pair: the retained catalog has 21 entries. The release verifies
these exceptions rather than silently averaging the aliases.

The nonlinear common quantile uses all population-representative validation
observations in one-sample and validation U only in two-sample. Raw q scores
are used for probability-corrected methods. The same selected train-only model
scores validation and test; no numerical threshold is transferred to a refit.
The threshold pilot contains quantile, PU-estimated-BA, and oracle-BA rules.
It does not establish a verified same-model fixed-probability-cutoff analysis.

The linear–neural real-data comparison has a fixed training budget. Some fits
hit its epoch cap; complete metric availability does not establish convergence.

## Numeric precision

Display CSVs may contain rounded values for typesetting. Use repetition files,
`source_data/*full_precision.csv`, or `audit/source_checks/*_numeric.csv` for
calculation. The reproduction wrapper reads floats using round-trip precision
and verifies rebuilt numerical results against the archives.
