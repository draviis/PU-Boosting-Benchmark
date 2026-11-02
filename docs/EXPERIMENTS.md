# Experiment code and reproduction scope

`reproduce.py` is the tested entry point for analyses. It never starts
training. Historical study drivers below are provided to document and
recreate the experiments; scheduler configuration and large archived score
arrays are not part of this source release. A fresh full fitting run was
not performed when preparing the repository.

## Source map

Paths are relative to `code/` and become workspace-relative after materializing.

| Purpose | Implementation |
| --- | --- |
| Original one-sample DGPs, sampling, tuning and fitting | `phase3_confirmatory_S001_S040_100reps/code/` |
| Original two-sample protocol | `phase5_two_sample_scar_confirmatory_100reps/code/` |
| Frozen scoring harmonization | `publication_preparation/scoring_harmonization_20261012/frozen/` and `publication_preparation/eta_scoring_20261013/frozen/` |
| Real-data final AdaPU workflow | `publication_preparation/adapu_workflow/worker.py` and `frozen_code/` |
| AdaPU Cython weak learner and implementation | `phase3_confirmatory_S001_S040_100reps/Ada-PU/` and `publication_preparation/adapu_rng_repair/` |
| Nonlinear DGPs, exact likelihood ratios, calibration | `mlj_revision/07_nonlinear_simulations/nonlinear_dgps.py`, `calibrate_dgps.py`, `test_nonlinear_dgps.py` |
| Nonlinear fitting and same-model threshold instrumentation | `mlj_revision/07_nonlinear_simulations/worker.py`, `sources/` |
| RN recovery and empty-refit handling | `worker_rn_recovery.py`, `worker_rn_empty_refit.py`, associated protocols under the nonlinear directory |
| RN mechanism / selector sensitivity | `mlj_revision/03_rn_mechanism/worker.py`, `mlj_revision/04_rn_selector_sensitivity/worker*.py` |
| Prior sensitivity | phase7/phase9 `code/` and `mlj_revision/06_estimated_prior/worker.py`, `prior_estimator.py` |
| Linear–neural simulation / real-data extensions | `nonlinear_pu_epochcheck_20260925/`, `pu_real_neural_20260925/` |
| Standardized timing | `mlj_revision/final_supplement_revision/source_code/timing_worker.py` |

## Frozen nonlinear environment

The recorded Linux/Python 3.10 environment used NumPy 2.2.6, SciPy 1.15.3,
pandas 2.3.3, scikit-learn 1.7.2, PyTorch 2.5.1+cpu, CatBoost 1.2.10,
xgboost-cpu 3.2.0, and LightGBM 4.7.0. These are recorded experiment versions,
not a claim that arbitrary newer versions yield identical fitted models.
CVXOPT 1.3.2 is additionally required for the official KM2 estimator wrapper.
Build the included AdaPU Cython extension locally with Cython, NumPy, and a
compatible C/C++ compiler. Compiled cluster binaries are not redistributed.

An explicit single-block fitting entry is provided:

```bash
python reproduce.py --output ../pu-fitting --materialize-only
# In ../pu-fitting/workspace/phase3_confirmatory_S001_S040_100reps/Ada-PU:
python setup.py build_ext --inplace
# Back in the release repository, after installing the recorded fitting environment:
python run_nonlinear_block.py --workspace ../pu-fitting/workspace --case 0 --block 0
```

This starts expensive fitting only when explicitly invoked. Cases 0–1199 and
their seeds are specified by `main_cases.json`. Blocks 0–7 partition the
original 27 configurations. The portable launcher removes exactly one archived
machine-specific compiled-binary hash assertion, records the new build hash,
and retains all scientific source and package checks. It does not change the
DGPs, objectives, tuning, thresholds, or failure rules. The launcher was
syntax/structure checked; fitting was not rerun during publication packaging.

Use the archived RN recovery protocol for failed RN candidates. A selected
final refit with zero negatives is unavailable; no fallback method or imputed
ranking score is permitted. Task-9 first-order risk variants, BaggingPU, and
nonlinear RN selectors are not added to the nonlinear catalog.

Some historical task workers refer to pre-existing score archives or
recovered data bundles. Those workers are source/protocol records, not a
promise that every cluster command runs against this lightweight release.
Regenerating their model-level inputs requires the corresponding original
experiment and preparation steps. The metric-level inputs needed for the
reported analyses are included and tested separately.

## Calibration

Do not recalibrate to reproduce the reported nonlinear study. Use the
released `frozen_dgp_parameters.json`. The archived calibration used common
random-number streams, 50,000 observations per class for candidate evaluation,
and 12 bisection iterations where numerical search was required. The radial
and threshold-interaction mechanisms have analytic calibration relationships;
NL05 inherits NL01's signal. Final Bayes evaluation used a separate large
population sample. The frozen calibration table and code specify numerical
values and uncertainty.

## Standardized timing

The timing archive has 20 cases: reference/p=500 × two designs × five fixed
realizations. It contains 440 fitted-pipeline records and 100 incremental
probability-correction records. One library thread per worker, distinct
physical cores on an exclusive AMD EPYC 7543 node, and discarded warm-up were
used. Timers include internal scoring/threshold preparation and redirected
fitting diagnostics, but exclude final result/prediction writes. Corrections
reuse the paired naive fit; its cost is not charged twice to a shared fit.
The records are wall-clock measurements, not hardware-independent complexity.
The exact protocol is also in the archived timing report.
