# Standardized timing study

Completed 20 data cases and 540 configuration records: reference and 500 predictors, both designs, five fixed realizations, 25 PU configurations plus two supervised references. No timing case failed. No predictive result was replaced.

OS: one-sample; TS: two-sample. Brackets give the 25th and 75th percentiles. Standalone reference times are seconds and high-dimensional times are minutes; incremental corrections in the final panel are milliseconds. Raw times are always retained in seconds. CPU: AMD EPYC 7543 32-Core Processor. Python 3.10.12. One library thread per worker; at most 20 case workers pinned to distinct physical cores on one exclusive node. Active concurrency decreases as cases finish. All methods share the same data within a case. Warm-up is discarded. Timers include internal scoring/threshold preparation and redirected fitting diagnostics. Corrected pipelines reuse the matching naive fit: standalone time adds the measured correction; the fit is not charged twice to the shared benchmark. Candidate counts and software versions are in the raw CSV. Candidate counts are validation candidates/checkpoints from the frozen selection hooks, not counts of separate trajectories; AdaPU can evaluate several stages from each beta trajectory. The two supervised references are separate comparators. The exploratory neural extension uses a distinct protocol and is excluded.

The worker metadata originally abbreviated the exclusion as "file I/O". The raw-table timer_definition makes the actual scope explicit: result and prediction files are outside the timer, while redirected fitting diagnostics may add logging cost. The original wording is retained in archived_timer_definition; no time is subtracted or estimated.

Methods ran in the frozen order recorded in the worker, not a randomized order. These are descriptive wall-clock timings on a shared-memory node, not hardware-independent complexity measures or single-process isolation estimates. Five repetitions give limited information on timing variability. Quartiles use the linear empirical-quantile convention. Candidate counts refer to evaluated validation candidates/checkpoints, not independent fitted trajectories. The timing uses generated reference/high-dimensional data with fixed seeds 2026102000 through 2026102019; all methods within a case share its fingerprint. There were 440 complete fitting pipelines and 100 incremental corrections.

Software versions:

- catboost: 1.2.10
- lightgbm: 4.7.0
- numpy: 2.2.6
- pandas: 2.3.3
- scikit-learn: 1.7.2
- scipy: 1.15.3
- torch: 2.5.1+cpu
- xgboost-cpu: 3.2.0
