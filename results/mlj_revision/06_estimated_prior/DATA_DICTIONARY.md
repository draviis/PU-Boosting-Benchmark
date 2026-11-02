# Task 6 result guide

All deltas are estimated-information minus oracle-information. Higher AUROC, AP and balanced accuracy are favorable; higher Brier score and log loss are unfavorable. Values in CSVs use natural units. Figures marked "pp" multiply AUROC/AP differences by 100.

## Units and identifiers

A case is a fixed training/validation/test construction identified by `key`, `fingerprint`, `seed`, design and repetition. Simulation cases use eight prespecified scenarios and 100 original repetitions per design. Real cases use eight datasets, two designs, two label budgets and 20 original repetitions.

Method numbers 003-027 identify the 25 practical PU configurations. The two supervised references are excluded. Method 003 is labelled "Probability-corrected logistic" across designs; `archived_method_label` retains the design-specific original name. Methods 004, 005, 026 and 027 depend on the supplied prior during fitting. Methods 003 and 011-015 use the prior in probability correction. All remaining methods reuse archived rankings.

`information_condition` separates oracle and estimated arms. Never merge them into a single ranking. `ranking_stage` is final training-plus-validation refit. `BA_stage` is the selected training-only model and has different coverage.

## Files

| File | Unit and interpretation |
| --- | --- |
| prior_estimation_accuracy_case.csv | One row per requested case, including raw/usable estimates, population and empirical reference quantities, counts, numerical warnings and provenance. |
| prior_estimation_accuracy_summary.csv | Accuracy for each scenario/design or dataset/design/budget and each applicable quantity (pi, pi_U, c). |
| prior_estimation_failures.csv | Failed estimator cases with the same schema as the case table. A header-only file means zero estimator failures, not zero requested cases. |
| estimated_prior_simulation_results.csv | One simulation case/method/information arm per row. |
| estimated_prior_simulation_differences.csv | One simulation case/method paired comparison per row. |
| estimated_prior_mc_uncertainty.csv | One simulation scenario/design/method/metric summary; retains requested and valid paired counts. |
| estimated_prior_real_results.csv | One real case/method/information arm per row. |
| estimated_prior_real_summary.csv | Condition means, equal-four-condition dataset means, and the equal-eight-dataset summary; select the appropriate summary_level before analysis. |
| prior_error_vs_performance.csv | Paired cases for four prior-dependent learners plus separately identified within-condition Spearman summaries. Filter record_type before use. |
| prior_sensitivity_coverage.csv | Estimated-prior coverage fractions plus separately identified archived misspecification context. Filter record_type before use. |

## Accuracy fields

`pi_error` and `absolute_prior_error` mean the signed error pi_hat - pi in native prevalence units; the latter follows the requested terminology and is not an absolute value. `pi_absolute_error` and `prior_absolute_error_magnitude` are nonnegative error magnitudes. `relative_prior_error` is (pi_hat - pi)/pi. `prior_ratio` is pi_hat/pi.

`absolute_bias` is abs(mean signed error), distinct from mean absolute error. RMSE is sqrt(mean squared error). Median relative absolute error is median(abs(estimate-truth)/truth). Accuracy summaries exclude invalid estimates from numerical error measures but retain all cases in failure-rate denominators.

In one-sample data `kappa_raw` estimates pi_U; `lambda_hat` uses all training observations. In two-sample data kappa estimates pi directly. One-sample c is lambda_hat/pi_hat. No natural c is estimated for two-sample sampling. Blank fields mark nonapplicability or missing metrics, never zero.

The retrospective empirical_train_pi field is the fraction of Y=1 in the stored training array. In two-sample data that array pools the deliberately enriched P sample with U, so this pooled fraction is not the population prior and is never used as KM2's target. The true_pi field records the generating prior (simulations) or original training-background prior (real data).

`search_lower_saturation` and `search_upper_saturation` identify numerical search boundary warnings. They are separate from solver failures. The saved official KM2 search resolution prevents reaching arbitrarily small mixture weights.

## Performance and uncertainty

`Delta_metric` = estimated_metric - oracle_metric for a paired case. MCSE uses the sample standard deviation of paired differences divided by sqrt(n_valid); intervals are mean +/-1.96 MCSE. These quantify simulation precision within a fixed DGP, not uncertainty over a random population of scenarios.

Real-data condition summaries average repetitions. Dataset means then weight the four design/budget conditions equally; the overall summary weights eight datasets equally. Aggregate dataset rows deliberately have no MC interval that would treat repeated cases as independent datasets.

For corrected probabilities, `oracle_Brier` and `oracle_LogLoss` in differences use a fresh evaluation of the same archived raw score under the oracle correction. Historical values remain in `oracle_archived_*` fields (and `archived_*` in oracle arm rows). Physical [0,1] clipping applies before Brier; 1e-7 endpoint clipping applies only to log loss. Ranking always uses raw q.

Oracle linear uPU/nnPU Brier metrics retain the archived 1e-7 endpoint convention, with a possible discrepancy bounded by 2e-7 in Brier units. The probability_convention / oracle_probability_convention and oracle_Brier_endpoint_error_bound fields disclose this limited exception; see ORACLE_LINEAR_PROBABILITY_NOTE.md. Newly evaluated probabilities use numerical clipping only for log loss.

Paired same-model BA is available only where archived oracle selected-model scores exist. BA_oracle_coverage distinguishes fixed_pilot, targeted_Task1B_audit and unavailable. The 1,194 fixed-pilot pairs alone enter BA summary means; eight additional targeted HTRU2 audit pairs are retained as case-level diagnostics and excluded from summaries. The targeted cases were selected for extreme original threshold behavior, so pooling them with the fixed pilot would introduce selection bias. Newly fitted selected-model BA elsewhere does not create a paired oracle comparison. MCSE is blank for n_valid < 2. See BA_COVERAGE_NOTE.md.

## Provenance and scope

Per-case JSON records store estimator parameters, solver calls, source fingerprints, selected model parameters, candidate validation criteria, random seeds, software versions and prediction hashes. Raw prediction NPZ files remain in the approved cluster Task 6 directory; the interpretation ZIP contains compact scientific result tables and validation evidence.

The archived +/-20% prior grid uses separate original seeds and is contextual, not paired with Task 6. Its one-sample arm held c fixed at the true value. Task 6 estimates c as well as pi.
