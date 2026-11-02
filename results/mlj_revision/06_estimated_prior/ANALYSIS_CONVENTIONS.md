# Task 6 analysis conventions

This document fixes analysis definitions before the benchmark experiment.

## Units and paired comparisons

AUROC, AP, BA, Brier and log loss use their natural units. Delta metrics mean estimated information minus oracle information. Higher AUROC/AP/BA is favorable; higher Brier/log loss is unfavorable. Percentage-point display multiplies AUROC/AP/BA differences by 100.

Simulation summaries keep scenario and design separate. MCSE is sample SD of the paired differences divided by sqrt(n_valid). A 95% Monte Carlo interval is mean +/-1.96*MCSE. Leave MCSE and intervals unavailable when n_valid<2. Preserve the full requested denominator and count missing/failed pairs explicitly.

Real-data summaries first average valid paired repetitions within dataset/design/budget, retaining valid/requested counts. Average the four condition means equally within each dataset, then weight the eight datasets equally. Report missing conditions explicitly; do not silently renormalize an incomplete dataset as a full-coverage eight-dataset result.

## Estimation accuracy

Bias is mean(estimate-truth). Absolute bias is abs(mean error), distinct from mean absolute error. RMSE is sqrt(mean squared error). Median absolute error, median relative absolute error and 90th-percentile absolute error summarize valid estimates. Failure rate uses all requested cases. Saturation warnings and safeguarded square roots are reported separately from failures.

Simulation truth is the generating pi, c and theoretical pi_U. Save empirical training fractions separately so sampling fluctuation is visible. Real-data oracle pi is the empirical training-background prevalence supplied by the original semi-synthetic construction, not a natural population claim.

The one-sample estimator output kappa is pi_U_hat. pi_hat and c_hat are derived using lambda_hat from the complete training population. Two-sample kappa is pi_hat; no natural c is estimated in that design. Raw and usable estimates are distinct fields. Invalid estimates remain failures and are never replaced by true priors.

## Failure and invariance coverage

A failed prior estimate prevents fitting the four prior-dependent methods and prevents corrected-probability evaluation. The prior-independent ranking models still have defined archived AUROC/AP; identify their mathematical invariance without pretending a successful prior estimate exists.

For the six corrected models, preserve raw-score hashes and independently verify that recomputing AUROC/AP from the identical raw scores gives exactly equal oracle and estimated values. Retain recovery differences versus archived rounded/numerically harmonized scores separately. No clipping of corrected eta may enter ranking metrics.

## Probability metrics

Physical probability correction clips eta to [0,1]. Brier uses that eta. Log loss applies the benchmark's 1e-7 stability clipping. Retain historical Brier/LogLoss values separately from the paired re-evaluated oracle probability values, because the historical corrected-probability helper also applied numerical endpoint clipping to eta. This numerical convention does not authorize overwriting archived results.

Use uPU/nnPU's defined sigmoid probabilities and archived metrics under the original implementation. AdaPU and PU gradient boosting have no invented probability outputs.

## Secondary BA

Use only validation/test scores of the same selected train-only model. One-sample quantiles use complete validation PU; two-sample quantiles use validation U only. Test Y is used only to evaluate the selected cutoff. Paired comparisons require oracle and estimated selected-model scores on the identical case. Missing oracle same-model scores remain unavailable. No old mixed thresholds, corrected-probability 0.5 main comparison or refit-transfer threshold is permitted.

## Relationship to the old sensitivity study

For each scenario/design, calculate pi_hat/pi and fractions within [0.8,1.2], [0.9,1.1] and [0.95,1.05]. Use valid estimates for these fractions and separately report estimator failures on all cases. Contextualize performance using archived prior-misspecification results; do not rerun the grid or pool the eight scenarios as one DGP.

Spearman correlations and scatterplots relate signed absolute prior error (pi_hat-pi), relative prior error ((pi_hat-pi)/pi), and one-sample c error to paired AUROC/AP differences within scenario/design/method. Call these descriptive associations, not causal identification.
