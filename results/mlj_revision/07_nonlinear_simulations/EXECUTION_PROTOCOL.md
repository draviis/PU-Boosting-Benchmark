# Approved nonlinear-extension execution protocol

This protocol incorporates the user's 17 October 2026 execution corrections and supersedes the catalog, threshold-transfer and approval-status passages in the historical NONLINEAR_DGP_PLAN.md. Its six DGP equations, frozen parameters and calibration are unchanged. The original plan and calibration archive remain preserved.

## Catalog and counts

Exactly the original 25 practical PU configurations plus two original supervised references, separately reported. No Task 9 first-order variant, BaggingPU, or Task 4 nonlinear-spy method. Retain original logistic-spy RN. Task 9 remains a feasibility/scope audit.

Six DGPs (NL01--NL06) x two designs x 100 repetitions = 1,200 cases.
Expected: 30,000 PU method rows plus 2,400 supervised-reference rows = 32,400 rows.
Training/tuning fits are more numerous than result rows; correction aliases reuse their original fitted engines when supported by the source protocol.

## Metrics and model stages

Primary ranking metrics AUROC and AP use raw ranking scores from the final train+validation refitted model, with the original frozen training, selection and refit definitions.

Secondary balanced accuracy, F1 and MCC use the selected TRAIN-ONLY model. Select the empirical (1-pi_tilde) quantile of that model's validation ranking scores (NumPy quantile, linear interpolation), then classify test scores from the SAME model using score >= threshold.
For one-sample use the full population-representative validation population; for two-sample use validation U only. Corrected methods use RAW q. No numerical threshold transfer to a refitted model. No old mixed threshold rules. Task 1C is not required.
Apply this same threshold diagnostic to the two supervised references as well, keeping those references outside practical PU rankings; their original training/selection may use true training/validation labels.

Save selected-model validation/test and refitted-model validation/test raw ranking scores, validation S masks, evaluation Y, selected parameters and candidate criteria. Refit validation scores are auditable in-sample predictions, never used to choose a threshold. Hidden test Y enters only downstream evaluation. For practical PU fitting, hidden train/validation/test Y is redacted.

## Information, randomness and source preservation

Supplied pi_tilde=.20; one-sample c=.30; two-sample canonical risk c=0. Sizes, SCAR samplers, seeds and phase-44 main repetition indices 0--99 remain as specified by nonlinear_dgps.py and frozen_dgp_parameters.json. No recalibration or scenario redesign.
Original source grids, selection criteria, preprocessing, tie handling and refit behavior are unchanged. Score-capture instrumentation must pass a scientific-AST preservation check. Engine versions are fixed across comparators. Nonlinear scenario IDs are separate from S001--S040.

## Preflight and dispatch

The plan and full 1,200-case run are approved. Cluster upload/execution on ravidaya's Augsburg account is already authorized. Execute smoke cases with repetition indices >=10000, outside inference; use their numerical checks and timing only, never their performance to change DGPs, grids or catalog.
Verify all score stages, all 27 methods, both threshold populations, finite scores, matching correction aliases, no hidden-label selection and source fidelity before main dispatch. Save hashes and environment. Failure recovery may replay failed cases with the identical frozen definition; completed results must not be overwritten.

## Analysis

Individual NL01--NL06 method/design results are primary. Also report the simple six-DGP mean and an equal-five-mechanism mean, averaging NL01 and NL05 first as the sinusoidal mechanism. Keep supervised references separate. Do not use best-of-group envelopes.

Predeclared paired repetition-level contrasts within each DGP/design:
- RN minus naive and cost-sensitive minus naive for each of five original boosting learners.
- Each of the three fixed nonboosting baselines minus naive CatBoost, cost-sensitive CatBoost, AdaPU and original PU gradient boosting.
- Probability-corrected minus naive same-engine checks for identical raw scores/hard predictions.
Report mean paired differences, MCSE=sd(differences)/sqrt(n), and mean +/- 1.96 MCSE; require n=100 for complete main cells. Report failures/coverage rather than treating incomplete cells as complete. No hypothesis testing with the six designed DGPs as a random population.
