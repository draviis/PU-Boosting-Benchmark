# Longer-training diagnostic

Triggered by the completed 500-epoch pilot: 537 of 960 selected controlled-model checkpoints occurred at the 500-epoch cap. This is a training-budget limitation, not a crashed job or proof of a faulty loss.

Increase only the maximum epoch allowance from 500 to 1500. Preserve optimizer, learning rate, width/decay grids, full batches, validation rules, patience, scaling, seeds, source generators and refit/threshold rules. Record validation trajectories and terminal reasons. This is a post-pilot sensitivity analysis, not a prespecified confirmatory result.

Rerun all eight controlled configurations on all 120 existing pilot datasets (960 fits), including originally uncapped ones, to avoid selecting cases by favorable test performance. Regenerated data fingerprints and runtime package versions must match the original pilot. No fresh statistical repetitions are added. Reuse the 240 original contextual CatBoost rows and predictions after identity and metric checks; do not present these as newly fitted models. Original 500-epoch results remain preserved.

Run on num with four CPUs/16 GB, at most four simultaneous tasks. First repetition of all 12 design/scenario combinations gates the other 108 tasks. Collection requires complete valid output and independently recomputed metrics. If a model still selects the new cap, report that unresolved limitation rather than claiming convergence.

No automatic expansion to 100 repetitions or additional datasets. Report paired 1500-minus-500 differences separately from the original pilot conclusions.
