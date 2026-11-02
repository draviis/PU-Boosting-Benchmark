# Real-data sources and preparation

Raw observation-level datasets are not redistributed here. Authoritative UCI
links and the exact recorded download hashes are supplied. Retain each
dataset's original attribution and terms when obtaining it.

| Task | UCI source | Positive definition | Clean n | Encoded p |
| --- | --- | --- | ---: | ---: |
| Spambase | [94](https://archive.ics.uci.edu/dataset/94/spambase) | Spam | 4204 | 57 |
| HTRU2 | [372](https://archive.ics.uci.edu/dataset/372/htru2) | Pulsar | 17898 | 8 |
| Rice | [545](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik) | Cammeo | 3810 | 7 |
| Phishing Websites | [327](https://archive.ics.uci.edu/dataset/327/phishing+websites) | Original label -1 | 5721 | 30 |
| Credit Default | [350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | Default | 29923 | 33 |
| Dry Bean | [602](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) | DERMASON | 13543 | 16 |
| Optical Digits | [80](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) | Even digits | 5620 | 64 |
| Online Shoppers | [468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) | Revenue | 12205 | 75 |

The release includes the recorded UCI metadata for the six table downloads,
with DOI and source URL. For those six datasets, first materialize the
repository and then call the source-preparation helper:

```bash
python reproduce.py --output ../pu-data --materialize-only
python prepare_real_data.py --workspace ../pu-data/workspace --dataset rice
```

Replace `rice` by `phishing`, `credit_default`, `dry_bean`, `optical_digits`,
or `online_shoppers`. The helper checks the downloaded file against the
recorded source SHA-256 before invoking the frozen preparation function.
It does not fit a model or overwrite an existing prepared dataset. If the
official source changes, review the discrepancy rather than bypassing the check.

Spambase and HTRU2 have direct acquisition functions in the released data
modules: `real_data_spambase/data.py:load_spambase` and
`real_data_htru2/htru2_math_20reps_20260914/real_data_htru2/data.py:load_htru2`.
Their paired split and label-hiding code is in the same modules and the
Spambase low-label module.

Preparation removes all retained-predictor groups with conflicting labels
and reduces other exact duplicates to one record. Credit Default excludes ID
and uses a fixed indicator schema for selected categorical fields. Online
Shoppers excludes PageValues before cleaning and applies fixed categorical
indicators. All learners receive the prepared numeric matrix; CatBoost does
not use native categorical processing.

Stratified 60/20/20 splits are used except Optical Digits, which retains the
official writer-disjoint 1,797-row test set and splits development 75/25.
Background and source pools are disjoint. The two-sample P/U counts match the
one-sample label budgets. Standardization is fitted on selected training data.
The main supplied prior is the complete training-background positive fraction,
not a validation/test-label estimate. These are real predictors with artificial
SCAR label hiding, not naturally collected PU datasets.
