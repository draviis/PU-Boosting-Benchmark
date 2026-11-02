# Third-party materials

## AdaPU

AdaPU code originates from Yawen Zhao's
[Ada-PU repository](https://github.com/Yawen-Zhao-820/Ada-PU), distributed
under the MIT license, copyright (c) 2022 Yawen Zhao. The full notice is
retained at `code/publication_preparation/adapu_rng_repair/LICENSE` and applies
to the AdaPU sources and derived versions, including those under
`code/phase3_confirmatory_S001_S040_100reps/Ada-PU/`.
Benchmark-specific repairs and wrappers do not remove the upstream notice.

## KM2 mixture-proportion estimator

The wrapper uses the official implementation accompanying Ramaswamy, Scott,
and Tewari (2016), [Mixture Proportion Estimation via Kernel Embedding of
Distributions](https://proceedings.mlr.press/v48/ramaswamy16.html).
The original source archive is linked from
[Clayton Scott's code page](https://web.eecs.umich.edu/~cscott/code.html).
No redistribution license was recorded in the archived benchmark copy, so
the upstream source is not included or relicensed under MIT. Use
`fetch_km2.py` to obtain and verify it from the author before running that
experiment. The benchmark wrapper and processed results are included.

## Datasets and dependencies

The original UCI datasets are not included. Obtain them from their authoritative
sources and retain the source-specific attribution and license terms.
Processed benchmark metrics contain no observation-level dataset records.
The repository's MIT license covers original code, not the original datasets.

NumPy, SciPy, pandas, scikit-learn, PyTorch, XGBoost, LightGBM, CatBoost,
CVXOPT, Cython, Matplotlib, and LaTeX components are external dependencies.
Their respective licenses continue to apply; their runtime binaries are not
redistributed here.
