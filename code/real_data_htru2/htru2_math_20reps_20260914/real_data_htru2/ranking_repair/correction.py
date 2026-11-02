"""Ranking-preserving one-sample probability correction.

Reuse the frozen naive learner's complete tuning/refit path. Add a corrected
probability only at final evaluation; never transform scores used for ranking,
model selection, or the classification threshold.
"""

import numpy as np


def fit_corrected(base, data, engine):
    original_evaluate = base.evaluate_test

    def evaluate(method, y_test, score_test, threshold, probability=None, extra=None):
        assert method == f"Block2 | Naive + {engine}"
        details = dict(extra or {})
        details.update(repair_revision="ranking_v1", ranking_score_type="raw_engine_score",
                       probability_score_type="clipped_raw_score_divided_by_c")
        return original_evaluate(
            f"Block2 | ProbCorrection + {engine}", y_test, score_test, threshold,
            probability=base.clip_prob(np.asarray(score_test, dtype=float) / data.c), extra=details,
        )

    base.evaluate_test = evaluate
    try:
        return base.fit_one_boosting_cell(data, "Naive", engine)
    finally:
        base.evaluate_test = original_evaluate
