"""Unclipped ranking/tuning/thresholds; corrected probabilities kept separate.

This overlay leaves the frozen benchmark sources and original results untouched.
It applies to one-sample main and supplied-prior experiments. The existing
two-sample implementation already keeps clipping outside its ranking score.
Workers must use one fit at a time because evaluate_test is temporarily wrapped.
"""
import numpy as np

VERSION = 'unclipped_ranking_v1'


def fit_corrected_boosting(base, data, engine):
    """Use exactly the naive fit/selection/threshold path, then correct probability."""
    evaluate = base.evaluate_test

    def separate_probability(method, y_test, score_test, threshold,
                             probability=None, extra=None):
        if method != f'Block2 | Naive + {engine}':
            raise RuntimeError('Unexpected evaluator call in corrected fit')
        details = dict(extra or {})
        details.update(scoring_revision=VERSION, ranking_score_type='raw_engine_score',
                       probability_score_type='clipped_raw_score_divided_by_c',
                       model_selection_score_type='raw_engine_score',
                       threshold_score_type='raw_engine_score')
        return evaluate(
            f'Block2 | ProbCorrection + {engine}', y_test, score_test, threshold,
            probability=base.clip_prob(np.asarray(score_test, dtype=float) / data.c),
            extra=details,
        )

    base.evaluate_test = separate_probability
    try:
        return base.fit_one_boosting_cell(data, 'Naive', engine)
    finally:
        base.evaluate_test = evaluate


def fit_logistic_probability_correction(base, data, C_grid=(0.1, 1.0, 10.0)):
    """Same logistic fitting as frozen M003, with unclipped q for ranking/cutoff.

    c is the true known labeling probability. A supplied population prior in
    data.pi is used only for threshold selection, as in the frozen protocol.
    """
    best = None
    for C in C_grid:
        model = base.LogisticRegression(C=C, max_iter=3000, random_state=data.seed)
        model.fit(data.train.X, data.train.s)
        q_val = model.predict_proba(data.val.X)[:, 1]
        val_score = base.pu_proxy_auc(data.val.s, q_val)
        if best is None or val_score > best['val_score']:
            best = {'C': C, 'val_score': val_score, 'model': model, 'q_val': q_val}

    final = base.LogisticRegression(C=best['C'], max_iter=3000, random_state=data.seed)
    final.fit(np.vstack([data.train.X, data.val.X]),
              np.concatenate([data.train.s, data.val.s]))
    q_test = final.predict_proba(data.test.X)[:, 1]
    threshold = base.prior_quantile_threshold(best['q_val'], data.pi)
    return base.evaluate_test(
        method='Block1 | Elkan-Noto known-c', y_test=data.test.y_true,
        score_test=q_test, threshold=threshold, probability=base.clip_prob(q_test/data.c),
        extra={'best_params': {'C': best['C']}, 'val_metric': best['val_score'],
               'scoring_revision': VERSION, 'ranking_score_type': 'raw_logistic_probability',
               'model_selection_score_type': 'raw_logistic_probability',
               'threshold_score_type': 'raw_logistic_probability',
               'probability_score_type': 'clipped_raw_score_divided_by_c'},
    )
