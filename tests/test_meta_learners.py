"""
Tests for the S-Learner and T-Learner meta-learners.

The key property, only checkable because the data is semi-synthetic: each learner
should recover the known ground-truth ATE and rank units by their true CATE
better than chance.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic_data import generate_uplift_data, FEATURES, true_ate
from src.meta_learners import SLearner, TLearner


def _fit_predict(learner_cls, n=20000, seed=0):
    df = generate_uplift_data(n=n, seed=seed)
    X = df[FEATURES]
    learner = learner_cls().fit(X, df["treatment"].values, df["outcome"].values)
    pred = learner.predict_uplift(X)
    return pred, df


def test_slearner_recovers_ate():
    pred, df = _fit_predict(SLearner)
    assert abs(pred.mean() - true_ate(df)) < 0.02


def test_tlearner_recovers_ate():
    pred, df = _fit_predict(TLearner)
    assert abs(pred.mean() - true_ate(df)) < 0.02


def test_slearner_ranks_cate_better_than_chance():
    pred, df = _fit_predict(SLearner)
    corr = np.corrcoef(pred, df["true_cate"].values)[0, 1]
    assert corr > 0.3


def test_tlearner_ranks_cate_better_than_chance():
    pred, df = _fit_predict(TLearner)
    corr = np.corrcoef(pred, df["true_cate"].values)[0, 1]
    assert corr > 0.3


def test_predict_uplift_shape():
    pred, df = _fit_predict(TLearner, n=5000)
    assert pred.shape == (5000,)
    assert np.isfinite(pred).all()
