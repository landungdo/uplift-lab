"""
Tests for uplift evaluation metrics (Qini, AUUC, uplift@k).

The central property, checkable because we have ground-truth CATE: on a held-out
test set, ranking by the true CATE (oracle) should score highest, a fitted
learner should beat random, and random should sit near zero.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic_data import generate_uplift_data, FEATURES
from src.meta_learners import TLearner
from src.evaluation import qini_curve, qini_coefficient, auuc, uplift_at_k


def _setup(n=30000, seed=0):
    df = generate_uplift_data(n=n, seed=seed)
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=seed)
    return train_df, test_df


def test_qini_curve_starts_at_origin():
    _, test_df = _setup()
    f, g = qini_curve(test_df["true_cate"].values,
                      test_df["treatment"].values, test_df["outcome"].values)
    assert f[0] == 0.0 and g[0] == 0.0
    assert f[-1] == 1.0  # ends at full population


def test_oracle_beats_random_qini():
    _, test_df = _setup()
    t, y = test_df["treatment"].values, test_df["outcome"].values
    q_oracle = qini_coefficient(test_df["true_cate"].values, t, y)
    rng = np.random.default_rng(0)
    q_random = qini_coefficient(rng.random(len(test_df)), t, y)
    assert q_oracle > q_random


def test_fitted_learner_beats_random():
    train_df, test_df = _setup()
    m = TLearner().fit(train_df[FEATURES], train_df["treatment"].values,
                       train_df["outcome"].values)
    pred = m.predict_uplift(test_df[FEATURES])
    t, y = test_df["treatment"].values, test_df["outcome"].values
    q_model = qini_coefficient(pred, t, y)
    rng = np.random.default_rng(0)
    q_random = qini_coefficient(rng.random(len(test_df)), t, y)
    assert q_model > q_random


def test_oracle_beats_fitted_learner_out_of_sample():
    """Out-of-sample, the true-CATE oracle should not be beaten by a learner."""
    train_df, test_df = _setup(n=40000)
    m = TLearner().fit(train_df[FEATURES], train_df["treatment"].values,
                       train_df["outcome"].values)
    pred = m.predict_uplift(test_df[FEATURES])
    t, y = test_df["treatment"].values, test_df["outcome"].values
    q_model = qini_coefficient(pred, t, y)
    q_oracle = qini_coefficient(test_df["true_cate"].values, t, y)
    assert q_oracle >= q_model


def test_uplift_at_k_positive_for_oracle():
    _, test_df = _setup()
    u = uplift_at_k(test_df["true_cate"].values,
                    test_df["treatment"].values, test_df["outcome"].values, k=0.3)
    assert u > 0


def test_qini_normalized_oracle_is_one():
    """Normalizing the oracle by itself gives 1.0."""
    _, test_df = _setup()
    t, y = test_df["treatment"].values, test_df["outcome"].values
    oracle = test_df["true_cate"].values
    from src.evaluation import qini_normalized
    assert abs(qini_normalized(oracle, t, y, oracle) - 1.0) < 1e-9


def test_qini_normalized_random_near_zero():
    """Random targeting normalizes to near zero."""
    _, test_df = _setup()
    t, y = test_df["treatment"].values, test_df["outcome"].values
    oracle = test_df["true_cate"].values
    from src.evaluation import qini_normalized
    rng = np.random.default_rng(0)
    qn = qini_normalized(rng.random(len(test_df)), t, y, oracle)
    assert abs(qn) < 0.2


def test_qini_normalized_model_between_random_and_oracle():
    """A fitted model should normalize between random (~0) and oracle (1)."""
    train_df, test_df = _setup()
    t, y = test_df["treatment"].values, test_df["outcome"].values
    oracle = test_df["true_cate"].values
    from src.evaluation import qini_normalized
    m = TLearner().fit(train_df[FEATURES], train_df["treatment"].values,
                       train_df["outcome"].values)
    pred = m.predict_uplift(test_df[FEATURES])
    qn = qini_normalized(pred, t, y, oracle)
    assert 0.2 < qn < 1.0
