"""
Tests for bootstrap confidence intervals.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic_data import generate_uplift_data, FEATURES, true_ate
from src.meta_learners import TLearner
from src.evaluation import qini_coefficient
from src.bootstrap import ate_ci, metric_ci


def test_ate_ci_ordering():
    df = generate_uplift_data(n=20000, seed=0)
    ate, lo, hi = ate_ci(df["treatment"].values, df["outcome"].values)
    assert lo < ate < hi


def test_ate_ci_contains_true_ate_at_scale():
    df = generate_uplift_data(n=60000, seed=0)
    _, lo, hi = ate_ci(df["treatment"].values, df["outcome"].values,
                       n_bootstrap=400)
    assert lo <= true_ate(df) <= hi


def test_qini_ci_ordering():
    df = generate_uplift_data(n=20000, seed=0)
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=0)
    m = TLearner().fit(train_df[FEATURES], train_df["treatment"].values,
                       train_df["outcome"].values)
    score = m.predict_uplift(test_df[FEATURES])
    q, lo, hi = metric_ci(score, test_df["treatment"].values,
                          test_df["outcome"].values, qini_coefficient,
                          n_bootstrap=200)
    assert lo < q < hi


def test_ci_narrows_with_more_data():
    """A larger sample should give a narrower ATE interval."""
    small = generate_uplift_data(n=5000, seed=1)
    large = generate_uplift_data(n=80000, seed=1)
    _, lo_s, hi_s = ate_ci(small["treatment"].values, small["outcome"].values,
                           n_bootstrap=300)
    _, lo_l, hi_l = ate_ci(large["treatment"].values, large["outcome"].values,
                           n_bootstrap=300)
    assert (hi_l - lo_l) < (hi_s - lo_s)
