"""
Tests for the targeting policy and budget optimisation.

Key properties, checkable via ground-truth CATE: targeting the top-k by a good
uplift model beats random targeting, and the profit-optimal budget is an
interior point (not "treat everyone") when treatment has a cost and sleeping
dogs exist.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic_data import generate_uplift_data, FEATURES
from src.meta_learners import XLearner
from src.policy import (
    select_top_k, policy_value_experimental, policy_value_true,
    profit_curve, optimal_budget,
)


def _fit_score(n=40000, seed=0):
    df = generate_uplift_data(n=n, seed=seed)
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=seed)
    model = XLearner().fit(train_df[FEATURES], train_df["treatment"].values,
                           train_df["outcome"].values)
    score = model.predict_uplift(test_df[FEATURES])
    return score, test_df


def test_select_top_k_size():
    score = np.arange(1000)[::-1]
    mask = select_top_k(score, 0.1)
    assert mask.sum() == 100
    # The selected ones should be the highest scores
    assert mask[:100].all()


def test_model_targeting_beats_random_true_value():
    score, test_df = _fit_score()
    true_cate = test_df["true_cate"].values
    model_mask = select_top_k(score, 0.3)
    rng = np.random.default_rng(0)
    random_mask = select_top_k(rng.random(len(test_df)), 0.3)
    assert policy_value_true(model_mask, true_cate) > policy_value_true(random_mask, true_cate)


def test_targeted_subset_can_beat_treating_everyone():
    """Targeting persuadables can yield more incremental conversions than
    treating everyone, because sleeping dogs have negative uplift."""
    score, test_df = _fit_score()
    true_cate = test_df["true_cate"].values
    top30 = policy_value_true(select_top_k(score, 0.3), true_cate)
    everyone = policy_value_true(np.ones(len(test_df), dtype=bool), true_cate)
    assert top30 > everyone


def test_optimal_budget_is_interior():
    """With treatment cost, the profit-optimal budget is not 100%."""
    score, test_df = _fit_score()
    curve = profit_curve(score, test_df["treatment"].values,
                         test_df["outcome"].values,
                         value_per_conversion=100, cost_per_treatment=10)
    best = optimal_budget(curve)
    assert best["budget_frac"] < 1.0


def test_experimental_value_approximates_true_value():
    """On a randomised set the experimental estimate tracks the true value."""
    score, test_df = _fit_score()
    mask = select_top_k(score, 0.3)
    exp = policy_value_experimental(mask, test_df["treatment"].values,
                                    test_df["outcome"].values)
    true = policy_value_true(mask, test_df["true_cate"].values)
    # Within 25% relative — experimental is a noisy but unbiased estimate
    assert abs(exp - true) / true < 0.25
