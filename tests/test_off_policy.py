"""
Tests for off-policy evaluation (IPS / SNIPS / Doubly Robust).

Key property, checkable via known potential outcomes: all three estimators
should be close to the true policy value on a randomised log, and Doubly Robust
should be at least as accurate as raw IPS.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic_data import generate_uplift_data, FEATURES
from src.meta_learners import TLearner
from src.off_policy import (
    _policy_actions, ips_value, snips_value, dr_value, true_policy_value,
)


def _setup(n=40000, seed=0, threshold=0.0):
    df = generate_uplift_data(n=n, seed=seed)
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=seed)
    tl = TLearner().fit(train_df[FEATURES], train_df["treatment"].values,
                        train_df["outcome"].values)
    score = tl.predict_uplift(test_df[FEATURES])
    mu1 = tl.model_treated.predict_proba(test_df[FEATURES])[:, 1]
    mu0 = tl.model_control.predict_proba(test_df[FEATURES])[:, 1]
    t = test_df["treatment"].values
    y = test_df["outcome"].values
    e = np.full(len(test_df), 0.5)  # exact design propensity
    actions = _policy_actions(score, threshold)
    true_val = true_policy_value(actions, test_df["p_treated"].values,
                                 test_df["p_control"].values)
    return dict(actions=actions, t=t, y=y, e=e, mu1=mu1, mu0=mu0, true=true_val)


def test_ips_close_to_true():
    d = _setup()
    est = ips_value(d["actions"], d["t"], d["y"], d["e"])
    assert abs(est - d["true"]) < 0.03


def test_snips_close_to_true():
    d = _setup()
    est = snips_value(d["actions"], d["t"], d["y"], d["e"])
    assert abs(est - d["true"]) < 0.03


def test_dr_close_to_true():
    d = _setup()
    est = dr_value(d["actions"], d["t"], d["y"], d["e"], d["mu1"], d["mu0"])
    assert abs(est - d["true"]) < 0.03


def test_dr_not_worse_than_ips():
    """Doubly Robust should be at least as accurate as raw IPS here."""
    d = _setup()
    ips_err = abs(ips_value(d["actions"], d["t"], d["y"], d["e"]) - d["true"])
    dr_err = abs(dr_value(d["actions"], d["t"], d["y"], d["e"],
                          d["mu1"], d["mu0"]) - d["true"])
    assert dr_err <= ips_err + 0.01  # small tolerance for noise


def test_treat_nobody_matches_control_outcome():
    """A policy that treats nobody should value near the control conversion rate."""
    d = _setup(threshold=10.0)  # threshold so high nobody is treated
    # everyone gets control; true value ~ mean control potential outcome
    est = snips_value(d["actions"], d["t"], d["y"], d["e"])
    assert abs(est - d["true"]) < 0.03
