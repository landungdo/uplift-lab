"""
Tests for the semi-synthetic uplift data generator.

These lock in the ground-truth properties the rest of the project relies on:
the four segments exist with the intended sign of uplift, randomised assignment
is balanced, and naive difference-in-means recovers the true ATE at scale.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic_data import generate_uplift_data, true_ate


def test_all_four_segments_present():
    df = generate_uplift_data(n=20000, seed=0)
    segs = set(df["segment"].unique())
    assert segs == {"persuadable", "sure_thing", "lost_cause", "sleeping_dog"}


def test_segment_uplift_signs():
    """Persuadables have positive uplift, sleeping dogs negative, others ~zero."""
    df = generate_uplift_data(n=40000, seed=0)
    means = df.groupby("segment")["true_cate"].mean()
    assert means["persuadable"] > 0.1
    assert means["sleeping_dog"] < -0.1
    assert abs(means["sure_thing"]) < 1e-9
    assert abs(means["lost_cause"]) < 1e-9


def test_randomised_assignment_is_balanced():
    df = generate_uplift_data(n=20000, seed=0, treatment_share=0.5)
    assert 0.47 < df["treatment"].mean() < 0.53


def test_naive_estimator_recovers_ate_at_scale():
    """Under randomisation, difference-in-means is unbiased for the true ATE."""
    df = generate_uplift_data(n=200000, seed=0)
    naive = (df.loc[df.treatment == 1, "outcome"].mean()
             - df.loc[df.treatment == 0, "outcome"].mean())
    assert abs(naive - true_ate(df)) < 0.005


def test_confounding_breaks_naive_balance():
    """With confounded assignment, treated and control differ on x0."""
    df = generate_uplift_data(n=20000, seed=0, confounding=True)
    x0_treated = df.loc[df.treatment == 1, "x1"].mean()
    x0_control = df.loc[df.treatment == 0, "x1"].mean()
    assert abs(x0_treated - x0_control) > 0.3  # a clear imbalance


def test_potential_outcomes_ordering():
    """p_treated and p_control are valid probabilities."""
    df = generate_uplift_data(n=10000, seed=0)
    assert df["p_control"].between(0, 1).all()
    assert df["p_treated"].between(0, 1).all()
