"""
Tests for experiment design and health checks.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic_data import generate_uplift_data
from src.experiment_design import (
    required_sample_size, minimum_detectable_effect, srm_check,
    two_proportion_test, aa_test_false_positive_rate,
)


def test_smaller_effect_needs_more_samples():
    n_small = required_sample_size(0.05, 0.01)
    n_large = required_sample_size(0.05, 0.05)
    assert n_small > n_large


def test_more_samples_gives_smaller_mde():
    mde_small_n = minimum_detectable_effect(0.05, 1000)
    mde_large_n = minimum_detectable_effect(0.05, 100000)
    assert mde_large_n < mde_small_n


def test_srm_not_flagged_for_balanced_split():
    res = srm_check(100000, 100000)
    assert not res["srm_detected"]


def test_srm_flagged_for_imbalanced_split():
    res = srm_check(105000, 95000)
    assert res["srm_detected"]


def test_two_proportion_detects_real_difference():
    # 5% vs 8% on large samples should be significant
    res = two_proportion_test(500, 10000, 800, 10000)
    assert res["p_value"] < 0.01
    assert res["effect"] > 0


def test_aa_false_positive_rate_near_alpha():
    """A correct pipeline yields A/A false positives near the alpha level."""
    df = generate_uplift_data(n=40000, seed=1)
    control = df.loc[df["treatment"] == 0, "outcome"].values
    fpr = aa_test_false_positive_rate(control, n_trials=400, alpha=0.05, seed=0)
    # Allow a tolerance band around 0.05 given finite trials
    assert 0.02 < fpr < 0.09
