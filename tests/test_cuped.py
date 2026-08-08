"""
Tests for CUPED variance reduction.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic_data import generate_uplift_data
from src.cuped import compute_theta, apply_cuped, cuped_ate


def _data(n=40000, seed=0):
    df = generate_uplift_data(n=n, seed=seed)
    rng = np.random.default_rng(seed)
    pre = df["p_control"].values + rng.normal(0, 0.15, len(df))
    return df["treatment"].values, df["outcome"].values, pre


def test_cuped_preserves_ate():
    """CUPED should not bias the ATE (same expectation)."""
    t, y, pre = _data()
    res = cuped_ate(t, y, pre)
    assert abs(res["ate_cuped"] - res["ate_raw"]) < 0.01


def test_cuped_reduces_standard_error():
    """With a correlated covariate, the SE should shrink."""
    t, y, pre = _data()
    res = cuped_ate(t, y, pre)
    assert res["se_cuped"] < res["se_raw"]
    assert res["se_reduction"] > 0


def test_cuped_adjusted_has_lower_variance():
    t, y, pre = _data()
    y_adj = apply_cuped(y, pre)
    assert np.var(y_adj) < np.var(y)


def test_zero_correlation_covariate_no_reduction():
    """An unrelated covariate gives ~no variance reduction."""
    t, y, _ = _data()
    rng = np.random.default_rng(99)
    noise = rng.normal(0, 1, len(y))  # independent of y
    res = cuped_ate(t, y, noise)
    # Reduction should be near zero (allow a small negative/positive wobble)
    assert abs(res["se_reduction"]) < 0.02


def test_theta_zero_for_constant_covariate():
    t, y, _ = _data()
    const = np.ones(len(y))
    assert compute_theta(y, const) == 0.0
