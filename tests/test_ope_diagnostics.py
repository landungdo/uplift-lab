"""
Tests for off-policy evaluation diagnostics.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ope_diagnostics import (
    overlap_diagnostics, importance_weights, effective_sample_size,
    ess_ratio, clip_weights,
)


def test_randomised_design_has_perfect_overlap():
    e = np.full(10000, 0.5)
    ov = overlap_diagnostics(e)
    assert ov["overlap_ok"]
    assert ov["frac_below_low"] == 0.0
    assert ov["frac_above_high"] == 0.0


def test_extreme_propensity_fails_overlap():
    # Half the units at 0.01, half at 0.99
    e = np.concatenate([np.full(5000, 0.01), np.full(5000, 0.99)])
    ov = overlap_diagnostics(e)
    assert not ov["overlap_ok"]


def test_ess_equals_n_for_uniform_weights():
    w = np.ones(1000)
    assert abs(effective_sample_size(w) - 1000) < 1e-6
    assert abs(ess_ratio(w) - 1.0) < 1e-6


def test_ess_drops_for_skewed_weights():
    # One huge weight dominates -> ESS much smaller than N
    w = np.concatenate([np.full(999, 1.0), [1000.0]])
    assert ess_ratio(w) < 0.5


def test_clipping_reduces_max_weight_and_reports_fraction():
    w = np.concatenate([np.full(990, 1.0), np.full(10, 100.0)])
    clipped, frac = clip_weights(w, percentile=95)
    assert clipped.max() < w.max()
    assert 0 < frac <= 0.05 + 1e-9


def test_clipping_raises_ess_ratio_for_skewed_weights():
    w = np.concatenate([np.full(990, 1.0), np.full(10, 500.0)])
    before = ess_ratio(w)
    clipped, _ = clip_weights(w, percentile=99)
    after = ess_ratio(clipped)
    assert after > before
