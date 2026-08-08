"""
Diagnostics for when off-policy evaluation can be trusted.

IPS-family estimators are only reliable under conditions on the propensities and
the importance weights. This module provides the standard checks an
experimentation DS runs before quoting an off-policy number:

  Overlap / positivity
    Every unit must have a non-trivial probability of each action. If estimated
    propensities pile up near 0 or 1, importance weights blow up and the
    estimate is dominated by a few units. We report the propensity range and the
    fraction of units in the extreme tails.

  Effective sample size (ESS)
    Skewed importance weights mean the effective number of samples backing the
    estimate is far below N:

        ESS = (sum w)^2 / sum(w^2)

    ESS / N near 1 is healthy; a small ratio warns that the estimate rests on a
    handful of high-weight units and has high variance.

  Weight clipping
    Capping weights at a percentile trades a little bias for a large variance
    reduction. We expose a helper to clip and report how much weight was capped.
"""

import numpy as np


def overlap_diagnostics(propensity, low: float = 0.05, high: float = 0.95) -> dict:
    """
    Positivity/overlap summary. Flags propensities in the extreme tails where
    importance weights become unstable.
    """
    e = np.asarray(propensity, dtype=float)
    frac_low = float(np.mean(e < low))
    frac_high = float(np.mean(e > high))
    return {
        "propensity_min": float(e.min()),
        "propensity_max": float(e.max()),
        "frac_below_low": frac_low,
        "frac_above_high": frac_high,
        "overlap_ok": bool(frac_low + frac_high < 0.01),
    }


def importance_weights(actions_pi, treatment, propensity) -> np.ndarray:
    """
    Importance weights for a deterministic policy: 1/P(logged action) where the
    logged action matches the policy's action, else 0.
    """
    a_pi = np.asarray(actions_pi)
    t = np.asarray(treatment)
    e = np.asarray(propensity, dtype=float)
    p_logged = np.where(t == 1, e, 1 - e)
    match = (a_pi == t).astype(float)
    return match / p_logged


def effective_sample_size(weights) -> float:
    """ESS = (sum w)^2 / sum(w^2). Returns the count (not the ratio)."""
    w = np.asarray(weights, dtype=float)
    denom = np.sum(w ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum(w) ** 2 / denom)


def ess_ratio(weights) -> float:
    """
    ESS / N: effective sample size as a fraction of the total number of units.
    1.0 means uniform weights (no variance inflation); values near 0 warn that
    the estimate rests on a few high-weight units.
    """
    w = np.asarray(weights, dtype=float)
    n = len(w)
    if n == 0:
        return 0.0
    return effective_sample_size(w) / n


def clip_weights(weights, percentile: float = 99.0) -> tuple:
    """
    Cap weights at the given percentile. Returns (clipped_weights, frac_capped).
    """
    w = np.asarray(weights, dtype=float)
    active = w[w > 0]
    if len(active) == 0:
        return w, 0.0
    cap = np.percentile(active, percentile)
    clipped = np.minimum(w, cap)
    frac_capped = float(np.mean(w > cap))
    return clipped, frac_capped


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sklearn.model_selection import train_test_split
    from src.synthetic_data import generate_uplift_data, FEATURES
    from src.meta_learners import TLearner
    from src.off_policy import _policy_actions

    # Randomised design: propensities are all 0.5 -> perfect overlap
    df = generate_uplift_data(n=40000, seed=42)
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=0)
    model = TLearner().fit(train_df[FEATURES], train_df["treatment"].values,
                           train_df["outcome"].values)
    score = model.predict_uplift(test_df[FEATURES])
    actions = _policy_actions(score, threshold=0.0)
    t = test_df["treatment"].values
    e = np.full(len(test_df), 0.5)

    print("=== Overlap (randomised design, e=0.5) ===")
    ov = overlap_diagnostics(e)
    for k, v in ov.items():
        print(f"  {k}: {v}")

    w = importance_weights(actions, t, e)
    print("\n=== Importance weights / ESS ===")
    n_active = int((w > 0).sum())
    print(f"  ESS:        {effective_sample_size(w):,.0f} "
          f"(of {n_active} units matching the policy action, {len(w)} total)")
    print(f"  ESS / N:    {ess_ratio(w):.3f}  "
          f"(a deterministic policy zeroes weights for ~half the log, so ~0.5 is")
    print(f"              expected here; among matched units the weights are uniform)")

    # Show what a confounded design does to overlap (propensity depends on x0)
    print("\n=== Confounded design (propensity varies) ===")
    df_c = generate_uplift_data(n=40000, seed=42, confounding=True)
    # Reconstruct the assignment propensity used by the generator
    x0 = df_c["x1"].values
    e_c = 1 / (1 + np.exp(-1.2 * x0))
    ov_c = overlap_diagnostics(e_c)
    for k, v in ov_c.items():
        print(f"  {k}: {v}")
    w_c = 1.0 / np.where(df_c["treatment"].values == 1, e_c, 1 - e_c)
    print(f"  ESS / N (confounded): {ess_ratio(w_c):.3f}  (all units active here)")
    clipped, frac = clip_weights(w_c, percentile=99)
    print(f"  after clipping at p99: {frac:.1%} of weights capped, "
          f"ESS/N -> {ess_ratio(clipped):.3f}")
