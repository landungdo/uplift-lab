"""
CUPED: variance reduction for experiment analysis.

CUPED (Controlled-experiment Using Pre-Experiment Data) reduces the variance of
an A/B metric by regressing out a pre-experiment covariate that is correlated
with the outcome but, crucially, unaffected by the treatment (e.g. a user's
pre-period activity). The adjusted metric has the same expectation (so the ATE
is unbiased) but lower variance, which tightens the confidence interval and
raises power — like getting a larger sample for free.

Mechanics. Given outcome Y and a pre-experiment covariate X (mean-centred),

    theta = Cov(Y, X) / Var(X)
    Y_cuped = Y - theta * (X - mean(X))

Because X is independent of treatment assignment, subtracting theta*X does not
change the treatment-control difference in expectation, but removes the part of
Y's variance explained by X. The variance reduction is approximately corr(Y, X)^2.

This module estimates theta on the pooled data and reports the ATE and its
standard error before and after CUPED.
"""

import numpy as np


def compute_theta(outcome, covariate) -> float:
    """theta = Cov(Y, X) / Var(X), estimated on the pooled sample."""
    y = np.asarray(outcome, dtype=float)
    x = np.asarray(covariate, dtype=float)
    var_x = np.var(x)
    if var_x == 0:
        return 0.0
    return float(np.cov(y, x, bias=True)[0, 1] / var_x)


def apply_cuped(outcome, covariate, theta: float = None):
    """Return the CUPED-adjusted outcome (same expectation, lower variance)."""
    y = np.asarray(outcome, dtype=float)
    x = np.asarray(covariate, dtype=float)
    if theta is None:
        theta = compute_theta(y, x)
    return y - theta * (x - x.mean())


def _ate_and_se(treatment, outcome):
    """Difference in means and its standard error."""
    t = np.asarray(treatment)
    y = np.asarray(outcome, dtype=float)
    yt, yc = y[t == 1], y[t == 0]
    ate = yt.mean() - yc.mean()
    se = np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
    return float(ate), float(se)


def cuped_ate(treatment, outcome, covariate) -> dict:
    """
    Estimate the ATE with and without CUPED adjustment, reporting the standard
    error reduction.
    """
    ate_raw, se_raw = _ate_and_se(treatment, outcome)
    y_adj = apply_cuped(outcome, covariate)
    ate_cuped, se_cuped = _ate_and_se(treatment, y_adj)
    return {
        "ate_raw": ate_raw,
        "se_raw": se_raw,
        "ate_cuped": ate_cuped,
        "se_cuped": se_cuped,
        "se_reduction": float(1 - se_cuped / se_raw) if se_raw > 0 else 0.0,
        "theta": compute_theta(outcome, covariate),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.synthetic_data import generate_uplift_data, FEATURES

    df = generate_uplift_data(n=40000, seed=42)

    # A realistic pre-experiment covariate: correlated with the outcome but
    # independent of treatment. A real pre-period metric (e.g. prior conversions)
    # is typically well correlated with the outcome; we emulate that with the
    # baseline conversion probability plus noise, and confirm it is balanced
    # across arms (as a genuine pre-period metric would be).
    rng = np.random.default_rng(0)
    pre_covariate = df["p_control"].values + rng.normal(0, 0.15, len(df))

    t = df["treatment"].values
    y = df["outcome"].values

    corr = np.corrcoef(y, pre_covariate)[0, 1]
    bal = abs(pre_covariate[t == 1].mean() - pre_covariate[t == 0].mean())
    print(f"Pre-experiment covariate: corr with outcome = {corr:.3f}, "
          f"arm imbalance = {bal:.4f} (should be ~0)\n")

    res = cuped_ate(t, y, pre_covariate)
    print(f"ATE (raw):   {res['ate_raw']:+.4f}  (SE {res['se_raw']:.4f})")
    print(f"ATE (CUPED): {res['ate_cuped']:+.4f}  (SE {res['se_cuped']:.4f})")
    print(f"\nStandard-error reduction: {res['se_reduction']:.1%}")
    print("Same ATE, smaller SE -> tighter CI and more power, for free.")
