"""
Bootstrap confidence intervals for uplift metrics.

A point estimate like "ATE = +0.024" or "Qini = 234" is incomplete without a
sense of its uncertainty. This module computes percentile bootstrap confidence
intervals by resampling the evaluation set with replacement and recomputing the
metric each time. It lets results be reported as "0.024 (95% CI 0.018-0.030)"
rather than a bare number — the difference between a claim and a defensible one.

The bootstrap is metric-agnostic: pass any function that maps a resampled frame
(or arrays) to a scalar.
"""

import numpy as np


def bootstrap_ci(statistic_fn, point_estimate: float = None,
                 n_bootstrap: int = 500, alpha: float = 0.05, seed: int = 0):
    """
    Generic percentile bootstrap.

    statistic_fn : callable(rng) -> float
        Given a numpy Generator, draws one resample and returns the statistic.
    point_estimate : float, optional
        The statistic on the ORIGINAL sample. This is the correct point estimate
        to report; the bootstrap replicates are used only to form the interval.
        If None, the mean of the replicates is used as a fallback.

    Returns (point, lo, hi).
    """
    rng = np.random.default_rng(seed)
    estimates = np.array([statistic_fn(rng) for _ in range(n_bootstrap)])
    lo = float(np.percentile(estimates, 100 * alpha / 2))
    hi = float(np.percentile(estimates, 100 * (1 - alpha / 2)))
    point = float(point_estimate) if point_estimate is not None else float(estimates.mean())
    return point, lo, hi


def ate_ci(treatment, outcome, n_bootstrap: int = 500, alpha: float = 0.05,
           seed: int = 0):
    """Bootstrap CI for the ATE via difference in means on resampled rows."""
    t = np.asarray(treatment)
    y = np.asarray(outcome)
    n = len(t)

    def stat(rng):
        idx = rng.integers(0, n, n)
        tt, yy = t[idx], y[idx]
        if tt.sum() == 0 or (1 - tt).sum() == 0:
            return 0.0
        return yy[tt == 1].mean() - yy[tt == 0].mean()

    # Point estimate on the ORIGINAL sample (not the bootstrap mean)
    point = (y[t == 1].mean() - y[t == 0].mean()
             if t.sum() > 0 and (1 - t).sum() > 0 else 0.0)
    return bootstrap_ci(stat, point_estimate=point,
                        n_bootstrap=n_bootstrap, alpha=alpha, seed=seed)


def metric_ci(score, treatment, outcome, metric_fn,
              n_bootstrap: int = 300, alpha: float = 0.05, seed: int = 0):
    """
    Bootstrap CI for any ranking metric of the form metric_fn(score, t, y)
    (e.g. qini_coefficient, auuc), resampling rows with replacement.
    """
    s = np.asarray(score)
    t = np.asarray(treatment)
    y = np.asarray(outcome)
    n = len(s)

    def stat(rng):
        idx = rng.integers(0, n, n)
        return metric_fn(s[idx], t[idx], y[idx])

    # Point estimate on the ORIGINAL sample
    point = metric_fn(s, t, y)
    return bootstrap_ci(stat, point_estimate=point,
                        n_bootstrap=n_bootstrap, alpha=alpha, seed=seed)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sklearn.model_selection import train_test_split
    from src.synthetic_data import generate_uplift_data, FEATURES
    from src.meta_learners import XLearner
    from src.evaluation import qini_coefficient, auuc

    df = generate_uplift_data(n=40000, seed=42)
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=0)
    t = test_df["treatment"].values
    y = test_df["outcome"].values

    print("Bootstrap 95% confidence intervals (test set)\n")

    ate, lo, hi = ate_ci(t, y)
    print(f"  ATE (naive diff-in-means): {ate:+.4f}  (95% CI {lo:+.4f} to {hi:+.4f})")

    model = XLearner().fit(train_df[FEATURES], train_df["treatment"].values,
                           train_df["outcome"].values)
    score = model.predict_uplift(test_df[FEATURES])

    q, qlo, qhi = metric_ci(score, t, y, qini_coefficient)
    print(f"  Qini area (X-Learner):     {q:.1f}  (95% CI {qlo:.1f} to {qhi:.1f})")

    a, alo, ahi = metric_ci(score, t, y, auuc)
    print(f"  AUUC (X-Learner):          {a:.1f}  (95% CI {alo:.1f} to {ahi:.1f})")
    print()
    print("Reporting intervals, not just point estimates, is what makes the")
    print("comparison between models and policies defensible.")
