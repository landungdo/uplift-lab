"""
Semi-synthetic data generator for uplift / causal inference.

The fundamental problem of causal inference is that for any individual we only
ever observe one outcome — treated or not, never both — so the true individual
treatment effect is unobservable in real data. That makes it impossible to check
directly whether an uplift estimator is right on a dataset like Criteo.

This generator sidesteps that: it *defines* each unit's treatment effect, then
generates outcomes consistent with it. Because we know the ground-truth
conditional average treatment effect (CATE) per unit, we can validate that an
estimator recovers it (e.g. via the true Qini/AUUC, or error against true CATE).

It also plants the four canonical uplift segments so policy evaluation has
something meaningful to find:

  - Persuadables   : buy only IF treated            (positive uplift)  <- the target
  - Sure things    : buy whether treated or not     (zero uplift)
  - Lost causes    : never buy                      (zero uplift)
  - Sleeping dogs  : buy UNLESS treated             (negative uplift)  <- avoid treating

Design choices:
  - Randomised treatment assignment (a clean A/B experiment) by default, so
    naive estimators are unbiased and the setup is easy to reason about. A
    `confounding` switch biases assignment on covariates, to demonstrate why
    naive difference-in-means then fails and adjustment is needed.
"""

import numpy as np
import pandas as pd

FEATURES = [f"x{i}" for i in range(1, 9)]  # 8 covariates


def _segment_from_latent(persuade_score, sleeping_score, rng):
    """Assign each unit to one of the four uplift segments from latent scores."""
    n = len(persuade_score)
    seg = np.empty(n, dtype=object)
    # Thresholds chosen to give a realistic mix (most people are not persuadable)
    for i in range(n):
        if sleeping_score[i] > 1.2:
            seg[i] = "sleeping_dog"
        elif persuade_score[i] > 0.8:
            seg[i] = "persuadable"
        elif persuade_score[i] > -0.3:
            seg[i] = "sure_thing"
        else:
            seg[i] = "lost_cause"
    return seg


def generate_uplift_data(n: int = 20000, seed: int = 42,
                         treatment_share: float = 0.5,
                         confounding: bool = False) -> pd.DataFrame:
    """
    Generate a semi-synthetic uplift dataset.

    Returns a DataFrame with:
      x1..x8        : covariates
      segment       : ground-truth uplift segment (for analysis, not a feature)
      treatment     : 0/1 assignment
      p_control     : true P(convert | not treated)
      p_treated     : true P(convert | treated)
      true_cate     : p_treated - p_control  (ground-truth individual uplift)
      outcome       : realised 0/1 conversion under the assigned treatment
    """
    rng = np.random.default_rng(seed)

    X = rng.normal(0, 1, size=(n, len(FEATURES)))
    df = pd.DataFrame(X, columns=FEATURES)

    # Latent scores that drive segment membership from covariates
    persuade_score = 0.9 * X[:, 0] - 0.5 * X[:, 1] + 0.4 * X[:, 2] + rng.normal(0, 0.5, n)
    sleeping_score = 0.8 * X[:, 3] - 0.6 * X[:, 4] + rng.normal(0, 0.5, n)
    segment = _segment_from_latent(persuade_score, sleeping_score, rng)
    df["segment"] = segment

    # Baseline conversion (untreated) depends on covariates
    base_logit = -0.5 + 0.6 * X[:, 0] + 0.3 * X[:, 5] - 0.4 * X[:, 6]
    p_control = 1 / (1 + np.exp(-base_logit))

    # Segment-specific treatment effect on the probability scale
    uplift = np.zeros(n)
    uplift[segment == "persuadable"] = 0.25
    uplift[segment == "sure_thing"] = 0.0
    uplift[segment == "lost_cause"] = 0.0
    uplift[segment == "sleeping_dog"] = -0.20

    # Add mild heterogeneity within persuadables so CATE isn't a step function
    persuadable_mask = segment == "persuadable"
    uplift[persuadable_mask] += 0.10 * (X[persuadable_mask, 0] - X[persuadable_mask, 0].mean())

    p_treated = np.clip(p_control + uplift, 0.01, 0.99)
    p_control = np.clip(p_control, 0.01, 0.99)

    df["p_control"] = p_control
    df["p_treated"] = p_treated
    df["true_cate"] = p_treated - p_control

    # Treatment assignment
    if confounding:
        # Bias assignment toward high-x0 units, so treated/control differ on x0
        assign_logit = 1.2 * X[:, 0]
        p_assign = 1 / (1 + np.exp(-assign_logit))
        treatment = (rng.random(n) < p_assign).astype(int)
    else:
        treatment = (rng.random(n) < treatment_share).astype(int)
    df["treatment"] = treatment

    # Realised outcome under assigned arm
    p_realized = np.where(treatment == 1, p_treated, p_control)
    df["outcome"] = (rng.random(n) < p_realized).astype(int)

    return df


def true_ate(df: pd.DataFrame) -> float:
    """Ground-truth average treatment effect (mean of true individual CATE)."""
    return float(df["true_cate"].mean())


if __name__ == "__main__":
    df = generate_uplift_data(n=20000, seed=42)
    print("Semi-synthetic uplift dataset")
    print(f"  rows: {len(df)}")
    print(f"  treatment share: {df['treatment'].mean():.1%}")
    print(f"  overall conversion: {df['outcome'].mean():.1%}")
    print()
    print("Segment mix (ground truth):")
    print(df["segment"].value_counts().to_string())
    print()
    print(f"True ATE (mean CATE): {true_ate(df):+.4f}")
    print()
    print("Naive difference in means (treated - control conversion):")
    naive = (df.loc[df.treatment == 1, "outcome"].mean()
             - df.loc[df.treatment == 0, "outcome"].mean())
    print(f"  {naive:+.4f}   (should be close to true ATE under randomisation)")
    print()
    print("Mean true uplift by segment (sanity check):")
    print(df.groupby("segment")["true_cate"].mean().to_string())
