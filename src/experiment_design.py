"""
Experiment design and health checks.

Before you can estimate uplift you need a trustworthy experiment. This module
covers the standard design and validation steps an experimentation DS owns:

  Power / MDE / sample size
    Given a baseline conversion rate and a target lift, how many users are
    needed per arm to detect the effect with adequate power? Or, for a fixed
    sample size, what is the Minimum Detectable Effect (MDE)?

  Sample Ratio Mismatch (SRM)
    If assignment is 50/50 by design but the observed split differs more than
    chance allows, the experiment is likely broken (logging bug, biased
    bucketing). A chi-square test on the observed counts flags this.

  A/A test
    Split the control group in two and run the analysis as if it were an A/B
    test. There is no real effect, so a correct pipeline should NOT find a
    significant difference more often than the false-positive rate. Repeated A/A
    tests should give roughly uniform p-values.

These are deliberately implemented from scipy primitives rather than a heavy
experimentation library, to show the underlying statistics.
"""

import numpy as np
from scipy import stats


def required_sample_size(baseline_rate: float, mde_abs: float,
                         alpha: float = 0.05, power: float = 0.8) -> int:
    """
    Sample size PER ARM to detect an absolute lift `mde_abs` on a binary
    conversion metric, using a two-proportion z-test approximation.

    baseline_rate : control conversion rate p0
    mde_abs       : absolute effect to detect (p1 - p0)
    """
    p0 = baseline_rate
    p1 = baseline_rate + mde_abs
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    pbar = (p0 + p1) / 2
    # Standard two-proportion sample size formula
    num = (z_alpha * np.sqrt(2 * pbar * (1 - pbar))
           + z_beta * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    denom = (p1 - p0) ** 2
    return int(np.ceil(num / denom))


def minimum_detectable_effect(baseline_rate: float, n_per_arm: int,
                              alpha: float = 0.05, power: float = 0.8) -> float:
    """
    Absolute MDE detectable with `n_per_arm` users per arm, inverting the
    two-proportion power formula (approximate, using baseline variance).
    """
    p0 = baseline_rate
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    se = np.sqrt(2 * p0 * (1 - p0) / n_per_arm)
    return float((z_alpha + z_beta) * se)


def srm_check(n_treated: int, n_control: int,
              expected_ratio: float = 0.5) -> dict:
    """
    Sample Ratio Mismatch test. Chi-square goodness-of-fit on observed arm
    counts against the expected split. A small p-value (< 0.001 by convention)
    signals a broken experiment.
    """
    total = n_treated + n_control
    expected_treated = total * expected_ratio
    expected_control = total * (1 - expected_ratio)
    chi2 = ((n_treated - expected_treated) ** 2 / expected_treated
            + (n_control - expected_control) ** 2 / expected_control)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "srm_detected": bool(p_value < 0.001),
        "observed_ratio": n_treated / total,
    }


def two_proportion_test(conv_a: int, n_a: int, conv_b: int, n_b: int) -> dict:
    """Two-proportion z-test; returns the effect and p-value."""
    p_a = conv_a / n_a
    p_b = conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return {"effect": 0.0, "p_value": 1.0}
    z = (p_b - p_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"effect": float(p_b - p_a), "z": float(z), "p_value": float(p_value)}


def aa_test_false_positive_rate(control_outcomes, n_trials: int = 500,
                                alpha: float = 0.05, seed: int = 0) -> float:
    """
    Run repeated A/A tests: randomly split the control group in two and test for
    a difference. Returns the fraction of trials that are "significant" — which
    should be close to `alpha` if the pipeline is calibrated.
    """
    y = np.asarray(control_outcomes)
    n = len(y)
    rng = np.random.default_rng(seed)
    false_positives = 0
    for _ in range(n_trials):
        perm = rng.permutation(n)
        half = n // 2
        a, b = y[perm[:half]], y[perm[half:2 * half]]
        res = two_proportion_test(int(a.sum()), len(a), int(b.sum()), len(b))
        if res["p_value"] < alpha:
            false_positives += 1
    return false_positives / n_trials


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.synthetic_data import generate_uplift_data

    print("=== Power / sample size ===")
    for mde in [0.01, 0.02, 0.05]:
        n = required_sample_size(baseline_rate=0.05, mde_abs=mde)
        print(f"  baseline 5%, detect +{mde:.0%} lift -> {n:,} users per arm")
    print()
    mde = minimum_detectable_effect(baseline_rate=0.05, n_per_arm=50000)
    print(f"  With 50,000/arm at baseline 5%: MDE = {mde:.4f} (~{mde*100:.2f} pts)")

    print("\n=== SRM check ===")
    clean = srm_check(100000, 100000)
    print(f"  50/50 split: p={clean['p_value']:.3f}, SRM detected={clean['srm_detected']}")
    broken = srm_check(105000, 95000)
    print(f"  105k/95k split: p={broken['p_value']:.2e}, SRM detected={broken['srm_detected']}")

    print("\n=== A/A test (pipeline calibration) ===")
    df = generate_uplift_data(n=40000, seed=1)
    control = df.loc[df["treatment"] == 0, "outcome"].values
    fpr = aa_test_false_positive_rate(control, n_trials=500, alpha=0.05)
    print(f"  A/A false-positive rate at alpha=0.05: {fpr:.3f} (should be ~0.05)")
