"""
Uplift evaluation: Qini curve, Qini coefficient, AUUC, and uplift@k.

Recovering the average treatment effect (ATE) is not enough for targeting — what
matters is whether the model *ranks* units so that treating the top-ranked ones
yields the most incremental conversions. These metrics measure that ranking.

Qini curve:
    Sort units by predicted uplift (descending). Walking down the ranked list,
    plot the cumulative *incremental* conversions gained versus the number of
    units targeted. Incremental gain at depth k is computed from the randomised
    experiment as:

        gain(k) = Y_treated(k) - Y_control(k) * (N_treated(k) / N_control(k))

    i.e. conversions among treated in the top-k, minus the control conversions
    scaled to the same exposure. A model that ranks persuadables first climbs
    steeply then flattens; random targeting gives the diagonal.

Qini coefficient:
    Area between the model's Qini curve and the random (diagonal) line,
    normalised. Higher is better; 0 means no better than random.

AUUC (Area Under the Uplift Curve):
    Area under the cumulative-gain curve. Closely related to Qini; reported
    alongside as it is the other common convention.

These let us compare S/T/X-learners on *targeting quality*, and — because the
data is semi-synthetic — against the best achievable ranking (true CATE).
"""

import numpy as np
import pandas as pd


def qini_curve(uplift_score, treatment, outcome):
    """
    Compute the Qini curve points.

    Returns (fractions, gains):
      fractions : fraction of population targeted (0..1)
      gains     : cumulative incremental conversions at each depth
    """
    order = np.argsort(-np.asarray(uplift_score))  # descending by predicted uplift
    t = np.asarray(treatment)[order]
    y = np.asarray(outcome)[order]

    # Cumulative counts down the ranked list
    cum_treated = np.cumsum(t)
    cum_control = np.cumsum(1 - t)
    cum_y_treated = np.cumsum(y * t)
    cum_y_control = np.cumsum(y * (1 - t))

    # Scale control conversions to treated exposure at each depth
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled_control = np.where(
            cum_control > 0, cum_y_control * (cum_treated / cum_control), 0.0
        )
    gains = cum_y_treated - scaled_control

    n = len(t)
    fractions = np.arange(1, n + 1) / n
    # Prepend the origin
    return np.concatenate([[0.0], fractions]), np.concatenate([[0.0], gains])


def qini_coefficient(uplift_score, treatment, outcome) -> float:
    """
    Qini coefficient: area between the model curve and the random diagonal,
    normalised by the total incremental gain. Higher is better; 0 ~ random.
    """
    fractions, gains = qini_curve(uplift_score, treatment, outcome)
    # Random baseline: straight line from origin to the final total gain
    total_gain = gains[-1]
    random_line = fractions * total_gain
    # Area between model and random, by trapezoidal integration over fractions
    area_between = np.trapezoid(gains - random_line, fractions)
    # Normalise by the area of the "perfect wedge" (0.5 * total_gain) if nonzero
    denom = 0.5 * abs(total_gain)
    return float(area_between / denom) if denom > 0 else 0.0


def auuc(uplift_score, treatment, outcome) -> float:
    """Area Under the Uplift Curve (cumulative gain), by trapezoidal rule."""
    fractions, gains = qini_curve(uplift_score, treatment, outcome)
    return float(np.trapezoid(gains, fractions))


def uplift_at_k(uplift_score, treatment, outcome, k: float = 0.3) -> float:
    """
    Incremental conversions captured by targeting the top-k fraction, expressed
    as a rate per targeted unit. k is a fraction of the population (e.g. 0.3).
    """
    order = np.argsort(-np.asarray(uplift_score))
    n = len(order)
    top = order[: max(1, int(k * n))]
    t = np.asarray(treatment)[top]
    y = np.asarray(outcome)[top]
    n_t, n_c = t.sum(), (1 - t).sum()
    if n_t == 0 or n_c == 0:
        return 0.0
    rate_treated = y[t == 1].mean()
    rate_control = y[t == 0].mean()
    return float(rate_treated - rate_control)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sklearn.model_selection import train_test_split
    from src.synthetic_data import generate_uplift_data, FEATURES
    from src.meta_learners import SLearner, TLearner, XLearner

    df = generate_uplift_data(n=40000, seed=42)

    # Uplift metrics must be computed out-of-sample: evaluating a learner on the
    # data it trained on lets it overfit the noise in realised outcomes and can
    # even score it above the true-CATE oracle. We split, fit on train, and score
    # every ranker on the held-out test set. (Same lesson as an out-of-time split
    # in a scoring model: measure on data the model has not seen.)
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=0)
    Xtr, ttr, ytr = train_df[FEATURES], train_df["treatment"].values, train_df["outcome"].values
    Xte, tte, yte = test_df[FEATURES], test_df["treatment"].values, test_df["outcome"].values

    print("Uplift ranking quality on held-out test set\n")
    print(f"{'Model':<14} {'Qini':>8} {'AUUC':>10} {'uplift@30%':>12}")
    print("-" * 46)

    rankers = [
        ("S-Learner", SLearner().fit(Xtr, ttr, ytr).predict_uplift(Xte)),
        ("T-Learner", TLearner().fit(Xtr, ttr, ytr).predict_uplift(Xte)),
        ("X-Learner", XLearner().fit(Xtr, ttr, ytr).predict_uplift(Xte)),
        ("Oracle(true)", test_df["true_cate"].values),
        ("Random", np.random.default_rng(0).random(len(test_df))),
    ]
    for name, score in rankers:
        q = qini_coefficient(score, tte, yte)
        a = auuc(score, tte, yte)
        u = uplift_at_k(score, tte, yte, k=0.3)
        print(f"{name:<14} {q:>8.3f} {a:>10.1f} {u:>12.4f}")

    print("\nThe true-CATE oracle sets the ceiling; X-Learner should approach it")
    print("and beat S/T; random sits near zero.")
