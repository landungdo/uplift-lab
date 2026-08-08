"""
Targeting policy and policy value under a budget.

An uplift score per user is only an input. The business decision is: given a
budget to treat at most a fraction of users (a campaign can't coupon everyone),
*whom* do we treat to maximise incremental conversions?

Policy: treat the top-k users by predicted uplift.

Evaluating a policy honestly is the same problem as uplift evaluation — we only
observe one arm per user. On a randomised test set we estimate the value of
"treat this set S" as the incremental conversion rate among S:

    value(S) = mean(outcome | treated, in S) - mean(outcome | control, in S)

multiplied by |S| to get total incremental conversions. Because the data is
semi-synthetic we can also compute the *true* policy value from the known CATE,
giving an exact benchmark:

    true_value(S) = sum(true_cate over users in S)

This module also finds the profit-optimal budget when treating has a cost:
each incremental conversion is worth `value_per_conversion`, each treatment
costs `cost_per_treatment`, so profit at budget k trades incremental revenue
against treatment spend — the same profit-curve idea as the credit policy
simulator, now for targeting.
"""

import numpy as np
import pandas as pd


def select_top_k(uplift_score, k_frac: float) -> np.ndarray:
    """Boolean mask selecting the top k fraction of users by predicted uplift."""
    n = len(uplift_score)
    k = max(1, int(round(k_frac * n)))
    order = np.argsort(-np.asarray(uplift_score))
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def policy_value_experimental(mask, treatment, outcome) -> float:
    """
    Estimate incremental conversions from treating the selected set, using the
    randomised experiment (difference in conversion rates within the set,
    scaled by set size).
    """
    treatment = np.asarray(treatment)
    outcome = np.asarray(outcome)
    sel = np.asarray(mask)
    t_in = sel & (treatment == 1)
    c_in = sel & (treatment == 0)
    if t_in.sum() == 0 or c_in.sum() == 0:
        return 0.0
    rate_t = outcome[t_in].mean()
    rate_c = outcome[c_in].mean()
    return float((rate_t - rate_c) * sel.sum())


def policy_value_true(mask, true_cate) -> float:
    """Exact incremental conversions from the known CATE (semi-synthetic only)."""
    return float(np.asarray(true_cate)[np.asarray(mask)].sum())


def profit_curve(uplift_score, treatment, outcome,
                 value_per_conversion: float = 100.0,
                 cost_per_treatment: float = 10.0,
                 grid=None) -> pd.DataFrame:
    """
    Profit as a function of targeting budget (fraction treated), using the
    experimental policy-value estimate for incremental conversions.

        profit(k) = value_per_conversion * incremental_conversions(k)
                    - cost_per_treatment * n_treated(k)
    """
    n = len(uplift_score)
    if grid is None:
        grid = np.round(np.arange(0.05, 1.01, 0.05), 2)

    rows = []
    for k in grid:
        mask = select_top_k(uplift_score, k)
        inc = policy_value_experimental(mask, treatment, outcome)
        n_treated = int(mask.sum())
        profit = value_per_conversion * inc - cost_per_treatment * n_treated
        rows.append({
            "budget_frac": k,
            "n_treated": n_treated,
            "incremental_conversions": inc,
            "profit": profit,
        })
    return pd.DataFrame(rows)


def optimal_budget(profit_df: pd.DataFrame) -> dict:
    """Budget row maximising profit."""
    return profit_df.loc[profit_df["profit"].idxmax()].to_dict()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sklearn.model_selection import train_test_split
    from src.synthetic_data import generate_uplift_data, FEATURES
    from src.meta_learners import XLearner

    df = generate_uplift_data(n=60000, seed=42)
    # Three-way split: train the model, SELECT the budget on validation, then
    # freeze it and report the final profit on the untouched test set. Selecting
    # and reporting the budget on the same set would bias the profit upward.
    train_df, rest = train_test_split(df, test_size=0.5, random_state=0)
    valid_df, test_df = train_test_split(rest, test_size=0.5, random_state=0)

    model = XLearner().fit(train_df[FEATURES], train_df["treatment"].values,
                           train_df["outcome"].values)

    def _scores(d):
        return model.predict_uplift(d[FEATURES])

    t = test_df["treatment"].values
    y = test_df["outcome"].values
    true_cate = test_df["true_cate"].values
    score = _scores(test_df)

    # Compare targeting the top 30% by model vs random vs treating everyone (test)
    print("Policy value: incremental conversions from treating a set (test set)\n")
    for name, sc in [("X-Learner top 30%", score),
                     ("Random 30%", np.random.default_rng(0).random(len(test_df)))]:
        mask = select_top_k(sc, 0.30)
        exp_val = policy_value_experimental(mask, t, y)
        true_val = policy_value_true(mask, true_cate)
        print(f"{name:<20} experimental={exp_val:8.1f}   true={true_val:8.1f}")
    all_mask = np.ones(len(test_df), dtype=bool)
    print(f"{'Treat everyone':<20} experimental={policy_value_experimental(all_mask,t,y):8.1f}"
          f"   true={policy_value_true(all_mask,true_cate):8.1f}")

    # SELECT the profit-optimal budget on VALIDATION
    val_score = _scores(valid_df)
    val_curve = profit_curve(val_score, valid_df["treatment"].values,
                             valid_df["outcome"].values,
                             value_per_conversion=100, cost_per_treatment=10)
    chosen = optimal_budget(val_curve)
    chosen_budget = chosen["budget_frac"]

    # FREEZE that budget and evaluate once on TEST
    test_curve = profit_curve(score, t, y,
                              value_per_conversion=100, cost_per_treatment=10,
                              grid=[chosen_budget])
    test_row = test_curve.iloc[0]

    print(f"\nBudget selected on validation: top {chosen_budget:.0%}")
    print(f"Frozen budget evaluated on test: top {chosen_budget:.0%}")
    print(f"  test profit:  ${test_row['profit']:,.0f}")
    print(f"  test n_treated: {int(test_row['n_treated'])}")
    print("\nSelecting the budget on validation and only then scoring it on test")
    print("removes the optimism bias of choosing and reporting on the same data.")
    print("Treating everyone is not optimal: extra treatments cost more than the")
    print("incremental conversions they buy, and sleeping dogs have negative uplift.")
