"""
Reproduce the full uplift pipeline end to end and write results to disk.

Running this script regenerates every headline number in the README from a fixed
seed, and saves them as machine-readable artifacts:

  results/metrics.json          - ATE, per-model Qini/AUUC/uplift@k, CIs
  results/model_comparison.csv  - the model comparison table
  results/profit_curve.csv      - the targeting profit curve

This makes the project reproducible: a reviewer runs one command and gets the
same tables the README cites.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.synthetic_data import generate_uplift_data, FEATURES, true_ate
from src.meta_learners import SLearner, TLearner, XLearner
from src.evaluation import qini_coefficient, qini_normalized, auuc, uplift_at_k
from src.policy import profit_curve, optimal_budget
from src.bootstrap import ate_ci, metric_ci

SEED = 42
OUTDIR = Path("results")


def main():
    OUTDIR.mkdir(exist_ok=True)

    df = generate_uplift_data(n=60000, seed=SEED)
    train_df, rest = train_test_split(df, test_size=0.5, random_state=0)
    valid_df, test_df = train_test_split(rest, test_size=0.5, random_state=0)

    Xtr = train_df[FEATURES]
    ttr = train_df["treatment"].values
    ytr = train_df["outcome"].values
    Xte = test_df[FEATURES]
    tte = test_df["treatment"].values
    yte = test_df["outcome"].values
    oracle = test_df["true_cate"].values

    # Model comparison on held-out test
    learners = {
        "S-Learner": SLearner(),
        "T-Learner": TLearner(),
        "X-Learner": XLearner(),
    }
    rows = []
    scores = {}
    for name, m in learners.items():
        m.fit(Xtr, ttr, ytr)
        s = m.predict_uplift(Xte)
        scores[name] = s
        rows.append({
            "model": name,
            "est_ate": float(s.mean()),
            "corr_true_cate": float(np.corrcoef(s, oracle)[0, 1]),
            "qini_area": qini_coefficient(s, tte, yte),
            "qini_norm": qini_normalized(s, tte, yte, oracle),
            "auuc": auuc(s, tte, yte),
            "uplift_at_30": uplift_at_k(s, tte, yte, 0.3),
        })
    # Oracle and random references
    for name, s in [("Oracle", oracle),
                    ("Random", np.random.default_rng(0).random(len(test_df)))]:
        rows.append({
            "model": name,
            "est_ate": float(s.mean()) if name != "Oracle" else float(oracle.mean()),
            "corr_true_cate": 1.0 if name == "Oracle" else 0.0,
            "qini_area": qini_coefficient(s, tte, yte),
            "qini_norm": qini_normalized(s, tte, yte, oracle),
            "auuc": auuc(s, tte, yte),
            "uplift_at_30": uplift_at_k(s, tte, yte, 0.3),
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTDIR / "model_comparison.csv", index=False)

    # Profit curve + budget selected on validation, evaluated on test
    xl = learners["X-Learner"]
    val_score = xl.predict_uplift(valid_df[FEATURES])
    val_curve = profit_curve(val_score, valid_df["treatment"].values,
                             valid_df["outcome"].values, 100, 10)
    chosen = optimal_budget(val_curve)
    test_curve = profit_curve(scores["X-Learner"], tte, yte, 100, 10)
    test_curve.to_csv(OUTDIR / "profit_curve.csv", index=False)

    # Confidence intervals
    ate, ate_lo, ate_hi = ate_ci(tte, yte, n_bootstrap=400)
    q, q_lo, q_hi = metric_ci(scores["X-Learner"], tte, yte, qini_coefficient,
                              n_bootstrap=300)

    metrics = {
        "seed": SEED,
        "true_ate_test": float(oracle.mean()),
        "ate_estimate": ate,
        "ate_95ci": [ate_lo, ate_hi],
        "xlearner_qini_area": q,
        "xlearner_qini_95ci": [q_lo, q_hi],
        "budget_selected_on_valid": float(chosen["budget_frac"]),
        "profit_at_selected_budget_test": float(
            test_curve.loc[
                (test_curve["budget_frac"] - chosen["budget_frac"]).abs().idxmin(),
                "profit"]
        ),
    }
    with open(OUTDIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Wrote:")
    print(f"  {OUTDIR/'model_comparison.csv'}")
    print(f"  {OUTDIR/'profit_curve.csv'}")
    print(f"  {OUTDIR/'metrics.json'}")
    print()
    print(comparison.to_string(index=False))
    print()
    print(f"ATE {ate:+.4f} (95% CI {ate_lo:+.4f} to {ate_hi:+.4f}); "
          f"budget {chosen['budget_frac']:.0%} selected on validation")


if __name__ == "__main__":
    main()
