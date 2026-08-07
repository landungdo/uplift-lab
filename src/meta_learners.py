"""
Meta-learners for uplift / CATE estimation: S-Learner and T-Learner.

A meta-learner estimates the conditional average treatment effect (CATE) —
tau(x) = E[Y(1) - Y(0) | X = x] — by combining standard supervised regressors
in a particular way. These two are the foundational baselines:

  S-Learner (Single model):
    Train ONE model on all data with treatment as an extra feature.
    tau(x) = mu(x, treated=1) - mu(x, treated=0).
    Simple, but the single model can "wash out" a weak treatment signal because
    the treatment is just one feature among many.

  T-Learner (Two models):
    Train TWO models, one on the treated group and one on the control group.
    tau(x) = mu_treated(x) - mu_control(x).
    Captures treatment-specific structure, but each model sees only half the
    data and the two error patterns can add up in the difference.

Both return a per-unit uplift prediction, which downstream code ranks and
evaluates (Qini/AUUC) and turns into a targeting policy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier


def _default_model():
    # A reasonable default base learner; kept small for speed on samples.
    return GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )


class SLearner:
    """Single-model meta-learner with treatment as a feature."""

    def __init__(self, base_model=None):
        self.model = base_model if base_model is not None else _default_model()

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> "SLearner":
        X = X.reset_index(drop=True)
        Xt = X.copy()
        Xt["treatment"] = np.asarray(treatment)
        self.model.fit(Xt, y)
        self._columns = Xt.columns
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reset_index(drop=True)
        X1 = X.copy(); X1["treatment"] = 1
        X0 = X.copy(); X0["treatment"] = 0
        X1 = X1[self._columns]; X0 = X0[self._columns]
        p1 = self.model.predict_proba(X1)[:, 1]
        p0 = self.model.predict_proba(X0)[:, 1]
        return p1 - p0


class TLearner:
    """Two-model meta-learner: separate models for treated and control."""

    def __init__(self, base_model=None):
        base = base_model if base_model is not None else _default_model()
        self.model_treated = clone(base)
        self.model_control = clone(base)

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> "TLearner":
        X = X.reset_index(drop=True)
        treatment = np.asarray(treatment)
        y = np.asarray(y)
        self.model_treated.fit(X[treatment == 1], y[treatment == 1])
        self.model_control.fit(X[treatment == 0], y[treatment == 0])
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        p1 = self.model_treated.predict_proba(X)[:, 1]
        p0 = self.model_control.predict_proba(X)[:, 1]
        return p1 - p0


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.synthetic_data import generate_uplift_data, FEATURES, true_ate

    df = generate_uplift_data(n=30000, seed=42)
    X = df[FEATURES]
    t = df["treatment"].values
    y = df["outcome"].values
    true_cate = df["true_cate"].values

    print(f"Ground truth ATE: {true_ate(df):+.4f}\n")

    for name, learner in [("S-Learner", SLearner()), ("T-Learner", TLearner())]:
        learner.fit(X, t, y)
        pred = learner.predict_uplift(X)
        est_ate = pred.mean()
        # Error against the KNOWN individual treatment effects
        cate_mae = np.mean(np.abs(pred - true_cate))
        corr = np.corrcoef(pred, true_cate)[0, 1]
        print(f"{name}")
        print(f"  estimated ATE:        {est_ate:+.4f}  (true {true_ate(df):+.4f})")
        print(f"  CATE MAE vs truth:    {cate_mae:.4f}")
        print(f"  corr(pred, true CATE): {corr:.3f}")
        print()
