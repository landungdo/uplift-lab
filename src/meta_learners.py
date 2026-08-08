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
    from sklearn.model_selection import train_test_split
    from src.synthetic_data import generate_uplift_data, FEATURES, true_ate

    df = generate_uplift_data(n=30000, seed=42)

    # Fit on train, evaluate CATE recovery on a held-out test set. Evaluating on
    # the training data would let a learner overfit the noise in realized
    # outcomes and overstate how well it recovers the true CATE.
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=0)
    Xtr, ttr, ytr = train_df[FEATURES], train_df["treatment"].values, train_df["outcome"].values
    Xte = test_df[FEATURES]
    true_cate_te = test_df["true_cate"].values

    print(f"Ground truth ATE (test): {test_df['true_cate'].mean():+.4f}\n")

    for name, learner in [("S-Learner", SLearner()),
                          ("T-Learner", TLearner()),
                          ("X-Learner", XLearner())]:
        learner.fit(Xtr, ttr, ytr)
        pred = learner.predict_uplift(Xte)          # evaluated out-of-sample
        est_ate = pred.mean()
        cate_mae = np.mean(np.abs(pred - true_cate_te))
        corr = np.corrcoef(pred, true_cate_te)[0, 1]
        print(f"{name}")
        print(f"  estimated ATE (test): {est_ate:+.4f}  (true {true_cate_te.mean():+.4f})")
        print(f"  CATE MAE vs truth:    {cate_mae:.4f}")
        print(f"  corr(pred, true CATE): {corr:.3f}")
        print()


class XLearner:
    """
    X-Learner (Kunzel et al. 2019). Improves on the T-Learner in three stages:

      1. Fit outcome models mu_treated, mu_control (like the T-Learner).
      2. Impute each unit's treatment effect using the *other* arm's model:
           treated units:  D1 = Y - mu_control(X)
           control units:  D0 = mu_treated(X) - Y
         then fit effect models tau_treated, tau_control on D1, D0.
      3. Combine with a propensity weight g(x):
           tau(x) = g(x) * tau_control(x) + (1 - g(x)) * tau_treated(x)

    The imputation step lets each arm's effect model borrow strength from the
    other arm, which helps when the treated and control groups are unbalanced —
    the case where the T-Learner's two independent models struggle most.
    """

    def __init__(self, base_model=None, propensity: float = 0.5):
        base = base_model if base_model is not None else _default_model()
        self.mu_treated = clone(base)
        self.mu_control = clone(base)
        self.tau_treated = clone(base)
        self.tau_control = clone(base)
        # Constant propensity is correct for a randomised experiment; a model
        # could be substituted under observational data.
        self.propensity = propensity

    def fit(self, X, treatment, y):
        X = X.reset_index(drop=True)
        treatment = np.asarray(treatment)
        y = np.asarray(y)

        Xt, yt = X[treatment == 1], y[treatment == 1]
        Xc, yc = X[treatment == 0], y[treatment == 0]

        # Stage 1: outcome models
        self.mu_treated.fit(Xt, yt)
        self.mu_control.fit(Xc, yc)

        # Stage 2: imputed effects, then effect models
        # For treated units: actual outcome minus predicted control outcome
        d1 = yt - self.mu_control.predict_proba(Xt)[:, 1]
        # For control units: predicted treated outcome minus actual outcome
        d0 = self.mu_treated.predict_proba(Xc)[:, 1] - yc

        # Effect models are regressors on the imputed effects
        from sklearn.ensemble import GradientBoostingRegressor
        self.tau_treated = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        self.tau_control = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        self.tau_treated.fit(Xt, d1)
        self.tau_control.fit(Xc, d0)
        return self

    def predict_uplift(self, X):
        g = self.propensity
        tau_t = self.tau_treated.predict(X)
        tau_c = self.tau_control.predict(X)
        return g * tau_c + (1 - g) * tau_t
