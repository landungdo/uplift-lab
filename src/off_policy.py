"""
Off-policy evaluation (OPE): estimate the value of a new targeting policy from
logged data, without deploying it.

Setup: we logged a randomised experiment under some behaviour policy (here,
random assignment with known propensity). We now have a *new* policy pi that
decides whom to treat (e.g. "treat if predicted uplift > threshold"). We want
the expected outcome if we had followed pi, using only the logged data.

Three estimators, in increasing sophistication:

  IPS (Inverse Propensity Scoring)
    Re-weight each logged unit by 1/propensity when the logged action matches
    what pi would do. Unbiased when propensities are known, but high variance
    when propensities are small.

  SNIPS (Self-Normalised IPS)
    Divide the IPS sum by the sum of weights instead of N. Slightly biased but
    much lower variance — usually preferred in practice.

  Doubly Robust (DR)
    Combine an outcome model with IPS correction. Consistent if *either* the
    propensity model or the outcome model is correct — hence "doubly robust".
    Lowest variance of the three when the outcome model is decent.

Because the data is semi-synthetic we can compare each estimate to the policy's
*true* value computed from the known potential outcomes, showing DR is closest.
"""

import numpy as np


def _policy_actions(uplift_score, threshold: float) -> np.ndarray:
    """Deterministic policy: treat (1) if predicted uplift exceeds threshold."""
    return (np.asarray(uplift_score) > threshold).astype(int)


def ips_value(actions_pi, treatment, outcome, propensity) -> float:
    """
    IPS estimate of E[outcome] under policy pi.

    For a deterministic policy, a logged unit contributes only when its logged
    treatment equals pi's action; the weight is 1/P(logged action).
    """
    a_pi = np.asarray(actions_pi)
    t = np.asarray(treatment)
    y = np.asarray(outcome)
    e = np.asarray(propensity)

    # Probability the behaviour policy took the logged action
    p_logged = np.where(t == 1, e, 1 - e)
    match = (a_pi == t).astype(float)
    weights = match / p_logged
    return float(np.mean(weights * y))


def snips_value(actions_pi, treatment, outcome, propensity) -> float:
    """Self-normalised IPS: divide by the sum of weights rather than N."""
    a_pi = np.asarray(actions_pi)
    t = np.asarray(treatment)
    y = np.asarray(outcome)
    e = np.asarray(propensity)

    p_logged = np.where(t == 1, e, 1 - e)
    match = (a_pi == t).astype(float)
    weights = match / p_logged
    if weights.sum() == 0:
        return 0.0
    return float(np.sum(weights * y) / np.sum(weights))


def dr_value(actions_pi, treatment, outcome, propensity,
             mu_treated_pred, mu_control_pred) -> float:
    """
    Doubly Robust estimate.

    mu_treated_pred / mu_control_pred are the outcome-model predictions for each
    unit under treatment / control. The DR estimator adds an IPS correction on
    the residual of the outcome model for the units where the logged action
    matches pi.
    """
    a_pi = np.asarray(actions_pi)
    t = np.asarray(treatment)
    y = np.asarray(outcome)
    e = np.asarray(propensity)
    mu1 = np.asarray(mu_treated_pred)
    mu0 = np.asarray(mu_control_pred)

    # Model's predicted outcome under the policy's chosen action
    mu_pi = np.where(a_pi == 1, mu1, mu0)

    # IPS correction only where logged action matches the policy action
    p_logged = np.where(t == 1, e, 1 - e)
    match = (a_pi == t).astype(float)
    correction = match / p_logged * (y - np.where(t == 1, mu1, mu0))
    return float(np.mean(mu_pi + correction))


def true_policy_value(actions_pi, p_treated, p_control) -> float:
    """
    Exact expected outcome under pi from known potential outcomes
    (semi-synthetic only): use p_treated where pi treats, p_control otherwise.
    """
    a_pi = np.asarray(actions_pi)
    return float(np.mean(np.where(a_pi == 1, p_treated, p_control)))


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sklearn.model_selection import train_test_split
    from src.synthetic_data import generate_uplift_data, FEATURES
    from src.meta_learners import TLearner

    df = generate_uplift_data(n=40000, seed=42)
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=0)

    # Fit an uplift model + outcome models on train
    tlearn = TLearner().fit(train_df[FEATURES], train_df["treatment"].values,
                            train_df["outcome"].values)
    score = tlearn.predict_uplift(test_df[FEATURES])

    # Outcome-model predictions on test (reuse the T-learner's two models)
    mu1 = tlearn.model_treated.predict_proba(test_df[FEATURES])[:, 1]
    mu0 = tlearn.model_control.predict_proba(test_df[FEATURES])[:, 1]

    t = test_df["treatment"].values
    y = test_df["outcome"].values
    # Known randomised propensity (share treated in the design)
    # Known design propensity of the randomised experiment (0.5 by construction).
    # Using the exact design probability — not the observed sample share — is the
    # correct choice: the propensity is a property of the assignment mechanism,
    # and in a real log it would be stored per unit rather than estimated.
    DESIGN_PROPENSITY = 0.5
    e = np.full(len(test_df), DESIGN_PROPENSITY)

    # New policy: treat users with predicted uplift above 0
    actions = _policy_actions(score, threshold=0.0)

    true_val = true_policy_value(actions, test_df["p_treated"].values,
                                 test_df["p_control"].values)
    ips = ips_value(actions, t, y, e)
    snips = snips_value(actions, t, y, e)
    dr = dr_value(actions, t, y, e, mu1, mu0)

    print("Off-policy evaluation of 'treat if predicted uplift > 0'\n")
    print(f"  True policy value (oracle): {true_val:.4f}")
    print(f"  IPS estimate:               {ips:.4f}   (err {abs(ips-true_val):.4f})")
    print(f"  SNIPS estimate:             {snips:.4f}   (err {abs(snips-true_val):.4f})")
    print(f"  Doubly Robust estimate:     {dr:.4f}   (err {abs(dr-true_val):.4f})")
    print()
    print("DR typically has the lowest error: it corrects the outcome model with")
    print("IPS, so it is accurate if either component is good.")
