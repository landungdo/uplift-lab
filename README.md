# Uplift Lab — Causal Experimentation & Treatment Optimization

![tests](https://img.shields.io/badge/tests-36%20passed-brightgreen)

An end-to-end **causal inference / uplift modeling** project that answers a
different question from a standard predictive model. Instead of *"who is likely
to convert?"* it asks *"who converts **because** we treated them, and whom
should we treat to maximize incremental profit under a budget?"*

> The whole pipeline is validated against **known ground-truth treatment
> effects** using a semi-synthetic data generator — because on real data the
> individual treatment effect is never observed, so an uplift estimator cannot
> be checked directly. This project makes that check possible and central.

## Why this is not just another classifier

A predictive model that targets "likely converters" wastes budget on **sure
things** (who convert anyway) and can even hurt by treating **sleeping dogs**
(who convert *unless* treated). Uplift modeling isolates the **persuadables** —
the only segment worth treating. This project builds the estimators, the honest
evaluation, and the budget-constrained decision layer around that idea.

## What's inside

| Component | Module | What it does |
|---|---|---|
| Semi-synthetic data | `src/synthetic_data.py` | Generates data with **known CATE** and the four uplift segments (persuadable / sure-thing / lost-cause / sleeping-dog) |
| Meta-learners | `src/meta_learners.py` | S-Learner, T-Learner, X-Learner for CATE estimation |
| Uplift evaluation | `src/evaluation.py` | Qini curve/coefficient, AUUC, uplift@k — **out-of-sample** |
| Targeting policy | `src/policy.py` | Budget-constrained top-k targeting + profit optimization |
| Experiment design | `src/experiment_design.py` | Power/MDE, sample size, SRM check, A/A calibration |
| Off-policy evaluation | `src/off_policy.py` | IPS, SNIPS, Doubly Robust — value a new policy from logged data |
| Bootstrap CIs | `src/bootstrap.py` | Percentile confidence intervals for ATE, Qini, AUUC |
| Reproduce | `scripts/reproduce.py` | One command regenerates every table into `results/` |

## Headline results (semi-synthetic, out-of-sample)

**Meta-learners recover the ground-truth ATE** (true ≈ +0.024) and rank CATE
progressively better — X-Learner best, as theory predicts:

| Model | corr with true CATE | Qini / Oracle | uplift@30% |
|---|---|---|---|
| S-Learner | 0.63 | 0.65 | 0.146 |
| T-Learner | 0.61 | 0.66 | 0.137 |
| X-Learner | **0.67** | **0.71** | **0.151** |
| Oracle (true CATE) | 1.00 | 1.00 | 0.194 |
| Random | 0.00 | -0.07 | 0.010 |

*Qini is reported normalised by the oracle's Qini area (0 ≈ random, 1 ≈ true
CATE), so it stays in [0, 1]. Metrics are computed out-of-sample; the ATE is
reported with a bootstrap 95% CI, e.g. ATE +0.023 (95% CI +0.008 to +0.039).*

**Targeting beats treating everyone.** Treating the top 30% by uplift yields
*more* incremental conversions than treating the entire population, because
treating everyone also hits sleeping dogs. The profit-optimal budget is selected
**on a validation split and frozen before scoring on test** (to avoid optimism
bias), and lands at an interior point (~top 15%), not "treat everyone".

**Off-policy evaluation** estimates a new policy's value from logged data without
deploying it; all three estimators track the true value, Doubly Robust closest:

| Estimator | error vs true policy value |
|---|---|
| IPS | 0.0092 |
| SNIPS | 0.0082 |
| Doubly Robust | **0.0079** |

## Two methodological findings worth calling out

- **In-sample evaluation can beat the oracle.** Scoring uplift on the training
  data let a learner appear to out-rank the true CATE — impossible in reality —
  because it overfits noise in realized outcomes. All metrics here are computed
  out-of-sample. (Same lesson as an out-of-time split in a scoring model.)
- **The naive difference-in-means recovers the ATE only at scale.** The
  generator is validated by confirming naive diff-in-means converges to the true
  ATE as n grows, and that confounded assignment breaks that balance.

## Running it

```bash
pip install -r requirements.txt
python src/synthetic_data.py      # inspect the generated data + segments
python src/meta_learners.py       # S/T/X-learners vs ground-truth CATE
python src/evaluation.py          # Qini/AUUC/uplift@k, out-of-sample
python src/policy.py              # budget-constrained targeting + profit curve
python src/experiment_design.py   # power/MDE, SRM, A/A calibration
python src/off_policy.py          # IPS / SNIPS / Doubly Robust
python src/bootstrap.py           # bootstrap confidence intervals
python scripts/reproduce.py       # regenerate all result tables into results/
pytest tests/ -v                  # full test suite (36 tests)
```

## Limitations & scope

- Results are on **semi-synthetic** data by design, so estimators can be checked
  against ground truth. Applying the same pipeline to the real **Criteo Uplift**
  benchmark is the natural next step (no ground-truth CATE there, so the
  semi-synthetic validation is what justifies trusting the estimators).
- Propensities are known (randomized design); observational data would need a
  fitted propensity model, which the off-policy code is structured to accept.
- Profit figures use illustrative unit economics (value per conversion, cost per
  treatment), not a calibrated business P&L.
