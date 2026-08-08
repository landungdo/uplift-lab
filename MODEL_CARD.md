# Model Card — Uplift Targeting System

A concise, honest description of what this uplift system does, the assumptions it
rests on, and where it should not be trusted. Modeled on the standard "model
card" format used for responsible ML documentation.

## Intended use

- **Task:** estimate the conditional average treatment effect (CATE) of a
  marketing treatment (e.g. a promotion) per user, and choose whom to treat
  under a budget to maximize incremental conversions / profit.
- **Intended users:** growth / product / experimentation data scientists deciding
  campaign targeting.
- **Out of scope:** anything requiring individual-level causal *guarantees*;
  high-stakes decisions about individuals (credit, employment, health). This is a
  population-targeting tool, not an individual adjudication tool.

## Data

- **Development data:** semi-synthetic, generated with a known ground-truth CATE
  and four canonical uplift segments (persuadable / sure-thing / lost-cause /
  sleeping-dog). This is deliberate: on real data the individual treatment effect
  is unobservable, so ground truth is needed to *validate* the estimators.
- **Assignment:** randomized (propensity = 0.5 by construction), i.e. a clean A/B
  experiment. The pipeline also supports a confounded-assignment mode to
  demonstrate failure of naive estimators.

## The five load-bearing assumptions

1. **Unconfoundedness / ignorability.** Given the observed covariates, treatment
   assignment is independent of the potential outcomes. Guaranteed here by
   randomization; on observational data it must be argued, not assumed.
2. **Overlap / positivity.** Every unit has a non-trivial probability of each
   arm. Checked explicitly in `src/ope_diagnostics.py` (propensity tails, ESS).
   Off-policy estimates are untrustworthy where this fails.
3. **SUTVA (no interference).** One unit's outcome is unaffected by another
   unit's treatment. Plausible for individual promotions; would break for
   strongly networked/viral effects.
4. **Correct propensities for OPE.** IPS/SNIPS/DR use known randomized
   propensities. Under observational data a propensity model is required and its
   misspecification biases IPS (DR is more robust).
5. **Stationarity.** The treatment-effect structure is stable between the data
   used to fit and the population the policy is deployed to. Drift would degrade
   targeting quality.

## Evaluation

- Metrics (Qini, AUUC, uplift@k) are computed **out-of-sample** on a held-out
  test split; the targeting budget is selected on a **separate validation split**
  and only then scored on test, to avoid optimism bias.
- Point estimates are reported with **bootstrap 95% confidence intervals**.
- Because ground-truth CATE is available, estimators are benchmarked against the
  **oracle** ranking; the normalized Qini expresses quality as a fraction of the
  oracle's.

## Known limitations

- Results are on semi-synthetic data; real-world performance on the Criteo
  benchmark (no ground truth) is the natural next validation step.
- Profit figures use illustrative unit economics (value/conversion, cost/
  treatment), not a calibrated business P&L.
- LGD-style downstream effects, long-term/retention effects, and treatment
  fatigue are not modeled.
- The uplift models use default hyperparameters; production use would tune them
  and add cross-fitting for the effect models.

## Ethical / responsible-use notes

- "Sleeping dogs" (negative uplift) exist: blindly treating everyone can *reduce*
  conversions and waste budget. The system explicitly avoids treating them.
- Targeting decisions should be audited for disparate impact if the covariates
  correlate with protected attributes — not addressed in this synthetic setting.
