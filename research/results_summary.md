# Results summary (synthetic validation and case study)

Raw outputs: `results/<experiment>/raw.csv`, summaries: `results/<experiment>/summary.json`.
Confidence intervals are 95 %. Configuration: `configs/experiments.yaml`.

## Exp 1 — Overlap and coverage (300 runs)

Spearman ρ across runs (bootstrap CI):

| Diagnostic | ρ vs true support violation | ρ vs \|WIS − V_true\| |
|---|---|---|
| Support-violation rate (SVR, ε=0.01) | **0.82** | 0.31 [0.21, 0.41] |
| AOI (candidate) | −0.83 | −0.31 [−0.41, −0.21] |
| Normalised action entropy | −0.81 | −0.28 [−0.38, −0.17] |
| kNN locally-deterministic share (model-free) | 0.66 | 0.21 [0.10, 0.32] |
| Propensity 5th percentile | 0.04 | 0.10 [−0.01, 0.20] |
| ESAC (candidate) | −0.32 | −0.21 [−0.32, −0.10] |
| Occupancy-weighted support | −0.35 | −0.21 [−0.32, −0.11] |
| IS ESS share (λ=0.25) | −0.61 | **−0.44 [−0.54, −0.34]** |
| Pareto-k̂ | 0.45 | 0.36 [0.25, 0.46] |

Findings
- AOI is (up to sign) the same information as SVR — confirms the analytical argument that it adds nothing.
- SVR underestimates the true violation rate by 0.14 on average: a smooth (logistic) behaviour model cannot represent hard support restrictions. Declared action spaces and the calibration gate mitigate but do not remove this.
- Coverage counts (ESAC and its occupancy-weighted variant) are weak proxies of both support and OPE error in this setting.
- The target-policy-specific diagnostics (ESS share, Pareto-k̂) anticipate OPE error best, but only moderately (|ρ| ≤ 0.44): WIS error is dominated by variance at horizon 10 even with the *true* behaviour policy. No pre-training diagnostic predicts OPE error precisely; they flag regimes of unreliability.

## Exp 3 — Temporal reliability (10 seeds per level)

| Perturbation | Targeted indicator response | Off-target indicators |
|---|---|---|
| Action recorded 1–2 steps late | action-effect gain 0.433 → 0.000 | unchanged |
| MCAR missingness 0→0.5 | missingness 0.00→0.50 exactly; observation age p90 0→3 | informative-missingness AUC stays 0.49–0.51 (correct specificity) |
| Informative missingness (outcome-driven) | outcome AUC 0.50 → 1.00 already at strength 1 (saturated design) | missingness ≈ 0.3–0.34 |
| Irregular sampling 0→0.6 | interval CV 0 → 0.44 → 0.61 → 0.75 (monotone) | gap-trajectory share non-monotone (0.11, 0.53, 0.27): relative gap threshold adapts to the median interval; action-effect gain drops (transitions span longer intervals) |
| Timestamp jitter, rounded | invisible at sd 0.1 (rounding restores the grid); CV 0.42 at sd 0.3 | heaping indicator uninformative here (integer grid by construction) |

Findings
- Separate temporal indicators respond selectively, supporting the decision to replace the single TOR.
- Negative results: the relative gap rate is not a monotone severity measure; heaping was not validated; the informative-missingness design needs weaker settings for a dose–response curve.

## Exp 2 — State representation (RQ1–RQ3; 1 400 runs, 20 seeds per cell)

GCM rejection rate at α = 0.05 (partial state = latent factor hidden):

| latent strength c | n=50 | n=100 | n=200 | n=400 | mean gain (n=400, nats) |
|---|---|---|---|---|---|
| 0 (Markov) | 0.10 | 0.15 | 0.05 | 0.10 | −0.002 |
| 0.1 | 0.20 | 0.75 | 0.90 | 0.95 | 0.003 |
| 0.2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.023 |
| 0.4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.053 |
| 0.8 | 1.00 | 1.00 | 1.00 | 1.00 | 0.188 |

| Pre-set target (proposal) | Result | Met? |
|---|---|---|
| RQ1 false-alarm rate ≤ 0.05 | full (Markov) state: 0.062 (25/400; Wilson CI ≈ [0.04, 0.09]); partial state with c = 0: 0.10 (8/80) | **slightly exceeded** — mild anti-conservativeness of the clustered max-statistic in small samples |
| RQ1 power ≥ 0.8 at moderate violation | ≥ 0.95 for c ≥ 0.2 at all n; c = 0.1 needs n ≥ 200 | met for c ≥ 0.2 |
| RQ2 Spearman ρ(gain, relative value error) ≥ 0.5 | **0.74 [0.69, 0.79]** | met |
| RQ3 recommended > random revision (CI excl. 0) | among 60 flagged cases: RER recommended (history augmentation) = 0.71, random revisions = −0.23; difference **0.94 [0.84, 1.05]**; per-case median 0.62 vs −0.04 | met; by strength: median RER −0.04 (c=0.2), 0.57 (c=0.4), 0.76 (c=0.8) |

Further observations
- The information gain is negative in expectation for Markov states at small n (≈ −0.02 nats at n = 50): cross-fitting penalises superfluous features, which makes the equivalence rule conservative.
- History augmentation (2 lags) reduces but does not eliminate dependence (rejection 0.45–1.0 at n = 400; gains ≤ 0.005 nats): a finite-memory approximation of the belief state. Below the relevance threshold, so classified as no relevant dependence.
- Random revisions are also flagged (the latent factor stays hidden), i.e. the diagnostic does not reward arbitrary changes.
- For small violations (c = 0.2) the revision does not reduce value error: statistical detectability precedes practical consequence — a reason for an effect-size threshold.

## Exp 2b — PK/PD anaesthesia (160 runs)

Mean history gain (nats/transition; GCM rejection = 1.0 in all cells):

| Representation | model | Δt = 2 min (n=50 / 150) | Δt = 6 min (n=50 / 150) |
|---|---|---|---|
| BIS only | linear | 0.071 / 0.074 | 0.589 / 0.600 |
| BIS only | gbm | 0.078 / 0.100 | 0.433 / 0.505 |
| BIS + previous infusion | linear | 0.148 / 0.156 | 0.170 / 0.198 |
| BIS + previous infusion | gbm | 0.138 / 0.166 | 0.057 / 0.141 |
| BIS + age, weight | linear | 0.063 / 0.068 | 0.526 / 0.543 |
| full compartments + covariates | linear | 0.749 / 0.797 | 0.826 / 1.212 |
| full compartments + covariates | gbm | 0.315 / 0.459 | 0.222 / 0.258 |

Findings
- All representations are flagged — correctly. The initial assumption that the compartment state is Markov was wrong for a *pooled* model: PK rate constants depend on unrecorded height/sex and nonlinearly on age/weight, and PD parameters (C50, γ, E0) vary per patient. Adding the true PD parameters reduced the gain only partly (linear 0.71 → 0.60; gbm 0.36 → 0.23, n = 150). History acts as patient identification; a sufficient state needs patient-specific parameters (mixed-effects / Bayesian individualisation).
- A coarser decision interval strongly increases history dependence of BIS-only states (0.07 → 0.6 nats): consistent with latent effect-site dynamics becoming relevant between decisions.
- Gains are *relative* to the base model's predictive power: richer states with very small residual noise can show larger gains than poor states. Gains should be compared across formulations only together with the out-of-fold R² (reported in `details`).
- The flexible model (gbm) lowers gains for the nonlinear system, confirming model-class dependence.

## Case study — eICU sepsis (3 362 stays, norepinephrine, 4 dose levels)

| Criterion | primary (Δt = 60 min) | Δt = 30 | Δt = 120 | + APACHE | gbm |
|---|---|---|---|---|---|
| B state sufficiency (gain, nats) | concern (0.086) | concern (0.092) | concern (0.063) | concern (0.085) | concern (0.095) |
| B action alignment (gain) | concern (0.0005) | concern (0.0008) | concern (0.0003) | concern (0.0005) | concern (0.0006) |
| B reward sufficiency / stationarity / estimability | no concern | no concern | no concern | no concern | no concern |
| B decision interval (steps with >1 raw action) | 6.3 % | 2.4 % | 18.6 % | 6.3 % | 6.3 % |
| C sampling (interval CV) | concern (1.45) | = | = | = | = |
| D behaviour history (gain, nats) | concern (0.95) | concern (1.07) | concern (0.74) | concern (0.94) | concern (0.96) |
| D support violation: state-only → history-aware | 0 % → 20 % | 0 % → 66 % | 0 % → 4 % | 0 % → 23 % | 0 % → 18 % |
| D action entropy: state-only → history-aware | 0.97 → 0.30 | 0.96 → 0.20 | 0.98 → 0.46 | 0.96 → 0.30 | 0.97 → 0.30 |
| D IS (λ=0.25): ESS share / Pareto-k̂, history-aware | 0.001 / 1.34 | 0.002 / 2.03 | 0.019 / 0.86 | 0.001 / 1.39 | 0.000 / 1.16 |

Interpretation (statistical, not clinical)
- **Question A:** (HR, MAP) is not a sufficient state (relevant history dependence in every formulation); reward and stationarity raise no concern. Recorded doses carry almost no information about the next HR/MAP given the current values — consistent with closed-loop titration (dose adjusted to keep MAP stable, confounding by indication) and/or misalignment; the data cannot distinguish these.
- **Question B:** treatment is strongly driven by the previous dose (0.95 nats). Conditioning on the state alone suggests perfect overlap (entropy 0.97) and acceptable IS diagnostics (ESS 0.33, k̂ 0.58), whereas the history-aware model reveals limited diversity and unreliable IS (k̂ > 0.7 everywhere). Without the history-aware check, the framework would have given a falsely reassuring Question-B answer.
- Finer grids (30 min) worsen IS reliability; coarser grids (120 min) aggregate more decisions per step (18.6 %) — a concrete trade-off for choosing Δt.
- 0 % missingness and observation age 0 reflect upstream preprocessing of the public dataset (pre-extracted, z-normalised series); the observation process of the source EHR cannot be assessed from it.

**Boundary case (eICU cardiac arrest, 5 000 stays):** no treatment variable → Level A `serious_concern` ("cannot be modelled as an MDP from these data"), consistent with the earlier interdisciplinary-project finding that action data are the bottleneck. Measurement patterns are weakly informative of mortality (AUC 0.57 [0.54, 0.60]).

## Implementation issues found by the experiments (fixed)
1. **Deterministic next-state components** (lag columns inside a history-augmented state, static covariates, noise-free compartment amounts) produced spurious information gains of > 2 nats and 100 % GCM rejections. Fix: target dimensions predicted with out-of-fold R² ≥ 0.999 by the base features (linear or configured model) are excluded and listed in `details.excluded_deterministic_dims`. Regression test added.
2. **Overlap measured on an insufficient state is over-optimistic.** In the eICU data, dose given (HR, MAP) looks almost random (entropy 0.97), while the previous dose predicts the next dose with a gain of 0.95 nats. Fix: when behaviour-history dependence is a concern, overlap and IS diagnostics are recomputed with a history-aware behaviour model and the worse status is reported.
3. **Unobserved actions are invisible** to overlap diagnostics unless the action space is declared (`action_levels`).
4. **Model-class dependence:** with a linear model, a reward that is quadratic in s′ is falsely flagged as history-dependent; report both model classes for reward sufficiency.
