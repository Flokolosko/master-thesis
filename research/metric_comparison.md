# Metric comparison and selection

Basis: the three literature inventories in `research/inventory/` (≈65 methods; categories 1–4 as
defined there) and the synthetic experiments (`research/results_summary.md`). Principle: prefer
established methods, adapt them to sequential clinical data, and add new formulations only where the
literature has a gap.

## 1. Selected core diagnostics (implemented)

| # | Diagnostic | Criterion | Literature basis | Why selected | Category |
|---|---|---|---|---|---|
| 1 | Cross-fitted **history information gain** (+ lag profile) | state sufficiency | predictive Markov vs history comparison (inventory M8); Gaussian CMI (M7); AIS value-loss link (Subramanian et al. 2022, M12) | effect size on an interpretable scale (nats), works for any model class, needs no target policy | 1–2 |
| 2 | **Clustered GCM test** | state/reward sufficiency | Shah & Peters 2020 (M4) | valid under weak regression conditions; cheap; trajectory clustering added | 2 |
| 3 | Reward-sufficiency / time-homogeneity / action-effect / delayed-effect gains | reward, stationarity, alignment, sequentiality | same machinery as 1 | one estimator, different conditioning sets → comparable units | 2 (adaptation) |
| 4 | **Support-violation rate** with ε-sensitivity curve, propensity quantiles | positivity | Petersen et al. 2012; propensity-distribution diagnostics (C1, C2) | standard, interpretable; curve avoids a single arbitrary ε | 1 |
| 5 | **Behaviour-policy calibration (ECE)** | gate for 4, 7 | C7 (Raghu et al. 2018 show OPE sensitivity to behaviour-model error) | propensity-based results are only as good as π̂_b | 1 |
| 6 | Normalised action entropy, kNN locally-deterministic share | positivity (descriptive / model-free) | C8; D5 | entropy summarises the propensity distribution; kNN version is independent of the propensity model | 2–3 |
| 7 | **Kish ESS share, Pareto-k̂, ESS-by-horizon** for reference/target policies | OPE reliability | Kish; Vehtari et al. PSIS (G1, G2, G8); Gottesman et al. 2018 (non-zero weights in sepsis) | pre-training (needs only a target policy); k̂ has a literature threshold (0.7) | 1–2 |
| 8 | Occupancy-weighted support | coverage | count-based coverage (D1) reweighted by visitation, as recommended in the inventory | fixes ESAC's equal weighting of rarely visited cells | 2 |
| 9 | Transition epistemic ratio (bootstrap) | transition estimability | ensemble/bootstrap disagreement (D6) | directly measures whether data pin down the dynamics | 2 |
| 10 | Temporal indicators (missingness, interval CV/gaps, heaping, observation age, action without observation, within-step action changes, informative missingness AUC/gain) | data quality | Kahn et al. 2016 frame (E1), Weiskopf & Weng (E2), Agniel et al. 2018 (E5), Lipton/Che (E5) | separate indicators call for different remedies; replaces the single TOR | 1–3 |

## 2. Candidate metrics from the original plan

### AOI — Action Overlap Index → *kept for comparison only*
- Algebraically $\mathrm{AOI}_\epsilon = 1 - \frac1n\sum_t \frac{1}{|\mathcal A|}\#\{a: \hat\pi_b(a\mid s_t)\le\epsilon\}$, i.e. a rescaled version of the per-action support shares already reported by diagnostic 4.
- Depends on $|\mathcal A|$ (adding a never-relevant action changes it) and on $\epsilon$; averages over actions that may be clinically irrelevant in a state; inherits all propensity-model errors.
- Experiment 1: compare its rank correlation with the true support violation and with true OPE error against SVR, entropy and ESS (see results summary).
- Verdict: no information beyond SVR + per-action shares; not used for statuses.

### ESAC — Effective State–Action Coverage → *replaced by occupancy-weighted support*
- Each (cluster, action) cell counts equally regardless of visitation; value depends on the arbitrary number of clusters and $N_{\min}$; scales badly with state dimension (most cells empty).
- Modern offline-RL theory requires coverage of the *target* policy's distribution (single-policy concentrability, inventory D3), not of all cells — a low ESAC is neither necessary nor sufficient for feasibility.
- Verdict: keep as baseline in experiments; use the occupancy-weighted variant and IS-based target-policy diagnostics instead.

### TOR — Temporal Observation Reliability (completeness only) → *split into separate indicators*
- Completeness alone conflates MCAR gaps (harmless except for power), informative observation (state omission), irregular sampling (wrong time model) and stale states (misalignment). Experiment 3 shows these vary independently.
- Verdict: replaced by the indicator set in row 10.

## 3. Investigated but not implemented (with reasons)

| Method | Reason | Status |
|---|---|---|
| Forward-Backward Learning Markov test (Shi et al. 2020, M1) | strongest formal test in the literature, but implementation effort (conditional characteristic functions, deep density models) is high; GCM covers the same hypothesis | **open: implement or call the authors' `TestMDP` code as a robustness check** |
| Single-policy concentrability in a clustered MDP (D3) | needs a fitted tabular MDP and a candidate target policy; overlaps with IS diagnostics | candidate extension |
| Multi-state survival Markov tests (Titman & Putter 2022, M11) | requires a discrete multi-state formulation; useful bridge to clinical biostatistics | candidate for discrete formulations |
| BIC Markov-order selection (M9) | only for small discrete state spaces | candidate for clustered formulations |
| Sensitivity analysis for unobserved confounding (MSM Γ, E-value; F4–F6, G7) | needs a target policy / estimator; it complements but does not replace the diagnostics | recommended next step in reports |
| λ-discrepancy (Allen et al. 2024, M15) | promising value-relevant Markov diagnostic but no offline validation | research candidate |
| Bellman–Wasserstein distance (Asadulaev et al. 2025, D8) | measures behaviour-policy quality relative to random, needs a trained critic, validated only on D4RL locomotion | excluded |
| Medical dead-end proportion | property of the cohort's prognosis, not of MDP adequacy | excluded |
| Bisimulation metrics, PSR rank (M14, M18) | conceptual grounding; estimation from clinical data impractical | excluded as metrics |

## 4. Thresholds
Only the Pareto-k̂ threshold (0.7) and α = 0.05 have a literature/statistical basis. All other thresholds
(`Thresholds` in `framework/assessment.py`) are documented conventions; the primary use mode is
**comparative** (alternative formulations of the same problem), which needs no absolute threshold.
