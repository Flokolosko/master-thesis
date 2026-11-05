# Research gap: what is established, what would be new

## Established (do not claim as contributions)
- **Qualitative guidance** for healthcare RL: Gottesman et al. 2019 (confounding, reward, OPE pitfalls).
- **Markov-property tests** for sequential decision data: Shi et al. 2020 (FBL test); generic CI tests (GCM, KCI, CRT/CPT); Markov-order selection for chains (BIC, CMI tests); Markov tests in multi-state survival models (Titman & Putter 2022).
- **Positivity diagnostics** in causal inference (Petersen et al. 2012; trimming, overlap weights; D'Amour et al. 2021 on high-dimensional overlap) and sequential positivity checks (PoRT-type).
- **Importance-sampling diagnostics** (Kish ESS; Pareto-k̂ in PSIS, Vehtari et al. 2024) and evidence that IS-based OPE fails in ICU data (Gottesman et al. 2018: few non-zero-weight trajectories in sepsis).
- **Coverage theory** in offline RL (concentrability; single-policy coverage suffices for pessimistic methods).
- **EHR data-quality frameworks** (Kahn et al. 2016) and evidence that observation processes are informative (Agniel et al. 2018).
- **Sensitivity analysis / negative controls** for unmeasured confounding, including sequential OPE (Namkoong et al. 2020; Kallus & Zhou 2020).
- **Empirical evidence of formulation sensitivity** (Luo et al. 2024) and of time-step sensitivity (Sun & Tang 2025, preprint).

## Gap
These pieces live in separate literatures and are applied, if at all, one at a time and *after* a
formulation has been chosen. No existing work (to the extent searched; see inventories):
1. organises the assumptions of a *specific MDP formulation of observational clinical data* into testable / partially assessable / non-testable classes, keeping MDP adequacy (Question A) separate from offline-RL feasibility (Question B);
2. measures the testable ones on a **common, interpretable effect-size scale** (out-of-sample nats for state, reward, time, action-effect and behaviour-history dependence), with an equivalence-style rule instead of p-values alone;
3. operationalises **temporal** assumptions of EHR-derived MDPs (decision interval vs decision frequency, action–state alignment, informative observation) as quantitative diagnostics;
4. **validates** such diagnostics against ground truth (power, false alarms, relation to value-estimation error, effect of recommended revisions), including a physiologically structured simulator (PK/PD anaesthesia).

## Candidate contributions (in decreasing confidence)
| Contribution | Novelty | Evidence needed |
|---|---|---|
| Structured, assumption-level framework separating Q-A/Q-B with explicit `not_assessable` items | synthesis; moderate novelty | coherence + case study |
| Unified cross-fitted information-gain family (state, reward, stationarity, action alignment, delayed effects, behaviour history) with equivalence rule | adaptation of predictive Markov comparison/CMI; moderate | Exp 2, 2b, 3 |
| Clustered GCM test for trajectory data | small adaptation | false-alarm rates (Exp 2) |
| Pareto-k̂ for trajectory IS weights in clinical OPE | new in context (inventory G2 found no RL use) | Exp 1 |
| Action-effect gain as misalignment detector | new formulation; simple | Exp 3 |
| Simulation validation with recommended vs random revisions (RQ3) | new evaluation design | Exp 2 |

## What would *not* be defensible
- Claiming that any diagnostic establishes Markovianity, causal identifiability or safety.
- Absolute thresholds as universal truths (only Pareto-k̂ 0.7 and α have external justification).
- An overall feasibility score.
- Generalising simulation results to clinical practice without the case-study caveats.
