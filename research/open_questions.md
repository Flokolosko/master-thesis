# Open questions for supervisor discussion

## Scope and alignment with the proposal
1. **Offline-RL layer.** The proposal centres on *empirical MDP adequacy* (Question A) with policy-learning feasibility as a separate, minimal layer. The implementation follows the A–E hierarchy with Question B (Levels C, D) fully implemented. Should Question B stay at this depth or be reduced in the thesis text?
2. **Level E (deployment).** Implemented as metadata questions only. Keep in the thesis, or drop as out of scope?
3. **Real data.** The proposal keeps real data generic; the case study uses the eICU sepsis cohort (credentialed PhysioNet data, kept outside the repo). Is reporting aggregate results from it acceptable in the thesis, and should a second dataset/published formulation be added (e.g. Luo et al. 2024's code base)?

## Methodology
4. **Relevance threshold δ for information gains** (default 0.01 nats/transition). Options: keep as convention with sensitivity curve; calibrate via simulation against value-error tolerance; tie to AIS bounds. Which is acceptable?
5. **Forward-Backward Learning test (Shi et al. 2020).** Implement (substantial effort) or use the authors' code as a robustness check alongside GCM?
6. **Model class dependence.** The reward-sufficiency smoke test showed a false "concern" with linear models when the reward is nonlinear in s′. Should the default model be gradient boosting (slower, less stable) or should both always be reported?
7. **"Moderate violation" for RQ1 targets.** Needs a definition per dimension (e.g. latent strength c = 0.4 in Exp 2; BIS-only state in Exp 2b). Agree on these before final runs.
8. **Reference target policies.** Without a clinical target policy, OPE diagnostics use (1−λ)π_b + λ·uniform. Is this an acceptable proxy, or should clinically motivated policy families be defined (e.g. "more/less vasopressor than practice")?
9. **Continuous actions.** Overlap is assessed after discretisation. Add generalised-propensity (conditional density) diagnostics for doses?

## Interpretation
10. **History-augmented states are still approximate.** In Exp 2 the recommended revision (2 lags) reduces but does not remove the gain. How should "adequate after revision" be phrased?
11. **Behaviour-history dependence as a confounding symptom.** Positive gain can also reflect legitimate history-dependent practice. Wording in the report is "symptom, not proof"; acceptable?
12. **PK/PD parameters** are taken from the commonly implemented Schnider model and illustrative PD values; should they be cross-checked against PAS (Aubouin-Pairault et al. 2023) and the original papers before final experiments?

13. **Patient heterogeneity.** The PK/PD experiment shows that pooled models are non-Markov whenever patient-specific parameters are latent; history then acts as patient identification. Should the framework add an explicit "patient-level heterogeneity" diagnostic (e.g. gain from trajectory-level random effects / early-trajectory summaries)?
14. **Comparability of gains across formulations.** Gains are relative to the base model's fit; should comparisons be reported jointly with out-of-fold R² or normalised?
15. **Case-study interpretation.** The near-zero action-effect gain in eICU sepsis is consistent with closed-loop titration (confounding by indication) or misalignment. Is there a way (e.g. raw timestamps from the source eICU tables) to separate these?

## Known limitations to discuss
- RQ1 false-alarm rate (0.062 for the Markov state) slightly exceeds the 0.05 target; consider a cluster-wild bootstrap or a permutation calibration.
- Diagnostics are blind to latent variables that leave no trace in recorded history.
- CI testing has no power against all alternatives (Shah & Peters 2020).
- Informative-missingness experiment saturates (AUC = 1 already at strength 1) — weaker settings needed for a dose–response curve.
- The relative gap rate in sampling diagnostics is not monotone in the drop probability because the median interval adapts; the interval CV is the more reliable indicator.
- Timestamp-heaping diagnostic was not meaningfully validated (synthetic times are on an integer grid).
