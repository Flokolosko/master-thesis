# Validation plan

All experiments: `uv run python -m mdp_diagnostics.experiments.run <name>` with
`configs/experiments.yaml` (`--quick` for smoke runs). Outputs: `results/<name>/raw.csv`,
`results/<name>/summary.json`. Design follows ADEMP (Morris et al. 2019): Aims, Data-generating
mechanisms, Estimands, Methods, Performance measures.

**Avoiding circularity.** Every diagnostic is validated against a ground-truth quantity that is *not*
computed with the diagnostic's own model: the true support from the simulator's behaviour policy, the
true policy value from Monte-Carlo rollouts in the true environment, or the known presence/strength of
an injected violation.

## Exp 1 — Action overlap and coverage (`overlap`)
- **Aim:** do overlap/coverage diagnostics track true support and anticipate OPE error?
- **DGM:** `LatentLinearEnv` (no latent effect), behaviour sharpness β ∈ {0.5,1,2,4,8}, hard support restriction of one action (none, x0>0.5, x0>0), n ∈ {100, 400} trajectories, horizon 10, 10 seeds.
- **Ground truth:** true support-violation rate from the simulator's π_b; true value of the reference policy π_λ (λ=0.25, defined on the *true* π_b) by 20 000 Monte-Carlo rollouts.
- **Methods compared:** SVR, AOI, entropy, kNN share, propensity 5th percentile, ESAC, occupancy-weighted support, ESS share, Pareto-k̂.
- **Performance:** Spearman ρ (bootstrap CI) of each diagnostic with (i) true support violation and (ii) |WIS − V_true| using the estimated π_b; bias of estimated vs true SVR.
- (Coverage, task Exp 4, is merged here: β and support restrictions shrink the covered state–action region; subsampling is represented by n.)

## Exp 2 — State representation (`state`) — proposal RQ1–RQ3
- **DGM:** `LatentLinearEnv` with latent AR(1) factor z (ρ=0.9), latent strength c ∈ {0, 0.1, 0.2, 0.4, 0.8}, n ∈ {50,100,200,400}, horizon 20, 20 seeds.
- **Representations:** full (x, z) — Markov; partial (x) — non-Markov for c>0; history (x + 2 lags of x and a) — recommended revision; two *random revisions* (add a noise feature; drop x1).
- **Ground truth:** c (violation present iff c>0 for the partial state); true value of a fixed target policy by Monte-Carlo.
- **Performance:**
  - RQ1: rejection rate of the GCM test at α=0.05 (Wilson CI) — power for partial/c>0, false-alarm rate for full and c=0; targets: false alarm ≤ 0.05, power ≥ 0.8 at moderate c.
  - RQ2: Spearman ρ between history gain and relative error of a model-based value estimate fitted on the representation (target ρ ≥ 0.5).
  - RQ3: among cases flagged by the framework, relative error reduction (E_orig − E_rev)/E_orig of the recommended revision vs. random revisions (target: difference with CI excluding 0).
- Note: the history representation is not exactly Markov either (it is a finite-memory approximation of a belief state); its residual gain quantifies that.

## Exp 2b — PK/PD anaesthesia (`propofol`)
- **DGM:** `PropofolEnv` (Schnider PK, effect site, Hill BIS, inter-patient variability in C50, γ, E0, covariates), simulated anaesthetist titrating towards BIS 50, decision interval ∈ {2, 6} min, 50/150 patients, 5 seeds.
- **Representations:** BIS only; BIS + previous infusion; BIS + age/weight; full compartments (+ covariates).
- **Ground truth:** BIS-only is non-Markov by construction (latent compartments). *Correction after running:* the full compartment state is also non-Markov for a pooled population model, because PK rate constants depend on unrecorded covariates (height, sex) and nonlinearly on age/weight, and PD parameters vary per patient (see results summary). A truly Markov reference needs patient-specific parameters in the state.
- **Performance:** rejection rates and gains per representation, model class (linear vs gbm) and interval.

## Exp 3 — Temporal reliability (`temporal`)
- **DGM:** `LatentLinearEnv`, 300 trajectories, horizon 20, binary outcome (worst 30% returns), 10 seeds.
- **Perturbations (one at a time):** irregular sampling (drop prob 0–0.6), MCAR missingness (0–0.5), informative missingness driven by the outcome (strength 0–3), timestamp jitter with rounding (sd 0–0.3), action recorded 0–2 steps late.
- **Performance:** monotone response of the targeted indicator; non-response of the others (specificity); action-effect gain under misalignment.

## Exp 4 — Gene / medication baseline with downstream agents (`gene`)
- **Origin:** baseline suggested by the supervisor: an arbitrary illness with three measurables, two medications and a partially hidden gene that makes medication 1 ineffective.
- **DGM:** `GeneTreatmentEnv`. Measurables m0–m2 (0 = healthy) evolve linearly with an illness drive; medication 1 (strong, few side effects) has no effect in gene carriers; medication 2 (weaker, more side effects) works for everyone. Simulated clinician: first-line medication 1, switch to medication 2 after non-response; known carriers never receive medication 1. Factors: carrier prevalence ∈ {0, 0.15, 0.3}; process noise σ ∈ {0.15, 0.3}; scenarios *hidden* (never tested), *partial* (50 % tested, result recorded), *confounded* (50 % tested, clinician uses the result, data do not record it); 300 patients × 12 steps, 5 seeds (90 datasets).
- **Ground truth:** naive state (m0–m2) is non-Markov iff prevalence > 0; state + true gene is Markov by construction (positive control). True policy values by 10 000 Monte-Carlo rollouts.
- **Pipeline per dataset:** (1) full framework assessment of the naive formulation (gbm); (2) state-sufficiency diagnostics for each formulation (naive, oracle, recorded test, history 2 lags, revised = recorded test + lags, two random revisions); (3) fitted Q iteration (gradient-boosted trees, γ = 0.95), with and without a support constraint (π̂_b ≥ 0.05), trained on each formulation and evaluated in the true simulator.
- **Performance:** verdict rates per formulation; Spearman ρ between the naive-state gain and the naive agent's regret relative to the oracle-formulation agent; value gain of recommended vs random revisions among flagged datasets; effect of the support constraint; optimism of the agents' own Q estimates.

## Known limitations of the validation design
- Synthetic DGMs are linear-Gaussian or compartmental; nonlinear clinical dynamics may behave differently (mitigation: PK/PD environment; gbm model class).
- The PK/PD parameters are illustrative (see `experiments/synthetic.py`); conclusions concern detectability of structural violations, not anaesthesia practice.
- Unmeasured confounding is not validated because the framework does not claim to detect it (only a symptom, behaviour-history dependence).
