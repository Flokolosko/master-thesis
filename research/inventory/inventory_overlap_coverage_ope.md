# Inventory: Action Overlap & Positivity (C), State-Action Coverage (D), OPE Reliability Diagnostics (G)

Thesis: *A Quantitative Framework for Assessing the Empirical Adequacy of MDP Models in Healthcare* (TU Wien, MSc Data Science).
Draft, literature-search output. Date of search: 2026-11-05.

---

## 1. Search log

**Date:** 2026-11-05. **Tools:** WebSearch (standard mode), WebFetch on arXiv abstract/HTML pages, PMC full text, and one arXiv PDF extracted locally with `pdftotext`.

| # | Query / source | What it established |
|---|---|---|
| 1 | D'Amour, Ding, Feller, Lei, Sekhon, overlap, high-dimensional covariates | J. Econometrics 221(2) 2021; arXiv:1711.02582 (abstract read) |
| 2 | Crump, Hotz, Imbens, Mitnik 2009, limited overlap | Biometrika 96(1):187–199 (secondary summaries) |
| 3 | Petersen et al. positivity (PMC full text, first ~100k chars) | ETA.Bias bootstrap, weight and propensity checks, responses |
| 4 | Vehtari et al. PSIS, JMLR 2024 | JMLR 25(72); arXiv:1507.02646 (abstract-level) |
| 5 | Chen & Jiang ICML 2019; Munos 2003; Munos & Szepesvári JMLR 2008 | Concentrability literature (abstracts only) |
| 6 | Rashidinejad et al. 2021; Xie et al. 2021 | Single-policy concentrability, pessimism (abstracts + secondary notes) |
| 7 | DualDICE; Liu et al. 2018 (curse of horizon) | Stationary distribution-ratio estimation (abstracts) |
| 8 | MOPO; MOReL; BEAR; BCQ | Uncertainty, ensembles and support constraints as coverage proxies (abstracts and reviews) |
| 9 | arXiv:2510.12638 (abstract + HTML full text via WebFetch) | BWD definition (Eqs. 5, 14–16) |
| 10 | Voloshin et al. 2021; Fu et al. 2021 DOPE; Tang & Wiens 2021; Luo et al. 2024; Hargrave et al. 2024 | Empirical OPE reliability evidence (abstracts) |
| 11 | Hao et al. 2021; Thomas et al. 2015; Hanna et al. 2017; Thomas & Brunskill 2016 | OPE confidence intervals and estimators (abstracts) |
| 12 | Gottesman et al. 2018 (full text read, Sec. 5 and recommendations); Gottesman et al. 2019; Gottesman et al. 2020 | Effective sample size in sepsis; influence diagnostics |
| 13 | Raghu et al. 2018 | Calibration of the behaviour policy (abstract) |
| 14 | Sequential positivity: sPoRT (Chatton et al.) | Longitudinal positivity diagnostic (abstract and snippets) |
| 15 | Hirano & Imbens 2004; Li, Morgan & Zaslavsky 2018; Elvira, Martino & Robert 2022; Kish 1965; Precup et al. 2000; Namkoong et al. 2020; Sun et al. 2022 | Supporting methods |

**Reading depth.** Full text (or most of it) was read only for Petersen et al. 2012, Gottesman et al. 2018 and Asadulaev et al. 2025. Everything else is at abstract or secondary-summary level, and is marked "(abstract)" where it matters.

**Timing taxonomy used below:**
- **T0, data only.** Uses the trajectories plus *nuisance* models such as a behaviour-policy classifier or density model. No target policy is needed.
- **T1, needs a target (evaluation) policy π_e** but no trained value or dynamics model. For pre-optimisation use, π_e can be a candidate class: deterministic or softened clinician-mode policies, simple rules, or "small deviation" policies.
- **T2, needs a trained evaluation model**, e.g. a Q-function (FQE), a dynamics ensemble, a critic, or density-ratio networks.

---

## 2. Method inventory

### C. Action overlap and positivity

#### C1. Behaviour-policy (propensity) distribution diagnostics
- **Metric:** Estimate π̂_β(a|s) from (s,a) pairs. Report:
  - the distribution of π̂_β(a_obs|s);
  - the minimum, min_{s,a∈A_clin} π̂_β(a|s);
  - the share of states with π̂_β(a|s) < ε for some clinically relevant a;
  - the IPW weight distribution 1/π̂_β.
- **Literature:** Petersen et al. 2012, doi:10.1177/0962280210386207; Crump et al. 2009, doi:10.1093/biomet/asn055.
- **Purpose:** Detect practical (near-)positivity violations. These are states where some action is (almost) never taken.
- **Requirements:** States, actions, and a probabilistic classifier for π_β.
- **Computation:** Fit a multinomial classifier (discrete actions), then compute summary statistics and histograms. The work is O(N·model cost).
- **Strengths:** Standard practice in causal inference, simple, and interpretable per action and per state cluster.
- **Limitations:**
  - Petersen et al. note that well-behaved weights do not rule out violations.
  - The diagnostic depends on how well the π̂_β model is specified and calibrated (see C7).
  - It cannot detect unmeasured confounding.
  - A low π̂_β can be a real *structural* zero (a contraindicated action), not a data gap. Clinical masks are needed to separate the two.
- **Applicability:** Discrete actions directly. Continuous actions via C6. High-dimensional states are fine with ML classifiers, but see C5 on overlap degrading with dimension.
- **Timing:** T0.
- **Validation:** Long-standing practice. Petersen et al. show that it does *not* quantify the bias caused.
- **Recommendation:** Implement (core).
- **Category:** 1.

#### C2. Overlap-based trimming and fraction-retained (Crump optimal subpopulation; overlap weights)
- **Metric:**
  - Crump et al.: keep units with α ≤ ê(x) ≤ 1−α. The rule of thumb is α = 0.1; the optimal α comes from the data under their conditions.
  - Framework statistic: the **fraction of transitions or trajectories retained**.
  - Overlap weights (Li et al.): weight ∝ probability of the *other* treatment. The weights are bounded in [0,1].
- **Literature:** Crump et al. 2009, doi:10.1093/biomet/asn055; Li, Morgan & Zaslavsky 2018, doi:10.1080/01621459.2016.1260466.
- **Purpose:** Quantify how much of the data lies in a region with adequate overlap, and so how much the estimand would shift after trimming.
- **Requirements:** Propensity estimates.
- **Computation:** Trivial once C1 is fitted.
- **Strengths:** Principled for binary treatment, with a variance-optimality motivation in both papers (abstract-level).
- **Limitations:**
  - Developed for binary point treatments. Extending it to |A|>2 and to sequences is non-trivial.
  - Trimming changes the target population, which may exclude exactly the sickest patients.
  - Sequential trimming compounds over time.
- **Applicability:** Binary or few actions. Multi-action use needs generalisation (e.g. min_a π̂_β(a|s) ≥ α).
- **Timing:** T0.
- **Validation:** Widely used in epidemiology and econometrics.
- **Recommendation:** Adapt. Report the retained fraction as a function of α as an "overlap curve".
- **Category:** 2.

#### C3. Parametric-bootstrap positivity bias diagnostic (ETA.Bias)
- **Metric:** Fit a data-generating distribution P̂ and resample datasets from it. Apply the estimator to each and compare the mean estimate with ψ(P̂). The gap is ETA.Bias.
- **Literature:** Petersen et al. 2012 (full text read).
- **Purpose:** Quantify the estimator bias attributable to positivity violations, which C1 cannot do.
- **Requirements:** A generative model of the full data (for MDPs: transitions, behaviour policy, rewards) and the estimator.
- **Computation:** B bootstrap fits, i.e. B times (simulation + estimator) cost.
- **Strengths:**
  - Translates overlap problems into the bias scale.
  - Petersen et al. suggest a red flag when ETA.Bias ≳ SE, or when a bias-corrected CI would change conclusions.
- **Limitations:**
  - The authors call it deliberately *optimistic*: it relies on the fitted model being close to the truth.
  - It was developed for point and longitudinal treatment-effect estimators, not RL OPE.
  - In the MDP setting it requires a fitted simulator, which drifts towards T2.
- **Applicability:** Any setting where a simulator can be fitted. In the tabular or clustered MDP case it is natural, because the estimated MDP *is* the generative model.
- **Timing:** T1/T2. It needs π_e and a fitted MDP model, but no RL policy optimisation.
- **Validation:** Simulations in Petersen et al. It was most informative with IPTW.
- **Recommendation:** Adapt. In the clustered-MDP setting, simulate from (P̂, π̂_β, R̂) and measure the bias of WIS/PDIS/FQE for candidate π_e.
- **Category:** 2.

#### C4. Sequential (longitudinal) positivity diagnostics: PoRT / sPoRT
- **Metric:** Regression trees that find covariate-defined subgroups, time-updated in the longitudinal case, in which some treatment level has (near-)zero empirical probability.
- **Literature:** Chatton et al., arXiv:2412.10245, later published in *Epidemiology* (DOI 10.1097/EDE.0000000000001902 per one secondary listing; not confirmed at the source). The PoRT predecessor was identified but not opened.
- **Purpose:** Positivity in a DTR must hold at every decision point given history. sPoRT gives interpretable subgroups in which it fails.
- **Requirements:** A longitudinal panel and the treatment at each step. Static or dynamic regimes.
- **Computation:** Tree fitting per time point (stratified) or pooled.
- **Strengths:**
  - Nonparametric, with no propensity functional-form assumptions.
  - Interpretable "where it fails" output, which suits clinical review.
  - R notebook available.
- **Limitations:**
  - Abstract-level reading only.
  - Designed for epidemiological DTRs with a small number of time points. Scaling to long ICU trajectories (tens of steps) and many actions is untested.
  - Same blindness to unmeasured confounding as C1.
- **Applicability:** Discrete treatments, moderate dimension.
- **Timing:** T0 (static regime) or T1 (dynamic regime = target policy).
- **Validation:** The authors' simulations and applied example (not read).
- **Recommendation:** Investigate. Possibly adapt as an interpretable companion to C1 on the clustered state.
- **Category:** 2.

#### C5. High-dimensional overlap caveat (D'Amour et al.)
- **Metric:**
  - There is no single diagnostic. The paper derives bounds showing that *strict overlap* (η ≤ e(x) ≤ 1−η) limits average covariate-mean imbalance between arms, and that the restriction tightens as dimension grows (abstract).
  - Practical corollary, our adaptation: treat high treatment predictability (e.g. AUROC of π̂_β near 1) as evidence against overlap.
- **Literature:** D'Amour et al. 2021, doi:10.1016/j.jeconom.2019.10.014; arXiv:1711.02582.
- **Purpose:** Theoretical justification for *why* positivity is likely violated in rich clinical state representations. It argues for doing overlap checks on the representation actually used by the MDP.
- **Requirements / computation:** Not applicable (theory). The AUROC proxy costs the same as C1.
- **Strengths:** A strong, citable argument that richer state representations (needed for the Markov property) trade off against overlap.
- **Limitations:** Not itself a metric. The AUROC proxy is our construction, not from the paper.
- **Applicability:** All high-dimensional settings.
- **Timing:** T0.
- **Validation:** Theoretical.
- **Recommendation:** Use as motivation, and as a design principle linking the Markov-adequacy and overlap criteria in the framework.
- **Category:** 1 (as theory); the AUROC proxy is 3.

#### C6. Continuous-action overlap: generalised propensity score / conditional density
- **Metric:**
  - GPS r(a,x) = f_{A|X}(a|x), the conditional treatment density (Hirano & Imbens 2004).
  - Overlap check: is r(a,x) bounded away from 0 over the action region of interest? For example, the fraction of states where f̂(a*|s) < ε for clinically relevant doses a*.
  - Alternatively, use the density of the observed dose relative to the mode.
- **Literature:** Hirano & Imbens 2004, Wiley book chapter (found via secondary sources; chapter not opened).
- **Purpose:** Positivity for continuous doses (fluids, vasopressors, insulin) without discretising.
- **Requirements:** A conditional density estimator, e.g. Gaussian/heteroscedastic regression, mixture density networks, normalising flows, or kernel CDE.
- **Computation:** Model fitting. Density values are unnormalised in high dimension, so calibrate thresholds via held-out quantiles.
- **Strengths:** Avoids arbitrary binning, which itself creates artificial positivity violations.
- **Limitations:**
  - Density estimates are unstable in the tails, which is exactly where positivity is judged.
  - The results are sensitive to model choice.
  - IS weights with continuous densities are prone to extreme variance.
- **Applicability:** Continuous actions, low to moderate action dimension.
- **Timing:** T0.
- **Validation:** Established in the causal dose-response literature. RL uses (BCQ's generative model, see D7) are related but serve a different purpose.
- **Recommendation:** Adapt. Implement if the thesis keeps continuous doses; otherwise document that discretisation is a modelling choice that interacts with overlap.
- **Category:** 2.

#### C7. Behaviour-policy calibration check
- **Metric:**
  - Calibration of π̂_β, e.g. expected calibration error or reliability diagrams per action, plus held-out log-loss.
  - Sensitivity of downstream IS estimates to the choice of π̂_β model (kNN vs NN vs logistic regression).
- **Literature:** Raghu et al. 2018, arXiv:1807.01066 (ICML 2018 workshop; abstract).
- **Purpose:** Every C-diagnostic and IS-based diagnostic depends on π̂_β. Raghu et al. report that behaviour-policy errors as small as 0.06 (average absolute) can produce >50% error in the value estimate. In their sepsis data, kNN gave better calibration than neural networks.
- **Requirements:** Held-out data.
- **Computation:** Cheap.
- **Strengths:** Directly addresses a known failure mode in healthcare RL.
- **Limitations:**
  - A workshop paper with a single dataset.
  - Calibration on observed actions says little about rare actions, where it matters most.
- **Applicability:** All settings.
- **Timing:** T0.
- **Validation:** Sepsis case study.
- **Recommendation:** Implement, as a prerequisite gate for C1, C2, G1 and G2.
- **Category:** 1.

#### C8. State-conditional action entropy / effective number of actions
- **Metric:**
  - H(s) = −Σ_a π̂_β(a|s) log π̂_β(a|s).
  - Effective number of actions N_eff(s) = exp(H(s)).
  - Report E_s[N_eff(s)/|A|] and its distribution.
- **Literature:** No dedicated validated source was found. It is a standard information-theoretic construction. Gottesman et al. 2018 discuss qualitatively that deterministic policies retain few non-zero-weight trajectories, which motivates measuring how stochastic clinicians are.
- **Purpose:** Measure the *variability* of clinical practice per state, i.e. the "natural experiment" content of the data.
- **Requirements / computation:** Same as C1.
- **Strengths:**
  - A continuous, threshold-free alternative to AOI (Section 3).
  - Distinguishes "two actions at 50/50" from "ten actions at 1% each".
- **Limitations:**
  - High entropy does not equal overlap *for the actions π_e will choose*.
  - Sensitive to π̂_β smoothing.
  - Ignores structural zeros.
  - Not validated as a predictor of OPE accuracy.
- **Applicability:** Discrete actions. For continuous actions use differential entropy (scale-dependent).
- **Timing:** T0.
- **Validation:** None found.
- **Recommendation:** Investigate, as a descriptive companion to C1.
- **Category:** 3.

### D. State-action coverage

#### D1. Count-based state-action coverage (tabular / clustered MDP)
- **Metric:**
  - N(s,a) visit counts.
  - The coverage fraction |{(s,a): N(s,a) ≥ N_min}| / |S||A| (= ESAC, see Section 3).
  - Count distribution (Gini, quantiles).
  - Transition-row support: the number of transitions per (s,a) available for estimating P̂(·|s,a).
- **Literature:** No single source; implicit in tabular offline RL analyses (e.g. the state-action counts underlying LCB in Rashidinejad et al. 2021, abstract level).
- **Purpose:** Data sufficiency for estimating P̂ and R̂ per cell.
- **Requirements:** A discretised state (clustering, as in the AI-Clinician-style pipelines).
- **Computation:** O(N).
- **Strengths:** Transparent, and it ties directly to the variance of the transition estimates.
- **Limitations:**
  - Depends completely on the discretisation, i.e. the number of clusters k.
  - Uniform coverage over all (s,a) is neither necessary nor sufficient: offline RL theory needs coverage of the *target* policy's occupancy (D3).
  - Ignores clinically infeasible pairs.
- **Applicability:** Discrete only. For continuous spaces see D5 and D6.
- **Timing:** T0.
- **Validation:** Indirect (concentration bounds for count-based estimates).
- **Recommendation:** Implement, but report it as a count *distribution* weighted by visitation and restricted to feasible actions, not just a single fraction.
- **Category:** 2.

#### D2. All-policy concentrability coefficient
- **Metric:**
  - C = sup_π sup_{s,a} d^π(s,a)/μ(s,a), or variants based on future-state distributions (Munos 2003; Munos & Szepesvári 2008, "discounted-average concentrability of future-state distributions").
  - Chen & Jiang (2019) discuss its necessity for batch value-function methods.
- **Literature:** Munos 2003 (ICML, pp. 560–567); Munos & Szepesvári 2008, JMLR 9:815–857; Chen & Jiang 2019, PMLR 97:1042–1051, arXiv:1905.00360.
- **Purpose:** The theoretical quantity that governs FQI/FQE error bounds under distribution shift.
- **Requirements:** In principle, all policies' occupancy measures and the true dynamics.
- **Computation:** Not computable from data in general. In a small tabular estimated MDP one could compute a plug-in version over a restricted policy class, but sup_π is dominated by poorly covered cells.
- **Strengths:** The canonical theoretical notion of coverage.
- **Limitations:**
  - Unobservable, with only proxies available.
  - Worst-case and usually infinite or vacuous for real clinical data.
  - Requires exploratory data, which clinical data almost never are.
- **Applicability:** Theory.
- **Timing:** N/A.
- **Validation:** Theoretical.
- **Recommendation:** Exclude as a metric. Cite as the conceptual target that all D-proxies approximate.
- **Category:** 4 (as a metric); 1 (as theory).

#### D3. Single-policy concentrability C^{π_e}
- **Metric:** C^{π} = max_{s,a} d^{π}(s,a)/μ(s,a). This definition comes from secondary lecture notes on Rashidinejad et al. (the paper itself uses a weaker variant relative to the expert policy); Xie et al. 2021 use related coverage notions.
- **Literature:** Rashidinejad et al. NeurIPS 2021 / IEEE TIT 2022, arXiv:2103.12021; Xie et al. NeurIPS 2021, arXiv:2106.06926.
- **Purpose:** The modern pessimism results need coverage of *one* comparator policy only. For the framework, that comparator can be "clinician-like but improved".
- **Requirements:** Target policy and occupancy d^{π_e}. A plug-in estimate needs a dynamics model (tabular P̂) or ratio estimation (D4).
- **Computation:**
  - Tabular: solve for d^{π_e} in the estimated MDP, which is O(|S|²|A|) per policy.
  - Report max and high quantiles of the ratio. The max is unstable, so use quantiles or the χ²-type average E_μ[(d/μ)²].
- **Strengths:** The theoretically correct notion of coverage for offline RL with pessimism.
- **Limitations:**
  - The max is dominated by rare cells.
  - Plug-in d^{π_e} inherits model error, which is circular when the MDP adequacy is the very question.
- **Applicability:** Tabular (direct); continuous via D4.
- **Timing:** T1, plus a fitted tabular model (lightweight T2).
- **Validation:** Theory. No empirical validation in healthcare was found.
- **Recommendation:** Adapt. Compute it for a small family of candidate π_e in the clustered MDP and report quantiles, not the max.
- **Category:** 2.

#### D4. Stationary or discounted distribution-ratio estimation (DualDICE; Liu et al. 2018)
- **Metric:**
  - w(s,a) = d^{π_e}(s,a)/d^D(s,a), estimated without knowing π_β (DualDICE) or via a minimax loss over state ratios (Liu et al.).
  - Coverage statistics: the distribution of ŵ, the Kish ESS of ŵ, max and quantiles.
- **Literature:** Nachum et al. NeurIPS 2019, arXiv:1906.04733; Liu et al. NeurIPS 2018 ("Breaking the curse of horizon").
- **Purpose:** An empirical proxy for single-policy concentrability that avoids the exponential-in-horizon variance of trajectory IS.
- **Requirements:** π_e, transitions (s,a,s′), and function approximators.
- **Computation:** Min-max optimisation, which is notoriously unstable. Hyperparameter sensitivity is reported in follow-up work (not verified here).
- **Strengths:** Behaviour-agnostic (DualDICE). Gives a coverage map over (s,a).
- **Limitations:**
  - Abstract-level reading.
  - Ratio estimates are least reliable exactly where coverage is poor.
  - Needs the Markov assumption, which the framework is still testing.
  - The infinite-horizon/stationary formulation may mismatch episodic ICU stays.
- **Applicability:** Continuous states. Discrete or continuous actions.
- **Timing:** T1 + T2 (trained ratio networks).
- **Validation:** OPE benchmarks in the original papers.
- **Recommendation:** Investigate. Optional advanced component; in the tabular case, D3 replaces it.
- **Category:** 3.

#### D5. Nearest-neighbour / density-based support estimation (OOD score)
- **Metric:**
  - For query (s, π_e(s)): the k-th nearest-neighbour distance in the (normalised or embedded) state-action space to the dataset, d_k(s,a). Thresholds are set at, e.g., the 95th percentile of in-sample held-out d_k.
  - Coverage = fraction of π_e's proposed (s,a) inside the support.
  - Alternatively, use kernel density or flow log-likelihood.
- **Literature:** Sun et al. ICML 2022, arXiv:2204.06507 (kNN OOD detection in deep feature space; image domain).
- **Purpose:** Model-free coverage check for continuous or high-dimensional (s,a).
- **Requirements:** π_e (or, at T0, held-out data to self-assess sparsity) and a distance metric or embedding.
- **Computation:** O(N log N) with KD-trees or approximate NN.
- **Strengths:** Nonparametric, simple, and the threshold is interpretable.
- **Limitations:**
  - Distance concentration in high dimension.
  - Metric choice is critical: mixed clinical feature types and missingness.
  - Validated for image OOD, not for RL coverage.
- **Applicability:** Continuous and mixed features, after scaling or embedding.
- **Timing:** T0 (data sparsity profile) or T1 (target-policy coverage).
- **Validation:** Image OOD benchmarks only.
- **Recommendation:** Adapt.
- **Category:** 2.

#### D6. Model-ensemble disagreement as a coverage proxy (MOPO / MOReL USAD)
- **Metric:**
  - u(s,a) = disagreement among M learned dynamics models: max pairwise prediction distance (MOReL's "unknown state-action detector") or predicted variance (MOPO).
  - Coverage = fraction of π_e-visited (s,a) with u < threshold.
- **Literature:** Kidambi et al. NeurIPS 2020, arXiv:2005.05951; Yu et al. NeurIPS 2020, arXiv:2005.13239 (abstracts/reviews; exact thresholds not read).
- **Purpose:** Identify regions where the data do not pin down the dynamics.
- **Requirements:** A trained dynamics ensemble.
- **Computation:** M model fits.
- **Strengths:**
  - Directly relevant to MDP adequacy, since it measures epistemic uncertainty of P̂.
  - Can be computed on observed (s,a) at T0-style use (in-sample uncertainty map).
- **Limitations:**
  - Ensemble disagreement is a heuristic for epistemic uncertainty, not a calibrated one.
  - In-distribution and out-of-distribution disagreement can be similar.
  - A NeurIPS reviewer criticised MOReL's binary penalty.
- **Applicability:** Continuous states. Discrete or continuous actions.
- **Timing:** T2 (dynamics models), but no policy optimisation.
- **Validation:** D4RL performance of the downstream algorithms, not direct coverage validation.
- **Recommendation:** Adapt. It fits well because the framework already fits transition models for Markov tests.
- **Category:** 2.

#### D7. Support-constraint divergences (BCQ generative model; BEAR MMD)
- **Metric:**
  - BCQ: likelihood of the action under a state-conditional generative model of π_β.
  - BEAR: MMD between π_e(·|s) and π_β(·|s) samples, used as a support-matching constraint.
  - As diagnostics: E_s[MMD(π_e(·|s), π̂_β(·|s))], or the generative-model likelihood of π_e's actions.
- **Literature:** Fujimoto et al. ICML 2019, arXiv:1812.02900; Kumar et al. NeurIPS 2019, arXiv:1906.00949.
- **Purpose:** Quantify how far π_e's actions leave the behaviour support (the "extrapolation error" cause).
- **Requirements:** π_e and a behaviour model.
- **Computation:** Moderate.
- **Strengths:** Directly targets the documented failure mechanism of offline RL.
- **Limitations:**
  - Designed as training constraints, not validated as standalone diagnostics.
  - The MMD value depends on the kernel bandwidth.
- **Applicability:** Continuous actions mainly. In discrete spaces it reduces to C1 evaluated at π_e's action.
- **Timing:** T1.
- **Validation:** Indirect (algorithm performance).
- **Recommendation:** Investigate. For discrete actions use the simpler π̂_β(π_e(s)|s).
- **Category:** 3.

#### D8. Bellman-Wasserstein Distance (BWD)
- **Metric (full text read):**
  - Fit a behavioural critic Q^β by SARSA-style TD on the dataset (Eq. 5).
  - Define the cost c((s,a),(s,a′)) = Q^β(s,a′) − ‖a′−a‖²₂ (Eq. 14), between dataset pairs and pairs whose actions come from a random policy (Gaussian) at dataset states.
  - BWD is the entropic dual OT objective with neural potentials g, f (Eqs. 15–16).
  - It measures how dissimilar the behaviour policy is from random behaviour in a value-aware way, i.e. dataset *quality / expertise*, not coverage.
- **Literature:** Asadulaev, Karray, Takáč 2025, arXiv:2510.12638 ("Expert or not? Assessing data quality in offline reinforcement learning").
- **Purpose:** Predict offline-RL performance before training. It correlates with an oracle aggregate score on D4RL MuJoCo.
- **Requirements:** Rewards, transitions, continuous actions (Euclidean action distance, Gaussian random reference), and three neural networks.
- **Computation:** About 10k updates, <10 min on a GPU (reported).
- **Strengths:** One of very few "pre-training dataset adequacy" scores for offline RL. Needs no target policy.
- **Limitations:**
  - Measures *expertise relative to random*, which is orthogonal to positivity and coverage. A near-deterministic expert dataset scores high yet has poor overlap.
  - Requires a trained critic, so it is T2 and reward-dependent; clinical reward design is itself contested (Luo et al. 2024).
  - The paper is internally inconsistent on the sign convention (Sec. 4 vs Table 1).
  - Not a proven metric.
  - Validated only on D4RL MuJoCo.
  - The random reference is ill-defined for discrete clinical action grids, and clinically meaningless.
  - Preprint, not peer-reviewed as far as found.
- **Applicability:** Continuous actions. Healthcare transfer is unvalidated.
- **Timing:** T0 with respect to policies, but T2 (trained critic).
- **Validation:** D4RL correlations only.
- **Recommendation:** Exclude from the core framework. Mention as related work on "dataset quality" and possibly as an exploratory experiment in a synthetic benchmark (EpiCare).
- **Category:** 4 for the healthcare core (3 if treated purely as exploratory).

### G. OPE reliability diagnostics

#### G1. Importance-weight effective sample size (Kish ESS; non-zero-weight count)
- **Metric:**
  - Trajectory weights ρ_i = Π_t π_e(a_t|s_t)/π̂_β(a_t|s_t).
  - Kish ESS = (Σρ_i)²/Σρ_i² (Kish 1965).
  - For deterministic π_e, also n_{ρ>0} = #trajectories with all actions matching (Gottesman et al. 2018).
  - Report per-decision ESS_t at each horizon t (Precup et al. 2000 per-decision IS).
- **Literature:**
  - Kish 1965 (book; formula verified via secondary sources).
  - Gottesman et al. 2018, arXiv:1805.12298 (full text read). In sepsis, 3855 test patients gave an average of **25** non-zero-weight patients for the learned policy, and only **167** even for the "most common physician action" deterministic policy. The non-zero fraction decays exponentially with horizon, and they recommend always examining the weight distribution.
  - Gottesman et al. 2019 Nature Medicine, doi:10.1038/s41591-018-0310-5.
  - Precup et al. ICML 2000.
  - Critique: Elvira, Martino & Robert 2022, doi:10.1111/insr.12500.
- **Purpose:** How many patients effectively inform the OPE estimate.
- **Requirements:** π_e and π̂_β.
- **Computation:** O(NT).
- **Strengths:** Cheap, interpretable in patient units, and widely recommended in healthcare RL.
- **Limitations:**
  - Elvira et al. argue the classical ESS approximation rests on assumptions that make it hard to justify.
  - High ESS does **not** imply unbiasedness: misspecified π̂_β, unmeasured confounding, or selection bias such as Gottesman's "only untreated low-risk patients retained" can all coexist with acceptable ESS.
  - Kish ESS is computed from the same heavy-tailed weights it tries to judge.
- **Applicability:** Discrete actions (continuous via density ratios).
- **Timing:** T1.
- **Validation:** Gottesman case study; general IS theory.
- **Recommendation:** Implement (core). Report ESS/N, ESS_t curves vs horizon, and the characteristics (e.g. length of stay, severity) of the retained patients.
- **Category:** 1.

#### G2. Pareto-smoothed IS k̂ diagnostic
- **Metric:**
  - Fit a generalised Pareto distribution to the upper tail of the importance ratios. The shape estimate k̂ indicates the number of finite moments and the finite-sample convergence rate.
  - The paper gives sample-size-dependent thresholds; the widely used rule is k̂ > 0.7 = unreliable (exact thresholds to be checked in the full text).
- **Literature:** Vehtari et al. JMLR 25(72) 2024, arXiv:1507.02646 (abstract-level). Implemented in ArviZ, posterior, and Pyro.
- **Purpose:** A tail-based reliability diagnostic that is more informative than ESS for heavy-tailed weights.
- **Requirements:** Weights from G1 (trajectory or per-decision).
- **Computation:** Cheap (GPD fit on the top M weights).
- **Strengths:**
  - Theoretically grounded.
  - Off-the-shelf implementations.
  - Gives both a diagnostic and stabilised (smoothed) weights.
- **Limitations:**
  - Developed for Bayesian computation, not RL OPE. No RL/healthcare validation was found.
  - Needs enough non-zero weights to fit the tail, which fails when n_{ρ>0} is tiny.
  - Says nothing about bias from π̂_β error or confounding.
- **Applicability:** Any weights.
- **Timing:** T1.
- **Validation:** Extensive in Bayesian contexts.
- **Recommendation:** Implement (adapt to RL weights). This is a strong, novel-in-context contribution for the thesis.
- **Category:** 2.

#### G3. High-confidence OPE lower bounds
- **Metric:** A (1−δ) lower confidence bound on V^{π_e} from importance-weighted returns, using concentration inequalities (Thomas et al. 2015).
- **Literature:** Thomas, Theocharous & Ghavamzadeh AAAI 2015, doi:10.1609/aaai.v29i1.9541 (abstract).
- **Purpose:** Turn OPE into a safety statement: is the lower bound above the behaviour policy's value?
- **Requirements:** π_e, π̂_β, bounded returns.
- **Computation:** Cheap.
- **Strengths:** Frequentist guarantee (under correct π_β and no confounding).
- **Limitations:**
  - Bounds can be very loose with small ESS. In the authors' marketing example the improvement was "guaranteed but minor" (secondary summary).
  - Guarantees assume known π_β.
- **Applicability:** Discrete or continuous actions.
- **Timing:** T1.
- **Validation:** Marketing data in the paper.
- **Recommendation:** Adapt. Report the CI width as a reliability measure.
- **Category:** 2.

#### G4. Bootstrap confidence intervals for OPE (FQE bootstrap; model-based / WDR bootstrap)
- **Metric:**
  - The bootstrap distribution of V̂^{π_e}, resampling *trajectories* (patients).
  - Hao et al. show that bootstrapped FQE is asymptotically efficient and distributionally consistent (abstract).
  - Hanna et al. propose MB-Bootstrap and WDR-Bootstrap lower bounds.
- **Literature:** Hao et al. ICML 2021, PMLR 139:4074–4084, arXiv:2102.03607; Hanna, Stone & Niekum AAAI 2017 (also AAMAS 2017), doi:10.1609/aaai.v31i1.11123, arXiv:1606.06126.
- **Purpose:** Quantify the statistical uncertainty of OPE estimates.
- **Requirements:** π_e and repeated FQE or model fits.
- **Computation:** B × (FQE cost); Hao et al.'s subsampling reduces this.
- **Strengths:** Practical, and FQE was the best-ranking OPE method in Tang & Wiens (abstract).
- **Limitations:**
  - Bootstrap CIs capture variance, not bias from poor coverage or confounding. Under extrapolation, FQE can be confidently wrong.
  - The asymptotics may not hold at clinical sample sizes.
- **Applicability:** All settings.
- **Timing:** T1 + T2.
- **Validation:** Classic control environments (Hao); small domains (Hanna).
- **Recommendation:** Adapt, as a post-hoc reliability layer, not a pre-training gate.
- **Category:** 2.

#### G5. Cross-estimator agreement and benchmark-calibrated reliability
- **Metric:**
  - Dispersion of V̂^{π_e} across estimators: WIS, PDIS, WDR (Thomas & Brunskill 2016), FQE, model-based.
  - Rank agreement (e.g. Spearman / Kendall τ) of candidate policies across estimators.
  - Optionally, agreement with ground truth in a simulator-calibrated setting.
- **Literature:**
  - Voloshin et al. NeurIPS D&B 2021 (COBS), arXiv:1911.06854.
  - Fu et al. ICLR 2021 (DOPE), arXiv:2103.16596.
  - Tang & Wiens MLHC 2021, PMLR 149, arXiv:2107.11003: FQE gave the best validation ranking for sepsis, at high computational cost.
  - Luo et al. ICML 2024, PMLR 235:33432–33465, arXiv:2405.18556: more than 17,000 evaluation experiments, in which RL can be beaten by random baselines depending on evaluation and reward choices.
  - Hargrave et al. NeurIPS D&B 2024 (EpiCare): several OPE methods common in medical RL performed poorly, and offline RL struggles in low-to-moderate data regimes.
  - Thomas & Brunskill ICML 2016, arXiv:1604.00923.
- **Purpose:** Empirical evidence that OPE is unreliable in healthcare-like settings. Estimator disagreement is a cheap warning sign.
- **Requirements:** π_e and several estimators (some T2).
- **Computation:** Moderate.
- **Strengths:** Strong recent evidence base. EpiCare provides a *ground-truth simulator* in which the framework's diagnostics can be validated against true OPE error.
- **Limitations:**
  - Agreement does not imply correctness: estimators share π̂_β and data biases.
  - The benchmarks are simulations.
- **Applicability:** All settings.
- **Timing:** T1 + T2.
- **Validation:** The benchmarks above.
- **Recommendation:** Implement disagreement as a diagnostic. **Use EpiCare (and/or a sepsis simulator) as the validation bed** for the whole framework, testing whether C/D/G diagnostics predict true OPE error.
- **Category:** 1 (benchmark evidence); 2 (disagreement metric).

#### G6. Influence-function diagnostics for OPE
- **Metric:** The influence of each transition or trajectory on the FQE estimate (kernel or linear FQE). Flag high-influence observations; summarise as the share of the estimate driven by the top-k patients.
- **Literature:** Gottesman et al. ICML 2020, arXiv:2002.03478 (abstract-level). On ICU data it surfaced measurement errors and temporal misalignment with an intensivist.
- **Purpose:** Detect estimates resting on a handful of patients, and support clinician review.
- **Requirements:** π_e and FQE with a kernel or linear function class.
- **Computation:** Moderate (closed form for linear/kernel models).
- **Strengths:** Interpretable, clinically actionable, healthcare-validated qualitatively.
- **Limitations:**
  - Restricted to function classes with tractable influence.
  - Detects fragility, not bias in well-supported but confounded regions.
- **Applicability:** Continuous states. Discrete actions.
- **Timing:** T1 + T2.
- **Validation:** Medical simulations and ICU data.
- **Recommendation:** Adapt (optional). A trajectory-level analogue for IS is trivial: the share of Σρ from the top-k weights.
- **Category:** 2.

#### G7. Sensitivity analysis for unobserved confounding in sequential OPE
- **Metric:** Worst-case bounds on V^{π_e} when unobserved confounders affect a single decision, under a sensitivity parameter Γ. Report the Γ at which conclusions flip.
- **Literature:** Namkoong et al. NeurIPS 2020 (abstract and reviews).
- **Purpose:** Addresses the main blind spot of all positivity and coverage diagnostics: no-unmeasured-confounding.
- **Requirements:** π_e and a sensitivity model.
- **Computation:** A loss-minimisation procedure (consistency proven).
- **Strengths:** Makes the untestable assumption explicit and quantitative. Demonstrated on simulated sepsis.
- **Limitations:**
  - The single-decision confounding assumption.
  - The bounds can be wide.
  - Implementation complexity.
- **Applicability:** Discrete actions.
- **Timing:** T1 (+ T2).
- **Validation:** Simulations.
- **Recommendation:** Investigate. At least discuss it as the complement that positivity diagnostics cannot replace.
- **Category:** 3.

#### G8. Horizon-dependence profile (per-decision vs trajectory weights)
- **Metric:**
  - ESS_t and k̂_t as functions of truncation horizon t, for trajectory weights versus per-decision weights (PDIS).
  - The "effective horizon" t* is where ESS_t/N < τ.
- **Literature:** Precup et al. 2000; Liu et al. 2018 (IS variance can be unbounded as horizon grows); Gottesman et al. 2018 (exponential decay of non-zero-weight sequences).
- **Purpose:** Find the horizon/time-step granularity at which OPE is still feasible. This informs MDP design (e.g. 4-h vs 1-h steps).
- **Requirements / computation:** As G1/G2. Cheap.
- **Strengths:** Directly links MDP formulation choices (time discretisation) to OPE feasibility.
- **Limitations:**
  - Per-decision weighting reduces variance but does not create support.
  - Truncation introduces bias.
- **Applicability:** Episodic data.
- **Timing:** T1.
- **Validation:** Theory plus the Gottesman case study.
- **Recommendation:** Implement, as part of G1/G2 reporting.
- **Category:** 2.

### Summary table

| ID | Method | Timing | Cat. | Rec. |
|---|---|---|---|---|
| C1 | Propensity / π̂_β distribution, min-prob, IPW weights | T0 | 1 | Implement |
| C2 | Crump trimming / fraction-retained curve | T0 | 2 | Adapt |
| C3 | ETA.Bias parametric bootstrap | T1/T2 | 2 | Adapt |
| C4 | sPoRT sequential positivity trees | T0/T1 | 2 | Investigate |
| C5 | D'Amour high-dimensional overlap (theory; AUROC proxy) | T0 | 1/3 | Motivation |
| C6 | GPS / conditional density overlap (continuous) | T0 | 2 | Adapt |
| C7 | π̂_β calibration | T0 | 1 | Implement (gate) |
| C8 | Action entropy / effective number of actions | T0 | 3 | Investigate |
| D1 | Count-based coverage | T0 | 2 | Implement (weighted) |
| D2 | All-policy concentrability | — | 4 | Exclude (theory only) |
| D3 | Single-policy concentrability (plug-in, quantiles) | T1(+T2) | 2 | Adapt |
| D4 | DualDICE-type ratio estimation | T1+T2 | 3 | Investigate |
| D5 | kNN / density support | T0/T1 | 2 | Adapt |
| D6 | Dynamics ensemble disagreement | T2 | 2 | Adapt |
| D7 | Support divergences (MMD / generative likelihood) | T1 | 3 | Investigate |
| D8 | Bellman-Wasserstein Distance | T2 | 4 (3) | Exclude from core |
| G1 | Kish ESS, non-zero-weight count | T1 | 1 | Implement |
| G2 | PSIS k̂ | T1 | 2 | Implement |
| G3 | HCOPE lower bound | T1 | 2 | Adapt |
| G4 | Bootstrap FQE / WDR CIs | T1+T2 | 2 | Adapt (post-hoc) |
| G5 | Cross-estimator agreement; EpiCare validation | T1+T2 | 1/2 | Implement |
| G6 | Influence diagnostics | T1+T2 | 2 | Adapt (optional) |
| G7 | Confounding sensitivity (Γ) | T1 | 3 | Investigate |
| G8 | Horizon profile of ESS / k̂ | T1 | 2 | Implement |

---

## 3. Assessment of earlier candidate metrics

### AOI = E_s[(1/|A|) Σ_a 1(π̂_β(a|s) > ε)]
- **What it is:** The average fraction of actions with non-negligible behaviour probability, i.e. a thresholded, state-averaged version of C1. Closely related to C8 (exp-entropy is a smooth analogue).
- **Literature support:** The underlying idea (propensity bounded away from 0, ε-positivity) is established (Petersen 2012; Crump 2009). The *specific* aggregate "AOI" was not found in the literature, so it is our construct.
- **Problems:**
  1. Dividing by |A| penalises large action spaces even when clinically only 2–3 actions are relevant. Normalise over clinically feasible actions A_feas(s) instead.
  2. Averaging over states hides the failing tail. Positivity is a *worst-case / where* property, so report the distribution and the share of states with fewer than 2 supported actions.
  3. ε is arbitrary. Report a curve over ε, as in Crump's α.
  4. It ignores *which* actions π_e needs. Overlap for the target's actions (D7/G1) matters more than overall diversity.
  5. Its accuracy depends on π̂_β calibration (C7).
- **Verdict:** It adds little beyond C1 + C8 as a headline number. Keep it as a descriptive statistic only if redefined over feasible actions, reported as a distribution and ε-curve, and paired with target-specific overlap. **Recommendation: adapt (merge into C1/C8). Category 2.**

### ESAC = |{(s,a): N(s,a) ≥ N_min}| / (|S_cluster|·|A|)
- **What it is:** D1 with a threshold.
- **Literature support:** Count-based coverage is implicit in tabular offline-RL analysis. The modern theory (Rashidinejad 2021; Xie 2021) says uniform coverage of all (s,a) is **not required**: only coverage of the comparator/target policy's occupancy matters. Uniform coverage corresponds to the outdated all-policy concentrability (D2).
- **Problems:**
  1. It is unweighted, so rare states count as much as common ones. Visitation-weight it: Σ_{s,a} μ(s) 1(N(s,a) ≥ N_min)/|A_feas(s)|.
  2. The denominator includes infeasible and contraindicated actions.
  3. It depends strongly on k (number of clusters) and N_min, so report a sensitivity grid.
  4. It says nothing about the *transition* estimation quality per cell (N(s,a) vs the number of successor states).
- **Verdict:** It is a reasonable, transparent tabular statistic but theoretically misaligned as a feasibility *gate*. **Recommendation: adapt.** Keep a weighted, feasibility-restricted version as a descriptive metric, and add D3 (target-specific coverage quantiles) as the theoretically aligned counterpart. Category 2.

### "Temporal observation reliability" (completeness only)
- **What it is:** The fraction of non-missing measurements per time step or trajectory.
- **Literature support (within this scope):** No paper in scope proposes completeness as an OPE or coverage reliability measure. The closest relevant evidence is Gottesman et al. 2020: influence analysis on ICU data surfaced *measurement errors and temporal misalignments*. These are reliability problems that completeness does not detect, because a fully observed but misaligned or mis-recorded value is complete.
- **Problems:**
  1. It does not tell you whether values are measured (vs imputed or carried forward), *when* they are measured (irregular sampling, informative missingness: sicker patients are measured more often), or whether timestamps align with actions. In ICU data, measurement frequency is itself treatment- and severity-dependent, which may confound.
  2. It is unrelated to the C/D/G constructs, so it is a data-quality metric, not an adequacy metric for overlap, coverage or OPE.
- **Verdict:** Completeness alone adds little for C/D/G. It belongs in a separate data-quality / observation-process module and should be extended with measurement frequency, staleness (time since last measurement at decision time), imputation share, and action-observation temporal alignment checks. **Recommendation: exclude from C/D/G; refer to the data-quality / partial-observability module.** Category 4 for this scope.

---

## 4. References

### Verified (existence confirmed via web search/fetch; reading depth noted)

1. Petersen, M. L., Porter, K. E., Gruber, S., Wang, Y., van der Laan, M. J. (2012). Diagnosing and responding to violations in the positivity assumption. *Statistical Methods in Medical Research* 21(1):31–54. doi:10.1177/0962280210386207. (Full text read, PMC.)
2. D'Amour, A., Ding, P., Feller, A., Lei, L., Sekhon, J. (2021). Overlap in observational studies with high-dimensional covariates. *Journal of Econometrics* 221(2):644–654. doi:10.1016/j.jeconom.2019.10.014; arXiv:1711.02582. (Abstract.)
3. Crump, R. K., Hotz, V. J., Imbens, G. W., Mitnik, O. A. (2009). Dealing with limited overlap in estimation of average treatment effects. *Biometrika* 96(1):187–199. doi:10.1093/biomet/asn055. (Secondary summaries.)
4. Li, F., Morgan, K. L., Zaslavsky, A. M. (2018). Balancing covariates via propensity score weighting. *JASA* 113(521):390–400. doi:10.1080/01621459.2016.1260466. (Abstract.)
5. Hirano, K., Imbens, G. W. (2004). The propensity score with continuous treatments. In Gelman & Meng (eds.), *Applied Bayesian Modeling and Causal Inference from Incomplete-Data Perspectives*, Wiley, pp. 73–84. (Secondary sources; chapter not opened.)
6. Chatton, A., Schomaker, M., Luque-Fernandez, M.-A., Platt, R. W., Schnitzer, M. E. Regression trees for nonparametric diagnostics of sequential positivity violations in longitudinal causal inference (later titled "Is checking for sequential positivity violations getting you down? Try sPoRT!"). arXiv:2412.10245. (Abstract/snippets. The *Epidemiology* DOI 10.1097/EDE.0000000000001902 comes from a secondary listing and should be checked.)
7. Raghu, A., Gottesman, O., Liu, Y., Komorowski, M., Faisal, A., Doshi-Velez, F., Brunskill, E. (2018). Behaviour policy estimation in off-policy policy evaluation: calibration matters. ICML 2018 Workshop (CausalML). arXiv:1807.01066. (Abstract.)
8. Vehtari, A., Simpson, D., Gelman, A., Yao, Y., Gabry, J. (2024). Pareto smoothed importance sampling. *JMLR* 25(72):1–58. arXiv:1507.02646. (Abstract.)
9. Kish, L. (1965). *Survey Sampling*. Wiley. (Formula confirmed via secondary sources.)
10. Elvira, V., Martino, L., Robert, C. P. (2022). Rethinking the effective sample size. *International Statistical Review* 90(3):525–550. doi:10.1111/insr.12500; arXiv:1809.04129. (Abstract.)
11. Precup, D., Sutton, R. S., Singh, S. (2000). Eligibility traces for off-policy policy evaluation. *ICML 2000*, pp. 759–766. (Abstract.)
12. Munos, R. (2003). Error bounds for approximate policy iteration. *ICML 2003*, pp. 560–567. (Bibliographic only.)
13. Munos, R., Szepesvári, Cs. (2008). Finite-time bounds for fitted value iteration. *JMLR* 9:815–857. (Abstract.)
14. Chen, J., Jiang, N. (2019). Information-theoretic considerations in batch reinforcement learning. *ICML 2019*, PMLR 97:1042–1051. arXiv:1905.00360. (Abstract.)
15. Rashidinejad, P., Zhu, B., Ma, C., Jiao, J., Russell, S. (2021). Bridging offline reinforcement learning and imitation learning: a tale of pessimism. *NeurIPS 2021*; *IEEE Trans. Inf. Theory* 2022. arXiv:2103.12021. (Abstract + secondary notes.)
16. Xie, T., Cheng, C.-A., Jiang, N., Mineiro, P., Agarwal, A. (2021). Bellman-consistent pessimism for offline reinforcement learning. *NeurIPS 2021* (oral). arXiv:2106.06926. (Abstract.)
17. Nachum, O., Chow, Y., Dai, B., Li, L. (2019). DualDICE: Behavior-agnostic estimation of discounted stationary distribution corrections. *NeurIPS 2019*. arXiv:1906.04733. (Abstract.)
18. Liu, Q., Li, L., Tang, Z., Zhou, D. (2018). Breaking the curse of horizon: infinite-horizon off-policy estimation. *NeurIPS 2018*. (Abstract.)
19. Yu, T., Thomas, G., Yu, L., Ermon, S., Zou, J., Levine, S., Finn, C., Ma, T. (2020). MOPO: Model-based offline policy optimization. *NeurIPS 2020*. arXiv:2005.13239. (Abstract.)
20. Kidambi, R., Rajeswaran, A., Netrapalli, P., Joachims, T. (2020). MOReL: Model-based offline reinforcement learning. *NeurIPS 2020*. arXiv:2005.05951. (Abstract + reviews.)
21. Fujimoto, S., Meger, D., Precup, D. (2019). Off-policy deep reinforcement learning without exploration. *ICML 2019*, PMLR 97:2052–2062. arXiv:1812.02900. (Abstract.)
22. Kumar, A., Fu, J., Soh, M., Tucker, G., Levine, S. (2019). Stabilizing off-policy Q-learning via bootstrapping error reduction. *NeurIPS 2019*. arXiv:1906.00949. (Abstract + reviews.)
23. Sun, Y., Ming, Y., Zhu, X., Li, Y. (2022). Out-of-distribution detection with deep nearest neighbors. *ICML 2022*, PMLR 162:20827–20840. arXiv:2204.06507. (Abstract.)
24. Asadulaev, A., Karray, F., Takáč, M. (2025). Expert or not? Assessing data quality in offline reinforcement learning. arXiv:2510.12638. (Full text via HTML; preprint.)
25. Gottesman, O., et al. (2018). Evaluating reinforcement learning algorithms in observational health settings. arXiv:1805.12298. (Full text read.)
26. Gottesman, O., Johansson, F., Komorowski, M., Faisal, A., Sontag, D., Doshi-Velez, F., Celi, L. A. (2019). Guidelines for reinforcement learning in healthcare. *Nature Medicine* 25:16–18. doi:10.1038/s41591-018-0310-5. (Abstract.)
27. Gottesman, O., Futoma, J., Liu, Y., Parbhoo, S., Celi, L. A., Brunskill, E., Doshi-Velez, F. (2020). Interpretable off-policy evaluation in reinforcement learning by highlighting influential transitions. *ICML 2020*. arXiv:2002.03478. (Abstract.)
28. Thomas, P. S., Theocharous, G., Ghavamzadeh, M. (2015). High-confidence off-policy evaluation. *AAAI 2015*, 29(1):3000–3006. doi:10.1609/aaai.v29i1.9541. (Abstract.)
29. Thomas, P. S., Brunskill, E. (2016). Data-efficient off-policy policy evaluation for reinforcement learning. *ICML 2016*. arXiv:1604.00923. (Abstract.)
30. Hanna, J. P., Stone, P., Niekum, S. (2017). Bootstrapping with models: confidence intervals for off-policy evaluation. *AAAI 2017* (short) / *AAMAS 2017*. doi:10.1609/aaai.v31i1.11123; arXiv:1606.06126. (Abstract; venue ambiguity noted.)
31. Hao, B., Ji, X., Duan, Y., Lu, H., Szepesvári, Cs., Wang, M. (2021). Bootstrapping fitted Q-evaluation for off-policy inference. *ICML 2021*, PMLR 139:4074–4084. arXiv:2102.03607. (Abstract.)
32. Voloshin, C., Le, H. M., Jiang, N., Yue, Y. (2021). Empirical study of off-policy policy evaluation for reinforcement learning. *NeurIPS Datasets & Benchmarks 2021*. arXiv:1911.06854. (Abstract.)
33. Fu, J., Norouzi, M., Nachum, O., Tucker, G., et al. (2021). Benchmarks for deep off-policy evaluation. *ICLR 2021*. arXiv:2103.16596. (Abstract.)
34. Tang, S., Wiens, J. (2021). Model selection for offline reinforcement learning: practical considerations for healthcare settings. *MLHC 2021*, PMLR 149. arXiv:2107.11003. (Abstract.)
35. Luo, Z., Pan, Y., Watkinson, P., Zhu, T. (2024). Position: Reinforcement learning in dynamic treatment regimes needs critical reexamination. *ICML 2024*, PMLR 235:33432–33465. arXiv:2405.18556. (Abstract.)
36. Hargrave, M., Spaeth, A., Grosenick, L. (2024). EpiCare: A reinforcement learning benchmark for dynamic treatment regimes. *NeurIPS 2024 Datasets & Benchmarks*. (Abstract.)
37. Namkoong, H., Keramati, R., Yadlowsky, S., Brunskill, E. (2020). Off-policy policy evaluation for sequential decisions under unobserved confounding. *NeurIPS 2020*. (Abstract + reviews.)

### Unverified leads (not confirmed or not opened; check before citing)
- **Munos (2007)**, "Performance bounds in Lp-norm for approximate value iteration" (SIAM J. Control Optim.). Mentioned in the brief; not searched.
- **Kong (1992)** technical report on IS effective sample size. Known only via Elvira et al.
- **PoRT** (single-time-point Positivity Regression Trees; Danelian et al.?). Seen only as a HAL title, "Identification of in-sample positivity violations using regression trees: The PoRT algorithm"; authors and venue not confirmed.
- **McClean & Díaz**, "Propensity score weighting across counterfactual worlds: longitudinal effects under positivity violations", arXiv:2507.10774. Seen in search results only.
- **"Non-overlap ratio"** positivity diagnostic. Mentioned in a search snippet; source not identified.
- **Farahmand et al. (2010)**, refined concentrability-based API/AVI bounds (NeurIPS 2010). Seen only as a citation in other papers.
- **Jiang & Li (2016)**, doubly robust OPE (ICML). Referenced by Thomas & Brunskill; not separately verified.
- **Kallus & Uehara / Uehara et al.**, double RL and OPE reviews. Not searched.
- **Applications of PSIS k̂ to RL importance weights.** No published RL/healthcare application was found; treat as a gap or opportunity, not an established practice.
