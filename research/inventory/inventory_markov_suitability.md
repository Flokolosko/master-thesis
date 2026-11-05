# Inventory: Sequential-decision suitability, temporal abstraction, Markov property & state sufficiency

Scope (A) sequential decision problem suitability and temporal abstraction; (B) Markov property / state sufficiency diagnostics.
Thesis: *A Quantitative Framework for Assessing the Empirical Adequacy of MDP Models in Healthcare* (TU Wien, Data Science MSc).
Status: draft literature inventory. Compiled 2026-11-05.

---

## 1. Search log

| # | Query / source | Result used for |
|---|---|---|
| 1 | WebSearch: Shi et al. "Does the MDP fit the data" (PMLR, arXiv 2002.01751, abstract) | M1. Author list corrected: Shi, Wan, Song, Lu, Leng. Chernozhukov is **not** an author |
| 2 | WebSearch: Chen & Hong 2012 Markov test, Econometric Theory | M3, plus the caveat from Zhou et al. 2023 |
| 3 | WebFetch arXiv 2305.19244 (abstract) | M2 |
| 4 | WebSearch: Shah & Peters GCM (Annals of Statistics) | M4, plus the no-free-lunch result for CI testing |
| 5 | WebSearch: KCI (Zhang et al. UAI 2011); Fukumizu et al. NIPS 2007 | M5 |
| 6 | WebSearch: Candès et al. 2018 model-X / CRT; Berrett et al. 2020 CPT | M6 |
| 7 | WebSearch: KSG 2004; Runge 2018 CMIknn; Mukherjee et al. CCMI | M7 |
| 8 | WebSearch: Katz 1981; Csiszár & Shields 2000; Papapetrou & Kugiumtzis (arXiv 1301.0148) | M9, M10 |
| 9 | WebSearch: Titman & Putter 2022; Rodríguez-Girondo & de Uña-Álvarez 2012 | M11 |
| 10 | WebSearch: Subramanian et al. JMLR 2022; Killian et al. ML4H 2020; Ni et al. ICLR 2024 | M8, M12 |
| 11 | WebSearch: Li, Walsh & Littman 2006; Abel et al. ICML 2016; Givan et al. 2003; Ferns et al. 2004; Castro AAAI 2020; Zhang et al. ICLR 2021; Gelada et al. ICML 2019 | M13, M14 |
| 12 | WebSearch: Allen et al. λ-discrepancy (NeurIPS 2024, arXiv 2407.07333) | M15 |
| 13 | WebSearch: Littman, Sutton & Singh PSR (NIPS 2001); Bennett & Kallus proximal RL | M18, M19 |
| 14 | WebSearch: Tallec et al. ICML 2019; Sutton, Precup & Singh 1999; Bradtke & Duff 1994; Doya 2000; Fatemi et al. CHIL 2022; WebFetch arXiv 2511.20913 (Sun & Tang) | M16, M17 |
| 15 | WebSearch: Komorowski 2018; Raghu 2017; Gottesman 2019; Steimle & Denton 2017; Siebert et al. ISPOR-SMDM 2012 (WebFetch of the ISPOR landing page) | Section 3 and context |

Depth of reading: unless a method entry says otherwise, I checked only **abstracts, landing pages and search-engine summaries**. I did not read full texts. Method descriptions stay at that level. Where an entry adds my own interpretation or a proposed adaptation for the thesis, it is labelled "(proposal)".

### Common notation

- Observed trajectory: $\tau_i=(X_{i0},A_{i0},R_{i0},\dots,X_{iT_i})$.
- Candidate state map: $S_t=\phi(H_t)$, where $H_t=(X_{0:t},A_{0:t-1},R_{0:t-1})$.
- Time grid: $\Delta t$. Action map: $\alpha$.
- **Markov null (state sufficiency):** $H_0:\ (S_{t+1},R_t)\ \perp\ H_{t-1}\mid (S_t,A_t)$.
- **k-th order null:** $H_0^{(k)}$ conditions on $(S_{t-k+1:t},A_{t-k+1:t})$.

### General caveats that apply to every method in Section 2

1. **No uniformly powerful CI test exists** (Shah & Peters 2020). A non-rejection is never evidence that the process is Markov. At most it shows "no detectable departure at this n against the alternatives this test is sensitive to". Report power or effect sizes, not only p-values.
2. All diagnostics are **predictive/statistical**. They test whether *observed* data satisfy conditional independence. They cannot establish causal or physiological Markovianity, and they cannot detect unobserved confounding that leaves no trace in the observed history (Section 3).
3. Tests of $S_{t+1}\perp H_{t-1}\mid S_t,A_t$ are conditional on the **behaviour policy**. If clinicians act on information missing from $S_t$, then $A_t$ carries history information. That dependence is part of the non-Markovianity and is also a confounding problem for later RL.
4. Clinical data have panel structure (many short episodes, irregular lengths). Asymptotics in $n$ (patients) are more relevant than asymptotics in $T$.

---

## 2. Method inventory

Category key: **1** established, directly applicable; **2** established, needs adaptation; **3** research candidate; **4** unsuitable as a pre-training diagnostic.

### M1. Forward-Backward Learning Markov test (Shi et al. 2020)

- **Metric:** Test of $H_0$ (and of $H_0^{(k)}$ for high-order MDPs) for sequential decision data. The p-value comes from a doubly-robust statistic.
- **Literature:** Shi, Wan, Song, Lu & Leng, ICML 2020, PMLR 119:8807–8817, arXiv:2002.01751.
- **Purpose:** Markov property of the state map $\phi$. Applied sequentially for $k=1,2,\dots$ it supports **order/lag selection** (the abstract mentions high-order MDPs and POMDPs).
- **Requirements:** Trajectories $(S_t,A_t,S_{t+1})$. The abstract says the test needs no parametric form for the joint distribution and claims validity as either the number of trajectories or the horizon grows.
- **Computation:** "Forward-backward learning". Per the abstract, this is a doubly-robust estimating equation with ML nuisance estimators. Details (from memory, unverified: conditional characteristic functions with cross-fitting) must be checked in the full text.
- **Complexity:** Dominated by fitting the nuisance learners (forward and backward models) under cross-fitting. Not quantified here.
- **Strengths:** The only test I found that targets exactly the MDP/RL null. Nonparametric. Handles high-dimensional states. Demonstrated on mobile-health data (OhioT1DM-type).
- **Limitations:** Power depends on nuisance-learner quality. The general CI-hardness result applies. Discrete or mixed state/action encodings and short ICU stays may need care. Code availability not verified.
- **Applicability:** Continuous states claimed. Discrete actions in the examples (to verify).
- **Timing:** Truly pre-training. No target policy needed.
- **Validation:** Theory (size/power claims in the abstract) and simulations, plus one mHealth application.
- **Recommendation:** **Implement / adapt.** Use it as the primary formal test. Run it on each candidate $(\phi,\Delta t)$ combination, applying it sequentially for $k$.
- **Category:** 2.

### M2. Deep conditional generative Markov test with order selection (Zhou, Shi, Li & Yao 2023)

- **Metric:** Doubly-robust test of the Markov property for high-dimensional time series, applied sequentially to select the Markov order.
- **Literature:** arXiv:2305.19244 (preprint; no venue stated on arXiv).
- **Purpose:** Markov order / lag selection.
- **Requirements:** Time-series data (no actions in the formulation). The abstract mentions sample splitting with cross-fitting.
- **Computation:** Deep conditional generative models estimate the conditional densities, which feed a doubly-robust statistic.
- **Complexity:** High, because deep generative models must be trained per fold and per order.
- **Strengths:** Asymptotic type-I control and power → 1 are claimed. The authors explicitly position the method against Chen & Hong 2012's size problems.
- **Limitations:** Designed for a single (long) time series, not for many short controlled episodes. Treating actions as covariates would be my own adaptation. The paper is only a preprint.
- **Applicability:** Continuous, high-dimensional.
- **Timing:** Pre-training.
- **Validation:** Simulations plus three data applications (per the abstract).
- **Recommendation:** **Investigate.** Use as a methodological reference for order selection, not as the main tool.
- **Category:** 3.

### M3. Conditional-characteristic-function Markov test (Chen & Hong 2012)

- **Metric:** Frequency-domain test of the Markov property using conditional characteristic functions over many lags.
- **Literature:** Chen & Hong, *Econometric Theory* 28(1):130–178, 2012, doi:10.1017/S0266466611000065.
- **Purpose:** Markov property (univariate or multivariate, discrete or continuous).
- **Requirements:** A single stationary time series. Smoothed nonparametric bootstrap.
- **Complexity:** Kernel smoothing plus bootstrap, so expensive in high dimension.
- **Strengths:** Classical. Has power against many non-Markov alternatives (per summaries).
- **Limitations:** Assumes a single stationary series, with no actions and no panel structure. Zhou et al. (2023) report it **fails to control type-I error** in their simulations. Kernel smoothing suffers from the curse of dimensionality.
- **Applicability:** Low-dimensional states.
- **Timing:** Pre-training.
- **Validation:** Simulations and financial applications (per summaries).
- **Recommendation:** **Exclude** from implementation. Cite as background only.
- **Category:** 4 (for this setting).

### M4. Generalised Covariance Measure (GCM) test of state sufficiency

- **Metric:** For a scalar $Y$ (e.g. one coordinate of $S_{t+1}$, or $R_t$), a scalar history feature $W=g(H_{t-1})$, and $Z=(S_t,A_t)$:
  $T=\sqrt{n}\,\bar{\rho}/\hat\sigma$ with $\rho_i=(Y_i-\hat f(Z_i))(W_i-\hat g(Z_i))$. $T$ is asymptotically $N(0,1)$ under $H_0$.
- **Literature:** Shah & Peters, *Annals of Statistics* 48(3):1514–1538, 2020, doi:10.1214/19-AOS1857, arXiv:1804.07203. R package `GeneralisedCovarianceMeasure`.
- **Purpose:** Markov property / whether the history adds information beyond $S_t$ (proposal: one test per state coordinate × lagged feature, with multiplicity control).
- **Requirements:** Two regressions, which must estimate the conditional means at a slow rate. That is the paper's stated validity condition.
- **Computation:** Fit $\hat f,\hat g$ with any ML regressor (e.g. gradient boosting), using cross-fitting (proposal) and patient-level splits.
- **Complexity:** Two regressions per tested pair. Cheap.
- **Strengths:** Simple and interpretable. Validity rests on regression quality, not on a density model. Scales to large $Z$. Also gives a key **negative result** that frames the whole framework (no CI test has power against all alternatives).
- **Limitations:** It only detects conditional **covariance**, so it has no power against dependence that leaves residual covariance at zero. Per-coordinate testing inflates multiplicity. A multivariate extension is mentioned in the literature but I have not checked it. Within-patient dependence needs patient-level sample splitting or clustering.
- **Applicability:** Continuous or discrete $Y$, $W$. Any $Z$.
- **Timing:** Pre-training.
- **Validation:** Theory in the paper. Widely used CI-test baseline.
- **Recommendation:** **Implement.** Use as the cheap, robust workhorse test and complement it with M1.
- **Category:** 2.

### M5. Kernel conditional independence tests (KCI; normalised conditional cross-covariance)

- **Metric:** Hilbert–Schmidt norm of a (normalised) conditional cross-covariance operator. The KCI statistic has a derived null distribution.
- **Literature:** Zhang, Peters, Janzing & Schölkopf, UAI 2011, arXiv:1202.3775. Fukumizu, Gretton, Sun & Schölkopf, NIPS 2007 (pp. 489–496).
- **Purpose:** Omnibus Markov / CI test for $H_0$.
- **Requirements:** Kernel and bandwidth choice. Moderate n.
- **Computation:** Kernel matrices with an analytic or gamma null approximation. Implementations exist in `causal-learn` and `hyppo`.
- **Complexity:** $O(n^2)$ memory and $O(n^3)$ time for naive implementations. Infeasible for MIMIC-scale transition counts without subsampling or approximation.
- **Strengths:** Sensitive to nonlinear dependence, not just covariance. Per the abstract, it performs well with large conditioning sets at modest n.
- **Limitations:** Calibration issues reported for non-smooth densities (Runge 2018). Kernel choice affects results. Poor scaling.
- **Applicability:** Continuous (and encoded discrete) data.
- **Timing:** Pre-training.
- **Validation:** Causal-discovery benchmarks.
- **Recommendation:** **Investigate** as a robustness check on subsamples.
- **Category:** 2.

### M6. Conditional randomisation / conditional permutation tests (CRT / CPT)

- **Metric:** p-value $=\frac{1+\sum_{b}\mathbb 1\{T(\tilde W^{(b)},Y,Z)\ge T(W,Y,Z)\}}{B+1}$, where $\tilde W^{(b)}\sim P(W\mid Z)$ (CRT) or is a weighted permutation of $W$ (CPT).
- **Literature:** Candès, Fan, Janson & Lv, *JRSSB* 80(3):551–577, 2018, doi:10.1111/rssb.12265, arXiv:1610.02351 (model-X framework; I did not confirm in the abstract that the CRT is introduced there). Berrett, Wang, Barber & Samworth, *JRSSB* 82(1):175–197, 2020, doi:10.1111/rssb.12340.
- **Purpose:** Markov test with finite-sample validity **if** $P(W\mid Z)$ is known.
- **Requirements:** A model of the history feature given the current state, $P(g(H_{t-1})\mid S_t,A_t)$. Type-I inflation depends on how well this is estimated (both papers discuss this).
- **Computation:** Any test statistic, e.g. the improvement in held-out prediction of $S_{t+1}$ from adding $W$, with B resamples.
- **Complexity:** B × cost of the statistic.
- **Strengths:** Flexible statistic, so the most powerful predictive models can be used.
- **Limitations:** In clinical data $P(\text{history}\mid\text{state})$ is unknown and hard to model. Misspecification then shifts error from the outcome model to the history model.
- **Applicability:** Any.
- **Timing:** Pre-training.
- **Validation:** Theory and genomics applications (not sequential).
- **Recommendation:** **Investigate.** Use only if a credible history model exists (e.g. a low-dimensional lag feature).
- **Category:** 3.

### M7. Conditional mutual information (CMI) of history given state

- **Metric:** $\mathrm{CMI}_k = I\big(S_{t+1};\,H_{t-k:t-1}\mid S_t,A_t\big)$ in nats. $H_0\iff \mathrm{CMI}=0$ (for all lags). Normalised variant (proposal): $\mathrm{CMI}/H(S_{t+1}\mid S_t,A_t)$.
- **Literature:**
  - Kraskov, Stögbauer & Grassberger, *Phys. Rev. E* 69:066138, 2004, doi:10.1103/PhysRevE.69.066138 (KSG MI estimator).
  - Runge, AISTATS 2018, PMLR 84:938–947, arXiv:1709.01447 (CMIknn with local-permutation test; `tigramite`).
  - Mukherjee, Asnani & Kannan, UAI 2019, PMLR 115:1083–1093, arXiv:1906.01824 (classifier-based CCMI).
- **Purpose:** **Effect size** of non-Markovianity, which complements the p-values from M1/M4. Comparing $k$ gives lag selection.
- **Requirements:** Continuous variables (kNN). For discrete state maps use plug-in entropy with bias correction.
- **Computation:**
  - KSG/CMIknn: kNN distances, with null by local permutation (Runge).
  - CCMI: classifier distinguishing joint from product samples, then a KL estimate.
- **Complexity:** kNN about $O(n^2)$ naively (tree structures help). CMIknn permutation testing is expensive. CCMI costs classifier training.
- **Strengths:** An interpretable, scale-free magnitude. CMIknn is reportedly better calibrated than kernel tests on non-smooth densities. CCMI targets high-dimensional cases where kNN degrades.
- **Limitations:** kNN estimators are biased in high dimensions (CCMI's stated motivation). Estimates are not comparable across different state dimensionalities without care. Mixed discrete/continuous variables need special handling. Within-patient correlation violates i.i.d. assumptions.
- **Applicability:** Continuous (KSG, CMIknn, CCMI) or discrete (plug-in).
- **Timing:** Pre-training.
- **Validation:** Estimator benchmarks in each paper. Not validated on clinical RL data.
- **Recommendation:** **Implement** CCMI or CMIknn on reduced representations, reported as an effect size with bootstrap CIs.
- **Category:** 2.

### M8. Predictive comparison: Markov vs history-augmented models

- **Metric:** Held-out predictive gain (proposal):
  $\Delta_k=\frac1n\sum\big[-\log \hat p_0(S_{t+1},R_t\mid S_t,A_t)\big]-\frac1n\sum\big[-\log \hat p_k(\cdot\mid S_t,A_t,H_{t-k:t-1})\big]$.
  An MSE analogue works for point predictions. With the same learner class, $\Delta_k$ estimates a lower bound on the CMI of M7.
- **Literature:** No single canonical paper. Grounded in:
  - AIS theory, which bounds the value loss of a representation by its prediction errors (Subramanian et al., JMLR 23(12):1–83, 2022, arXiv:2010.08843).
  - Killian et al., ML4H 2020, PMLR 136:139–160, arXiv:2011.11235. They found sequentially formed (recurrent) state representations supported better policy learning on MIMIC-III sepsis data.
  - Ni et al., ICLR 2024, arXiv:2401.08898 (self-predictive abstraction unifies state/history representations).
- **Purpose:** State sufficiency, order selection and choice of $\phi$. Runs before RL.
- **Requirements:** Patient-level train/test split. Matched model capacity, e.g. Markov MLP vs MLP with lag features vs GRU over the full history.
- **Computation:** Fit both models with K-fold patient-grouped CV. Report $\Delta_k$ with bootstrap CI over patients and plot $\Delta_k$ against $k$.
- **Complexity:** Moderate (several supervised model fits).
- **Strengths:** Intuitive for clinicians. Gives magnitude. Uses standard tooling. Directly links to AIS value-loss bounds.
- **Limitations:** $\Delta_k\approx0$ can reflect an under-powered or under-fitted history model, not true sufficiency. Overfitting of the history model makes the gain negative. It is a predictive criterion, not causal. Gains in predicting irrelevant state coordinates may not matter for decisions; weighting by reward or value relevance remains open.
- **Applicability:** Any.
- **Timing:** Pre-training. No policy needed.
- **Validation:** Indirect (Killian et al.; AIS theory).
- **Recommendation:** **Implement.** This is the core practical diagnostic.
- **Category:** 1–2.

### M9. Markov-order selection by information criteria (AIC/BIC) for discretised state maps

- **Metric:** $\hat k=\arg\min_k\big[-2\log L_k + c(n)\,|\mathcal S|^k(|\mathcal S|-1)\big]$, with $c(n)=\log n$ (BIC) or 2 (AIC). For MDPs (proposal), condition on actions: $|\mathcal S|^k|\mathcal A|^k(|\mathcal S|-1)$ parameters.
- **Literature:**
  - Katz, *Technometrics* 23(3):243–249, 1981. Later work reports he showed AIC is inconsistent (it overestimates the order).
  - Csiszár & Shields, *Annals of Statistics* 28(6):1601–1619, 2000, doi:10.1214/aos/1015957472. BIC order estimator consistent with no prior bound on the order.
- **Purpose:** Order/lag selection for **discrete** state maps, e.g. k-means clusters à la Komorowski et al. 2018 (750 clusters).
- **Requirements:** Finite state space. Enough counts.
- **Computation:** Count transition tuples, compute the likelihood for each $k$, take the argmin.
- **Complexity:** $O(n)$ counting, but the parameter count explodes ($750^2$ already for $k=2$).
- **Strengths:** Classical with a consistency theory. Trivial to compute.
- **Limitations:** With large $|\mathcal S|$ and limited data, BIC will almost always choose $k=1$ because of the parameter penalty. That reflects lack of data, not Markovianity. It also assumes a stationary, homogeneous chain.
- **Applicability:** Discrete states and actions only.
- **Timing:** Pre-training.
- **Validation:** Theory. Classical applications (e.g. meteorology).
- **Recommendation:** **Implement** for small or aggregated state spaces only, and report it together with the data-sufficiency warning.
- **Category:** 1.

### M10. Sequential CMI significance tests for Markov-chain order (Papapetrou & Kugiumtzis)

- **Metric:** Test $I(X_{t};X_{t-k}\mid X_{t-1:t-k+1})=0$ for increasing $k$. $\hat k$ is the last order that is rejected.
- **Literature:** Papapetrou & Kugiumtzis, *Physica A* 392(7):1593–1601, 2013, arXiv:1301.0148. Parametric-significance follow-up: arXiv:1511.02339.
- **Purpose:** Order selection for discrete symbol sequences.
- **Requirements:** Discrete states. Randomisation (surrogate) tests or analytic bias-based limits.
- **Complexity:** Low to moderate.
- **Strengths:** Reported as better than AIC/BIC for orders >1 (per abstract). Less penalty-driven than BIC.
- **Limitations:** Effectiveness at large orders depends on data availability (per abstract). Single-sequence setting. No actions.
- **Applicability:** Discrete.
- **Timing:** Pre-training.
- **Validation:** Simulations (per abstract).
- **Recommendation:** **Adapt.** Condition on $A$ and pool across patients. This pairs naturally with M9.
- **Category:** 2.

### M11. Markov tests from multi-state survival models (clinical biostatistics)

- **Metric:**
  - Cox-based test: add the time of entry into the current state (or sojourn time) as a covariate in the transition hazards and test its coefficient.
  - Log-rank-type tests that group patients by state occupancy at varying landmark times.
  - Illness–death future/past association trace with a supremum test.
- **Literature:**
  - Titman & Putter, *Biostatistics* 23(2):380–396, 2022, doi:10.1093/biostatistics/kxaa030 (R package `markovMSM`).
  - Rodríguez-Girondo & de Uña-Álvarez, *Statistics in Medicine* 31(30):4416–4427, 2012, doi:10.1002/sim.5619.
- **Purpose:** Markov property in **continuous time with right-censoring**. Also detects **semi-Markov** (duration) dependence, which feeds the time-grid decisions in M16/M17.
- **Requirements:** Data expressed as transitions between a small set of clinical states (e.g. SOFA bands, ventilated/not, discharge/death) with exact or interval times.
- **Computation:** Standard survival-model fits. Bootstrap for the global tests.
- **Complexity:** Low.
- **Strengths:** Established in medical statistics and familiar to clinical reviewers. Handles censoring and irregular timing natively. Per the abstract, the Cox approach keeps good power across scenarios.
- **Limitations:** Only for coarse discrete states. Does not natively handle actions (they would enter as time-varying covariates, which is my proposal). Which test is most powerful depends on the departure (per abstract).
- **Applicability:** Discrete states in continuous time.
- **Timing:** Pre-training.
- **Validation:** Simulations and biomedical datasets.
- **Recommendation:** **Adapt.** Strongly recommended as the clinically accepted bridge between ISPOR-style state-transition modelling and MDPs.
- **Category:** 2.

### M12. Approximate-information-state (AIS) error as a state-sufficiency score

- **Metric:** For a representation $Z_t=\sigma(H_t)$:
  - $\varepsilon=\sup|\mathbb E[R_t\mid H_t,A_t]-\hat r(Z_t,A_t)|$
  - $\delta = d_{\mathfrak F}\big(P(Z_{t+1}\mid H_t,A_t),\hat P(Z_{t+1}\mid Z_t,A_t)\big)$, an IPM such as Wasserstein or MMD.
  
  Per the abstract, approximately satisfying the information-state properties yields an approximate DP with bounded optimality loss (the bound is a function of ε, δ; exact form to be taken from the paper).
- **Literature:** Subramanian, Sinha, Seraj & Mahajan, *JMLR* 23(12):1–83, 2022, arXiv:2010.08843.
- **Purpose:** Turns "Markov or not" into a graded **value-loss-relevant** sufficiency score for a candidate state map.
- **Requirements:** Estimates of the history-conditional quantities, which in practice means a rich history model (e.g. RNN) as reference.
- **Computation (proposal):** Fit a history model $\hat p(\cdot\mid H_t,A_t)$ and a state model $\hat p(\cdot\mid S_t,A_t)$. Estimate ε (reward) and δ (MMD or Wasserstein between their predictive distributions) on held-out data.
- **Complexity:** Moderate (two generative/predictive models plus IPM estimation).
- **Strengths:** The theory directly links sufficiency errors to sub-optimality. Does not require training a policy.
- **Limitations:** The sup-norm quantities are not estimable from finite data; average-case surrogates lose the guarantee. The reference history model can itself be insufficient. Bounds are likely loose with clinical discounting.
- **Applicability:** Any.
- **Timing:** Pre-training (no policy).
- **Validation:** Theory. Numerical experiments in RL benchmarks (per abstract). No clinical validation found.
- **Recommendation:** **Adapt.** Report it as a graded score alongside M8.
- **Category:** 2–3.

### M13. Approximate state-abstraction criteria and value-loss bounds

- **Metric:** For an aggregation $\phi:\mathcal S\to\bar{\mathcal S}$, model-irrelevance ε-criteria (proposal: estimated from data):
  - $\varepsilon_R=\max_{\phi(s)=\phi(s')}\max_a|r(s,a)-r(s',a)|$
  - $\varepsilon_T=\max_{\phi(s)=\phi(s')}\max_a\|P(\phi(\cdot)\mid s,a)-P(\phi(\cdot)\mid s',a)\|_1$
  
  Q*-, a*- and π*-irrelevance variants also exist. Value-loss bounds hold for approximate abstractions.
- **Literature:**
  - Li, Walsh & Littman, ISAIM 2006 ("Towards a unified theory of state abstraction for MDPs"; exact-abstraction hierarchy).
  - Abel, Hershkowitz & Littman, ICML 2016, PMLR 48:2915–2923, arXiv:1701.04113 (bounded-loss guarantees for four approximate abstraction types).
- **Purpose:** Adequacy of a **coarsening** state map, e.g. raw features → 750 clusters. Measures how much the clustering mixes states with different dynamics or rewards.
- **Requirements:** A finer reference state (e.g. raw continuous features or many small clusters) and estimates of $r$ and $P$ at that finer level.
- **Computation (proposal):** Within each abstract state, estimate the dispersion of reward and next-abstract-state distributions across fine-level neighbourhoods (kNN or sub-clusters). Report quantiles rather than the max.
- **Complexity:** Moderate.
- **Strengths:** Theory-backed. Interpretable per cluster ("cluster 312 mixes improving and deteriorating patients").
- **Limitations:** The Q*-irrelevance variants need $Q^*$, which is not available before training (Category 4 for those). Max-based bounds are dominated by sparse cells. The criteria depend on the choice of fine reference.
- **Applicability:** Discrete abstract states. Fine level discrete or continuous.
- **Timing:** Model-irrelevance variants are pre-training. Q*/π* variants are not.
- **Validation:** Theory and toy RL domains.
- **Recommendation:** **Adapt** the model-irrelevance variant. **Exclude** the Q*-based variants from the pre-training framework.
- **Category:** 2 (model-irrelevance), 4 (Q*-based).

### M14. Bisimulation and bisimulation metrics

- **Metric:** Exact bisimulation (Givan et al.) partitions states with equal rewards and equal block transition probabilities. The bisimulation metric (Ferns et al.) is the fixed point of
  $d(s,s')=\max_a\big[c_R|r(s,a)-r(s',a)|+c_T\,W_d(P(\cdot\mid s,a),P(\cdot\mid s',a))\big]$.
- **Literature:**
  - Givan, Dean & Greig, *Artificial Intelligence* 147(1–2):163–223, 2003, doi:10.1016/S0004-3702(02)00376-4.
  - Ferns, Panangaden & Precup, UAI 2004, pp. 162–169. Continuous-state version: arXiv:1207.1386.
  - Castro, AAAI 2020, doi:10.1609/aaai.v34i06.6564, arXiv:1911.09291 (sampling-based and differentiable approximations, *deterministic* MDPs).
  - Zhang et al., ICLR 2021, arXiv:2006.10742 (DBM representation learning).
  - Gelada et al., ICML 2019, arXiv:1906.02736 (DeepMDP; links latent-model losses to bisimulation).
- **Purpose:** Gives a principled definition of the *ideal* abstraction. As a diagnostic it can check whether states merged by $\phi$ are behaviourally close.
- **Requirements:** A transition model at the fine level. Note that the max over actions requires the model for all actions, including those clinicians rarely take (an offline-support problem).
- **Computation:** Fixed-point iteration with Wasserstein (Kantorovich) distances. This is expensive and the estimates are noisy under stochastic transitions. Castro's scalable method targets deterministic MDPs.
- **Complexity:** Polynomial but heavy: $O(|\mathcal S|^2)$ pairs × one OT problem per pair per iteration.
- **Strengths:** The strongest theoretical notion of behavioural equivalence.
- **Limitations:** Fragile under estimation error, which motivated the metric version in the first place. Unsupported actions in observational data make the max over actions unidentifiable. Learned-representation versions (DBM, DeepMDP) are training procedures, not diagnostics.
- **Applicability:** Finite (exact computation) or continuous (approximate).
- **Timing:** Pre-training in principle, but needs a dynamics model.
- **Validation:** Theory. Control and vision RL benchmarks.
- **Recommendation:** **Investigate.** Possibly restrict to an on-support (behaviour-policy) variant. Cite mainly as conceptual grounding for M13.
- **Category:** 3.

### M15. λ-discrepancy (TD(λ) value disagreement)

- **Metric:** $\Lambda=\|V^{\pi}_{\lambda_1}-V^{\pi}_{\lambda_2}\|$, typically comparing λ=0 (TD, which relies on the Markov assumption) with λ=1 (Monte Carlo). The paper proves Λ=0 for all MDPs and Λ≠0 almost always for a broad class of POMDPs.
- **Literature:** Allen, Kirtland, et al., NeurIPS 2024, arXiv:2407.07333. Code: github.com/brownirl/lambda_discrepancy.
- **Purpose:** **Decision-relevant** Markov diagnostic, i.e. non-Markovianity that affects value estimation.
- **Requirements:** A policy whose value is estimated. Offline, the natural choice (proposal) is the estimated **behaviour policy** with on-policy TD(0) vs MC on the observed trajectories.
- **Computation (proposal):** Tabular or fitted TD(0) and MC estimates of $V^{\pi_b}$ on the discretised MDP. Bootstrap over patients for the CI.
- **Complexity:** Low (tabular) to moderate (function approximation).
- **Strengths:** Directly targets the consequence that matters for RL (biased bootstrapped values). Uses the reward, so it ignores irrelevant non-Markovian coordinates.
- **Limitations:** The paper's guarantee is "almost always" non-zero, so specific POMDPs can give Λ=0. Finite-sample noise in MC returns is large with sparse terminal rewards (e.g. 90-day mortality). Function approximation adds its own discrepancy. Using it offline with the behaviour policy is my adaptation and is not validated. I saw only the abstract and summaries.
- **Applicability:** Discrete (tabular) or continuous (approximate).
- **Timing:** Needs value estimation of $\pi_b$ but no RL training of a target policy, so it is pre-training with respect to the target.
- **Validation:** Theory plus POMDP benchmark experiments. None in healthcare.
- **Recommendation:** **Investigate.** A promising research candidate and a nice complement to the purely statistical tests.
- **Category:** 3.

### M16. Time-grid (Δt) sensitivity analysis

- **Metric (proposal):** For a set of grids $\Delta t\in\{1,2,4,8\}$ h, report a vector of diagnostics:
  - Markov test results (M1/M4) and $\Delta_k$ (M8).
  - Fraction of bins with ≥1 action change or with >1 distinct action.
  - Fraction of bins requiring imputation.
  - Self-transition rate $P(S_{t+1}=S_t)$.
  - Within-bin action–outcome ordering violations (outcome measured before action within the same bin).
- **Literature:**
  - Komorowski et al., *Nature Medicine* 24:1716–1720, 2018, doi:10.1038/s41591-018-0213-5 (4-h steps, the de-facto standard).
  - Sun & Tang, arXiv:2511.20913 (2025 preprint). Compares 1/2/4/8 h steps in a fixed offline-RL pipeline with action re-mapping. Finer steps (1–2 h) performed best and were most stable in their static-behaviour-policy setup.
  - Tallec, Blier & Ollivier, ICML 2019, PMLR 97:6096–6104, arXiv:1901.09732 (Q-learning collapses as Δt→0; Q-function has no continuous-time limit).
- **Purpose:** Choosing and justifying the decision interval (temporal abstraction).
- **Requirements:** Raw time-stamped events. Re-binning pipeline. An action re-mapping that makes doses comparable across Δt (e.g. rates).
- **Computation:** Re-run the data pipeline and the diagnostics for each Δt.
- **Complexity:** Linear in the number of grids × diagnostics.
- **Strengths:** Cheap. Directly addresses an arbitrary and influential design choice. Tallec et al. show that too-fine grids also harm value-based RL, so the trade-off is real in both directions.
- **Limitations:** No single optimality criterion exists; it is multi-objective and needs clinical judgement about the decision cadence. Markov diagnostics are not comparable across Δt without care, since test power changes with n and conditioning sets differ. Sun & Tang is a preprint and evaluates *after* training.
- **Applicability:** Any.
- **Timing:** Pre-training, provided only the data diagnostics are used, not Sun & Tang's policy evaluations.
- **Validation:** Sun & Tang (empirical, sepsis). Tallec et al. (theory and benchmarks).
- **Recommendation:** **Implement.** This is a core component of the framework.
- **Category:** 2.

### M17. Decision-timing / semi-Markov adequacy check

- **Metric (proposal):**
  1. Distribution of inter-decision intervals (CV, multimodality).
  2. Test whether elapsed time since the last decision or state entry predicts the next state or action given $(S_t,A_t)$. Use the M4/M8 machinery or the Cox entry-time test of M11.
  3. Fraction of decisions that fall off-grid.
- **Literature:**
  - Sutton, Precup & Singh, *Artificial Intelligence* 112(1–2):181–211, 1999 (SMDP / options framework).
  - Bradtke & Duff, NIPS 7 (1994), pp. 393–400 (RL for SMDPs / continuous-time MDPs).
  - Doya, *Neural Computation* 12(1):219–245, 2000, doi:10.1162/089976600300015961 (continuous-time RL via HJB, no discretisation).
  - Fatemi et al., CHIL 2022, PMLR 174:119–137, arXiv:2203.09365. Per the abstract, only SMDP-based offline algorithms recovered the optimal policy in variable-timing environments; warfarin dosing application.
  - Titman & Putter 2022 (entry-time covariate test).
- **Purpose:** Decide between a fixed-grid MDP, an SMDP (event-driven decisions with variable durations) and continuous-time formulations.
- **Requirements:** Exact timestamps of actions and measurements.
- **Complexity:** Low.
- **Strengths:** Directly addresses irregular clinical decision timing. SMDP algorithms exist if the check fails.
- **Limitations:** Decision timing is itself a clinician choice, so it may be informative or confounded. Measurement times are not decision times. The test only detects duration effects visible in the data.
- **Applicability:** Any.
- **Timing:** Pre-training.
- **Validation:** Fatemi et al. (empirical). Classical SMDP theory.
- **Recommendation:** **Implement** the descriptive part and test 2 (via M4/M11).
- **Category:** 2.

### M18. Predictive state representations / latent-dimensionality (rank) diagnostics

- **Metric:** System-dynamics-matrix rank. Per the abstract of Littman, Sutton & Singh, a linear PSR exists with no more predictions than the number of states in the minimal POMDP. A proposed diagnostic is the numerical rank of an empirical (test × history) prediction matrix, compared with the rank implied by $\phi$.
- **Literature:** Littman, Sutton & Singh, NIPS 2001 (pp. 1555–1561 per mlanthology). Author lists differ across indexes (Singh is missing in some).
- **Purpose:** Indicates the size of a sufficient state, which helps diagnose POMDP structure.
- **Requirements:** Discrete observations/actions and a defined test/history set. Large data.
- **Complexity:** SVD of a large sparse matrix.
- **Strengths:** A principled sufficiency notion defined purely by observable predictions.
- **Limitations:** I did not verify spectral learning methods. Numerical rank under noise is ill-posed. Clinical observations are continuous and high-dimensional.
- **Applicability:** Discrete, small spaces.
- **Timing:** Pre-training.
- **Validation:** Mostly synthetic.
- **Recommendation:** **Investigate**, at most as a conceptual reference.
- **Category:** 3–4.

### M19. Proximal / POMDP-confounding identification (context, not a diagnostic)

- **Metric:** Not a test. Proximal RL gives identification conditions (bridge functions) for policy value when the behaviour policy depends on unobserved state.
- **Literature:** Bennett & Kallus, *Operations Research* 2024, pp. 1071–1086, doi:10.1287/opre.2021.0781, arXiv:2110.15332 (sepsis application).
- **Purpose:** Explains why a Markov test that passes on observables does not guarantee unconfounded OPE or RL.
- **Timing:** Requires a target policy (OPE), so not pre-training.
- **Recommendation:** **Exclude** as a metric. Cite in the limitations/conceptual section.
- **Category:** 4.

### Summary table

| ID | Method | Purpose | Category | Recommendation |
|---|---|---|---|---|
| M1 | FBL Markov test (Shi et al. 2020) | Markov, order | 2 | Implement/adapt |
| M2 | Deep generative Markov test (Zhou et al. 2023) | Order | 3 | Investigate |
| M3 | CCF test (Chen & Hong 2012) | Markov | 4 | Exclude (background) |
| M4 | GCM (Shah & Peters 2020) | Markov / CI | 2 | Implement |
| M5 | KCI / kernel CI | Markov / CI | 2 | Investigate (subsample) |
| M6 | CRT / CPT | Markov / CI | 3 | Investigate |
| M7 | CMI (KSG / CMIknn / CCMI) | Effect size, lag | 2 | Implement |
| M8 | Markov vs history predictive gain | Sufficiency, order | 1–2 | Implement (core) |
| M9 | AIC/BIC Markov order | Order (discrete) | 1 | Implement (small S) |
| M10 | Sequential CMI order tests | Order (discrete) | 2 | Adapt |
| M11 | Multi-state model Markov tests | Markov / semi-Markov | 2 | Adapt |
| M12 | AIS error | Graded sufficiency | 2–3 | Adapt |
| M13 | ε-model-irrelevance of abstraction | Clustering adequacy | 2 (Q*: 4) | Adapt |
| M14 | Bisimulation metrics | Ideal abstraction | 3 | Investigate |
| M15 | λ-discrepancy | Value-relevant non-Markov | 3 | Investigate |
| M16 | Δt sensitivity analysis | Time grid | 2 | Implement (core) |
| M17 | Decision-timing / SMDP check | Time grid vs SMDP | 2 | Implement |
| M18 | PSR rank | Latent dimension | 3–4 | Investigate |
| M19 | Proximal RL | Confounding (context) | 4 | Exclude |

---

## 3. Conceptual (clinical judgement) vs empirically evaluable criteria

### Fundamentally conceptual (data cannot settle them)

- **Is the problem sequential?** Decisions must be repeated, with later options depending on earlier actions and outcomes, and the objective must be long-horizon rather than a one-shot classification. Steimle & Denton (2017) frame MDPs/POMDPs for repeated screening and treatment decisions in chronic disease (chapter abstract only seen).
- **Decision cadence that is clinically meaningful and actionable.** What frequency can clinicians actually act at? This sets the admissible range of Δt for M16.
- **Clinically valid action space.** Which interventions are controllable decisions rather than protocol-driven? Are dose bins clinically distinct?
- **Reward/objective validity.** Whether a proxy (e.g. 90-day mortality, SOFA change) reflects patient value. Trade-offs and ethics.
- **No unmeasured confounding / sufficiency with respect to unobserved clinician information.** This is untestable from observational data alone (Section 1, caveat 3; Bennett & Kallus 2024).
- **Causal vs predictive Markovianity.** Whether the physiology is "really" Markov at some resolution.
- **Stationarity and transportability** across hospitals and eras (the data can only partially inform this).
- **Good-practice transparency** of state-transition model structure. ISPOR-SMDM Task Force-3 (Siebert et al. 2012) gives recommendations on model type and structure. I saw only the landing page; it did not reveal specific Markov-assumption guidance. Gottesman et al. (2019) give RL-in-healthcare guidelines (I did not read beyond the opening).

### Empirically evaluable (Section 2 methods)

| Criterion | Methods |
|---|---|
| Markov property / state sufficiency of $\phi$ given observables | M1, M4, M5, M6, M7, M8, M12 |
| Markov order / history length | M1 (sequential), M2, M8 ($\Delta_k$ vs k), M9, M10 |
| Abstraction (clustering) adequacy | M13, M14 |
| Value-relevant non-Markovianity | M15, M12 |
| Time-grid adequacy | M16 |
| Fixed-grid vs semi-Markov / continuous-time timing | M17, M11 |
| Duration dependence (semi-Markov) | M11, M17 |

**Mixed:** "Is the problem rich enough to need RL?" is conceptual, but it can be partly informed by data, e.g. action variability across similar states (overlap) and whether actions affect future states. That belongs in other scopes (action/overlap inventory).

---

## 4. References

### Verified

Existence confirmed via WebSearch/WebFetch (title, authors, venue/year).

1. Shi, C., Wan, R., Song, R., Lu, W., Leng, L. (2020). Does the Markov Decision Process Fit the Data: Testing for the Markov Property in Sequential Decision Making. *ICML 2020*, PMLR 119:8807–8817. arXiv:2002.01751.
2. Zhou, Y., Shi, C., Li, L., Yao, Q. (2023). Testing for the Markov Property in Time Series via Deep Conditional Generative Learning. arXiv:2305.19244 (preprint).
3. Chen, B., Hong, Y. (2012). Testing for the Markov Property in Time Series. *Econometric Theory* 28(1):130–178. doi:10.1017/S0266466611000065.
4. Shah, R. D., Peters, J. (2020). The Hardness of Conditional Independence Testing and the Generalised Covariance Measure. *Annals of Statistics* 48(3):1514–1538. doi:10.1214/19-AOS1857. arXiv:1804.07203.
5. Zhang, K., Peters, J., Janzing, D., Schölkopf, B. (2011). Kernel-based Conditional Independence Test and Application in Causal Discovery. *UAI 2011*. arXiv:1202.3775.
6. Fukumizu, K., Gretton, A., Sun, X., Schölkopf, B. (2007). Kernel Measures of Conditional Dependence. *NIPS 20*, pp. 489–496.
7. Candès, E., Fan, Y., Janson, L., Lv, J. (2018). Panning for Gold: 'Model-X' Knockoffs for High Dimensional Controlled Variable Selection. *JRSSB* 80(3):551–577. doi:10.1111/rssb.12265. arXiv:1610.02351.
8. Berrett, T. B., Wang, Y., Barber, R. F., Samworth, R. J. (2020). The Conditional Permutation Test for Independence While Controlling for Confounders. *JRSSB* 82(1):175–197. doi:10.1111/rssb.12340. arXiv:1807.05405.
9. Kraskov, A., Stögbauer, H., Grassberger, P. (2004). Estimating Mutual Information. *Physical Review E* 69:066138. doi:10.1103/PhysRevE.69.066138. (Erratum: PRE 83:019903, 2011.)
10. Runge, J. (2018). Conditional Independence Testing Based on a Nearest-Neighbor Estimator of Conditional Mutual Information. *AISTATS 2018*, PMLR 84:938–947. arXiv:1709.01447.
11. Mukherjee, S., Asnani, H., Kannan, S. (2019). CCMI: Classifier Based Conditional Mutual Information Estimation. *UAI 2019*, PMLR 115:1083–1093. arXiv:1906.01824.
12. Katz, R. W. (1981). On Some Criteria for Estimating the Order of a Markov Chain. *Technometrics* 23(3):243–249. (Bibliographic record only.)
13. Csiszár, I., Shields, P. C. (2000). The Consistency of the BIC Markov Order Estimator. *Annals of Statistics* 28(6):1601–1619. doi:10.1214/aos/1015957472.
14. Papapetrou, M., Kugiumtzis, D. (2013). Markov Chain Order Estimation with Conditional Mutual Information. *Physica A* 392(7):1593–1601. arXiv:1301.0148.
15. Titman, A. C., Putter, H. (2022). General Tests of the Markov Property in Multi-State Models. *Biostatistics* 23(2):380–396. doi:10.1093/biostatistics/kxaa030.
16. Rodríguez-Girondo, M., de Uña-Álvarez, J. (2012). A Nonparametric Test for Markovianity in the Illness-Death Model. *Statistics in Medicine* 31(30):4416–4427. doi:10.1002/sim.5619.
17. Subramanian, J., Sinha, A., Seraj, R., Mahajan, A. (2022). Approximate Information State for Approximate Planning and Reinforcement Learning in Partially Observed Systems. *JMLR* 23(12):1–83. arXiv:2010.08843.
18. Killian, T. W., Zhang, H., Subramanian, J., Fatemi, M., Ghassemi, M. (2020). An Empirical Study of Representation Learning for Reinforcement Learning in Healthcare. *ML4H 2020*, PMLR 136:139–160. arXiv:2011.11235.
19. Ni, T., Eysenbach, B., Seyedsalehi, E., Ma, M., Gehring, C., Mahajan, A., Bacon, P.-L. (2024). Bridging State and History Representations: Understanding Self-Predictive RL. *ICLR 2024*. arXiv:2401.08898.
20. Li, L., Walsh, T. J., Littman, M. L. (2006). Towards a Unified Theory of State Abstraction for MDPs. *ISAIM 2006* (9th Int. Symp. on AI and Mathematics).
21. Abel, D., Hershkowitz, D. E., Littman, M. L. (2016). Near Optimal Behavior via Approximate State Abstraction. *ICML 2016*, PMLR 48:2915–2923. arXiv:1701.04113.
22. Givan, R., Dean, T., Greig, M. (2003). Equivalence Notions and Model Minimization in Markov Decision Processes. *Artificial Intelligence* 147(1–2):163–223. doi:10.1016/S0004-3702(02)00376-4.
23. Ferns, N., Panangaden, P., Precup, D. (2004). Metrics for Finite Markov Decision Processes. *UAI 2004*, pp. 162–169. (Continuous-state version: arXiv:1207.1386.)
24. Castro, P. S. (2020). Scalable Methods for Computing State Similarity in Deterministic Markov Decision Processes. *AAAI 2020* 34(6):10069–10076. doi:10.1609/aaai.v34i06.6564. arXiv:1911.09291.
25. Zhang, A., McAllister, R., Calandra, R., Gal, Y., Levine, S. (2021). Learning Invariant Representations for Reinforcement Learning without Reconstruction. *ICLR 2021*. arXiv:2006.10742.
26. Gelada, C., Kumar, S., Buckman, J., Nachum, O., Bellemare, M. G. (2019). DeepMDP: Learning Continuous Latent Space Models for Representation Learning. *ICML 2019*, PMLR 97:2170–2179. arXiv:1906.02736.
27. Allen, C., Kirtland, A., et al. (2024). Mitigating Partial Observability in Sequential Decision Processes via the Lambda Discrepancy. *NeurIPS 2024*. arXiv:2407.07333. (Full author list not checked.)
28. Littman, M. L., Sutton, R. S., Singh, S. (2001). Predictive Representations of State. *NIPS 14*. (Singh is listed in the PDF but not in every index.)
29. Bennett, A., Kallus, N. (2024). Proximal Reinforcement Learning: Efficient Off-Policy Evaluation in Partially Observed Markov Decision Processes. *Operations Research*, 1071–1086. doi:10.1287/opre.2021.0781. arXiv:2110.15332.
30. Tallec, C., Blier, L., Ollivier, Y. (2019). Making Deep Q-learning Methods Robust to Time Discretization. *ICML 2019*, PMLR 97:6096–6104. arXiv:1901.09732.
31. Sutton, R. S., Precup, D., Singh, S. (1999). Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning. *Artificial Intelligence* 112(1–2):181–211.
32. Bradtke, S. J., Duff, M. O. (1995). Reinforcement Learning Methods for Continuous-Time Markov Decision Problems. *NIPS 7* (1994 conference), pp. 393–400.
33. Doya, K. (2000). Reinforcement Learning in Continuous Time and Space. *Neural Computation* 12(1):219–245. doi:10.1162/089976600300015961.
34. Fatemi, M., Wu, M., Petch, J., Nelson, W., Connolly, S. J., Benz, A., Carnicelli, A., Ghassemi, M. (2022). Semi-Markov Offline Reinforcement Learning for Healthcare. *CHIL 2022*, PMLR 174:119–137. arXiv:2203.09365.
35. Sun, Y., Tang, S. (2025). Exploring Time-Step Size in Reinforcement Learning for Sepsis Treatment. arXiv:2511.20913 (preprint).
36. Komorowski, M., Celi, L. A., Badawi, O., Gordon, A. C., Faisal, A. A. (2018). The Artificial Intelligence Clinician Learns Optimal Treatment Strategies for Sepsis in Intensive Care. *Nature Medicine* 24:1716–1720. doi:10.1038/s41591-018-0213-5. (4-h steps and 750 k-means states per secondary/replication sources.)
37. Raghu, A., Komorowski, M., Celi, L. A., Szolovits, P., Ghassemi, M. (2017). Continuous State-Space Models for Optimal Sepsis Treatment: A Deep Reinforcement Learning Approach. *MLHC 2017*, PMLR 68:147–163. arXiv:1705.08422.
38. Gottesman, O., Johansson, F., Komorowski, M., Faisal, A., Sontag, D., Doshi-Velez, F., Celi, L. A. (2019). Guidelines for Reinforcement Learning in Healthcare. *Nature Medicine* 25(1):16–18. doi:10.1038/s41591-018-0310-5.
39. Steimle, L. N., Denton, B. T. (2017). Markov Decision Processes for Screening and Treatment of Chronic Diseases. In *Markov Decision Processes in Practice*, Springer, pp. 189–222. doi:10.1007/978-3-319-47766-4_6.
40. Siebert, U., Alagoz, O., Bayoumi, A. M., Jahn, B., Owens, D. K., Cohen, D. J., Kuntz, K. M. (2012). State-Transition Modeling: A Report of the ISPOR-SMDM Modeling Good Research Practices Task Force-3. *Medical Decision Making* 32(5):690–700, doi:10.1177/0272989X12455463. Also *Value in Health* 15(6):812–820, doi:10.1016/j.jval.2012.06.014 (issue number differs between sources: 15(5) vs 15(6); check).

### Unverified leads (not confirmed in this session; do not cite without checking)

- Papapetrou, Kugiumtzis et al. (2016). Parametric significance tests of CMI for Markov chain order. arXiv:1511.02339 appeared in search results; journal version not confirmed.
- A multivariate/vector extension of the GCM. Mentioned in later literature; not checked.
- The claim that the CRT was introduced in Candès et al. (2018). This is the common attribution, but I did not confirm it in the abstract.
- Implementation details of Shi et al. (2020), such as conditional characteristic functions and cross-fitting. These come from memory; check the full text.
- Exact form of the AIS value-loss bound (Subramanian et al. 2022, Theorem numbers and constants).
- Rodríguez-Girondo et al. (2016). Comparative study of Markov-condition tests in the illness–death model, *Statistics in Medicine* 35(20). Seen only as a portal listing.
- Denton, B. T. (2018). Optimization of Sequential Decision Making for Chronic Diseases: From Data to Decisions. INFORMS TutORials, doi:10.1287/educ.2018.0184. Seen in search results only.
- Spectral learning of PSRs / observable operator models (e.g. Boots, Siddiqi & Gordon; Jaeger). Not searched.
- medDreamer (arXiv:2505.19785) and Neural-CDE offline RL for sepsis (arXiv:2506.15019). Continuous-time / irregular-sampling clinical RL; seen only in search summaries.
- Multi-output GP deep recurrent Q-networks for sepsis (Futoma et al.?). Title seen; authors/venue not confirmed.
