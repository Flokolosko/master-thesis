# Inventory: Temporal Data Quality (E) and Confounding & Identifiability (F)

Thesis context: pre-training assessment of whether observational ICU/EHR trajectories support an MDP model and offline RL.
Status: draft literature inventory (2026-11-05). Depth-of-reading is stated per item; most items were checked at the level of the abstract plus bibliographic record only.

---

## 1. Search log (short)

- Tools: WebSearch (standard), WebFetch for the Kahn et al. 2016 full text on PMC, the Namkoong et al. 2020 arXiv page and the Dorfman et al. 2021 NeurIPS abstract page.
- Queries (≈35): EHR DQ frameworks (Kahn 2016, Weiskopf & Weng 2013); informative missingness (Agniel 2018, Che 2018, Lipton 2016, Sisk et al. 2021 review); MCAR test (Little 1988); irregular sampling (Shukla & Marlin 2020; Pullenayegum & Lim 2016); charting lag (MIMIC `charttime`/`storetime`); digit heaping (Whipple/Myers); sepsis RL time-binning (Komorowski 2018, Jeter et al. 2019, Gottesman 2018/2019, Luo et al. 2024); g-methods (Robins 1986, Robins et al. 2000, Hernán & Robins, Murphy 2003, Chakraborty & Moodie 2013); confounded OPE (Namkoong 2020, Kallus & Zhou 2018/2020, Tennenholtz 2020, Oberst & Sontag 2019, Bennett & Kallus proximal RL); sensitivity analysis (Tan 2006, Rosenbaum 2002, VanderWeele & Ding 2017, Brumback 2004); negative controls (Lipsitch 2010, Shi et al. 2020); proximal inference (Tchetgen Tchetgen et al.); positivity (Petersen 2012); balance (Austin 2009); deconfounding (Bica 2020); MDP ambiguity (Dorfman 2021).
- Gaps:
  - No peer-reviewed ICU study that directly measures **charting time vs. event time** came up in this search. Only indirect evidence (MIMIC docs; a pediatric sepsis study seen as a snippet, not identified).
  - No published **timestamp-heaping** metric for clinical event times was found. Whipple/Myers come from demography.
  - Label-leakage papers (JAMIA 2023, JAMA Netw Open 2025) appeared only as search snippets and are not identified well enough to cite.

---

## 2. Method inventory

Category key: **1** established, usable directly; **2** established, needs adaptation; **3** research candidate; **4** unsuitable as a dataset-level pre-training metric (may still be useful for framing).

### Part E: temporal data quality

#### E1. Harmonized DQ framework: conformance / completeness / plausibility
- **Metric:** Checklist of rule-based checks in three categories: (i) conformance (value, relational, computational); (ii) completeness; (iii) plausibility (uniqueness, atemporal, **temporal**). Each is assessed in two contexts: *verification* (internal expectations) and *validation* (external benchmarks). Typical output per check: failure rate = #violations / #applicable records.
- **Literature:** Kahn et al. 2016, eGEMs, doi:10.13063/2327-9214.1244. Full text read via PMC.
- **Purpose:** Shared vocabulary for reporting DQ checks. "Temporal plausibility" covers expected changes over time, state transitions and ordering (e.g. admission before discharge).
- **Requirements:** Schema metadata and domain rules (ranges, ordering constraints).
- **Computation:** SQL/pandas rule checks.
- **Complexity:** O(N) per rule.
- **Strengths:** Widely adopted terminology that makes reports comparable.
- **Limitations:** It is a taxonomy, not a metric set. **Timeliness and currency are explicitly excluded** as extrinsic, task-dependent features. Completeness does not distinguish *why* data are missing. Cannot detect informative missingness or confounding.
- **Applicability:** High, as the organizing scaffold for the E-indicators.
- **Timing:** Pre-training. No policy or model needed.
- **Validation:** Authors aligned the terms to 10 published DQ terminologies (coverage validation, not predictive validity).
- **Recommendation:** Implement as the reporting frame.
- **Category:** 1

#### E2. EHR DQ dimensions including "currency"
- **Metric:** Five dimensions: completeness, correctness, concordance, plausibility, currency. Seven method classes, e.g. element presence, distribution comparison, validity checks, log review.
- **Literature:** Weiskopf & Weng 2013, JAMIA 20(1):144–151, doi:10.1136/amiajnl-2011-000681 (PMID 22733976, PMC3555312). Abstract-level only. The DOI was not confirmed by the searches and should be double-checked.
- **Purpose:** Adds *currency* (timeliness), which Kahn omits. This dimension corresponds to documentation lag (E9).
- **Requirements / Computation / Complexity:** As E1. Currency needs two timestamps (event and recording).
- **Strengths:** Names the temporal dimension that is relevant to us.
- **Limitations:** The authors report little consistency across studies. Their dimensions are not operationalized as formulas.
- **Applicability:** Justifies a separate timeliness indicator.
- **Timing:** Pre-training.
- **Validation:** Literature review only.
- **Recommendation:** Adapt (cite for E9).
- **Category:** 2

#### E3. Variable-level missingness / completeness rate
- **Metric:** For variable v over the time grid (bins b of width Δ) across trajectories i:
  `Miss_v = 1 − (1/Σ_i B_i) Σ_i Σ_b 1[v observed in (i,b)]`.
  Also report a per-trajectory version and a version stratified by time since admission.
- **Literature:** Completeness in Kahn 2016 and Weiskopf & Weng 2013. The formula is ours, a standard operationalization.
- **Purpose:** Basic coverage of each state feature at the chosen decision resolution.
- **Requirements:** Raw event table; bin width Δ.
- **Computation:** Group-by count.
- **Complexity:** O(N_events).
- **Strengths:** Trivial and interpretable.
- **Limitations:** Depends on Δ. Says nothing about the missingness mechanism.
- **Applicability:** Direct.
- **Timing:** Pre-training.
- **Validation:** Not applicable (descriptive).
- **Recommendation:** Implement.
- **Category:** 1

#### E4. Little's MCAR test
- **Metric:** For missingness patterns j = 1..J with n_j cases and p_j observed variables:
  `d² = Σ_j n_j (ȳ_j − μ̂_j)ᵀ Σ̂_j⁻¹ (ȳ_j − μ̂_j)`
  Here μ̂, Σ̂ are ML (EM) estimates restricted to the observed variables. Under MCAR (and multivariate normality), d² is asymptotically χ² with Σ_j p_j − p degrees of freedom.
- **Literature:** Little 1988, JASA 83(404):1198–1202, doi:10.2307/2290157. Abstract plus secondary documentation only. The formula above follows standard textbook and software descriptions, not a reading of the original.
- **Purpose:** Global test of MCAR versus "not MCAR".
- **Requirements:** A cross-sectional matrix, e.g. a per-bin snapshot or per-patient summaries. Approximate normality.
- **Computation:** EM estimation plus a quadratic form. Implementations exist: R `naniar::mcar_test`, `misty::na.test`.
- **Complexity:** EM is O(N·p²) per iteration. It becomes unstable when there are many patterns.
- **Strengths:** One global statistic, so no multiple-testing problem.
- **Limitations:** Reported to be conservative in small samples, sensitive to non-normality, and low-powered. **Failing to reject does not establish MCAR.** It cannot separate MAR from MNAR. It is not designed for longitudinal or irregular data.
- **Applicability:** ICU data are almost certainly not MCAR, so the test mostly confirms the obvious. It is useful as a baseline sanity check only.
- **Timing:** Pre-training.
- **Validation:** Original simulations. Later critiques (e.g. Aleksić 2025, seen only as a snippet) report type-I error distortion.
- **Recommendation:** Adapt (per-bin snapshots), but report as secondary.
- **Category:** 2

#### E5. Informativeness of the observation process (predictive test)
- **Metric:** Informativeness score
  `IM = AUROC(f(R_{1:t}, δ_{1:t}) → Y) − 0.5`.
  R are missingness indicators, δ are time-since-last-observation, and Y is the outcome (mortality) or the *next action* A_t. A conditional version compares models with and without R: `ΔIM = AUROC(S_t, R_t) − AUROC(S_t)`. A significant ΔIM means the observation process carries information beyond the observed values.
- **Literature:**
  - Agniel, Kohane & Weber 2018, BMJ 361:k1479, doi:10.1136/bmj.k1479: lab ordering *timing* predicted 3-year survival better than the result in 118 of 174 tests.
  - Lipton, Kale & Wetzel 2016, MLHC, PMLR 56 (arXiv:1606.04130): missingness indicators improve classification; *which* tests are run can be more predictive than their values.
  - Che et al. 2018, Sci Rep 8:6085, doi:10.1038/s41598-018-24271-9 (GRU-D): uses masking and time intervals; reports missingness correlated with labels.
  - Sisk et al. 2021, JAMIA 28:155–166: review of "informative presence/observation".
  - All abstract-level only. Our score is a formalization *derived from* these empirical demonstrations. None of the papers defines it as a metric.
- **Purpose:** Quantify how much the measurement process encodes clinician beliefs. For an MDP this matters in two ways: (a) state definitions that drop R lose information, so Markov violation is likely; (b) R is a proxy for unrecorded clinician information, which signals confounding risk.
- **Requirements:** Outcome or action labels, and a cross-fitted predictor (logistic regression or gradient-boosted trees).
- **Computation:** K-fold cross-fitting; bootstrap CI for ΔIM.
- **Complexity:** Cost of model training. Moderate.
- **Strengths:** Directly grounded in clinical evidence. Model-agnostic.
- **Limitations:** Predictive association, not mechanism. It cannot distinguish MAR (explained by observed history) from MNAR without the conditional version, and even then only relative to the covariates used. Depends on model capacity.
- **Applicability:** High.
- **Timing:** Pre-training. Needs a simple auxiliary predictor, not an RL policy.
- **Validation:** Synthetic injection, e.g. simulate MCAR versus action-dependent observation and check that ΔIM separates them.
- **Recommendation:** Implement.
- **Category:** 2

#### E6. Visit/observation-process modelling (irregular observation)
- **Metric:** Model the observation intensity λ_v(t | history), e.g. Andersen–Gill or Poisson-type regression of measurement counts on past values and actions. Test whether past outcome values or actions predict observation intensity (the "visit process" assumptions: completely random, at random, not at random).
- **Literature:**
  - Pullenayegum & Lim 2016, Stat Methods Med Res 25(6):2992–3014, doi:10.1177/0962280214536537. Abstract only.
  - Inverse-intensity weighting extension: Tompkins, Dubin & Wallace, arXiv:2405.15740. Title and snippet only, so listed as an unverified lead.
- **Purpose:** A more principled version of E5. It also enables inverse-intensity weighting.
- **Requirements:** Event-level timestamps.
- **Computation:** Recurrent-event regression.
- **Complexity:** Moderate.
- **Strengths:** Makes visit-process assumptions explicit.
- **Limitations:** The review concludes that no single method fits all visit processes. Dependence on *unobserved* values (MNAR) is untestable.
- **Applicability:** Medium. Useful for lab tests; less so for near-continuous vitals.
- **Timing:** Pre-training.
- **Validation:** Simulation.
- **Recommendation:** Investigate.
- **Category:** 2–3

#### E7. Sampling-interval irregularity
- **Metric:** For variable v and patient i, let the inter-observation gaps be Δ_k = t_{k+1} − t_k. Compute:
  - `CV_v = sd(Δ)/mean(Δ)`
  - gap-tail ratio `G_v = P(Δ > c·median(Δ))`
  - coverage at decision resolution: fraction of decision bins with ≥1 observation (equals 1 − E3).
- **Literature:** Shukla & Marlin 2020, arXiv:2012.00168. Survey; abstract only. It frames irregular sampling as breaking fixed-dimensional representations and lists temporal discretization as one primitive. The formulas are ours.
- **Purpose:** Shows whether a fixed-Δ MDP time grid is reasonable, and how much binning/imputation is needed.
- **Requirements:** Timestamps.
- **Computation:** Per-variable diffs.
- **Complexity:** O(N).
- **Strengths:** Simple. Directly informs the choice of Δ.
- **Limitations:** Descriptive only. Irregularity itself may be informative (see E5).
- **Applicability:** High.
- **Timing:** Pre-training.
- **Validation:** Not applicable.
- **Recommendation:** Implement.
- **Category:** 2

#### E8. Timestamp precision and heaping
- **Metric:** Minute-digit heaping index adapted from Whipple:
  `H_m = (n(minute ∈ M)/n) / (|M|/60)`, with M = {0, 30} or {0, 15, 30, 45}. H = 1 means no heaping; H = 60/|M| means all timestamps are heaped.
  Also Myers-type blended index over all 60 minute values: `0.5·Σ_m |share_m − 1/60|`, i.e. total-variation distance from uniform.
  Also the fraction of timestamps with seconds = 0 and the fraction with time = 00:00 (date-only).
- **Literature:** Whipple's and Myers' indices come from demographic age heaping (secondary sources only; no primary citation verified). Terminal-digit preference in blood-pressure *values* (Nieter et al. 2006, Am J Hypertens) was seen only as a citation in a reference list. **No clinical-timestamp application was found; this is a research candidate.**
- **Purpose:** Detect coarse or retrospective charting. If timestamps are heaped, intra-bin ordering of state and action is unreliable.
- **Requirements:** Raw timestamps.
- **Computation:** Histogram.
- **Complexity:** O(N).
- **Strengths:** Cheap, and the result is easy to interpret.
- **Limitations:** Rounded timestamps may reflect a protocol (e.g. hourly vitals) rather than error. It shows imprecision, not its direction or magnitude.
- **Applicability:** High for manually charted items such as MIMIC `chartevents` and input events.
- **Timing:** Pre-training.
- **Validation:** Compare device-streamed and manually charted items.
- **Recommendation:** Implement as a novel adaptation.
- **Category:** 3

#### E9. Documentation lag (charting delay)
- **Metric:** `L = storetime − charttime` per manually entered record. Report median, P90, and the fraction with L > Δ (records that would *not be available* at the decision time of their bin).
- **Literature:** The MIMIC-III/IV `chartevents` documentation defines `charttime` as the closest proxy for measurement time and `storetime` as the time of manual entry/validation (mimic.mit.edu; documentation, not a paper). Conceptual link to "currency" (Weiskopf & Weng 2013). A pediatric sepsis study reporting mean EHR entry delays appeared only as an unidentified snippet. **No ICU charting-lag study was verified.**
- **Purpose:** Measures timeliness. Also detects look-ahead risk: data used at time t may not have existed in the record until later.
- **Requirements:** Two timestamps. These exist in MIMIC for manually validated items, not for all.
- **Computation:** Difference of the two timestamps.
- **Complexity:** O(N).
- **Strengths:** Directly measurable in MIMIC.
- **Limitations:** `charttime` is itself a proxy whose error is unknown, so L is a lower bound on true lag. Semantics differ for auto-validated device data.
- **Applicability:** High (MIMIC). Dataset-specific elsewhere.
- **Timing:** Pre-training.
- **Validation:** Cross-check against high-frequency waveform data where available.
- **Recommendation:** Implement.
- **Category:** 2

#### E10. State–action temporal alignment / look-ahead leakage audit
- **Metric:** For each decision bin [τ_b, τ_b + Δ):
  - (a) leakage rate: fraction of state features aggregated over the bin whose timestamp is later than the first action in the bin, `LR = #{obs: t_obs > t_first_action(b), obs ∈ s_b} / #{obs ∈ s_b}`;
  - (b) reverse-causality rate: fraction of transitions where the action precedes the state measurements it is supposed to respond to.
- **Literature:**
  - Komorowski et al. 2018, Nat Med 24:1716–1720, doi:10.1038/s41591-018-0213-5. The 4-h bins, 750 k-means states and 25 actions are confirmed only via secondary sources.
  - Gottesman et al. 2018, arXiv:1805.12298: choices in summarizing patient history and confounders can make policy-value estimates unreliable. Abstract only.
  - Gottesman et al. 2019, Nat Med 25:16–18, doi:10.1038/s41591-018-0310-5.
  - Luo et al. 2024, ICML, PMLR 235:33432–33465 (arXiv:2405.18556): RL results vary substantially with the MDP formulation and the evaluation metric. Abstract only.
  - Jeter et al. 2019, arXiv:1902.03271: critique of the AI Clinician. Only the abstract/intro was visible; specific technical objections were **not** verified.
- **Purpose:** With coarse bins, the state can contain post-action physiology. A policy may then appear to "respond" to effects of its own treatment.
- **Requirements:** Event-level timestamps for states and actions.
- **Computation:** Interval joins.
- **Complexity:** O(N log N).
- **Strengths:** Directly targets an MDP-specific failure mode.
- **Limitations:** The formula is ours. Its validity depends on timestamp accuracy (E8, E9).
- **Applicability:** High.
- **Timing:** Pre-training, given a candidate MDP discretization.
- **Validation:** Sweep Δ and report LR(Δ).
- **Recommendation:** Implement.
- **Category:** 3 (formalization new; motivation established)

#### E11. Trajectory fragmentation and censoring
- **Metric:** Per trajectory:
  - number of gaps with no observations of key features longer than G hours, `F_i`;
  - segment-length distribution;
  - censoring rate = fraction of trajectories ending without a terminal outcome (discharge/death) inside the window;
  - check whether censoring is informative by regressing the censoring hazard on history (as in IPCW).
- **Literature:** Robins, Hernán & Brumback 2000, Epidemiology 11(5):550–560 (MSM/IPTW). The IPCW concept is supported only by secondary sources, e.g. Howe et al., PMC3105434, not read. Formulas are ours.
- **Purpose:** Fragmented episodes break the assumption of a terminal reward and bias returns. Informative censoring is a selection-bias analogue of confounding.
- **Requirements:** Admission/discharge timestamps and outcome.
- **Computation:** Survival-style tabulation.
- **Complexity:** O(N).
- **Strengths:** Straightforward.
- **Limitations:** Informative censoring on unobservables is not testable.
- **Applicability:** High.
- **Timing:** Pre-training.
- **Validation:** Not applicable.
- **Recommendation:** Implement.
- **Category:** 2

### Part F: confounding and identifiability

#### F1. Identification assumptions for DTRs: sequential exchangeability, positivity, consistency (framework)
- **Metric:** None directly. These are the assumptions under which the g-formula identifies the value of a regime:
  - sequential exchangeability `Y^g ⊥ A_t | H_t`;
  - sequential positivity `P(A_t = a | H_t = h) > 0` for all (a, h) reachable under g;
  - consistency.
- **Literature:**
  - Robins 1986, Mathematical Modelling 7:1393–1512, doi:10.1016/0270-0255(86)90088-6 (origin of the g-formula; secondary sources only).
  - Hernán & Robins, *Causal Inference: What If* (2020), Chapman & Hall/CRC, ch. 19. Secondary summaries only; wording not checked against the book.
  - Murphy 2003, JRSS-B 65(2):331–355, doi:10.1111/1467-9868.00389.
  - Chakraborty & Moodie 2013, Springer, doi:10.1007/978-1-4614-7428-9.
- **Purpose:** Defines what the data must satisfy for offline policy evaluation and learning to be causal.
- **Strengths / Limitations:** Positivity is partially checkable (F2). **Exchangeability and consistency are not testable from observational data.**
- **Applicability:** Conceptual backbone for Part F.
- **Timing:** Not applicable.
- **Validation:** Not applicable.
- **Recommendation:** Adopt as the framework.
- **Category:** 1 (framework); 4 (as a test)

#### F2. Sequential positivity / overlap diagnostics
- **Metric:**
  - Estimate the behavior policy π̂_b(a | s_t), cross-fitted.
  - Report the fraction of (state, action) visits with π̂_b < ε.
  - Report the per-action distribution of π̂_b.
  - Report a state-cluster action-support count: number of actions with ≥k samples per discrete state.
  - Report the Kish effective sample size of cumulative importance ratios for a reference policy: `ESS = (Σ w_i)² / Σ w_i²`.
- **Literature:**
  - Petersen et al. 2012, Stat Methods Med Res 21(1):31–54, doi:10.1177/0962280210386207: parametric-bootstrap diagnostic for sparsity; responses include restricting covariates, sample or target parameter. Abstract only.
  - Komorowski's own response (arXiv:1903.02345) reports how often the suggested action was observed per state. This is an example of support checking; snippet only.
  - Kish ESS is standard; no citation verified.
- **Purpose:** Detect regions where no policy other than the behavior policy can be evaluated.
- **Requirements:** A propensity model.
- **Computation:** Classifier plus summaries.
- **Complexity:** Moderate.
- **Strengths:** The only identification assumption that is (partially) checkable from the data.
- **Limitations:** Estimated propensities inherit model misspecification. ESS needs a candidate target policy (e.g. "clinician-majority" or uniform). Near-deterministic clinical practice makes violations common.
- **Applicability:** High.
- **Timing:** Pre-training. The behavior-policy model only; ESS for reference policies.
- **Validation:** Simulation with known π_b.
- **Recommendation:** Implement.
- **Category:** 1–2

#### F3. Covariate balance (standardized mean difference)
- **Metric:** `SMD_x = (x̄_treated − x̄_control)/sqrt((s²_t + s²_c)/2)`, computed after weighting by 1/π̂_b, per time step and per action contrast. The common rule of thumb is |SMD| < 0.1.
- **Literature:** Austin 2009, Stat Med 28 (balance diagnostics; abstract only; DOI not confirmed in the search). Brookhart et al. 2010, Med Care 48(6):S114–S120 (confounding in database studies).
- **Purpose:** Checks whether the propensity model balances the *measured* confounders.
- **Requirements:** Binary or dichotomized actions.
- **Computation:** Simple.
- **Complexity:** O(N·p).
- **Strengths:** Standard practice in pharmacoepidemiology.
- **Limitations:** **Says nothing about unmeasured confounders.** The 0.1 threshold is arbitrary and sensitive to sample size. Multi-step extension is ad hoc.
- **Applicability:** Medium. Needs adaptation to sequential actions with more than two levels.
- **Timing:** Pre-training.
- **Validation:** Not applicable.
- **Recommendation:** Adapt.
- **Category:** 2

#### F4. E-value
- **Metric:** For an observed risk ratio RR > 1: `E = RR + sqrt(RR·(RR − 1))`. This is the minimum strength of association (on the RR scale) that an unmeasured confounder would need with both treatment and outcome to explain away the effect. It can also be computed for a CI limit.
- **Literature:** VanderWeele & Ding 2017, Ann Intern Med 167(4):268–274, doi:10.7326/M16-2607. Abstract/bibliographic only; the formula is the standard one.
- **Purpose:** Single-number robustness to unmeasured confounding of a point-exposure association.
- **Requirements:** An effect estimate (RR, or OR/HR approximations).
- **Computation:** Closed form.
- **Complexity:** O(1).
- **Strengths:** Assumption-light and interpretable.
- **Limitations:** Point exposure, not sequential. It needs an effect estimate, e.g. "treated vs. not at step t" or an OPE contrast. It does not tell whether such a confounder exists.
- **Applicability:** Medium: for per-step treatment–outcome associations, or as a summary of an OPE contrast.
- **Timing:** Pre-training, for associations. Needs an estimate otherwise.
- **Validation:** Not applicable.
- **Recommendation:** Adapt.
- **Category:** 2

#### F5. Rosenbaum bounds / marginal sensitivity model (MSM, Γ)
- **Metric:**
  - Rosenbaum: for units with equal x, the treatment odds may differ by at most a factor Γ ≥ 1.
  - MSM (Tan 2006): `Γ⁻¹ ≤ [e(x,u)/(1−e(x,u))] / [e(x)/(1−e(x))] ≤ Γ`, with e the propensity score.
  - Report Γ*, the smallest Γ at which the conclusion changes.
- **Literature:**
  - Rosenbaum 2002, *Observational Studies* (2nd ed.), Springer, doi:10.1007/978-1-4757-3692-2. Confirmed only via R `rbounds` documentation.
  - Tan 2006, JASA 101(476):1619–1637, doi:10.1198/016214506000000023. Abstract plus secondary description.
- **Purpose:** Calibrated sensitivity analysis. Basis of most confounded-OPE work (F6).
- **Requirements:** Propensity model; an estimand.
- **Computation:** Rosenbaum needs matched designs. MSM bounds via linear programming or sharp estimators.
- **Complexity:** Moderate.
- **Strengths:** Interpretable Γ.
- **Limitations:** Γ itself is not identified; plausible Γ must come from domain knowledge. Point-treatment in its original form.
- **Applicability:** Medium (per-step).
- **Timing:** Needs an estimand.
- **Validation:** Not applicable.
- **Recommendation:** Adapt.
- **Category:** 2

#### F6. Sequential / longitudinal sensitivity analysis and confounding-robust OPE
- **Metric:** Worst-case lower and upper bounds on V(π_e) under a per-step sensitivity model with parameter Γ (or a bias function).
- **Literature:**
  - Brumback et al. 2004, Stat Med 23(5):749–767, doi:10.1002/sim.1657: MSM for repeated measures; a sensitivity parameter plus a user-specified bias function; augmented IPTW.
  - Namkoong et al. 2020, NeurIPS (arXiv:2003.05623): worst-case bounds. When confounders affect *every* decision, even small per-step confounding can badly bias OPE. They propose a "one-decision confounding" model with a consistent loss-minimization procedure; sepsis and autism simulations. Abstract read; the exact parameterization was not read.
  - Kallus & Zhou 2020, NeurIPS (arXiv:2002.04518): sharp bounds for infinite-horizon OPE under a sensitivity model. Set membership is a linear program, but the support function is non-convex, so they use projected gradient descent. Abstract only.
  - Kallus & Zhou 2018, NeurIPS (arXiv:1805.08593): confounding-robust policy *improvement*, a minimax-regret policy relative to a baseline over an uncertainty set of propensity weights.
- **Purpose:** Turns "is there confounding?" into "how much confounding would overturn the conclusion?"
- **Requirements:** **A target policy π_e** and a behavior-policy model.
- **Computation:** Optimization. Non-convex for Kallus & Zhou 2020.
- **Complexity:** High.
- **Strengths:** RL-specific; sharp bounds.
- **Limitations:** Not purely pre-training. Γ must be chosen externally. Bounds can be vacuous with small datasets.
- **Applicability:** Post-hoc evaluation layer, or with reference policies (e.g. "always treat more").
- **Timing:** Needs a target policy.
- **Validation:** Simulators with known confounding.
- **Recommendation:** Investigate. Possibly a "Γ needed to flip value ranking of two reference policies" indicator.
- **Category:** 3

#### F7. Negative controls (exposure and outcome)
- **Metric:**
  - Negative-control outcome (NCO): an outcome not plausibly caused by the action but sharing its confounders. Test the association A_t → NCO conditional on H_t; a non-null association indicates residual confounding or bias.
  - Negative-control exposure (NCE): analogous, with an exposure that cannot affect the outcome.
- **Literature:**
  - Lipsitch, Tchetgen Tchetgen & Cohen 2010, Epidemiology 21(3):383–388, doi:10.1097/EDE.0b013e3181d61eeb.
  - Shi, Miao & Tchetgen Tchetgen 2020, Curr Epidemiol Rep 7(4):190–202, doi:10.1007/s40471-020-00243-4 (arXiv:2009.05641).
  - Both abstract-level.
- **Purpose:** The main *empirical* falsification test for unmeasured confounding.
- **Requirements:** Domain-justified NC variables. ICU candidates (ours, untested): an outcome measured before the action (a "pre-treatment outcome"), or labs unrelated to vasopressors.
- **Computation:** Regression.
- **Complexity:** Low.
- **Strengths:** Falsification-oriented.
- **Limitations:** Can detect but not prove absence of bias. NCs may lack both sensitivity and specificity. Validity rests on untestable assumptions (U-comparability).
- **Applicability:** Medium. The work is in choosing NCs.
- **Timing:** Pre-training.
- **Validation:** Not applicable.
- **Recommendation:** Adapt.
- **Category:** 2

#### F8. Proximal causal inference / proximal RL
- **Metric:** Identification of the policy value via treatment-side and outcome-side proxies (Z, W) of the latent confounder, through bridge functions (proximal g-formula).
- **Literature:**
  - Tchetgen Tchetgen et al. 2020, arXiv:2009.10982; published as Stat Sci 39(3):375–390 (2024), doi:10.1214/23-STS911. Covers point and time-varying treatments.
  - Bennett & Kallus, arXiv:2110.15332; Oper Res 72(3):1071–1086 (2024), doi:10.1287/opre.2021.0781: proximal RL for OPE in POMDPs, efficient estimators, sepsis simulation.
  - Abstract-level.
- **Purpose:** An estimation route if proxies exist. For our framework, the useful idea is a "proxy availability" check (e.g. the missingness indicators from E5 as candidate proxies; ours, speculative).
- **Requirements:** Valid proxies satisfying completeness conditions. These conditions are untestable.
- **Computation:** Min-max / integral-equation estimation.
- **Complexity:** High.
- **Strengths:** Allows identification despite unmeasured confounding.
- **Limitations:** Strong untestable assumptions. Not a diagnostic.
- **Applicability:** Low as a metric.
- **Timing:** Needs a target policy.
- **Validation:** Not applicable.
- **Recommendation:** Investigate (discussion only).
- **Category:** 3–4

#### F9. OPE in partially observable environments (decoupled POMDP)
- **Metric:** None for diagnostics. Identification result for OPE in a "decoupled POMDP" with separate observed and unobserved processes.
- **Literature:** Tennenholtz, Mannor & Shalit 2020, AAAI 34(06):10276–10283, doi:10.1609/aaai.v34i06.6590 (arXiv:1909.03739). Abstract only.
- **Purpose:** Formalizes that ICU data under unobserved state are POMDP data, so standard MDP OPE is biased.
- **Limitations:** Needs structural assumptions; not a data check.
- **Applicability:** Conceptual (motivates Markov and confounding tests).
- **Timing:** Not applicable.
- **Recommendation:** Cite only.
- **Category:** 4

#### F10. Counterfactual non-identifiability (Gumbel-max SCM)
- **Metric:** None. The result is that multiple SCMs reproduce the same observed transition distributions yet imply different counterfactuals. The authors impose counterfactual stability via a Gumbel-max SCM.
- **Literature:** Oberst & Sontag 2019, ICML, PMLR 97:4881–4890 (arXiv:1905.05824). Abstract only.
- **Purpose:** Shows that trajectory-level counterfactual claims are not identified from data even with no confounding. The data constrain only interventional, not counterfactual, quantities.
- **Limitations:** A modelling assumption, not a diagnostic.
- **Applicability:** Supports the "cannot be assessed" section.
- **Recommendation:** Cite only.
- **Category:** 4

#### F11. MDP ambiguity in offline meta-RL
- **Metric:** None.
- **Literature:** Dorfman, Shenfeld & Tamar 2021, NeurIPS 34 (proceedings hash 248024541dbda1d3fd75fe49d1a4df4d; code github.com/Rondorf/BOReL). Abstract read.
- **What it shows:** The setting is *offline meta-RL*. A meta-agent learns from training logs of N agents, each trained on a different task, aiming for a Bayes-optimal (VariBAD-style) policy. Learning the belief from offline data creates an identifiability problem they call "MDP ambiguity": per a paper snippet, without suitably collected data it can make learning an effective offline Bayesian RL solution impossible. They propose data-collection/modification fixes, e.g. reward relabelling when ambiguity comes only from reward differences.
- **Relevance:** Analogy only. Behavior-policy-dependent coverage can leave the environment (here: the patient's MDP) unidentifiable from the data. This is **not** a confounding or clinical result and should not be cited as one.
- **Recommendation:** Cite cautiously as an analogy.
- **Category:** 4

#### F12. Substitute-confounder methods (Time Series Deconfounder)
- **Metric:** Latent factor model over multiple treatments that infers "substitute confounders" rendering treatments conditionally independent.
- **Literature:** Bica, Alaa & van der Schaar 2020, ICML, PMLR 119:884–895 (arXiv:1902.00450). Abstract only.
- **Purpose:** Estimate treatment effects despite multi-cause hidden confounders.
- **Limitations:** Requires multiple causes and specific factor-model assumptions. Deconfounder-type identification claims have been debated in the literature (not verified here; see unverified leads). Not a diagnostic.
- **Recommendation:** Exclude as a metric; mention.
- **Category:** 4

---

## 3. Proposed separate temporal indicators

Notation: trajectories i = 1..N; decision bins b of width Δ; state variables v; R_{i,b,v} = 1 if v is observed in bin (i, b); event timestamps t; actions A_{i,b}.

| # | Indicator | Formula (proposal) | Derived from |
|---|---|---|---|
| T1 | **Missingness rate** | `MR_v(Δ) = 1 − Σ_{i,b} R_{i,b,v} / Σ_i B_i`. Aggregate as feature-weighted mean or max over core state features. | Completeness (Kahn 2016; Weiskopf & Weng 2013) |
| T2 | **Informative missingness** | `IM_Y = AUROC_cv(R_{≤b}, δ_{≤b} → Y) − 0.5` (marginal); `ΔIM_A = AUROC_cv(S_b, R_b → A_b) − AUROC_cv(S_b → A_b)` (conditional, on the next action). Secondary: Little's d² p-value on per-bin snapshots. | Agniel 2018; Lipton 2016; Che 2018; Sisk 2021; Little 1988 |
| T3 | **Sampling-interval irregularity** | `CV_v = sd(Δt)/mean(Δt)` pooled within patient; `GTR_v = P(Δt > 3·median Δt)`; `Δ-mismatch_v = median(Δt_v)/Δ` (>1 means the decision grid is finer than the measurement frequency, so imputation dominates the state). | Shukla & Marlin 2020 (framing); formulas ours |
| T4 | **Timestamp precision / heaping** | `H_{M} = (n_{min∈M}/n)/(|M|/60)` for M = {0,30}; `TV_min = ½ Σ_{m=0}^{59} |p_m − 1/60|`; `Z_sec = P(seconds = 0)`. Report separately for manual and device items. | Whipple/Myers demographic indices (adapted; no clinical precedent found) |
| T5 | **Documentation lag** | `L = storetime − charttime`; report `median(L)`, `P90(L)`, `LateFrac = P(L > Δ)`. | MIMIC documentation; "currency" (Weiskopf & Weng 2013) |
| T6 | **State–action alignment / leakage** | `LR(Δ) = #{obs ∈ s_b : t_obs > t_firstaction(b)} / #{obs ∈ s_b}`; `RC(Δ) = P(t_firstaction(b) < min t_obs∈s_b)`; plot LR over Δ ∈ {1, 2, 4} h. Optionally combine with T5: an observation counts as leaked if `storetime > t_action`. | Komorowski 2018 (4-h bins); Gottesman 2018; Luo et al. 2024 (formulation sensitivity); formulas ours |
| T7 | **Fragmentation / censoring** | `FR = mean_i #{gaps > G h in core vitals}`; `SegLen` distribution; `CensRate = P(no terminal event within window)`; informative-censoring check: AUROC of history → censoring at b. | IPCW logic (Robins et al. 2000 and later); formulas ours |

Companion confounding-related indicators (Part F, pre-training):
- **C1 positivity:** `P(π̂_b(A_t | S_t) < 0.01)` and action-support per state cluster.
- **C2 ESS** of reference policies.
- **C3 negative-control association** p-value or effect size.
- **C4 Γ\*:** robustness of a per-step treatment–outcome association, via MSM or E-value.

Reporting rule: none of C1–C4 certifies no-confounding. C1 and C3 can *falsify*; C4 *quantifies fragility*.

---

## 4. What cannot be assessed from data alone (confounding)

1. **Sequential exchangeability** (no unmeasured confounding) is untestable from observational data. Every confounding test conditions on observed variables, and any pattern in the observed distribution is compatible with some model that has a hidden confounder. Hernán & Robins frame it as an assumption justified by subject-matter knowledge, guaranteed only by (sequential) randomization.
2. **Consistency / well-defined interventions** (e.g. "vasopressor dose quartile" as a single treatment version) is a definitional, design-level matter.
3. **Negative controls** depend on untestable assumptions (that the NC shares the same confounding structure). A null result does not prove absence of bias (Lipsitch et al. 2010; Shi et al. 2020).
4. **Sensitivity parameters (Γ, E-value thresholds, bias functions)** are not identified. The data give the Γ at which conclusions flip, but not whether that Γ is plausible.
5. **MNAR missingness:** dependence of observation on *unobserved* values cannot be tested. Little's test can reject only MCAR, and conditional informativeness tests assess dependence on observed history only.
6. **Proxy validity / completeness** in proximal methods is untestable.
7. **Counterfactual (trajectory-level) quantities** are not identified even without confounding. Different SCMs fit the same data (Oberst & Sontag 2019).
8. **Partial observability vs. confounding:** a hidden state that drives both clinician actions and outcomes makes the logged process a POMDP (Tennenholtz et al. 2020). Markov tests can show *a* violation but cannot attribute it to confounding versus mis-specified state.
9. **Positivity** is only partially checkable. Estimated propensities near 0 may reflect model error, and structural zeros cannot be told apart from finite-sample sparsity without domain knowledge.
10. **Consequence for the framework:** report confounding as *falsification checks + fragility quantification + an explicit assumption register* (a DAG and a domain-expert questionnaire), never as a pass/fail certification.

---

## 5. References

### Verified
Existence was confirmed via search results giving title, authors, venue and year. Depth of reading was the abstract only unless stated otherwise.

**Temporal data quality**
- Kahn MG et al. (2016). A Harmonized Data Quality Assessment Terminology and Framework for the Secondary Use of Electronic Health Record Data. *eGEMs* 4(1):1244. doi:10.13063/2327-9214.1244. *(full text read via PMC5051581)*
- Weiskopf NG, Weng C (2013). Methods and dimensions of electronic health record data quality assessment: enabling reuse for clinical research. *JAMIA* 20(1):144–151. PMID 22733976. *(DOI 10.1136/amiajnl-2011-000681 is from memory; check it.)*
- Agniel D, Kohane IS, Weber GM (2018). Biases in electronic health record data due to processes within the healthcare system: retrospective observational study. *BMJ* 361:k1479. doi:10.1136/bmj.k1479.
- Che Z, Purushotham S, Cho K, Sontag D, Liu Y (2018). Recurrent Neural Networks for Multivariate Time Series with Missing Values. *Sci Rep* 8:6085. doi:10.1038/s41598-018-24271-9. arXiv:1606.01865.
- Lipton ZC, Kale DC, Wetzel R (2016). Directly Modeling Missing Data in Sequences with RNNs: Improved Classification of Clinical Time Series. *MLHC*, PMLR 56. arXiv:1606.04130.
- Sisk R, Lin L, Sperrin M, Barrett JK, Tom B, Diaz-Ordaz K, Peek N, Martin GP (2021). Informative presence and observation in routine health data: a review of methodology for clinical risk prediction. *JAMIA* 28:155–166.
- Little RJA (1988). A Test of Missing Completely at Random for Multivariate Data with Missing Values. *JASA* 83(404):1198–1202. doi:10.2307/2290157.
- Pullenayegum EM, Lim LS (2016). Longitudinal data subject to irregular observation: a review of methods with a focus on visit processes, assumptions, and study design. *Stat Methods Med Res* 25(6):2992–3014. doi:10.1177/0962280214536537.
- Shukla SN, Marlin BM (2020). A Survey on Principles, Models and Methods for Learning from Irregularly Sampled Time Series. arXiv:2012.00168.
- Hripcsak G, Albers DJ (2013). Correlating electronic health record concepts with healthcare process events. *JAMIA* 20(e2):e311–e318. doi:10.1136/amiajnl-2013-001922.
- Komorowski M, Celi LA, Badawi O, Gordon AC, Faisal AA (2018). The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care. *Nat Med* 24:1716–1720. doi:10.1038/s41591-018-0213-5. *(4-h/750/25 details only via secondary sources)*
- Jeter R, Josef C, Shashikumar S, Nemati S (2019). Does the "Artificial Intelligence Clinician" learn optimal treatment strategies for sepsis in intensive care? arXiv:1902.03271.
- Komorowski M et al. (2019). Understanding the Artificial Intelligence Clinician and optimal treatment strategies for sepsis in intensive care. arXiv:1903.02345.
- Gottesman O et al. (2018). Evaluating Reinforcement Learning Algorithms in Observational Health Settings. arXiv:1805.12298.
- Gottesman O, Johansson F, Komorowski M, Faisal A, Sontag D, Doshi-Velez F, Celi LA (2019). Guidelines for reinforcement learning in healthcare. *Nat Med* 25:16–18. doi:10.1038/s41591-018-0310-5.
- Luo Z, Pan Y, Watkinson P, Zhu T (2024). Position: Reinforcement Learning in Dynamic Treatment Regimes Needs Critical Reexamination. *ICML*, PMLR 235:33432–33465. arXiv:2405.18556.
- MIMIC-III/IV documentation, `chartevents` table (`charttime`, `storetime`). https://mimic.mit.edu/docs/iv/modules/icu/chartevents (documentation, not peer-reviewed).

**Confounding and identifiability**
- Robins J (1986). A new approach to causal inference in mortality studies with a sustained exposure period—application to control of the healthy worker survivor effect. *Mathematical Modelling* 7(9–12):1393–1512. doi:10.1016/0270-0255(86)90088-6.
- Robins JM, Hernán MA, Brumback B (2000). Marginal structural models and causal inference in epidemiology. *Epidemiology* 11(5):550–560.
- Hernán MA, Robins JM (2020). *Causal Inference: What If*. Boca Raton: Chapman & Hall/CRC. *(chapter content via secondary summaries)*
- Murphy SA (2003). Optimal dynamic treatment regimes. *JRSS-B* 65(2):331–355. doi:10.1111/1467-9868.00389.
- Chakraborty B, Moodie EEM (2013). *Statistical Methods for Dynamic Treatment Regimes*. Springer. doi:10.1007/978-1-4614-7428-9.
- Petersen ML, Porter KE, Gruber S, Wang Y, van der Laan MJ (2012). Diagnosing and responding to violations in the positivity assumption. *Stat Methods Med Res* 21(1):31–54. doi:10.1177/0962280210386207.
- Austin PC (2009). Balance diagnostics for comparing the distribution of baseline covariates between treatment groups in propensity-score matched samples. *Stat Med* 28. *(volume/pages/DOI not confirmed)*
- Brookhart MA et al. (2010). Confounding control in healthcare database research: challenges and potential approaches. *Med Care* 48(6):S114–S120.
- Tan Z (2006). A Distributional Approach for Causal Inference Using Propensity Scores. *JASA* 101(476):1619–1637. doi:10.1198/016214506000000023.
- Rosenbaum PR (2002). *Observational Studies* (2nd ed.). Springer. doi:10.1007/978-1-4757-3692-2. *(confirmed only via R `rbounds` docs)*
- VanderWeele TJ, Ding P (2017). Sensitivity Analysis in Observational Research: Introducing the E-Value. *Ann Intern Med* 167(4):268–274. doi:10.7326/M16-2607.
- Brumback BA, Hernán MA, Haneuse SJ, Robins JM (2004). Sensitivity analyses for unmeasured confounding assuming a marginal structural model for repeated measures. *Stat Med* 23(5):749–767. doi:10.1002/sim.1657.
- Namkoong H, Keramati R, Yadlowsky S, Brunskill E (2020). Off-policy Policy Evaluation For Sequential Decisions Under Unobserved Confounding. *NeurIPS 33*. arXiv:2003.05623.
- Kallus N, Zhou A (2020). Confounding-Robust Policy Evaluation in Infinite-Horizon Reinforcement Learning. *NeurIPS 33*. arXiv:2002.04518.
- Kallus N, Zhou A (2018). Confounding-Robust Policy Improvement. *NeurIPS 31*. arXiv:1805.08593.
- Tennenholtz G, Mannor S, Shalit U (2020). Off-Policy Evaluation in Partially Observable Environments. *AAAI* 34(06):10276–10283. doi:10.1609/aaai.v34i06.6590. arXiv:1909.03739.
- Oberst M, Sontag D (2019). Counterfactual Off-Policy Evaluation with Gumbel-Max Structural Causal Models. *ICML*, PMLR 97:4881–4890. arXiv:1905.05824.
- Lipsitch M, Tchetgen Tchetgen E, Cohen T (2010). Negative Controls: A Tool for Detecting Confounding and Bias in Observational Studies. *Epidemiology* 21(3):383–388. doi:10.1097/EDE.0b013e3181d61eeb.
- Shi X, Miao W, Tchetgen Tchetgen E (2020). A Selective Review of Negative Control Methods in Epidemiology. *Curr Epidemiol Rep* 7(4):190–202. doi:10.1007/s40471-020-00243-4. arXiv:2009.05641.
- Tchetgen Tchetgen EJ, Ying A, Cui Y, Shi X, Miao W (2020/2024). An Introduction to Proximal Causal Learning. arXiv:2009.10982; published as "An Introduction to Proximal Causal Inference", *Stat Sci* 39(3):375–390. doi:10.1214/23-STS911.
- Bennett A, Kallus N (2024). Proximal Reinforcement Learning: Efficient Off-Policy Evaluation in Partially Observed Markov Decision Processes. *Oper Res* 72(3):1071–1086. doi:10.1287/opre.2021.0781. arXiv:2110.15332.
- Dorfman R, Shenfeld I, Tamar A (2021). Offline Meta Reinforcement Learning – Identifiability Challenges and Effective Data Collection Strategies. *NeurIPS 34*. *(abstract read)*
- Bica I, Alaa A, van der Schaar M (2020). Time Series Deconfounder: Estimating Treatment Effects over Time in the Presence of Hidden Confounders. *ICML*, PMLR 119:884–895. arXiv:1902.00450.

### Unverified leads (do not cite without checking)
- Nieter et al. (2006). Effect of terminal digit preference on blood pressure measurement and treatment in primary care. *Am J Hypertens*. Seen only as a citation in a reference list.
- Whipple's and Myers' indices: primary demographic sources not identified.
- Aleksić D (2025): critique of Little's MCAR test under non-normality (conference PDF snippet).
- Tompkins G, Dubin JA, Wallace M: flexible inverse probability of treatment and intensity weighting (FIPTIW). arXiv:2405.15740.
- Bonvini M, Kennedy EH, Ventura V, Wasserman L: Sensitivity Analysis for Marginal Structural Models. arXiv:2210.04681. Nonparametric sensitivity for time-varying treatments; looks highly relevant.
- Howe CJ et al.: IPCW and selection bias (PMC3105434). Not opened.
- JAMIA 2023 paper on label leakage beyond the "no time machine" rule, and JAMA Netw Open 2025 (Ramadan, Liu et al.) on ICD-code label leakage. Seen only as snippets.
- Pediatric sepsis study comparing EHR-documented vitals with streamed physiology (documentation delay). Not identified.
- Wong D et al. (2017), JAMIA: time to record vital signs, electronic vs paper, on wards (not ICU).
- Critiques of deconfounder-type identification (e.g. Ogburn, Shpitser & Tchetgen Tchetgen comments on Wang & Blei). From memory only.
- Raghu et al. 2017 (deep RL for sepsis) and Killian et al. 2020 (representation learning for healthcare RL). Not searched.
- Weiskopf & Weng 2013 DOI and Austin 2009 volume/pages. To confirm.
