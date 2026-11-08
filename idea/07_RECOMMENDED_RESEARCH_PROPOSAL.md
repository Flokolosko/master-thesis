# Recommended Research Proposal

## Provisional Title
*A Quantitative Feasibility and Readiness Framework for Markov Decision Process Modeling in Retrospective Healthcare Data*

## 1. Research Questions
1. How can qualitative clinical requirements for Reinforcement Learning (e.g., action support, Markovian states) be translated into computationally verifiable quantitative metrics?
2. Can a hierarchical diagnostic pipeline reliably identify datasets that will fail Off-Policy Evaluation (OPE) or exhibit Out-Of-Distribution (OOD) extrapolation *before* policy optimization begins?
3. How do these diagnostic metrics behave across different clinical decision-making problems (e.g., sepsis vs. cardiac arrest)?

## 2. Hypotheses
- **H1:** Datasets scoring low on the Action Overlap Index (AOI) will demonstrate exponentially higher variance in Fitted Q-Evaluation (FQE).
- **H2:** Applying the proposed hierarchical framework will filter out fundamentally unidentifiable MDP problems without requiring computationally expensive RL training loops.

## 3. Methodology
1. **Framework Construction:** Formalize the 5-level checklist and define 3-5 calculable metrics (AOI, ESAC, TOR).
2. **Implementation:** Develop a Python toolkit (`mdp-diagnostics`) capable of taking a standard state-action-reward trajectory dataframe and outputting the feasibility report.
3. **Empirical Validation (Ablation):** 
   - Baseline: Run the toolkit on the existing MIMIC-III Sepsis and eICU Norepinephrine datasets.
   - Perturbation: Systematically degrade the datasets (e.g., dropping actions to reduce overlap, hiding variables to induce confounding).
   - Evaluation: Measure the correlation between the diagnostic scores and the subsequent failure of CQL and FQE.

## 4. Expected Contributions
1. A rigorous, mathematically grounded checklist bridging clinical intuition and statistical identifiability.
2. An open-source Python library for computing *a priori* dataset readiness for offline RL.
3. An empirical demonstration proving that early diagnostics predict downstream OPE and safety failures.

## 5. Execution Plan
- **Month 1:** Finalize mathematical definitions and write the literature review.
- **Month 2:** Implement the metrics in Python.
- **Month 3:** Run baseline validations on existing MIMIC/eICU cohorts.
- **Month 4:** Perform perturbation experiments (synthetic confounding/support removal).
- **Month 5:** Analyze results and draft the thesis.
- **Month 6:** Final review and defense preparation.
