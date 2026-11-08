# Research Gap and Novelty Assessment

## 1. The Current State of the Art
The literature (e.g., Gottesman et al., 2019) provides robust *conceptual* guidelines for applying Reinforcement Learning in healthcare. Similarly, technical literature provides algorithms for handling safety (e.g., ICRL, CQL) and evaluating policies (e.g., FQE). 

## 2. The Identified Gap
Despite these advancements, there is a distinct **methodological gap** between high-level clinical checklists and post-hoc technical evaluations:
1. **Lack of A Priori Quantitative Diagnostics:** Most frameworks ask qualitative questions (e.g., "Is there confounding?"). There is no standardized toolkit of *quantitative* metrics to compute *before* training to determine if the data supports an MDP.
2. **Disconnected Checklists:** Clinical suitability, mathematical MDP formulation, dataset quality, and offline RL identifiability are often evaluated in silos.
3. **Over-reliance on OPE:** Researchers often train a model and then struggle with Off-Policy Evaluation (OPE), rather than identifying fundamental dataset flaws (e.g., lack of action overlap) beforehand.

## 3. Proposed Scientific Contributions
This thesis proposes a unified, quantitative framework that addresses these gaps. The novel contributions include:
- **Translation of Qualitative Concepts into Quantitative Metrics:** Developing concrete mathematical formulations for abstract requirements like "adequate action support" or "state representation sufficiency."
- **A Hierarchical Feasibility Pipeline:** Structuring the assessment sequentially to filter out unsuitable problems early, saving computational and research resources.
- **Empirical Validation of Diagnostics:** Testing the proposed metrics on real clinical datasets (e.g., MIMIC-III, eICU) to prove they reliably predict downstream RL failure modes (e.g., MDP ambiguity).

## 4. Genuinely Novel Components vs. Existing Methods
- **Existing:** Gottesman checklist, Bellman-Wasserstein Distance, Propensity scores.
- **Novel Synthesis:** Creating an open-source, reproducible diagnostic pipeline specifically tuned for clinical EHR data.
- **Potential New Metrics:** Developing a "Clinical Action Overlap Index" or a "Sequential Confounding Risk Score" that extends standard causal inference metrics to longitudinal MDPs.
