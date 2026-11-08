# Critical Assessment of Thesis Feasibility

## 1. Is there a clear research gap?
**Yes.** While there are conceptual guidelines (e.g., Gottesman et al.) for medical RL and advanced offline safety algorithms, there is a lack of *a priori, computationally verifiable diagnostics* for clinical EHR datasets. Researchers currently find out their data is unsuitable *after* training models and failing OPE.

## 2. What already exists?
- Clinical checklists (Gottesman, Evidence Ladder).
- Theoretical bounds (MDP Ambiguity).
- Safety filters and conservative RL algorithms (CQL, ICRL).

## 3. Which parts of the framework are genuinely novel?
The primary novelty lies in **operationalizing abstract clinical requirements into calculable metrics** (e.g., translating "sufficient support" into an Action Overlap Index) and proving that these metrics predict downstream failure modes across diverse clinical domains (sepsis, cardiac arrest).

## 4. Are new metrics necessary?
We do not strictly *need* to invent completely new mathematics; adapting and rigorously validating existing metrics (like Bellman-Wasserstein Distance or Positivity bounds from causal inference) specifically for the sequential EHR context is a strong contribution.

## 5. Main Methodological Risks
- **Circular Validation:** Risk of defining a metric that just measures the same thing as the evaluation loss. (Mitigated by testing against policy *divergence* and OPE *variance*, rather than just "accuracy").
- **Computational Overhead:** Calculating some metrics (like exact state-action coverage in continuous spaces) might require complex density estimation.
- **Scope Creep:** Trying to solve RL safety while also building a diagnostic framework. The thesis must strictly focus on *diagnostics*, not building the ultimate RL agent.

## 6. Suitability for a Data Science Master's Thesis at TU Wien
**Highly Suitable.** It combines:
- Advanced statistical learning (causal inference, offline RL).
- Applied data science (working with MIMIC/eICU).
- Methodological engineering (building a reusable framework/pipeline).

## 7. Minimum Viable Scope
1. Define the 5-level hierarchical checklist.
2. Implement 3 core metrics mathematically.
3. Validate them using ONE existing dataset (e.g., Norepinephrine titration) by showing how the metrics respond to artificial data degradation.
