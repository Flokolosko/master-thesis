# Literature Review: Feasibility Frameworks for RL in Healthcare

This review covers existing guidelines, frameworks, and criteria for applying Markov Decision Processes (MDPs) and Reinforcement Learning (RL) to healthcare.

## 1. Existing Clinical Guidelines and Frameworks

### The "Evidence Ladder" Framework
Healthcare RL feasibility has been categorized into an "Evidence Ladder," proposing that readiness progresses through:
1. **Problem Formulation:** Defining states, actions, and rewards.
2. **Retrospective Identification:** Can historical data estimate the policy?
3. **Policy Estimation:** Algorithm development.
4. **Stress Testing:** Checking edge cases.
5. **Prospective Evaluation:** Real-world pilot.
6. **Lifecycle Monitoring:** Post-deployment surveillance.

### Gottesman et al. (Nature Medicine, 2019)
The foundational "Guidelines for Reinforcement Learning in Healthcare" outlines crucial pitfalls in transitioning from cohorts to practice:
- **Reward Misspecification:** Ensuring rewards align with clinical goals (e.g., survival vs. intermediate markers).
- **Confounding:** Addressing unobserved variables influencing past clinical decisions.
- **Unsafe Exploration:** The impossibility of random "exploration" on real patients.
- **Off-Policy Evaluation (OPE):** The difficulty of estimating new policies from old data reliably.

### Feasibility-Guided RL (FCAF-RL)
This approach imposes technical constraints:
- **Clinician-Defined Feasible Regions:** Bounding the agent's actions to standard medical practice.
- **Conservative Objectives:** Penalizing actions that lack support in the data.

## 2. Suitability of Clinical Problems for MDP Modeling

According to operations research and clinical decision analysis (e.g., Steimle & Denton), a problem fits an MDP when:
- **Sequential Nature:** The problem features a series of decisions over time (e.g., chronic disease management, continuous ICU titration) rather than one-off interventions (e.g., surgery).
- **Stochastic Transitions:** Health states transition probabilistically.
- **Markov Property (Memoryless):** The next state depends only on the current state and action. If history is needed, the state representation must be augmented.
- **Trade-offs:** Short-term vs. long-term competing risks (e.g., toxicity vs. tumor suppression).
- **Observability:** If true states are hidden, Partially Observable MDPs (POMDPs) are necessary.

## 3. Dataset Suitability for Offline RL

Offline RL requires datasets with:
- **Sufficient Coverage (Support):** A diverse set of observed actions to evaluate counterfactuals. Narrow clinical guidelines often yield "action imbalance," making alternative actions unidentifiable.
- **Markovian State Representation:** The EHR must capture the same physiological variables the clinician used to make the decision. Missing data breaks the Markov assumption, causing unobserved confounding.

**Evaluation & Benchmarks:**
- **Off-Policy Evaluation (OPE):** Methods like Fitted Q-Evaluation (FQE) are standard but notoriously fragile in clinical regimes.
- **EpiCare Benchmark:** Evaluates offline RL against healthcare-specific challenges (sparse rewards, variable trajectories).
- **Bellman-Wasserstein Distance (BWD):** An a priori diagnostic to estimate dataset quality for dynamic programming.

## 4. Identifiability and Safety

### Identifiability and MDP Ambiguity
- **BOReL (2021):** Formalized "MDP Ambiguity" — different underlying MDPs can generate the exact same observational dataset, preventing the identification of true causal effects.
- **Medical Dead-ends (Killian et al., 2023):** Emphasized identifying high-risk states where mortality becomes inevitable regardless of policy, a critical aspect of safety-aware identifiability.

### Safety Frameworks
Recent advances focus on robust offline safety constraints:
- **Two-Stage Frameworks (e.g., CPQ-IQL, 2025):** Combine constraint-penalized Q-learning with runtime safety filters aligned to medical guidelines (like the Surviving Sepsis Campaign).
- **Offline Guarded Safe RL (OGSRL, 2025):** Implements an "OOD Guardian" to restrict policies from leading patients into Out-Of-Distribution downstream states.
- **Inverse Constrained RL (ICRL, 2024):** Infers safety constraints directly from expert trajectories using Constraint Transformers instead of relying on manually specified reward penalties.

## Summary

While there are conceptual checklists (e.g., Gottesman et al.) and advanced safety algorithms (e.g., OGSRL), there remains a gap between *conceptual clinical guidelines* and *measurable statistical diagnostics* that a data scientist can compute *before* training an RL agent.
