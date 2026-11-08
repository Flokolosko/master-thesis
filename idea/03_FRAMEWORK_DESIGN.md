# Framework Design: Hierarchical Feasibility Checklists

The proposed framework follows a sequential, hierarchical structure. A failure at any level suggests the problem is either fundamentally unsuitable or requires significant reformulation before proceeding to the next level.

## Level A: Clinical Problem Suitability
*Goal: Determine if the clinical problem is inherently sequential and stochastic.*
- **A1. Sequentiality:** Are there multiple decision points over time?
- **A2. Action Multiplicity:** Are there $\ge 2$ viable, distinct clinical actions at decision points?
- **A3. Probabilistic Transitions:** Is patient response to treatment stochastic?
- **A4. Competing Risks:** Is there a trade-off between immediate and long-term outcomes?
- **Decision:** If no to any, use supervised learning, single-point causal inference, or decision trees.

## Level B: MDP Formulation Validity
*Goal: Assess if the problem can be mathematically expressed as an MDP.*
- **B1. State Definition:** Can the state space be defined using available EHR variables?
- **B2. Action Granularity:** Can continuous treatments (e.g., drug infusions) be meaningfully discretized without losing clinical relevance?
- **B3. Reward Alignment:** Can a quantitative reward function be defined that strictly aligns with clinical objectives (avoiding reward hacking)?
- **B4. Markov Assumption:** Does the defined state vector satisfy $P(S_{t+1} | S_t, A_t) \approx P(S_{t+1} | S_{0:t}, A_{0:t})$?

## Level C: Data Suitability and Quality
*Goal: Determine if the specific dataset is adequate for the formulation.*
- **C1. Trajectory Length:** Are patient trajectories long enough to capture sequential effects?
- **C2. Missingness:** Is the rate of missing data manageable without introducing severe informative missingness bias?
- **C3. Temporal Resolution:** Are timestamps accurate enough to maintain the true sequence of states and actions?

## Level D: Offline RL Feasibility (Identifiability & Support)
*Goal: Verify if a counterfactual policy can be safely learned and evaluated.*
- **D1. Action Overlap (Positivity):** Do clinicians naturally vary their treatments enough in similar states to learn alternative policies?
- **D2. Unobserved Confounding Risk:** Are all variables that influence the clinician's policy $b(A|S)$ recorded in the state $S$?
- **D3. State-Action Coverage:** Are the clinically relevant regions of the state-action space sufficiently populated?

## Level E: Clinical Deployment & Safety Readiness
*Goal: Assess prerequisites for future real-world application.*
- **E1. Interpretability:** Can the resulting policy be explained to a clinician?
- **E2. Safety Constraints:** Can strict "do not harm" rules be programmatically enforced?
- **E3. Fallback Mechanisms:** Is there a safe baseline policy if the agent encounters an Out-Of-Distribution (OOD) state?

## Output Generation
The framework will output a comprehensive **Feasibility Report**, categorizing the problem as:
1. Unsuitable for RL.
2. Suitable for Contextual Bandits (single-step).
3. Suitable for Behavioral Cloning (but not counterfactual optimization).
4. Feasible for Offline RL (with high confidence in OPE).
