# Empirical Validation Strategy

To prove that the proposed framework and its quantitative metrics are not arbitrary, they must be empirically validated against known RL outcomes.

## 1. Case Studies Using Existing Repositories
The current repository contains implementations for several clinical problems that can serve as testbeds:
1. **Norepinephrine Titration (Cardiac Arrest):** High-frequency, continuous data, known confounding issues.
2. **Delirium Treatment:** Sparse data, highly subjective state representation.
3. **Sepsis Treatment (MIMIC-III):** The classic, widely studied RL benchmark.

## 2. Validation Methodology
Instead of training *new* RL models to maximize reward, we will use the datasets to test the *diagnostics*:

**Experiment 1: Correlating Overlap with OPE Stability**
- Calculate the **Action Overlap Index (AOI)** for different clinical problems.
- Run standard Fitted Q-Evaluation (FQE).
- *Hypothesis:* Problems with lower AOI will exhibit mathematically provable higher variance and wider confidence intervals in OPE estimates.

**Experiment 2: Synthetic Confounding Injection**
- Take a high-quality trajectory dataset.
- Artificially hide key variables (e.g., blood pressure) to induce unobserved confounding.
- Measure the change in **Temporal Observation Reliability (TOR)** and check if it successfully flags the broken Markov assumption before model training fails.

**Experiment 3: State-Action Coverage vs. Extrapolation Error**
- Measure **ESAC** across different datasets.
- Train an unconstrained DQN and a Conservative Q-Learning (CQL) model.
- *Hypothesis:* The difference in policy recommendations between DQN and CQL will be strictly bounded by the ESAC score (low coverage = high divergence).

## 3. Feasibility of Validation
These experiments do not require prospective clinical trials. They rely entirely on retrospective data and simulated ablations, making them highly feasible for a master's thesis within standard academic timeframes.
