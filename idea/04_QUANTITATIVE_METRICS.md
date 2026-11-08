# Quantitative Metrics for MDP Feasibility

This document proposes mathematical formulations for evaluating the feasibility criteria defined in the framework.

## 1. Action Overlap Index (Positivity Metric)
**Definition:** Measures whether clinicians naturally explore different actions within similar patient states, essential for counterfactual learning.
**Mathematical Formulation:** 
Let $\pi_\beta(a|s)$ be the estimated behavior policy.
$$ \text{AOI} = \mathbb{E}_{s \sim \mathcal{D}} \left[ \frac{\sum_{a \in \mathcal{A}} \mathbb{I}(\pi_\beta(a|s) > \epsilon)}{|\mathcal{A}|} \right] $$
Where $\epsilon$ is a threshold for minimum support (e.g., 0.05).
**Interpretation:** 1.0 means all actions are supported everywhere; values near 0 mean rigid protocols with no overlap.

## 2. Temporal Observation Reliability (TOR)
**Definition:** Measures the regularity and completeness of state observations between decision points.
**Mathematical Formulation:**
$$ \text{TOR} = 1 - \frac{1}{N} \sum_{i=1}^N \frac{\text{MissingVariables}(s_i)}{|S|} $$
**Required Data:** Discretized patient trajectories.
**Interpretation:** A high TOR indicates reliable trajectory sampling; a low TOR suggests the Markov property will be violated due to unobserved changes.

## 3. Effective State-Action Coverage (ESAC)
**Definition:** Evaluates how much of the clinically relevant state-action space is actually represented in the dataset.
**Mathematical Formulation:**
Given a discretized or clustered state space $\mathcal{S}_{clust}$,
$$ \text{ESAC} = \frac{|\{(s, a) \in \mathcal{S}_{clust} \times \mathcal{A} : N(s, a) \ge N_{min} \}|}{|\mathcal{S}_{clust}| \times |\mathcal{A}|} $$
**Threshold:** If $\text{ESAC} < 0.1$, training standard offline RL will likely result in severe OOD extrapolation errors.

## 4. Bellman-Wasserstein Distance (BWD) Proxy
**Definition:** An existing metric adapted to measure how much "useful information" a dataset contains for dynamic programming relative to a random dataset.
**Interpretation:** High BWD implies high OPE error bounds; the dataset cannot reliably evaluate new policies.

## 5. Medical Dead-end Proportion (MDP_Dead)
**Definition:** The fraction of states from which mortality is inevitable ($P(\text{survival}|s, a) < \delta \ \forall a$).
**Interpretation:** High proportions indicate the dataset represents a severely ill cohort where policy optimization may be futile or highly sensitive.
