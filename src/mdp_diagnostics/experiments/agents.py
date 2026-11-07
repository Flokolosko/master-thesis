"""Offline RL agents used *only* to measure downstream consequences in experiments.

Fitted Q iteration (Ernst et al. 2005; Riedmiller 2005) with gradient-boosted
trees over a given state representation, optionally restricted to actions the
behaviour policy supports (a discrete batch-constrained variant in the spirit
of BCQ, Fujimoto et al. 2019). The agents are not part of the diagnostic
framework; they show what happens when one trains a policy on a formulation
the framework has assessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


@dataclass
class FQIAgent:
    """Greedy policy from fitted Q iteration; ``support`` restricts actions to pi_b >= tau."""

    q: HistGradientBoostingRegressor
    n_actions: int
    support: HistGradientBoostingClassifier | None = None
    tau: float = 0.05

    def q_values(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        q = np.column_stack([self.q.predict(np.hstack([X, np.tile(np.eye(self.n_actions)[k], (n, 1))])) for k in range(self.n_actions)])
        if self.support is not None:
            pb = np.zeros((n, self.n_actions))
            pb[:, self.support.classes_.astype(int)] = self.support.predict_proba(X)
            q = np.where(pb >= self.tau, q, -np.inf)
        return q

    def act(self, X: np.ndarray) -> np.ndarray:
        """One-hot greedy action probabilities."""
        a = self.q_values(X).argmax(axis=1)
        return np.eye(self.n_actions)[a]


def fit_fqi(
    X: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    X_next: np.ndarray,
    terminal: np.ndarray,
    n_actions: int,
    gamma: float = 0.95,
    iterations: int = 15,
    constrained: bool = False,
    tau: float = 0.05,
    seed: int = 0,
) -> FQIAgent:
    """Fit Q(s, a) by repeated regression on r + gamma * max_a' Q(s', a')."""
    a = action.astype(int)
    XA = np.hstack([X, np.eye(n_actions)[a]])
    support = None
    if constrained:
        support = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_leaf_nodes=15, random_state=seed).fit(X, a)
    agent = None
    y = reward.copy()
    for _ in range(iterations):
        q = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_leaf_nodes=15, random_state=seed).fit(XA, y)
        agent = FQIAgent(q, n_actions, support, tau)
        y = reward + gamma * np.where(terminal, 0.0, agent.q_values(X_next).max(axis=1))
    return agent
