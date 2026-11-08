"""Offline RL agents used *only* to measure downstream consequences in experiments.

Fitted Q iteration (Ernst et al. 2005; Riedmiller 2005) with gradient-boosted
trees over a given state representation, optionally restricted to actions the
behaviour policy supports (a discrete batch-constrained variant in the spirit
of BCQ, Fujimoto et al. 2019).

Two Q-models are available. ``"gbm"`` refits gradient boosting in every
iteration; it is flexible but FQI with it is not a contraction, so Q keeps
changing slightly between iterations. ``"frozen_trees"`` follows Ernst et al.
(2005, Sec. 4): an extra-trees ensemble is refit for a few warm-up iterations,
then its structure is frozen and only leaf values are updated. The model is
then a fixed averager (Gordon 1995), the FQI operator is a gamma-contraction
and Q converges geometrically. The agents are not part of the diagnostic
framework; they show what happens when one trains a policy on a formulation
the framework has assessed.

``fit_dqn`` adds an offline DQN (Double DQN with target network, Huber loss,
Adam) on the same transitions, implemented in numpy to avoid a deep-learning
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Callable

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor


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


class FrozenTreeQ:
    """Extra-trees ensemble with frozen structure; leaf values are training-target means (an averager)."""

    def __init__(self, forest: ExtraTreesRegressor, XA_train: np.ndarray):
        self.forest = forest
        self.leaves = forest.apply(XA_train).astype(np.int32)
        self.sizes = [est.tree_.node_count for est in forest.estimators_]
        self.values = [np.zeros(n) for n in self.sizes]

    def update(self, y: np.ndarray) -> "FrozenTreeQ":
        for t, n in enumerate(self.sizes):
            cnt = np.bincount(self.leaves[:, t], minlength=n)
            self.values[t] = np.bincount(self.leaves[:, t], weights=y, minlength=n) / np.maximum(cnt, 1)
        return self

    def predict(self, XA: np.ndarray) -> np.ndarray:
        L = self.forest.apply(XA)
        return np.mean([self.values[t][L[:, t]] for t in range(len(self.sizes))], axis=0)


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
    model: str = "gbm",
    smooth: float = 1.0,
    max_iter: int = 100,
    leaves: int = 15,
    n_trees: int = 50,
    min_samples_leaf: int = 10,
    freeze_after: int = 10,
    callback: Callable[[int, FQIAgent], None] | None = None,
) -> FQIAgent:
    """Fit Q(s, a) by repeated regression on r + gamma * max_a' Q(s', a').

    ``smooth`` < 1 damps the regression targets across iterations
    (y <- (1 - smooth) y_prev + smooth y_new). ``callback(iteration, agent)``
    is called after every iteration (used for learning curves). Defaults
    reproduce the original gradient-boosting FQI.
    """
    a = action.astype(int)
    XA = np.hstack([X, np.eye(n_actions)[a]])
    support = None
    if constrained:
        support = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_leaf_nodes=15, random_state=seed).fit(X, a)
    agent, frozen = None, None
    y = reward.copy()
    for it in range(iterations):
        if model == "gbm":
            q = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=leaves, random_state=seed).fit(XA, y)
        elif model == "frozen_trees":
            if frozen is None or it < freeze_after:
                forest = ExtraTreesRegressor(n_estimators=n_trees, min_samples_leaf=min_samples_leaf, random_state=seed, n_jobs=1).fit(XA, y)
                frozen = FrozenTreeQ(forest, XA)
            q = frozen.update(y)
        else:
            raise ValueError(f"unknown model {model!r}")
        agent = FQIAgent(q, n_actions, support, tau)
        if callback is not None:
            callback(it, agent)
        y_new = reward + gamma * np.where(terminal, 0.0, agent.q_values(X_next).max(axis=1))
        y = y_new if it == 0 or smooth >= 1.0 else (1 - smooth) * y + smooth * y_new
    return agent


# --------------------------------------------------------------------------------------
# Offline DQN (numpy implementation, no deep-learning dependency)
# --------------------------------------------------------------------------------------


class MLPQ:
    """Two-hidden-layer ReLU network mapping standardised states to one Q-value per action."""

    def __init__(self, n_in: int, n_actions: int, hidden: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        he = lambda fan_in, fan_out: rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))  # noqa: E731
        self.params = {"W1": he(n_in, hidden), "b1": np.zeros(hidden), "W2": he(hidden, hidden), "b2": np.zeros(hidden),
                       "W3": rng.normal(0.0, 0.01, (hidden, n_actions)), "b3": np.zeros(n_actions)}
        self.mean = np.zeros(n_in)
        self.scale = np.ones(n_in)

    def copy(self) -> "MLPQ":
        other = object.__new__(MLPQ)
        other.params = {k: v.copy() for k, v in self.params.items()}
        other.mean, other.scale = self.mean, self.scale
        return other

    def forward(self, X: np.ndarray, cache: bool = False):
        p = self.params
        x = (X - self.mean) / self.scale
        z1 = x @ p["W1"] + p["b1"]
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ p["W2"] + p["b2"]
        h2 = np.maximum(z2, 0.0)
        q = h2 @ p["W3"] + p["b3"]
        return (q, (x, z1, h1, z2, h2)) if cache else q

    def backward(self, dq: np.ndarray, cache) -> dict:
        p = self.params
        x, z1, h1, z2, h2 = cache
        g = {"W3": h2.T @ dq, "b3": dq.sum(0)}
        dh2 = (dq @ p["W3"].T) * (z2 > 0)
        g |= {"W2": h1.T @ dh2, "b2": dh2.sum(0)}
        dh1 = (dh2 @ p["W2"].T) * (z1 > 0)
        g |= {"W1": x.T @ dh1, "b1": dh1.sum(0)}
        return g


def huber_grad(q: np.ndarray, action: np.ndarray, target: np.ndarray, delta: float = 1.0):
    """Mean Huber loss on Q(s, a) - target and its gradient with respect to all Q outputs."""
    idx = np.arange(len(q))
    diff = q[idx, action] - target
    loss = np.where(np.abs(diff) <= delta, 0.5 * diff**2, delta * (np.abs(diff) - 0.5 * delta)).mean()
    dq = np.zeros_like(q)
    dq[idx, action] = np.clip(diff, -delta, delta) / len(q)
    return float(loss), dq


@dataclass
class DQNAgent:
    """Greedy policy from a Q-network; ``support`` restricts actions to pi_b >= tau (as in FQIAgent)."""

    net: MLPQ
    n_actions: int
    support: HistGradientBoostingClassifier | None = None
    tau: float = 0.05

    def _mask(self, X: np.ndarray, q: np.ndarray) -> np.ndarray:
        if self.support is None:
            return q
        pb = np.zeros((len(X), self.n_actions))
        pb[:, self.support.classes_.astype(int)] = self.support.predict_proba(X)
        return np.where(pb >= self.tau, q, -np.inf)

    def q_values(self, X: np.ndarray) -> np.ndarray:
        return self._mask(X, self.net.forward(X))

    def act(self, X: np.ndarray) -> np.ndarray:
        return np.eye(self.n_actions)[self.q_values(X).argmax(axis=1)]


def fit_dqn(
    X: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    X_next: np.ndarray,
    terminal: np.ndarray,
    n_actions: int,
    gamma: float = 0.95,
    steps: int = 20000,
    batch: int = 256,
    lr: float = 1e-3,
    hidden: int = 64,
    target_update: int = 500,
    polyak: float = 0.0,
    lr_end: float | None = None,
    double: bool = True,
    constrained: bool = False,
    tau: float = 0.05,
    seed: int = 0,
    callback: Callable[[int, DQNAgent], None] | None = None,
    callback_every: int = 1000,
) -> DQNAgent:
    """Offline (batch) DQN: minibatch SGD on a fixed transition set.

    Huber loss, Adam, a hard-updated target network every ``target_update``
    steps (Mnih et al. 2015) and the Double-DQN target (van Hasselt et al.
    2016). With ``constrained`` the bootstrap maximum and the greedy policy
    only use actions the behaviour policy supports (discrete BCQ-style mask).
    ``callback(step, agent)`` runs every ``callback_every`` gradient steps.

    Convergence options: ``polyak`` > 0 replaces the hard target update by a
    soft update theta' <- (1 - polyak) theta' + polyak theta every step
    (Lillicrap et al. 2016); ``lr_end`` decays the learning rate from ``lr``
    to ``lr_end`` with a cosine schedule (Loshchilov & Hutter 2017), so the
    SGD noise vanishes towards the end of training.
    """
    rng = np.random.default_rng(seed)
    a = action.astype(int)
    net = MLPQ(X.shape[1], n_actions, hidden, seed)
    net.mean = X.mean(0)
    net.scale = np.where(X.std(0) > 1e-8, X.std(0), 1.0)
    support = None
    if constrained:
        support = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_leaf_nodes=15, random_state=seed).fit(X, a)
    agent = DQNAgent(net, n_actions, support, tau)
    next_mask = agent._mask(X_next, np.zeros((len(X_next), n_actions)))  # 0 for allowed, -inf for unsupported
    target = net.copy()
    m = {k: np.zeros_like(v) for k, v in net.params.items()}
    v = {k: np.zeros_like(w) for k, w in net.params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    for step in range(1, steps + 1):
        i = rng.integers(0, len(X), batch)
        q_next_t = target.forward(X_next[i]) + next_mask[i]
        if double:
            a_star = (net.forward(X_next[i]) + next_mask[i]).argmax(axis=1)
            boot = q_next_t[np.arange(batch), a_star]
        else:
            boot = q_next_t.max(axis=1)
        y = reward[i] + gamma * np.where(terminal[i], 0.0, boot)
        q, cache = net.forward(X[i], cache=True)
        _, dq = huber_grad(q, a[i], y)
        grads = net.backward(dq, cache)
        lr_t = lr if lr_end is None else lr_end + 0.5 * (lr - lr_end) * (1 + np.cos(np.pi * (step - 1) / steps))
        for k, gk in grads.items():
            m[k] = b1 * m[k] + (1 - b1) * gk
            v[k] = b2 * v[k] + (1 - b2) * gk**2
            net.params[k] -= lr_t * (m[k] / (1 - b1**step)) / (np.sqrt(v[k] / (1 - b2**step)) + eps)
        if polyak > 0:
            for k in net.params:
                target.params[k] += polyak * (net.params[k] - target.params[k])
        elif step % target_update == 0:
            target = net.copy()
        if callback is not None and step % callback_every == 0:
            callback(step, agent)
    return agent
