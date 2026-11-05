"""Lightweight downstream estimators used *only* to measure consequences in experiments.

They provide the independent criterion (true error) against which diagnostics
are validated; none of them is part of the diagnostic framework itself.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from mdp_diagnostics.data.schema import Transitions


def wis_estimate(trans: Transitions, pb_obs: np.ndarray, pt_obs: np.ndarray, gamma: float = 1.0) -> float:
    """Weighted (self-normalised) trajectory-wise importance sampling estimate."""
    _, inv = np.unique(trans.groups, return_inverse=True)
    log_w = np.bincount(inv, weights=np.log(np.maximum(pt_obs, 1e-300)) - np.log(np.maximum(pb_obs, 1e-12)))
    ret = np.bincount(inv, weights=gamma ** trans.step * trans.reward)
    w = np.exp(log_w - log_w.max())
    return float(np.sum(w * ret) / np.sum(w)) if w.sum() > 0 else float("nan")


def model_based_value(
    trans: Transitions,
    policy,
    n_rollouts: int = 4000,
    horizon: int | None = None,
    gamma: float = 1.0,
    seed: int = 0,
) -> float:
    """Value of ``policy(state_matrix) -> probabilities`` under a fitted linear-Gaussian model.

    The model treats ``trans.state`` as a Markov state: next state ~ N(W [s, a], Sigma),
    reward ~ Ridge([s, a, s']). Rollouts start from observed first states. The
    resulting error relative to the true value is the consequence of an
    inadequate state representation that the experiments measure.
    """
    rng = np.random.default_rng(seed)
    X = np.hstack([trans.state, trans.action_features])
    dyn = Ridge(alpha=1e-3).fit(X, trans.next_state)
    d = trans.next_state.shape[1]
    resid = trans.next_state - dyn.predict(X).reshape(-1, d)
    chol = np.linalg.cholesky(np.atleast_2d(np.cov(resid.T)) + 1e-9 * np.eye(d))
    rew = Ridge(alpha=1e-3).fit(np.hstack([X, trans.next_state, trans.next_state**2]), trans.reward)
    starts = trans.state[trans.step == trans.step.min()]
    H = horizon or int(trans.step.max() - trans.step.min() + 1)
    s = starts[rng.integers(0, len(starts), n_rollouts)]
    k = trans.action_features.shape[1]
    total = np.zeros(n_rollouts)
    for t in range(H):
        p = policy(s)
        a = np.minimum((p.cumsum(axis=1) < rng.random(n_rollouts)[:, None]).sum(axis=1), k - 1)
        af = np.eye(k)[a]
        xa = np.hstack([s, af])
        s_next = dyn.predict(xa).reshape(-1, d) + rng.standard_normal(s.shape) @ chol.T
        total += gamma**t * rew.predict(np.hstack([xa, s_next, s_next**2]))
        s = s_next
    return float(total.mean())
