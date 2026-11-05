"""Importance-sampling based reliability diagnostics for off-policy evaluation.

These need a *target policy* but no trained value model, so they can be
computed before policy optimisation. When no clinically motivated target policy
is available, a family of reference policies pi_lambda = (1 - lambda) pi_b +
lambda * uniform is used: lambda measures how far a candidate policy deviates
from observed practice, and the resulting effective-sample-size curve shows how
quickly evaluation becomes unreliable as the deviation grows.

High ESS / low Pareto-k are necessary, not sufficient, for reliable OPE: they
do not detect bias from confounding or a misspecified behaviour model.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from mdp_diagnostics.data.schema import Transitions
from mdp_diagnostics.metrics.base import MetricResult
from mdp_diagnostics.metrics.overlap import BehaviourPolicy
from mdp_diagnostics.utils.statistics import kish_ess, pareto_khat

TargetPolicy = Callable[[np.ndarray], np.ndarray]  # states (n, d) -> probabilities (n, K)

OPE_ASSUMPTIONS = [
    "Importance weights use the estimated behaviour policy; its errors propagate into the weights.",
    "Weights are only valid under sequential exchangeability given the recorded state (untestable from data).",
    "Diagnostics describe variance/stability of IS-type estimators, not their bias.",
]


def _trajectory_log_weights(trans: Transitions, bp: BehaviourPolicy, target_proba: np.ndarray, floor: float = 1e-6):
    pb = bp.observed_proba
    pt = target_proba[np.arange(trans.n), bp.observed_index]
    n_floored = int((pb < floor).sum())
    log_rho = np.log(np.clip(pt, 1e-300, None)) - np.log(np.maximum(pb, floor))
    _, inv = np.unique(trans.groups, return_inverse=True)
    log_w = np.bincount(inv, weights=log_rho)
    return log_w, inv, n_floored


def is_weight_diagnostics(
    trans: Transitions,
    bp: BehaviourPolicy,
    target_proba: np.ndarray,
    returns: np.ndarray | None = None,
    label: str = "target",
    n_boot: int = 300,
    seed: int = 0,
) -> MetricResult:
    """ESS share, Pareto-k and (optionally) WIS bootstrap CI width for one target policy.

    ``returns`` are per-trajectory returns aligned with sorted unique groups.
    """
    log_w, inv, n_floored = _trajectory_log_weights(trans, bp, target_proba)
    w = np.exp(log_w - log_w.max())  # scale-free for ESS and k-hat
    n_traj = len(w)
    ess = kish_ess(w)
    # Horizon profile: ESS share of weights truncated after h decisions.
    pb = np.maximum(bp.observed_proba, 1e-6)
    pt = target_proba[np.arange(trans.n), bp.observed_index]
    log_rho = np.log(np.clip(pt, 1e-300, None)) - np.log(pb)
    rank = trans.step - np.array([trans.step[inv == g].min() for g in range(n_traj)])[inv]
    horizon_profile = {}
    for h in (1, 2, 5, 10, 20):
        if h > rank.max() + 1:
            break
        lw = np.bincount(inv, weights=np.where(rank < h, log_rho, 0.0), minlength=n_traj)
        horizon_profile[str(h)] = kish_ess(np.exp(lw - lw.max())) / n_traj
    details = {
        "ess_share_by_horizon": horizon_profile,
        "target": label,
        "ess": ess,
        "pareto_k": pareto_khat(w),
        "max_weight_share": float(w.max() / w.sum()),
        "n_behaviour_prob_floored": n_floored,
    }
    if returns is not None:
        rng = np.random.default_rng(seed)
        wn = np.exp(log_w - log_w.max())
        est = float(np.sum(wn * returns) / np.sum(wn))
        reps = []
        for _ in range(n_boot):
            idx = rng.integers(0, n_traj, n_traj)
            if wn[idx].sum() > 0:
                reps.append(np.sum(wn[idx] * returns[idx]) / np.sum(wn[idx]))
        lo, hi = np.quantile(reps, [0.025, 0.975])
        sd = float(np.std(returns)) or 1.0
        details.update({"wis_estimate": est, "wis_ci": [float(lo), float(hi)],
                        "wis_ci_width_in_return_sd": float((hi - lo) / sd)})
    return MetricResult(
        f"is_ess_share[{label}]", ess / n_traj, unit="ESS / n_trajectories",
        n_transitions=trans.n, n_trajectories=n_traj, details=details,
        assumptions=list(OPE_ASSUMPTIONS) + ["Pareto-k > 0.7 indicates unreliable IS (Vehtari et al., PSIS)."],
    )


def reference_policy_family(bp: BehaviourPolicy, lam: float) -> np.ndarray:
    k = bp.proba.shape[1]
    return (1 - lam) * bp.proba + lam / k


def ess_deviation_curve(
    trans: Transitions,
    bp: BehaviourPolicy,
    lambdas: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0),
    returns: np.ndarray | None = None,
    seed: int = 0,
) -> list[MetricResult]:
    return [
        is_weight_diagnostics(trans, bp, reference_policy_family(bp, lam), returns=returns, label=f"mix_uniform_{lam:g}", seed=seed)
        for lam in lambdas
    ]


def trajectory_returns(trans: Transitions, gamma: float = 1.0) -> np.ndarray | None:
    """Discounted per-trajectory returns from transition rewards (sorted group order)."""
    if trans.reward is None:
        return None
    _, inv = np.unique(trans.groups, return_inverse=True)
    disc = gamma ** trans.step.astype(float)
    return np.bincount(inv, weights=disc * trans.reward)
