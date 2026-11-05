"""Action overlap / positivity diagnostics for discrete actions.

The behaviour policy pi_b(a | s) is estimated with trajectory-level
cross-fitting so that every decision is scored by a model that never saw its
trajectory. Positivity concerns whether *every action of interest* has non-zero
probability in the visited states; diagnostics here quantify how close the
estimated policy comes to violating it. They are conditional on the recorded
state and therefore say nothing about confounding by unrecorded information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from mdp_diagnostics.data.schema import Transitions
from mdp_diagnostics.metrics.base import MetricResult
from mdp_diagnostics.metrics.markov import ModelKind, _classifier
from mdp_diagnostics.utils.statistics import cluster_bootstrap_mean, group_folds

OVERLAP_ASSUMPTIONS = [
    "Positivity is assessed relative to the recorded state only (no statement about unmeasured confounders).",
    "Results depend on the behaviour-policy model; misspecification can hide or exaggerate support problems.",
    "Probabilities near the threshold are estimation-noise dominated; see the threshold sensitivity curve.",
]


@dataclass
class BehaviourPolicy:
    """Cross-fitted behaviour-policy probabilities for every decision."""

    proba: np.ndarray  # (n, K) out-of-fold probabilities
    levels: np.ndarray
    observed_index: np.ndarray  # column of the observed action per row
    log_loss: float
    marginal_log_loss: float

    @property
    def observed_proba(self) -> np.ndarray:
        return self.proba[np.arange(len(self.proba)), self.observed_index]


def fit_behaviour_policy(
    trans: Transitions,
    features: np.ndarray | None = None,
    model: ModelKind = "linear",
    n_splits: int = 5,
    seed: int = 0,
) -> BehaviourPolicy:
    if trans.action_levels is None:
        raise ValueError("Overlap diagnostics need discrete actions (discretise continuous actions first).")
    X = trans.state if features is None else features
    y = trans.action
    levels = trans.action_levels
    k = len(levels)
    proba = np.zeros((len(y), k))
    for train, test in group_folds(trans.groups, n_splits, seed):
        present = np.unique(y[train])
        if len(present) == 1:
            proba[test, np.searchsorted(levels, present[0])] = 1.0
            continue
        m = _classifier(model, seed).fit(X[train], y[train])
        cols = np.searchsorted(levels, m.classes_)
        proba[np.ix_(test, cols)] = m.predict_proba(X[test])
    obs = np.searchsorted(levels, y)
    p_obs = np.clip(proba[np.arange(len(y)), obs], 1e-12, 1)
    freq = np.bincount(obs, minlength=k) / len(y)
    return BehaviourPolicy(
        proba=proba, levels=levels, observed_index=obs,
        log_loss=float(-np.log(p_obs).mean()),
        marginal_log_loss=float(-np.log(np.clip(freq[obs], 1e-12, 1)).mean()),
    )


def support_violation_rate(trans: Transitions, bp: BehaviourPolicy, epsilon: float = 0.01, n_boot: int = 300, seed: int = 0) -> MetricResult:
    """Share of decisions in which at least one action has pi_b(a|s) < epsilon."""
    viol = (bp.proba < epsilon).any(axis=1).astype(float)
    est = cluster_bootstrap_mean(viol, trans.groups, n_boot=n_boot, seed=seed)
    per_action = {f"{lv:g}": float((bp.proba[:, j] < epsilon).mean()) for j, lv in enumerate(bp.levels)}
    grid = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    sens = {f"{e:g}": float((bp.proba < e).any(axis=1).mean()) for e in grid}
    return MetricResult.from_estimate(
        "support_violation_rate", est, unit="share of decisions", n_transitions=trans.n,
        n_trajectories=trans.n_trajectories,
        details={"epsilon": epsilon, "per_action_share_below_epsilon": per_action, "threshold_sensitivity": sens},
        assumptions=list(OVERLAP_ASSUMPTIONS),
    )


def observed_action_propensity(trans: Transitions, bp: BehaviourPolicy) -> MetricResult:
    """Lower quantiles of pi_b(a_obs | s); small values produce extreme IS weights."""
    p = bp.observed_proba
    q = np.quantile(p, [0.0, 0.01, 0.05, 0.5])
    return MetricResult(
        "observed_action_propensity", float(q[2]), unit="5th percentile of pi_b(a_obs|s)",
        n_transitions=trans.n, n_trajectories=trans.n_trajectories,
        details={"min": q[0], "p01": q[1], "p05": q[2], "median": q[3],
                 "behaviour_log_loss": bp.log_loss, "marginal_log_loss": bp.marginal_log_loss},
        assumptions=list(OVERLAP_ASSUMPTIONS),
    )


def action_entropy(trans: Transitions, bp: BehaviourPolicy, n_boot: int = 300, seed: int = 0) -> MetricResult:
    """Mean normalised entropy H(pi_b(.|s)) / log K and effective number of actions."""
    p = np.clip(bp.proba, 1e-12, 1)
    h = -(p * np.log(p)).sum(axis=1)
    k = len(bp.levels)
    norm = h / np.log(k) if k > 1 else np.zeros_like(h)
    est = cluster_bootstrap_mean(norm, trans.groups, n_boot=n_boot, seed=seed)
    return MetricResult.from_estimate(
        "normalised_action_entropy", est, unit="0 (deterministic) .. 1 (uniform)",
        n_transitions=trans.n, n_trajectories=trans.n_trajectories,
        details={"effective_number_of_actions_mean": float(np.exp(h).mean()), "n_actions": k},
        assumptions=list(OVERLAP_ASSUMPTIONS),
    )


def behaviour_calibration(trans: Transitions, bp: BehaviourPolicy, n_bins: int = 10) -> MetricResult:
    """Expected calibration error of the cross-fitted behaviour policy (one-vs-rest, pooled).

    All propensity-based diagnostics (support, ESS, Pareto-k) inherit errors of
    pi_b; a poorly calibrated model should be treated as a gate that makes them
    inconclusive.
    """
    k = len(bp.levels)
    onehot = np.eye(k)[bp.observed_index]
    p, y = bp.proba.ravel(), onehot.ravel()
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return MetricResult(
        "behaviour_calibration_ece", float(ece), unit="expected calibration error",
        n_transitions=trans.n, n_trajectories=trans.n_trajectories,
        details={"log_loss": bp.log_loss, "marginal_log_loss": bp.marginal_log_loss, "n_bins": n_bins},
        assumptions=["Calibration is measured out-of-fold over all (decision, action) pairs."],
    )


def action_overlap_index(trans: Transitions, bp: BehaviourPolicy, epsilon: float = 0.05) -> MetricResult:
    """Candidate metric AOI = E_s[(1/|A|) sum_a 1(pi_b(a|s) > epsilon)] (kept for comparison).

    Note: equals 1 - mean share of actions below epsilon; it depends on the
    action discretisation (|A|) and on epsilon, and averages over actions that
    may be clinically irrelevant in a given state.
    """
    aoi = (bp.proba > epsilon).mean(axis=1)
    return MetricResult(
        "action_overlap_index", float(aoi.mean()), unit="share of supported actions",
        n_transitions=trans.n, n_trajectories=trans.n_trajectories, details={"epsilon": epsilon},
        assumptions=list(OVERLAP_ASSUMPTIONS) + ["Depends on |A| and epsilon; not comparable across action discretisations."],
    )


def knn_local_action_diversity(
    trans: Transitions,
    k: int = 25,
    max_queries: int = 5000,
    seed: int = 0,
) -> MetricResult:
    """Model-free overlap check: action diversity among k nearest neighbours.

    For each (sub-sampled) decision, the k nearest decisions from *other*
    trajectories in standardised state space are retrieved. Reports the share of
    decisions whose neighbourhood contains a single action only (locally
    deterministic behaviour) and the mean normalised neighbourhood entropy.
    Independent of any propensity model, but subject to the curse of
    dimensionality and to the chosen distance metric.
    """
    if trans.action_levels is None:
        return MetricResult.failed("knn_local_action_diversity", "Discrete actions required.")
    X = StandardScaler().fit_transform(trans.state)
    rng = np.random.default_rng(seed)
    q = rng.choice(trans.n, size=min(max_queries, trans.n), replace=False)
    extra = min(trans.n - 1, 4 * k)
    nn = NearestNeighbors(n_neighbors=extra + 1).fit(X)
    _, idx = nn.kneighbors(X[q])
    levels = trans.action_levels
    kk = len(levels)
    single, ent = [], []
    for row, qi in zip(idx, q):
        other = row[trans.groups[row] != trans.groups[qi]][:k]
        if len(other) < max(5, k // 2):
            continue
        counts = np.bincount(np.searchsorted(levels, trans.action[other]), minlength=kk)
        p = counts / counts.sum()
        single.append(float((counts > 0).sum() == 1))
        nz = p[p > 0]
        ent.append(float(-(nz * np.log(nz)).sum() / np.log(kk)) if kk > 1 else 0.0)
    if not single:
        return MetricResult.failed("knn_local_action_diversity", "Too few neighbours from other trajectories.")
    return MetricResult(
        "knn_locally_deterministic_share", float(np.mean(single)), unit="share of decisions",
        n_transitions=trans.n, n_trajectories=trans.n_trajectories,
        details={"k": k, "mean_neighbourhood_entropy": float(np.mean(ent)), "n_queries": len(single)},
        assumptions=["Euclidean distance on standardised states is a meaningful similarity.",
                     "Neighbourhoods become uninformative in high-dimensional state spaces."],
    )
