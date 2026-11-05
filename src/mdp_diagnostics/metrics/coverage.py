"""State-action coverage and transition-estimability diagnostics."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from mdp_diagnostics.data.schema import Transitions
from mdp_diagnostics.metrics.base import MetricResult


def _cells(trans: Transitions, n_clusters: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    X = StandardScaler().fit_transform(trans.state)
    n_clusters = int(min(n_clusters, max(2, trans.n // 5)))
    labels = KMeans(n_clusters=n_clusters, n_init=4, random_state=seed).fit_predict(X)
    a_idx = np.searchsorted(trans.action_levels, trans.action)
    return labels, a_idx, n_clusters


def esac(trans: Transitions, n_clusters: int = 50, n_min: int = 10, seed: int = 0) -> MetricResult:
    """Candidate ESAC = |{(s,a): N(s,a) >= N_min}| / (|S_cluster| |A|) (kept for comparison).

    Every cell counts equally, regardless of how often the state cluster is
    visited or whether the action is clinically plausible there, and the value
    depends on the arbitrary number of clusters.
    """
    if trans.action_levels is None:
        return MetricResult.failed("esac", "Discrete actions required.")
    labels, a_idx, c = _cells(trans, n_clusters, seed)
    k = len(trans.action_levels)
    counts = np.zeros((c, k))
    np.add.at(counts, (labels, a_idx), 1)
    return MetricResult(
        "esac", float((counts >= n_min).mean()), unit="share of (cluster, action) cells",
        n_transitions=trans.n, n_trajectories=trans.n_trajectories,
        details={"n_clusters": c, "n_min": n_min},
        assumptions=["Depends on the clustering granularity and N_min; cells are unweighted."],
    )


def occupancy_weighted_support(trans: Transitions, n_clusters: int = 50, n_min: int = 10, seed: int = 0) -> MetricResult:
    """For each decision: share of actions with >= n_min observations in its state cluster.

    Averaged over decisions (i.e. weighted by state occupancy), so rarely
    visited regions do not dominate. Also reports the share of decisions whose
    own (cluster, action) cell is under-supported.
    """
    if trans.action_levels is None:
        return MetricResult.failed("occupancy_weighted_support", "Discrete actions required.")
    labels, a_idx, c = _cells(trans, n_clusters, seed)
    k = len(trans.action_levels)
    counts = np.zeros((c, k))
    np.add.at(counts, (labels, a_idx), 1)
    supported = counts >= n_min
    per_decision = supported[labels].mean(axis=1)
    own_unsupported = float((~supported[labels, a_idx]).mean())
    return MetricResult(
        "occupancy_weighted_support", float(per_decision.mean()), unit="mean share of supported actions",
        n_transitions=trans.n, n_trajectories=trans.n_trajectories,
        details={"n_clusters": c, "n_min": n_min, "share_decisions_in_unsupported_cell": own_unsupported},
        assumptions=["State clusters approximate regions of similar patients.", "Result should be checked for sensitivity to n_clusters and n_min."],
    )


def transition_model_stability(trans: Transitions, n_boot: int = 50, alpha: float = 1.0, seed: int = 0) -> MetricResult:
    """Estimability of the transition model from a trajectory-level bootstrap.

    Refits a ridge next-state model on bootstrap samples of trajectories and
    reports the ratio of the mean bootstrap prediction SD (epistemic) to the
    residual SD (aleatoric), averaged over state dimensions. Ratios well below
    1 mean the data pin down the (linear) dynamics; large ratios mean the
    transition estimate is data-limited.
    """
    Xs = StandardScaler().fit_transform(trans.state)
    X = np.hstack([Xs, trans.action_features])
    Y = trans.next_state
    unique, inv = np.unique(trans.groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(unique))]
    rng = np.random.default_rng(seed)
    preds = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(unique), len(unique))
        idx = np.concatenate([members[g] for g in pick])
        preds.append(Ridge(alpha=alpha).fit(X[idx], Y[idx]).predict(X).reshape(len(X), -1))
    preds = np.stack(preds)
    epi = preds.std(axis=0).mean(axis=0)
    resid = (Y - Ridge(alpha=alpha).fit(X, Y).predict(X).reshape(Y.shape)).std(axis=0)
    ratio = epi / np.maximum(resid, 1e-12)
    per_point = (preds.std(axis=0) / np.maximum(resid, 1e-12)).mean(axis=1)
    return MetricResult(
        "transition_epistemic_ratio", float(ratio.mean()), unit="bootstrap SD / residual SD",
        n_transitions=trans.n, n_trajectories=trans.n_trajectories,
        details={"per_dimension": ratio.tolist(), "p90_over_decisions": float(np.quantile(per_point, 0.9))},
        assumptions=["Linear-in-features transition model; estimability of richer models can differ."],
    )


def knn_distance_ratio(reference: np.ndarray, query: np.ndarray, k: int = 10) -> MetricResult:
    """Distribution-shift proxy: median kNN distance of query points to the data,
    relative to the median within-data kNN distance (1 = in-distribution)."""
    sc = StandardScaler().fit(reference)
    R, Q = sc.transform(reference), sc.transform(query)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(R)
    d_ref = nn.kneighbors(R)[0][:, 1:].mean(axis=1)
    d_q = nn.kneighbors(Q, n_neighbors=k)[0].mean(axis=1)
    return MetricResult(
        "knn_distance_ratio", float(np.median(d_q) / np.median(d_ref)), unit="ratio",
        details={"share_beyond_p99_reference": float((d_q > np.quantile(d_ref, 0.99)).mean())},
        assumptions=["Euclidean distance on standardised features."],
    )
