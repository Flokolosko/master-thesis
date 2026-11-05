"""Statistical helpers shared by all diagnostics.

All resampling is done at the level of trajectories (patients), never at the
level of individual transitions, because transitions within one trajectory are
dependent. Cross-fitting likewise splits by trajectory to prevent leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a (1 - alpha) confidence interval."""

    value: float
    lower: float = float("nan")
    upper: float = float("nan")

    def to_dict(self) -> dict:
        return {"value": self.value, "lower": self.lower, "upper": self.upper}


def group_folds(groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Trajectory-level K-fold splits (no trajectory appears in train and test)."""
    unique = np.unique(groups)
    n_splits = int(min(n_splits, len(unique)))
    if n_splits < 2:
        raise ValueError("Cross-fitting needs at least two trajectories.")
    # Shuffle group labels so that fold assignment does not depend on ID order.
    rng = np.random.default_rng(seed)
    relabel = dict(zip(unique, rng.permutation(len(unique))))
    shuffled = np.array([relabel[g] for g in groups])
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=shuffled))


def cluster_bootstrap_mean(
    values: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Mean of ``values`` with a percentile CI from a trajectory-level bootstrap.

    Each replicate draws trajectories with replacement and recomputes the
    transition-weighted mean. Implemented with multinomial trajectory counts,
    which is equivalent to resampling trajectory IDs.
    """
    values = np.asarray(values, dtype=float)
    _, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv).astype(float)
    point = float(sums.sum() / counts.sum())
    if n_boot <= 0 or len(sums) < 2:
        return Estimate(point)
    rng = np.random.default_rng(seed)
    g = len(sums)
    draws = rng.multinomial(g, np.full(g, 1.0 / g), size=n_boot)
    reps = (draws @ sums) / (draws @ counts)
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return Estimate(point, float(lo), float(hi))


def cluster_bootstrap_statistic(
    statistic,
    groups: np.ndarray,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Generic trajectory-level bootstrap for ``statistic(index_array)``."""
    unique, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(unique))]
    point = float(statistic(np.arange(len(groups))))
    if n_boot <= 0 or len(unique) < 2:
        return Estimate(point)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([members[k] for k in pick])
        val = statistic(idx)
        if np.isfinite(val):
            reps.append(val)
    if len(reps) < 10:
        return Estimate(point)
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return Estimate(point, float(lo), float(hi))


def kish_ess(weights: np.ndarray) -> float:
    """Kish effective sample size (sum w)^2 / sum w^2."""
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w)]
    denom = float(np.sum(w**2))
    return float(np.sum(w) ** 2 / denom) if denom > 0 else 0.0


def _gpd_fit_zhang_stephens(excess: np.ndarray) -> tuple[float, float]:
    """Generalized Pareto fit by Zhang & Stephens (2009), as used in PSIS.

    Returns (k, sigma). ``excess`` must be positive and sorted ascending. A weak
    prior shrinks k towards 0.5, following Vehtari et al. (PSIS).
    """
    x = np.sort(np.asarray(excess, dtype=float))
    n = len(x)
    prior = 3.0
    m = 30 + int(np.sqrt(n))
    bs = 1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    bs /= prior * x[int(n / 4 + 0.5) - 1]
    bs += 1.0 / x[-1]
    ks = np.log1p(-bs[:, None] * x[None, :]).mean(axis=1)
    log_lik = n * (np.log(-bs / ks) - ks - 1.0)
    weights = 1.0 / np.exp(log_lik - log_lik[:, None]).sum(axis=1)
    b = float(np.sum(bs * weights))
    k = float(np.mean(np.log1p(-b * x)))
    sigma = -k / b
    k = (k * n + 0.5 * 10) / (n + 10)
    return k, sigma


def pareto_khat(weights: np.ndarray) -> float:
    """Pareto-k diagnostic for importance weights (Vehtari et al., PSIS).

    Fits a generalized Pareto distribution to the largest weights. Values above
    ~0.7 indicate that the importance-sampling estimate is unreliable because
    the weight distribution has (near) infinite variance.
    """
    w = np.sort(np.asarray(weights, dtype=float)[np.isfinite(weights)])
    s = len(w)
    if s < 20 or np.all(w == w[0]):
        return float("nan")
    m = int(np.ceil(min(0.2 * s, 3.0 * np.sqrt(s))))
    tail = w[-m:]
    threshold = w[-m - 1]
    excess = tail - threshold
    excess = excess[excess > 0]
    if len(excess) < 5:
        return float("nan")
    k, _ = _gpd_fit_zhang_stephens(excess)
    return float(k)


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> Estimate:
    """Wilson score interval for a binomial proportion (e.g. power estimates)."""
    if n == 0:
        return Estimate(float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    lo, hi = centre - half, centre + half
    return Estimate(p, 0.0 if lo < 1e-12 else lo, 1.0 if hi > 1 - 1e-12 else hi)


def spearman_with_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> Estimate:
    """Spearman rank correlation with a percentile bootstrap CI over units."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return Estimate(float("nan"))
    point = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if np.ptp(x[idx]) == 0 or np.ptp(y[idx]) == 0:
            continue
        reps.append(stats.spearmanr(x[idx], y[idx]).statistic)
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2]) if reps else (np.nan, np.nan)
    return Estimate(point, float(lo), float(hi))
