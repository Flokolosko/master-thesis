import numpy as np
from scipy import stats

from mdp_diagnostics.utils.statistics import (
    cluster_bootstrap_mean, group_folds, kish_ess, pareto_khat, spearman_with_ci, wilson_interval,
)


def test_kish_ess_analytical():
    assert kish_ess(np.ones(100)) == 100
    w = np.array([1.0, 0, 0, 0])
    assert kish_ess(w) == 1
    w = np.array([1.0, 1.0, 2.0])  # (4)^2 / 6
    assert np.isclose(kish_ess(w), 16 / 6)


def test_pareto_khat_recovers_known_shape():
    # Weights with a generalized Pareto tail of known shape k.
    for k_true in (0.2, 0.5, 0.9):
        w = stats.genpareto.rvs(c=k_true, size=20000, random_state=1)
        k_hat = pareto_khat(w)
        assert abs(k_hat - k_true) < 0.12, (k_true, k_hat)


def test_pareto_khat_light_tail_is_small():
    w = np.random.default_rng(0).uniform(0.5, 1.5, 5000)
    assert pareto_khat(w) < 0.2


def test_cluster_bootstrap_mean_point_and_coverage():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(200), 10)
    values = rng.normal(2.0, 1.0, len(groups))
    est = cluster_bootstrap_mean(values, groups, n_boot=500, seed=1)
    assert np.isclose(est.value, values.mean())
    assert est.lower < 2.0 < est.upper


def test_group_folds_no_leakage():
    groups = np.repeat(np.arange(30), 5)
    for train, test in group_folds(groups, 5, seed=0):
        assert not set(groups[train]) & set(groups[test])


def test_wilson_interval_bounds():
    e = wilson_interval(0, 50)
    assert e.lower == 0 and 0 < e.upper < 0.1
    e = wilson_interval(50, 50)
    assert e.upper == 1


def test_spearman_perfect():
    x = np.arange(20.0)
    assert np.isclose(spearman_with_ci(x, x**2).value, 1.0)
