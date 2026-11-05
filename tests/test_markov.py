import numpy as np

from mdp_diagnostics.metrics.markov import conditional_information_gain, gcm_test, history_gain, history_gcm


def test_gain_matches_gaussian_cmi():
    """Y = b*H + e with e ~ N(0, 1): I(Y; H) = 0.5 log(1 + b^2 Var H)."""
    rng = np.random.default_rng(0)
    n, b = 6000, 1.0
    groups = np.repeat(np.arange(n // 10), 10)
    base = rng.normal(size=(n, 1))  # irrelevant
    h = rng.normal(size=(n, 1))
    y = b * h[:, 0] + rng.normal(size=n)
    res = conditional_information_gain(base, h, y, groups, "regression", n_boot=200)
    expected = 0.5 * np.log(1 + b**2)
    assert abs(res.value - expected) < 0.03
    assert res.lower < expected < res.upper


def test_gain_near_zero_for_markov(markov_data):
    tr = markov_data.transitions(lags=2)
    res = history_gain(tr, "state", n_boot=200)
    assert res.upper < 0.01


def test_gain_positive_for_second_order(ar2_data):
    tr = ar2_data.transitions(lags=2)
    res = history_gain(tr, "state", n_boot=200)
    assert res.lower > 0.02


def test_gcm_detects_and_controls():
    rng = np.random.default_rng(1)
    n = 3000
    groups = np.repeat(np.arange(n // 10), 10)
    z = rng.normal(size=(n, 1))
    x = z + rng.normal(size=(n, 1))
    y_null = z[:, 0] + rng.normal(size=n)  # y _||_ x | z
    y_alt = z[:, 0] + 0.3 * x[:, 0] + rng.normal(size=n)
    assert gcm_test(z, x, y_null, groups, seed=0).p_value > 0.01
    assert gcm_test(z, x, y_alt, groups, seed=0).p_value < 0.01


def test_gcm_false_alarm_rate_markov():
    from tests.conftest import SCHEMA, make_ar_data
    from mdp_diagnostics import TrajectoryData, TrajectorySchema

    rejections = 0
    reps = 20
    for seed in range(reps):
        d = TrajectoryData(make_ar_data(n_traj=80, T=20, seed=seed), TrajectorySchema(**SCHEMA))
        rejections += history_gcm(d.transitions(lags=2), "state", n_mult=500, seed=seed).p_value < 0.05
    # Expected ~1 of 20; allow sampling noise.
    assert rejections <= 4


def test_action_target_classification(markov_data):
    tr = markov_data.transitions(lags=1)
    res = history_gain(tr, "action", n_boot=100)
    assert res.ok and res.unit == "nats/transition"


def test_history_augmented_state_has_no_spurious_gain():
    """Lag columns inside the state are deterministic next-state components and must be excluded."""
    from tests.conftest import SCHEMA, make_ar_data
    from mdp_diagnostics import TrajectoryData, TrajectorySchema
    from mdp_diagnostics.experiments.perturbations import add_lag_features

    df, cols = add_lag_features(make_ar_data(phi2=0.35), "id", ["s"], 1)
    d = TrajectoryData(df, TrajectorySchema(**{**SCHEMA, "state_cols": ["s", *cols]}))
    tr = d.transitions(lags=2)
    g = history_gain(tr, "state", n_boot=100)
    assert g.details["excluded_deterministic_dims"] == [1]
    assert g.upper < 0.01  # the AR(2) process is Markov in (s_t, s_{t-1})
    assert history_gcm(tr, "state", n_mult=500).p_value > 0.01
