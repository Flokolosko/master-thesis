import numpy as np
import pandas as pd

from mdp_diagnostics import TrajectoryData, TrajectorySchema
from mdp_diagnostics.metrics import coverage, ope_reliability, overlap
from tests.conftest import SCHEMA, make_ar_data


def test_deterministic_policy_violates_support():
    # Action 1 is never taken; declaring the action space makes that visible.
    d = TrajectoryData(make_ar_data(deterministic=True), TrajectorySchema(**SCHEMA, action_levels=[0.0, 1.0, 2.0]))
    tr = d.transitions()
    bp = overlap.fit_behaviour_policy(tr)
    svr = overlap.support_violation_rate(tr, bp, epsilon=0.01, n_boot=50)
    assert svr.value == 1.0
    assert svr.details["per_action_share_below_epsilon"]["1"] == 1.0


def test_deterministic_policy_without_declared_actions_is_smoothed():
    # Known limitation: a regularised logistic model softens a deterministic
    # threshold rule near the boundary, so the violation share is below 1.
    d = TrajectoryData(make_ar_data(deterministic=True), TrajectorySchema(**SCHEMA))
    tr = d.transitions()
    bp = overlap.fit_behaviour_policy(tr)
    svr = overlap.support_violation_rate(tr, bp, epsilon=0.01, n_boot=50)
    assert 0.5 < svr.value < 1.0
    ent = overlap.action_entropy(tr, bp, n_boot=50)
    assert ent.value < 0.2
    knn = overlap.knn_local_action_diversity(tr, k=15)
    assert knn.value > 0.8


def test_uniform_random_policy_has_full_overlap():
    rng = np.random.default_rng(0)
    df = make_ar_data(seed=3)
    df.loc[df["action"].notna(), "action"] = rng.integers(0, 3, df["action"].notna().sum()).astype(float)
    tr = TrajectoryData(df, TrajectorySchema(**SCHEMA)).transitions()
    bp = overlap.fit_behaviour_policy(tr)
    assert overlap.support_violation_rate(tr, bp, n_boot=50).value == 0
    assert overlap.action_entropy(tr, bp, n_boot=50).value > 0.95
    assert np.isclose(overlap.action_overlap_index(tr, bp).value, 1.0)


def test_ess_equals_n_when_target_equals_behaviour(markov_data):
    tr = markov_data.transitions()
    bp = overlap.fit_behaviour_policy(tr)
    res = ope_reliability.is_weight_diagnostics(tr, bp, bp.proba)
    assert np.isclose(res.value, 1.0)


def test_ess_drops_with_deviation():
    # Short horizon; the curve need not be monotone in lambda (mixing with the
    # uniform policy shrinks some weights), but large deviations cost ESS.
    tr = TrajectoryData(make_ar_data(n_traj=300, T=5), TrajectorySchema(**SCHEMA)).transitions()
    bp = overlap.fit_behaviour_policy(tr)
    curve = ope_reliability.ess_deviation_curve(tr, bp, lambdas=(0.05, 1.0))
    assert 1.0 > curve[0].value > curve[1].value


def test_esac_and_occupancy_bounds(markov_data):
    tr = markov_data.transitions()
    e = coverage.esac(tr, n_clusters=10, n_min=5)
    o = coverage.occupancy_weighted_support(tr, n_clusters=10, n_min=5)
    assert 0 <= e.value <= 1 and 0 <= o.value <= 1


def test_transition_stability_decreases_with_data():
    small = TrajectoryData(make_ar_data(n_traj=20), TrajectorySchema(**SCHEMA)).transitions()
    large = TrajectoryData(make_ar_data(n_traj=400), TrajectorySchema(**SCHEMA)).transitions()
    assert coverage.transition_model_stability(large, n_boot=30).value < coverage.transition_model_stability(small, n_boot=30).value


def test_calibration_and_horizon_profile(markov_data):
    # The fixture's behaviour policy is a step function of sign(s), which a
    # linear-logistic model cannot represent; a random policy is well specified.
    tr = markov_data.transitions()
    bp = overlap.fit_behaviour_policy(tr)
    rng = np.random.default_rng(0)
    df = make_ar_data(seed=3)
    df.loc[df["action"].notna(), "action"] = rng.integers(0, 3, df["action"].notna().sum()).astype(float)
    tr_u = TrajectoryData(df, TrajectorySchema(**SCHEMA)).transitions()
    ece_u = overlap.behaviour_calibration(tr_u, overlap.fit_behaviour_policy(tr_u)).value
    assert ece_u < 0.03 < overlap.behaviour_calibration(tr, bp).value
    res = ope_reliability.is_weight_diagnostics(tr, bp, ope_reliability.reference_policy_family(bp, 0.5))
    prof = res.details["ess_share_by_horizon"]
    assert prof["1"] >= prof["5"] >= prof["20"]
