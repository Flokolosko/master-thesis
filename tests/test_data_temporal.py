import numpy as np
import pandas as pd
import pytest

from mdp_diagnostics import TrajectoryData, TrajectorySchema
from mdp_diagnostics.data import discretize_actions, resample_to_grid
from mdp_diagnostics.experiments import perturbations as P
from mdp_diagnostics.metrics import temporal
from tests.conftest import SCHEMA, make_ar_data


def test_missing_columns_rejected():
    with pytest.raises(ValueError):
        TrajectoryData(pd.DataFrame({"id": [1]}), TrajectorySchema(**SCHEMA))


def test_empty_dataframe_rejected():
    df = make_ar_data().iloc[:0]
    with pytest.raises(ValueError):
        TrajectoryData(df, TrajectorySchema(**SCHEMA))


def test_variable_length_and_single_row_trajectories():
    df = make_ar_data(n_traj=10, T=5)
    df = pd.concat([df, pd.DataFrame({"id": [99], "time": [0.0], "s": [0.1], "action": [1.0], "reward": [0.0]})])
    d = TrajectoryData(df, TrajectorySchema(**SCHEMA))
    assert any("fewer than two rows" in i.message for i in d.issues)
    tr = d.transitions(lags=1)
    assert 99 not in set(tr.groups)
    assert tr.n == 10 * 4  # each trajectory: 5 transitions, minus 1 for the lag


def test_missing_values_not_imputed():
    df = make_ar_data(n_traj=20, T=10)
    df.loc[5, "s"] = np.nan
    d = TrajectoryData(df, TrajectorySchema(**SCHEMA))
    tr = d.transitions()
    assert tr.n_dropped_missing == 2  # row 5 as state and as next state
    assert np.isfinite(tr.state).all()


def test_history_alignment():
    df = make_ar_data(n_traj=3, T=6)
    tr = TrajectoryData(df, TrajectorySchema(**SCHEMA)).transitions(lags=1)
    # history column 0 is s[t-1]; it must equal the previous row's state of the same trajectory
    for i in range(1, tr.n):
        if tr.groups[i] == tr.groups[i - 1]:
            assert tr.history[i, 0] == tr.state[i - 1, 0]


def test_irregular_sampling_metric_responds():
    df = make_ar_data(n_traj=50, T=30)
    base = temporal.sampling_intervals(TrajectoryData(df, TrajectorySchema(**SCHEMA)))
    irr = temporal.sampling_intervals(TrajectoryData(P.irregular_sampling(df, "id", 0.5), TrajectorySchema(**SCHEMA)))
    assert base.value == 0
    assert irr.value > 0.3


def test_informative_missingness_detected():
    df = make_ar_data(n_traj=300, T=15, seed=2)
    outcome = df.groupby("id")["s"].transform(lambda s: float(s.abs().mean() > 0.8))
    df["outcome"] = outcome
    schema = TrajectorySchema(**SCHEMA, outcome_col="outcome")
    mcar = P.mcar_missing(df, ["s"], 0.3, seed=1)
    inf = P.informative_missing(df, ["s"], 0.3, strength=-3.0, driver="outcome", seed=1)
    a_mcar = temporal.informative_missingness_outcome(TrajectoryData(mcar, schema)).value
    a_inf = temporal.informative_missingness_outcome(TrajectoryData(inf, schema)).value
    assert a_mcar < 0.62 and a_inf > 0.8


def test_resample_and_discretise():
    df = make_ar_data(n_traj=5, T=20)
    df["dose"] = np.where(df["action"] > 0, df["action"] * 1.7, 0.0)
    d = TrajectoryData(df, TrajectorySchema(id_col="id", time_col="time", state_cols=["s"], action_col="dose", action_type="continuous"))
    g, summary = resample_to_grid(d, dt=4.0)
    assert g.df.groupby("id").size().max() == 6
    assert summary["n_distinct_actions"].max() >= 1
    dd = discretize_actions(g, n_bins=3)
    assert dd.schema.action_type == "discrete"
    assert set(np.unique(dd.df[dd.schema.action_col].dropna())) <= {0.0, 1.0, 2.0}


def test_observation_age():
    df = make_ar_data(n_traj=5, T=10)
    df.loc[df["time"] % 2 == 1, "s"] = np.nan
    age = temporal.observation_age(TrajectoryData(df, TrajectorySchema(**SCHEMA)))
    assert age.details["per_variable"]["s"]["p90"] == 1.0
