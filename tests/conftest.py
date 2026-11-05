import numpy as np
import pandas as pd
import pytest

from mdp_diagnostics import TrajectoryData, TrajectorySchema


def make_ar_data(n_traj=200, T=30, order=1, phi2=0.0, n_actions=3, seed=0, deterministic=False):
    """Scalar process s_{t+1} = 0.6 s_t + phi2 s_{t-1} + 0.3 * (a - 1) + noise.

    With phi2 = 0 the state s is Markov; with phi2 != 0 it is second-order.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_traj):
        s_prev, s = 0.0, rng.normal()
        for t in range(T):
            if deterministic:
                a = 0 if s > 0 else 2
            else:
                p = np.exp(-np.abs(np.arange(n_actions) - (1 - np.sign(s))))
                a = rng.choice(n_actions, p=p / p.sum())
            r = -s**2
            rows.append((i, float(t), s, float(a), r))
            s_prev, s = s, 0.6 * s + phi2 * s_prev + 0.3 * (a - 1) + 0.5 * rng.normal()
        rows.append((i, float(T), s, np.nan, np.nan))
    return pd.DataFrame(rows, columns=["id", "time", "s", "action", "reward"])


SCHEMA = dict(id_col="id", time_col="time", state_cols=["s"], action_col="action", reward_col="reward")


@pytest.fixture
def markov_data():
    return TrajectoryData(make_ar_data(), TrajectorySchema(**SCHEMA))


@pytest.fixture
def ar2_data():
    return TrajectoryData(make_ar_data(phi2=0.35), TrajectorySchema(**SCHEMA))
