"""Temporal data-quality indicators computed on the raw (un-gridded) trajectories.

Instead of one composite "temporal reliability" score, separate indicators are
reported because they call for different remedies: missingness (imputation or
POMDP), informative observation (include measurement indicators in the state),
irregular sampling and fragmentation (event-driven or semi-MDP time grid),
timestamp heaping (limits the usable time resolution) and stale state
information at decision times (state-action misalignment).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mdp_diagnostics.data.schema import TrajectoryData
from mdp_diagnostics.metrics.base import MetricResult
from mdp_diagnostics.metrics.markov import conditional_information_gain
from mdp_diagnostics.utils.statistics import cluster_bootstrap_mean


def missingness(data: TrajectoryData) -> MetricResult:
    s = data.schema
    m = data.df[s.state_cols].isna()
    per_traj = m.mean(axis=1).groupby(data.df[s.id_col]).mean()
    return MetricResult(
        "state_missingness", float(m.to_numpy().mean()), unit="share of missing state values",
        n_trajectories=data.n_trajectories,
        details={"per_variable": m.mean().to_dict(),
                 "rows_with_any_missing": float(m.any(axis=1).mean()),
                 "rows_all_missing": float(m.all(axis=1).mean()),
                 "per_trajectory_p90": float(per_traj.quantile(0.9))},
        assumptions=["Counts missing cells on the decision rows provided; absent rows are not counted (see sampling intervals)."],
    )


def sampling_intervals(data: TrajectoryData, gap_factor: float = 3.0) -> MetricResult:
    """Irregularity of decision times: coefficient of variation of intervals and gap rate."""
    s = data.schema
    dt = data.df.groupby(s.id_col)[s.time_col].diff().dropna().to_numpy(float)
    if len(dt) == 0:
        return MetricResult.failed("sampling_interval_cv", "No consecutive rows.")
    med = float(np.median(dt))
    gaps = dt > gap_factor * med
    tid = data.df[s.id_col].to_numpy()[data.df.groupby(s.id_col)[s.time_col].diff().notna().to_numpy()]
    traj_with_gap = pd.Series(gaps).groupby(tid).any().mean()
    return MetricResult(
        "sampling_interval_cv", float(dt.std() / dt.mean()) if dt.mean() > 0 else float("nan"),
        unit="CV of inter-decision intervals (0 = regular grid)", n_trajectories=data.n_trajectories,
        details={"median_interval": med, "iqr": np.quantile(dt, [0.25, 0.75]).tolist(),
                 "p95_interval": float(np.quantile(dt, 0.95)), "gap_factor": gap_factor,
                 "gap_rate": float(gaps.mean()), "share_trajectories_with_gap": float(traj_with_gap),
                 "share_zero_intervals": float((dt == 0).mean())},
        assumptions=[f"A gap is an interval longer than {gap_factor:g} x the median interval."],
    )


def timestamp_heaping(data: TrajectoryData, units: tuple[float, ...] = (5, 15, 30, 60)) -> MetricResult:
    """Share of timestamps falling on multiples of round units.

    Strong heaping on e.g. full hours indicates charting at documentation times
    rather than event times and bounds the meaningful time resolution.
    """
    t = data.df[data.schema.time_col].to_numpy(float)
    shares = {f"{u:g}": float(np.mean(np.isclose(np.mod(t, u), 0) | np.isclose(np.mod(t, u), u))) for u in units}
    finest = min(units)
    return MetricResult(
        "timestamp_heaping", shares[f"{max(units):g}"], unit=f"share of times on multiples of {max(units):g}",
        details={"shares_by_unit": shares,
                 "expected_if_uniform_on_finest_unit": {f"{u:g}": finest / u for u in units}},
        assumptions=["Time unit of the data must be known (e.g. minutes).",
                     "Heaping can also be a deliberate grid of the source dataset."],
    )


def observation_age(data: TrajectoryData) -> MetricResult:
    """Time since the last actual observation of each state variable at decision times.

    Large ages mean decisions are made on stale recorded information, a form of
    state-action temporal misalignment.
    """
    s = data.schema
    df = data.df
    t = df[s.time_col].astype(float)
    ages = {}
    all_ages = []
    for c in s.state_cols:
        last = t.where(df[c].notna()).groupby(df[s.id_col]).ffill()
        age = (t - last).to_numpy(float)
        ages[c] = {"median": float(np.nanmedian(age)) if np.isfinite(age).any() else None,
                   "p90": float(np.nanquantile(age, 0.9)) if np.isfinite(age).any() else None,
                   "never_observed_share": float(np.isnan(age).mean())}
        all_ages.append(age)
    stacked = np.concatenate(all_ages)
    return MetricResult(
        "observation_age_p90", float(np.nanquantile(stacked, 0.9)) if np.isfinite(stacked).any() else float("nan"),
        unit="time units since last observation (90th pct over variables/decisions)",
        n_trajectories=data.n_trajectories, details={"per_variable": ages},
        assumptions=["Missing cells are treated as 'not measured at this time', not as data errors."],
    )


def action_without_fresh_observation(data: TrajectoryData) -> MetricResult:
    """Share of action changes at decision rows where no state variable was observed."""
    s = data.schema
    if s.action_col is None:
        return MetricResult.failed("action_change_without_observation", "No action column.")
    df = data.df
    prev = df.groupby(s.id_col)[s.action_col].shift(1)
    change = (df[s.action_col] != prev) & prev.notna()
    no_obs = df[s.state_cols].isna().all(axis=1)
    n_change = int(change.sum())
    val = float((change & no_obs).sum() / n_change) if n_change else float("nan")
    return MetricResult(
        "action_change_without_observation", val, unit="share of action changes",
        n_trajectories=data.n_trajectories, details={"n_action_changes": n_change},
        assumptions=["Assumes state and action share the row's timestamp."],
    )


def within_step_action_changes(summary: pd.DataFrame) -> MetricResult:
    """From ``resample_to_grid`` summary: share of grid steps with >1 distinct raw action.

    Indicates that the decision interval is coarser than the decision process
    (several decisions are aggregated into one MDP action).
    """
    if "n_distinct_actions" not in summary:
        return MetricResult.failed("within_step_action_changes", "No action information in grid summary.")
    v = summary["n_distinct_actions"].to_numpy()
    return MetricResult(
        "within_step_action_changes", float((v > 1).mean()), unit="share of grid steps",
        details={"mean_raw_rows_per_step": float(summary["n_raw_rows"].mean())},
        assumptions=["Raw action values are recorded at their true times."],
    )


def informative_missingness_outcome(data: TrajectoryData, n_splits: int = 5, seed: int = 0) -> MetricResult:
    """Cross-validated AUC for predicting the trajectory outcome from measurement patterns alone.

    Features: per-variable missing share and number of rows per time unit.
    AUC ~ 0.5 means the observation process carries no outcome information;
    AUC clearly above 0.5 means measurement patterns are informative (cf. test
    ordering predicting survival) and should not be ignored as MCAR.
    """
    s = data.schema
    y = data.outcomes()
    if y is None or y.nunique() < 2:
        return MetricResult.failed("informative_missingness_outcome_auc", "Binary trajectory outcome required.")
    df = data.df
    g = df.groupby(s.id_col)
    X = df[s.state_cols].isna().groupby(df[s.id_col]).mean()
    span = (g[s.time_col].max() - g[s.time_col].min()).replace(0, np.nan)
    X["rows_per_time"] = (g.size() / span).fillna(0)
    X = X.loc[y.index].to_numpy(float)
    yv = y.to_numpy().astype(int)
    if np.ptp(X, axis=0).max() == 0:
        return MetricResult("informative_missingness_outcome_auc", 0.5, unit="AUC", details={"note": "No variation in measurement patterns."})
    oof = np.zeros(len(yv))
    skf = StratifiedKFold(n_splits=min(n_splits, int(np.bincount(yv).min())), shuffle=True, random_state=seed)
    for tr, te in skf.split(X, yv):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], yv[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(yv, oof)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(300):
        i = rng.integers(0, len(yv), len(yv))
        if len(np.unique(yv[i])) == 2:
            reps.append(roc_auc_score(yv[i], oof[i]))
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return MetricResult(
        "informative_missingness_outcome_auc", float(auc), unit="AUC", lower=float(lo), upper=float(hi),
        n_trajectories=len(yv),
        assumptions=["Association only; informative measurement can reflect clinician suspicion (confounding by indication)."],
    )


def informative_missingness_action(data: TrajectoryData, model: str = "linear", seed: int = 0, n_boot: int = 300) -> MetricResult:
    """Gain in predicting the action from missingness indicators beyond the (carried-forward) state.

    The state is carried forward within trajectory for this diagnostic only
    (remaining gaps set to the variable mean); a positive gain indicates that
    *whether* something was measured informs treatment, so a Markov state built
    from values alone omits decision-relevant information.
    """
    s = data.schema
    if s.action_col is None or s.action_type != "discrete":
        return MetricResult.failed("informative_missingness_action_gain", "Discrete action required.")
    df = data.df
    miss = df[s.state_cols].isna().to_numpy(float)
    if miss.sum() == 0:
        return MetricResult("informative_missingness_action_gain", 0.0, unit="nats/decision", details={"note": "No missing values."})
    filled = df.groupby(s.id_col)[s.state_cols].ffill()
    filled = filled.fillna(filled.mean()).to_numpy(float)
    ok = df[s.action_col].notna().to_numpy()
    res = conditional_information_gain(
        filled[ok], miss[ok][:, miss.std(axis=0) > 0], df[s.action_col].to_numpy()[ok], df[s.id_col].to_numpy()[ok],
        "classification", model=model, seed=seed, n_boot=n_boot, name="informative_missingness_action_gain",
    )
    res.assumptions = ["Carried-forward values are used as the 'value' information.",
                       "Association only (measurement and treatment may share an unrecorded cause)."]
    return res


def trajectory_length(data: TrajectoryData, min_length: int = 3) -> MetricResult:
    lengths = data.trajectory_lengths()
    return MetricResult(
        "trajectory_length_median", float(lengths.median()), unit="decision rows",
        n_trajectories=len(lengths),
        details={"iqr": lengths.quantile([0.25, 0.75]).tolist(), "min_length": min_length,
                 "share_shorter_than_min": float((lengths < min_length).mean())},
    )


def bootstrap_share(flags: np.ndarray, groups: np.ndarray, name: str, unit: str, seed: int = 0) -> MetricResult:
    est = cluster_bootstrap_mean(flags.astype(float), groups, seed=seed)
    return MetricResult.from_estimate(name, est, unit=unit)
