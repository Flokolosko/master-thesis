"""Explicit, documented preprocessing steps.

Every step that changes the information content (regridding, imputation,
discretisation) records what it did, so that reports can show it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mdp_diagnostics.data.schema import TrajectoryData


@dataclass
class PreprocessingLog:
    steps: list[dict] = field(default_factory=list)

    def add(self, name: str, **info) -> None:
        self.steps.append({"step": name, **info})


def discretize_actions(
    data: TrajectoryData,
    n_bins: int,
    zero_as_own_bin: bool = True,
    new_col: str | None = None,
    log: PreprocessingLog | None = None,
) -> TrajectoryData:
    """Quantile-discretise a continuous action (e.g. a drug dose).

    If ``zero_as_own_bin`` the value 0 (no treatment) forms bin 0 and the
    positive values are split into ``n_bins - 1`` quantile bins, as is common
    for vasopressor/fluid actions in clinical RL.
    """
    s = data.schema
    col = new_col or f"{s.action_col}_bin"
    a = data.df[s.action_col].to_numpy(float)
    out = np.full(len(a), np.nan)
    if zero_as_own_bin:
        pos = a > 0
        out[a == 0] = 0
        if pos.any() and n_bins > 1:
            edges = np.unique(np.quantile(a[pos], np.linspace(0, 1, n_bins)))
            out[pos] = 1 + np.clip(np.searchsorted(edges, a[pos], side="right") - 1, 0, len(edges) - 2)
    else:
        edges = np.unique(np.nanquantile(a, np.linspace(0, 1, n_bins + 1)))
        ok = np.isfinite(a)
        out[ok] = np.clip(np.searchsorted(edges, a[ok], side="right") - 1, 0, len(edges) - 2)
    df = data.df.copy()
    df[col] = out
    if log is not None:
        log.add("discretize_actions", source=s.action_col, target=col, n_levels=int(np.unique(out[np.isfinite(out)]).size))
    return data.with_df(df, action_col=col, action_type="discrete")


def add_terminal_outcome_reward(
    data: TrajectoryData,
    scale: float = -1.0,
    col: str = "terminal_reward",
    log: PreprocessingLog | None = None,
) -> TrajectoryData:
    """Sparse reward: ``scale * outcome`` on the last transition of each trajectory, 0 elsewhere.

    The last row of a trajectory is the terminal observation and has no
    transition, so the reward is placed on the second-to-last row.
    """
    s = data.schema
    if s.outcome_col is None:
        raise ValueError("A trajectory outcome column is required.")
    df = data.df.copy()
    rev = df.groupby(s.id_col).cumcount(ascending=False)
    outcome = df.groupby(s.id_col)[s.outcome_col].transform("first").astype(float)
    df[col] = np.where(rev == 1, scale * outcome, 0.0)
    df.loc[rev == 0, col] = np.nan
    if log is not None:
        log.add("add_terminal_outcome_reward", outcome=s.outcome_col, scale=scale)
    return data.with_df(df, reward_col=col)


def resample_to_grid(
    data: TrajectoryData,
    dt: float,
    action_agg: str = "mean",
    impute: str | None = None,
    log: PreprocessingLog | None = None,
) -> tuple[TrajectoryData, pd.DataFrame]:
    """Aggregate irregular observations onto a fixed decision grid of width ``dt``.

    States: last observed value within each bin. Actions: aggregated with
    ``action_agg`` ("mean", "max" or "last"). Rewards: summed. Empty bins stay
    missing unless ``impute="locf"`` (last observation carried forward), in which
    case an indicator column ``<state>__imputed`` is added for every state.

    Returns the new data and a per-bin summary with the number of raw rows and
    distinct raw actions per bin (used by the temporal diagnostics).
    """
    s = data.schema
    df = data.df.copy()
    df["_bin"] = np.floor(df[s.time_col] / dt).astype(int)
    keys = [s.id_col, "_bin"]
    agg = {c: "last" for c in s.state_cols}
    if s.action_col:
        agg[s.action_col] = action_agg
    if s.reward_col:
        agg[s.reward_col] = "sum"
    if s.outcome_col:
        agg[s.outcome_col] = "first"
    grouped = df.groupby(keys, sort=True)
    binned = grouped.agg(agg)
    summary = pd.DataFrame({"n_raw_rows": grouped.size()})
    if s.action_col:
        summary["n_distinct_actions"] = grouped[s.action_col].nunique()
    binned = binned.reset_index()

    # Re-insert empty bins between first and last observation of each trajectory.
    full = []
    for tid, part in binned.groupby(s.id_col, sort=False):
        idx = pd.RangeIndex(part["_bin"].min(), part["_bin"].max() + 1, name="_bin")
        p = part.set_index("_bin").reindex(idx)
        p[s.id_col] = tid
        full.append(p.reset_index())
    out = pd.concat(full, ignore_index=True)
    n_empty = int(out[s.state_cols].isna().all(axis=1).sum())
    if impute == "locf":
        for c in s.state_cols:
            out[f"{c}__imputed"] = out[c].isna().astype(int)
        out[s.state_cols] = out.groupby(s.id_col)[s.state_cols].ffill()
        if s.outcome_col:
            out[s.outcome_col] = out.groupby(s.id_col)[s.outcome_col].ffill()
    elif impute is not None:
        raise ValueError(f"Unknown imputation '{impute}'.")
    out[s.time_col] = out["_bin"] * dt
    out = out.drop(columns="_bin")
    if log is not None:
        log.add("resample_to_grid", dt=dt, action_agg=action_agg, impute=impute, n_bins=len(out), n_empty_bins=n_empty)
    return data.with_df(out), summary.reset_index()
