"""Trajectory data container and transition construction.

Input format: a long pandas DataFrame with one row per decision time of a
trajectory (patient stay). State columns may contain NaN; missing values are
never imputed silently. Model-based diagnostics work on complete cases unless
the caller explicitly preprocesses (see ``preprocessing``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from mdp_diagnostics.data.validation import ValidationIssue, validate_dataframe

ActionType = Literal["discrete", "continuous"]


@dataclass
class TrajectorySchema:
    """Column roles of a trajectory DataFrame."""

    id_col: str
    time_col: str
    state_cols: list[str]
    action_col: str | None = None
    action_type: ActionType = "discrete"
    reward_col: str | None = None
    outcome_col: str | None = None  # trajectory-level outcome, e.g. mortality
    # Declared action space for discrete actions. If None, only observed levels
    # are known, and actions that are never taken are invisible to overlap
    # diagnostics; declaring the clinically relevant set avoids that blind spot.
    action_levels: list[float] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectorySchema":
        return cls(**d)


@dataclass
class Transitions:
    """Aligned arrays of transitions (s_t, a_t, r_t, s_{t+1}) plus history.

    Row i describes the transition from decision step ``step[i]`` of trajectory
    ``groups[i]``. ``history`` holds the lagged states and actions
    s_{t-1..t-k}, a_{t-1..t-k}; rows without a full history of ``lags`` steps
    are dropped so that Markov and history models are fitted on identical rows.
    """

    groups: np.ndarray
    step: np.ndarray
    time: np.ndarray
    state: np.ndarray
    action: np.ndarray  # raw action values
    action_features: np.ndarray  # one-hot (discrete) or numeric (continuous)
    next_state: np.ndarray
    reward: np.ndarray | None
    history: np.ndarray
    state_names: list[str]
    history_names: list[str]
    action_levels: np.ndarray | None
    lags: int
    n_candidates: int  # transitions before dropping missing/short-history rows
    n_dropped_missing: int

    @property
    def n(self) -> int:
        return len(self.groups)

    @property
    def n_trajectories(self) -> int:
        return len(np.unique(self.groups))


@dataclass
class TrajectoryData:
    df: pd.DataFrame
    schema: TrajectorySchema
    issues: list[ValidationIssue] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.issues = validate_dataframe(self.df, self.schema)
        fatal = [i for i in self.issues if i.severity == "error"]
        if fatal:
            raise ValueError("Invalid trajectory data: " + "; ".join(i.message for i in fatal))
        s = self.schema
        self.df = self.df.sort_values([s.id_col, s.time_col], kind="mergesort").reset_index(drop=True)

    # ------------------------------------------------------------------ basics
    @property
    def groups(self) -> np.ndarray:
        return self.df[self.schema.id_col].to_numpy()

    @property
    def n_trajectories(self) -> int:
        return int(self.df[self.schema.id_col].nunique())

    def trajectory_lengths(self) -> pd.Series:
        return self.df.groupby(self.schema.id_col).size()

    def action_levels(self) -> np.ndarray | None:
        s = self.schema
        if s.action_col is None or s.action_type != "discrete":
            return None
        observed = self.df[s.action_col].dropna().unique()
        if s.action_levels is not None:
            undeclared = set(observed) - set(s.action_levels)
            if undeclared:
                raise ValueError(f"Observed actions {sorted(undeclared)} are not in the declared action_levels.")
            return np.sort(np.asarray(s.action_levels, dtype=float))
        return np.sort(observed)

    def outcomes(self) -> pd.Series | None:
        s = self.schema
        if s.outcome_col is None:
            return None
        return self.df.groupby(s.id_col)[s.outcome_col].first()

    def with_df(self, df: pd.DataFrame, **schema_changes) -> "TrajectoryData":
        """Copy with a new DataFrame (and optionally modified schema)."""
        schema = TrajectorySchema(**{**self.schema.__dict__, **schema_changes})
        return TrajectoryData(df, schema)

    # ------------------------------------------------------------- transitions
    def _encode_actions(self, values: np.ndarray, levels: np.ndarray | None) -> np.ndarray:
        if levels is None:
            return values.reshape(-1, 1).astype(float)
        onehot = (values.reshape(-1, 1) == levels.reshape(1, -1)).astype(float)
        onehot[np.isnan(values.astype(float)), :] = np.nan
        return onehot

    def transitions(self, lags: int = 0, state_cols: list[str] | None = None) -> Transitions:
        """Build complete-case transitions with ``lags`` steps of history."""
        s = self.schema
        if s.action_col is None:
            raise ValueError("Transitions require an action column.")
        cols = list(state_cols or s.state_cols)
        df = self.df
        g = df.groupby(s.id_col, sort=False)
        levels = self.action_levels()

        state = df[cols].to_numpy(float)
        action = df[s.action_col].to_numpy(float)
        act_feat = self._encode_actions(action, levels)
        nxt = g[cols].shift(-1).to_numpy(float)
        reward = df[s.reward_col].to_numpy(float) if s.reward_col else None
        step = g.cumcount().to_numpy()

        hist_parts, hist_names = [], []
        act_names = [f"a={lv:g}" for lv in levels] if levels is not None else [s.action_col]
        for k in range(1, lags + 1):
            hist_parts.append(g[cols].shift(k).to_numpy(float))
            hist_names += [f"{c}[t-{k}]" for c in cols]
            lagged_action = g[s.action_col].shift(k).to_numpy(float)
            hist_parts.append(self._encode_actions(lagged_action, levels))
            hist_names += [f"{n}[t-{k}]" for n in act_names]
        history = np.hstack(hist_parts) if hist_parts else np.empty((len(df), 0))

        has_next = g.cumcount(ascending=False).to_numpy() > 0
        candidates = has_next & (step >= lags)
        complete = (
            np.isfinite(state).all(1)
            & np.isfinite(nxt).all(1)
            & np.isfinite(act_feat).all(1)
            & np.isfinite(history).all(1)
        )
        if reward is not None:
            complete &= np.isfinite(reward)
        keep = candidates & complete

        return Transitions(
            groups=df[s.id_col].to_numpy()[keep],
            step=step[keep],
            time=df[s.time_col].to_numpy(float)[keep],
            state=state[keep],
            action=action[keep],
            action_features=act_feat[keep],
            next_state=nxt[keep],
            reward=reward[keep] if reward is not None else None,
            history=history[keep],
            state_names=cols,
            history_names=hist_names,
            action_levels=levels,
            lags=lags,
            n_candidates=int(candidates.sum()),
            n_dropped_missing=int((candidates & ~complete).sum()),
        )
