"""Structural validation of trajectory DataFrames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from mdp_diagnostics.data.schema import TrajectorySchema


@dataclass(frozen=True)
class ValidationIssue:
    severity: Literal["error", "warning", "info"]
    message: str

    def to_dict(self) -> dict:
        return {"severity": self.severity, "message": self.message}


def validate_dataframe(df: pd.DataFrame, schema: "TrajectorySchema") -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    needed = [schema.id_col, schema.time_col, *schema.state_cols]
    needed += [c for c in (schema.action_col, schema.reward_col, schema.outcome_col) if c]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return [ValidationIssue("error", f"Missing columns: {missing}")]
    if len(df) == 0:
        return [ValidationIssue("error", "DataFrame is empty.")]
    if not schema.state_cols:
        issues.append(ValidationIssue("error", "No state columns specified."))
    if df[schema.id_col].isna().any():
        issues.append(ValidationIssue("error", "Trajectory identifier contains missing values."))
    if df[schema.time_col].isna().any():
        issues.append(ValidationIssue("error", "Decision timestamps contain missing values."))
    if not pd.api.types.is_numeric_dtype(df[schema.time_col]):
        issues.append(ValidationIssue("error", "Time column must be numeric (e.g. minutes since admission)."))
        return issues

    dup = df.duplicated([schema.id_col, schema.time_col]).sum()
    if dup:
        issues.append(ValidationIssue("warning", f"{dup} duplicated (trajectory, time) rows; order within ties is arbitrary."))

    lengths = df.groupby(schema.id_col).size()
    n_single = int((lengths < 2).sum())
    if n_single:
        issues.append(ValidationIssue("warning", f"{n_single} trajectories have fewer than two rows and yield no transitions."))

    for col in schema.state_cols:
        frac = float(df[col].isna().mean())
        if frac == 1.0:
            issues.append(ValidationIssue("error", f"State column '{col}' is entirely missing."))
        elif frac > 0:
            issues.append(ValidationIssue("info", f"State column '{col}' has {frac:.1%} missing values (not imputed)."))

    if schema.action_col:
        a = df[schema.action_col]
        if a.isna().all():
            issues.append(ValidationIssue("error", "Action column is entirely missing."))
        elif schema.action_type == "discrete" and a.nunique() > 50:
            issues.append(ValidationIssue("warning", f"Discrete action column has {a.nunique()} levels; consider discretising."))
        if a.nunique(dropna=True) == 1:
            issues.append(ValidationIssue("warning", "Only one action value is observed (degenerate behaviour policy)."))
    else:
        issues.append(ValidationIssue("warning", "No action column: MDP and offline-RL diagnostics are not assessable."))

    if schema.outcome_col:
        varies = df.groupby(schema.id_col)[schema.outcome_col].nunique(dropna=True).max()
        if varies is not None and varies > 1:
            issues.append(ValidationIssue("warning", "Outcome column varies within trajectories; the first value is used."))

    t = df[schema.time_col].to_numpy(float)
    if np.isfinite(t).all() and (t < 0).any():
        issues.append(ValidationIssue("info", "Negative timestamps present (allowed, interpreted relative to an index time)."))
    return issues
