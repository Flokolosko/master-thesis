"""Controlled perturbations of trajectory DataFrames (one manipulated factor each)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mcar_missing(df: pd.DataFrame, cols: list[str], p: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    for c in cols:
        out.loc[rng.random(len(out)) < p, c] = np.nan
    return out


def informative_missing(df: pd.DataFrame, cols: list[str], p: float, strength: float, driver: str, seed: int = 0) -> pd.DataFrame:
    """Missing with probability depending on ``driver`` (e.g. the value itself or an outcome).

    logit P(missing) = logit(p) - strength * standardised(driver): with
    strength > 0, high driver values are measured more often (sicker patients
    are monitored more closely).
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    d = out[driver].to_numpy(float)
    zd = (d - np.nanmean(d)) / (np.nanstd(d) or 1.0)
    logit = np.log(p / (1 - p)) - strength * np.nan_to_num(zd)
    pm = 1 / (1 + np.exp(-logit))
    for c in cols:
        out.loc[rng.random(len(out)) < pm, c] = np.nan
    return out


def irregular_sampling(df: pd.DataFrame, id_col: str, p_drop: float, seed: int = 0) -> pd.DataFrame:
    """Drop interior decision rows at random (first and last row of each trajectory kept)."""
    rng = np.random.default_rng(seed)
    g = df.groupby(id_col)
    interior = (g.cumcount() > 0) & (g.cumcount(ascending=False) > 0)
    drop = interior & (rng.random(len(df)) < p_drop)
    return df.loc[~drop].reset_index(drop=True)


def timestamp_jitter(df: pd.DataFrame, time_col: str, sd: float, round_to: float | None = None, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    t = out[time_col].to_numpy(float) + rng.normal(0, sd, len(out))
    if round_to:
        t = np.round(t / round_to) * round_to
    out[time_col] = t
    return out


def action_misalignment(df: pd.DataFrame, id_col: str, action_col: str, shift: int = 1) -> pd.DataFrame:
    """Record each action ``shift`` rows late (documentation delay)."""
    out = df.copy()
    out[action_col] = out.groupby(id_col)[action_col].shift(shift)
    return out


def subsample_trajectories(df: pd.DataFrame, id_col: str, frac: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = df[id_col].unique()
    keep = rng.choice(ids, size=max(2, int(round(frac * len(ids)))), replace=False)
    return df[df[id_col].isin(keep)].reset_index(drop=True)


def add_lag_features(df: pd.DataFrame, id_col: str, cols: list[str], lags: int,
                     pad: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """History-augmented state: append lagged copies of ``cols``.

    Without ``pad`` rows lacking a full history are dropped. With ``pad``
    (column -> fill value, or "first" for the trajectory's first value) the
    start of each trajectory is padded so that trajectories keep their length.
    """
    out = df.copy()
    names = []
    g = out.groupby(id_col)
    for k in range(1, lags + 1):
        for c in cols:
            name = f"{c}_lag{k}"
            out[name] = g[c].shift(k)
            if pad is not None and c in pad:
                fill = g[c].transform("first") if pad[c] == "first" else pad[c]
                out[name] = out[name].fillna(fill)
            names.append(name)
    out = out.dropna(subset=names).reset_index(drop=True)
    return out, names
