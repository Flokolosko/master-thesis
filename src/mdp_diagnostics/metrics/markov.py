"""State-sufficiency (Markov) diagnostics.

Two complementary diagnostics are provided:

* **History information gain** G = E[log p(Y | Z, H) - log p(Y | Z)], estimated
  out-of-sample with trajectory-level cross-fitting. With Z = (s_t, a_t),
  Y = s_{t+1} and H = lagged states/actions, G estimates the conditional mutual
  information I(S_{t+1}; H_t | S_t, A_t) under the working model (Gaussian
  residuals for continuous targets, a probabilistic classifier for discrete
  targets). G is an effect size in nats per transition; G ~ 0 is what a Markov
  state predicts, G > 0 means observable history improves prediction.
* **Generalised covariance measure (GCM) test** of Y _||_ H | Z (Shah & Peters,
  2020), using cross-fitted residuals and a trajectory-clustered multiplier
  bootstrap for the max statistic over all (target, history) pairs.

Both only detect history dependence that is *visible in recorded variables* and
expressible by the chosen regression models. Neither proves that the process is
Markov in a causal or physical sense.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mdp_diagnostics.data.schema import Transitions
from mdp_diagnostics.metrics.base import MetricResult
from mdp_diagnostics.utils.statistics import cluster_bootstrap_mean, group_folds

ModelKind = Literal["linear", "gbm"]
_EPS = 1e-6
DETERMINISTIC_R2 = 0.999  # target dims predicted better than this by the base model are excluded

MARKOV_ASSUMPTIONS = [
    "Detects only dependence on *recorded* history; latent variables that leave no trace in observed history are undetectable.",
    "Effect size is relative to the regression model class; a too-weak model under-detects, a mis-tuned flexible model can over- or under-detect.",
    "Continuous targets: Gaussian, homoscedastic residuals per dimension (gain = Gaussian conditional mutual information).",
    "Transitions are pooled over time (time-homogeneity is assessed separately).",
]


# ----------------------------------------------------------------- models
def _regressor(model: ModelKind, seed: int):
    if model == "linear":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, random_state=seed)


def _classifier(model: ModelKind, seed: int):
    if model == "linear":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, random_state=seed)


def _oof_regression(X: np.ndarray, Y: np.ndarray, folds, model: ModelKind, seed: int) -> np.ndarray:
    """Out-of-fold predictions for a (n, d) target."""
    pred = np.zeros_like(Y, dtype=float)
    for train, test in folds:
        for j in range(Y.shape[1]):
            if X.shape[1] == 0:
                pred[test, j] = Y[train, j].mean()
                continue
            m = _regressor(model, seed).fit(X[train], Y[train, j])
            pred[test, j] = m.predict(X[test])
    return pred


def _oof_class_logprob(X: np.ndarray, y: np.ndarray, folds, model: ModelKind, seed: int) -> np.ndarray:
    """Out-of-fold log-probability of the observed class."""
    classes = np.unique(y)
    logp = np.zeros(len(y))
    for train, test in folds:
        train_classes = np.unique(y[train])
        if len(train_classes) == 1 or X.shape[1] == 0:
            freq = np.array([(y[train] == c).mean() for c in classes])
            p = np.clip(freq[np.searchsorted(classes, y[test])], _EPS, 1)
            logp[test] = np.log(p)
            continue
        m = _classifier(model, seed).fit(X[train], y[train])
        proba = m.predict_proba(X[test])
        col = {c: i for i, c in enumerate(m.classes_)}
        idx = np.array([col.get(c, -1) for c in y[test]])
        p = np.where(idx >= 0, proba[np.arange(len(test)), np.maximum(idx, 0)], _EPS)
        logp[test] = np.log(np.clip(p, _EPS, 1))
    return logp


def _r2(Y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return 1 - (Y - pred).var(axis=0) / np.maximum(Y.var(axis=0), 1e-12)


# --------------------------------------------------- information gain core
def conditional_information_gain(
    base: np.ndarray,
    extra: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    kind: Literal["regression", "classification"],
    model: ModelKind = "linear",
    n_splits: int = 5,
    n_boot: int = 300,
    seed: int = 0,
    name: str = "conditional_information_gain",
) -> MetricResult:
    """Out-of-sample gain in log predictive density from adding ``extra`` to ``base``."""
    n = len(groups)
    if n < 20 or len(np.unique(groups)) < 4:
        return MetricResult.failed(name, "Too few transitions/trajectories for cross-fitting.")
    if extra.shape[1] == 0:
        return MetricResult.failed(name, "No additional features to test.")
    folds = group_folds(groups, n_splits, seed)
    full = np.hstack([base, extra])
    details: dict = {}
    if kind == "regression":
        Y = target.reshape(n, -1).astype(float)
        pb = _oof_regression(base, Y, folds, model, seed)
        # Target dimensions that the base features determine (almost) exactly, e.g.
        # shifted lag columns of a history-augmented state, carry no testable
        # stochasticity and would turn numerical noise into huge log-ratios.
        r2_base = _r2(Y, pb)
        if model != "linear":  # flexible models approximate exact linear relations imperfectly
            r2_base = np.maximum(r2_base, _r2(Y, _oof_regression(base, Y, folds, "linear", seed)))
        keep = r2_base < DETERMINISTIC_R2
        details["excluded_deterministic_dims"] = np.flatnonzero(~keep).tolist()
        if not keep.any():
            return MetricResult.failed(name, "All target dimensions are deterministic given the base features.")
        Y, pb = Y[:, keep], pb[:, keep]
        pf = _oof_regression(full, Y, folds, model, seed)
        rb, rf = Y - pb, Y - pf
        var_b = np.maximum(rb.var(axis=0), 1e-12)
        var_f = np.maximum(rf.var(axis=0), 1e-12)
        # Per-transition log-density difference under pooled Gaussian residuals.
        ll_b = -0.5 * np.log(var_b) - rb**2 / (2 * var_b)
        ll_f = -0.5 * np.log(var_f) - rf**2 / (2 * var_f)
        per_sample = (ll_f - ll_b).sum(axis=1)
        tot_var = np.maximum(Y.var(axis=0), 1e-12)
        details["per_dimension_gain_nats"] = (0.5 * np.log(var_b / var_f)).tolist()
        details["oof_r2_base"] = (1 - var_b / tot_var).tolist()
        details["oof_r2_full"] = (1 - var_f / tot_var).tolist()
    else:
        y = target.ravel()
        per_sample = _oof_class_logprob(full, y, folds, model, seed) - _oof_class_logprob(base, y, folds, model, seed)
    est = cluster_bootstrap_mean(per_sample, groups, n_boot=n_boot, seed=seed)
    return MetricResult.from_estimate(
        name, est, unit="nats/transition", n_transitions=n,
        n_trajectories=int(len(np.unique(groups))), details=details,
        assumptions=list(MARKOV_ASSUMPTIONS),
    )


# ------------------------------------------------------------- GCM test
def gcm_test(
    base: np.ndarray,
    extra: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    model: ModelKind = "linear",
    n_splits: int = 5,
    n_mult: int = 2000,
    seed: int = 0,
    name: str = "gcm_test",
) -> MetricResult:
    """Clustered GCM test of target _||_ extra | base (Shah & Peters, 2020).

    Residuals of every target column and every ``extra`` column on ``base`` are
    computed out-of-fold; the statistic is the max over pairs of the absolute
    standardised residual-product sum, with a cluster-robust variance and a
    Gaussian multiplier bootstrap applied per trajectory.
    """
    n = len(groups)
    if n < 20 or len(np.unique(groups)) < 4 or extra.shape[1] == 0:
        return MetricResult.failed(name, "Too few transitions/trajectories or no extra features.")
    folds = group_folds(groups, n_splits, seed)
    Y = target.reshape(n, -1).astype(float)
    X = extra.astype(float)
    # Drop constant columns (e.g. never-used action levels).
    X = X[:, X.std(axis=0) > 1e-12]
    if X.shape[1] == 0:
        return MetricResult.failed(name, "All extra features are constant.")
    ry = Y - _oof_regression(base, Y, folds, model, seed)
    r2 = _r2(Y, Y - ry)
    if model != "linear":
        r2 = np.maximum(r2, _r2(Y, _oof_regression(base, Y, folds, "linear", seed)))
    keep = r2 < DETERMINISTIC_R2
    if not keep.any():
        return MetricResult.failed(name, "All target dimensions are deterministic given the base features.")
    ry = ry[:, keep]
    rx = X - _oof_regression(base, X, folds, model, seed)
    prod = (ry[:, :, None] * rx[:, None, :]).reshape(n, -1)
    _, inv = np.unique(groups, return_inverse=True)
    g = inv.max() + 1
    sums = np.zeros((g, prod.shape[1]))
    np.add.at(sums, inv, prod)
    counts = np.bincount(inv).astype(float)
    centred = sums - counts[:, None] * prod.mean(axis=0, keepdims=True)
    scale = np.sqrt((centred**2).sum(axis=0))
    scale[scale == 0] = np.inf
    t_stat = np.abs(sums.sum(axis=0)) / scale
    stat = float(t_stat.max())
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, n_mult, 250):
        e = rng.standard_normal((min(250, n_mult - start), g))
        boot = np.abs(e @ centred) / scale
        exceed += int((boot.max(axis=1) >= stat).sum())
    p = (exceed + 1) / (n_mult + 1)
    return MetricResult(
        name=name, value=stat, unit="max |T|", p_value=float(p), n_transitions=n,
        n_trajectories=g, details={"n_pairs": int(prod.shape[1])},
        assumptions=[
            "Validity requires the regressions on the conditioning set to be consistent (Shah & Peters, 2020).",
            "No conditional-independence test has power against all alternatives; non-rejection is not evidence of Markovianity.",
            "Trajectories are independent; dependence within a trajectory is handled by clustering.",
        ],
    )


# ------------------------------------------------------------- wrappers
def _design(trans: Transitions, target: str):
    if target == "state":
        return np.hstack([trans.state, trans.action_features]), trans.next_state, "regression"
    if target == "reward":
        if trans.reward is None:
            raise ValueError("No reward column.")
        return np.hstack([trans.state, trans.action_features, trans.next_state]), trans.reward, "regression"
    if target == "action":
        if trans.action_levels is not None:
            return trans.state, trans.action, "classification"
        return trans.state, trans.action, "regression"
    raise ValueError(f"Unknown target '{target}'.")


def history_gain(
    trans: Transitions,
    target: Literal["state", "reward", "action"] = "state",
    lags: int | None = None,
    **kw,
) -> MetricResult:
    """History information gain for next state, reward or behaviour action."""
    base, y, kind = _design(trans, target)
    width = trans.history.shape[1] // max(trans.lags, 1)
    k = trans.lags if lags is None else lags
    extra = trans.history[:, : k * width]
    res = conditional_information_gain(base, extra, y, trans.groups, kind, name=f"history_gain[{target}]", **kw)
    res.details["lags"] = k
    return res


def history_gcm(trans: Transitions, target: Literal["state", "reward", "action"] = "state", **kw) -> MetricResult:
    base, y, kind = _design(trans, target)
    if kind == "classification":
        levels = trans.action_levels
        y = (y.reshape(-1, 1) == levels.reshape(1, -1)).astype(float)
    return gcm_test(base, trans.history, y, trans.groups, name=f"history_gcm[{target}]", **kw)


def lag_profile(trans: Transitions, target: str = "state", **kw) -> list[MetricResult]:
    """History gain for 1..trans.lags lags on identical rows."""
    return [history_gain(trans, target, lags=k, **kw) for k in range(1, trans.lags + 1)]


def time_inhomogeneity_gain(trans: Transitions, **kw) -> MetricResult:
    """Gain from adding decision step and time to the next-state model.

    Positive values indicate non-stationary dynamics given (s, a) — or a state
    that omits a time-varying factor.
    """
    base = np.hstack([trans.state, trans.action_features])
    extra = np.column_stack([trans.step, trans.time]).astype(float)
    res = conditional_information_gain(base, extra, trans.next_state, trans.groups, "regression", name="time_inhomogeneity_gain", **kw)
    res.assumptions = res.assumptions[:3] + ["Non-stationarity is only detected along decision step / elapsed time."]
    return res


def action_effect_gain(trans: Transitions, **kw) -> MetricResult:
    """Gain from adding the recorded action a_t to a next-state model given s_t.

    If treatments are believed to act within one decision interval, a gain near
    zero suggests that recorded actions are misaligned with the states (e.g.
    documentation delay, wrong time binning) or that the interval is too short
    for effects to appear.
    """
    res = conditional_information_gain(trans.state, trans.action_features, trans.next_state, trans.groups, "regression",
                                       name="action_effect_gain", **kw)
    res.assumptions = res.assumptions[:3] + ["Interpretation requires prior knowledge that the action affects the state within one interval."]
    return res


def delayed_action_effect_gain(trans: Transitions, **kw) -> MetricResult:
    """Gain from adding only *lagged actions* to the next-state model.

    Positive values mean past treatments affect the next state beyond the
    current state and action (delayed effects, e.g. drug accumulation).
    """
    d = len(trans.state_names)
    width = trans.history.shape[1] // max(trans.lags, 1)
    cols = [k * width + j for k in range(trans.lags) for j in range(d, width)]
    base = np.hstack([trans.state, trans.action_features])
    res = conditional_information_gain(base, trans.history[:, cols], trans.next_state, trans.groups, "regression", name="delayed_action_effect_gain", **kw)
    return res
