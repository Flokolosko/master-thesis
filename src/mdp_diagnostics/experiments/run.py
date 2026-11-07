"""Controlled validation experiments.

Usage:  uv run python -m mdp_diagnostics.experiments.run <experiment> [--config configs/experiments.yaml] [--quick]

Experiments (see research/validation_plan.md):
  overlap      Exp 1/4  action support & coverage vs. true OPE error
  state        Exp 2    hidden latent state: power, false alarms, value error (RQ1-RQ3)
  propofol     Exp 2b   PK/PD anaesthesia: BIS-only vs. augmented vs. full state, decision interval
  temporal     Exp 3    irregular sampling, (informative) missingness, action misalignment
  gene         Exp 4    illness with 3 measurables, 2 medications and a partially recorded gene;
                        framework verdicts vs. offline RL agents trained on each formulation
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from mdp_diagnostics import AssessmentConfig, FeasibilityAssessment, TrajectoryData, TrajectorySchema
from mdp_diagnostics.experiments import perturbations as P
from mdp_diagnostics.experiments.agents import fit_fqi
from mdp_diagnostics.experiments.ope import model_based_value, wis_estimate
from mdp_diagnostics.experiments.synthetic import GeneTreatmentEnv, LatentLinearEnv, PropofolEnv, sample_patients
from mdp_diagnostics.framework.criteria import Status, classify_gain
from mdp_diagnostics.metrics import coverage, markov, ope_reliability, overlap, temporal
from mdp_diagnostics.utils.statistics import spearman_with_ci, wilson_interval

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"


def _lin_schema(states=("x0", "x1"), **kw) -> TrajectorySchema:
    return TrajectorySchema(id_col="id", time_col="time", state_cols=list(states), action_col="action",
                            reward_col="reward", action_levels=[0.0, 1.0, 2.0], **kw)


# ======================================================== Exp 1: overlap
def exp_overlap(cfg: dict) -> pd.DataFrame:
    rows = []
    for beta in cfg["betas"]:
        for forbid in cfg["forbid_above"]:
            for n in cfg["n_traj"]:
                for seed in range(cfg["seeds"]):
                    env = LatentLinearEnv(beta=beta, forbid_above=forbid, horizon=cfg["horizon"])
                    df = env.generate(n, seed=seed)
                    tr = TrajectoryData(df, _lin_schema()).transitions()
                    bp = overlap.fit_behaviour_policy(tr, seed=seed)
                    lam = cfg["target_lambda"]
                    target = ope_reliability.reference_policy_family(bp, lam)
                    diag = ope_reliability.is_weight_diagnostics(tr, bp, target)
                    # Ground truth: the same reference policy defined on the TRUE behaviour policy.
                    true_pb = df.loc[df["action"].notna(), [f"pb_{k}" for k in range(3)]].to_numpy()
                    true_target_obs = ((1 - lam) * true_pb + lam / 3)[np.arange(tr.n), bp.observed_index]
                    v_true = env.policy_value(lambda x, z: (1 - lam) * env.behaviour_proba(x, z) + lam / 3, n_traj=cfg["mc"], seed=seed + 999)
                    v_wis_true_pb = wis_estimate(tr, true_pb[np.arange(tr.n), bp.observed_index], true_target_obs)
                    v_wis_est = wis_estimate(tr, bp.observed_proba, target[np.arange(tr.n), bp.observed_index])
                    rows.append({
                        "beta": beta, "forbid_above": forbid if forbid is not None else "none", "n_traj": n, "seed": seed,
                        "true_support_violation": float((true_pb < 0.01).any(axis=1).mean()),
                        "support_violation": overlap.support_violation_rate(tr, bp, 0.01, n_boot=0).value,
                        "aoi": overlap.action_overlap_index(tr, bp, 0.05).value,
                        "entropy": overlap.action_entropy(tr, bp, n_boot=0).value,
                        "knn_deterministic": overlap.knn_local_action_diversity(tr, seed=seed).value,
                        "propensity_p05": overlap.observed_action_propensity(tr, bp).value,
                        "esac": coverage.esac(tr, seed=seed).value,
                        "occupancy_support": coverage.occupancy_weighted_support(tr, seed=seed).value,
                        "ess_share": diag.value, "pareto_k": diag.details["pareto_k"],
                        "v_true": v_true, "wis_err_est_pb": abs(v_wis_est - v_true), "wis_err_true_pb": abs(v_wis_true_pb - v_true),
                    })
    return pd.DataFrame(rows)


# =================================================== Exp 2: hidden state
def _target_policy_x(x):
    """Fixed evaluation policy acting on x0 only (column 0 of any state matrix)."""
    logits = 3.0 * (-x[:, :1] * np.array([-1.0, 0.0, 1.0])[None, :])
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _representations(df: pd.DataFrame, lags: int, rng: np.random.Generator) -> dict:
    hist_df, hist_cols = P.add_lag_features(df, "id", ["x0", "x1", "action"], lags, pad={"x0": "first", "x1": "first", "action": 1.0})
    noise_df = df.copy()
    noise_df["noise"] = rng.normal(size=len(df))
    return {
        "full": (df, ["x0", "x1", "z"]),
        "partial": (df, ["x0", "x1"]),
        "history": (hist_df, ["x0", "x1", *hist_cols]),
        # "random revisions": plausible-looking changes not targeted at the diagnosed problem
        "random_noise_feature": (noise_df, ["x0", "x1", "noise"]),
        "random_drop_x1": (df, ["x0"]),
    }


def exp_state(cfg: dict) -> pd.DataFrame:
    rows = []
    for strength in cfg["latent_strengths"]:
        for n in cfg["n_traj"]:
            for seed in range(cfg["seeds"]):
                env = LatentLinearEnv(latent_strength=strength, rho=cfg["rho"], horizon=cfg["horizon"])
                df = env.generate(n, seed=seed)
                v_true = env.policy_value(lambda x, z: _target_policy_x(x), n_traj=cfg["mc"], seed=seed + 777)
                rng = np.random.default_rng(seed)
                for rep, (d, cols) in _representations(df, cfg["lags"], rng).items():
                    if rep.startswith("random") and n != max(cfg["n_traj"]):
                        continue
                    data = TrajectoryData(d, _lin_schema(cols))
                    tr = data.transitions(lags=cfg["lags"])
                    g = markov.history_gain(tr, "state", n_boot=cfg["n_boot"], seed=seed)
                    t = markov.history_gcm(tr, "state", n_mult=cfg["n_mult"], seed=seed)
                    tr0 = data.transitions(lags=0)
                    v_hat = model_based_value(tr0, _target_policy_x, n_rollouts=cfg["rollouts"], horizon=cfg["horizon"], seed=seed)
                    rows.append({"latent_strength": strength, "n_traj": n, "seed": seed, "representation": rep,
                                 "gain": g.value, "gain_lower": g.lower, "gain_upper": g.upper, "gcm_p": t.p_value,
                                 "reject": t.p_value < 0.05, "v_true": v_true, "v_model": v_hat,
                                 "value_error": abs(v_hat - v_true), "rel_value_error": abs(v_hat - v_true) / abs(v_true)})
    return pd.DataFrame(rows)


# ===================================================== Exp 2b: propofol
def exp_propofol(cfg: dict) -> pd.DataFrame:
    rows = []
    for interval in cfg["decision_intervals"]:
        for n in cfg["n_patients"]:
            for seed in range(cfg["seeds"]):
                env = PropofolEnv(decision_interval=interval, duration=cfg["duration"])
                df = env.simulate(sample_patients(n, np.random.default_rng(seed)), seed=seed)
                df["prev_action"] = df.groupby("id")["action"].shift(1).fillna(0.0)
                reps = {
                    "bis_only": ["bis"],
                    "bis_prev_action": ["bis", "prev_action"],
                    "bis_covariates": ["bis", "age", "weight"],
                    "full_compartments": ["bis", "ce", "c1", "a2", "a3", "age", "weight"],
                }
                for rep, cols in reps.items():
                    schema = TrajectorySchema(id_col="id", time_col="time", state_cols=cols, action_col="action",
                                              reward_col="reward", action_levels=[0.0, 1.0, 2.0, 3.0])
                    tr = TrajectoryData(df, schema).transitions(lags=cfg["lags"])
                    for model in cfg["models"]:
                        g = markov.history_gain(tr, "state", model=model, n_boot=cfg["n_boot"], seed=seed)
                        t = markov.history_gcm(tr, "state", model=model, n_mult=cfg["n_mult"], seed=seed)
                        rows.append({"decision_interval": interval, "n_patients": n, "seed": seed, "representation": rep,
                                     "model": model, "gain": g.value, "gain_lower": g.lower, "gain_upper": g.upper,
                                     "gcm_p": t.p_value, "reject": t.p_value < 0.05})
    return pd.DataFrame(rows)


# ===================================================== Exp 3: temporal
def exp_temporal(cfg: dict) -> pd.DataFrame:
    rows = []
    for seed in range(cfg["seeds"]):
        env = LatentLinearEnv(horizon=cfg["horizon"])
        base = env.generate(cfg["n_traj"], seed=seed)
        ret = base.groupby("id")["reward"].sum()
        base["outcome"] = base["id"].map((ret < ret.quantile(0.3)).astype(float))
        schema = _lin_schema(outcome_col="outcome")

        def record(kind, level, df, extra=None):
            d = TrajectoryData(df, schema)
            r = {"seed": seed, "perturbation": kind, "level": level,
                 "missingness": temporal.missingness(d).value,
                 "interval_cv": temporal.sampling_intervals(d).value,
                 "gap_traj_share": temporal.sampling_intervals(d).details.get("share_trajectories_with_gap"),
                 "obs_age_p90": temporal.observation_age(d).value,
                 "inf_miss_auc": temporal.informative_missingness_outcome(d, seed=seed).value,
                 "heaping_1": temporal.timestamp_heaping(d, units=(0.25, 1.0)).value}
            tr = d.transitions()
            r["action_effect_gain"] = markov.action_effect_gain(tr, n_boot=0, seed=seed).value if tr.n > 50 else np.nan
            rows.append(r | (extra or {}))

        for p in cfg["drop_probs"]:
            record("irregular_sampling", p, P.irregular_sampling(base, "id", p, seed=seed))
        for p in cfg["mcar_probs"]:
            record("mcar_missing", p, P.mcar_missing(base, ["x0", "x1"], p, seed=seed))
        for s in cfg["informative_strengths"]:
            record("informative_missing", s, P.informative_missing(base, ["x0", "x1"], 0.3, -s, "outcome", seed=seed))
        for sd in cfg["jitter_sd"]:
            record("timestamp_jitter_rounded", sd, P.timestamp_jitter(base, "time", sd, round_to=1.0 if sd > 0 else None, seed=seed))
        for k in cfg["action_shift"]:
            df = P.action_misalignment(base, "id", "action", k) if k else base
            record("action_misalignment", k, df)
    return pd.DataFrame(rows)


# ===================================================== Exp 4: gene / medication
M_COLS = ["m0", "m1", "m2"]
GENE_REC = ["gene_tested", "gene_positive"]
# Formulations a researcher could choose; "revised" follows the framework's recommendations
# (add recorded variables and the history that carries the gain).
GENE_REPS = {
    "naive": {"cols": M_COLS},
    "oracle": {"cols": [*M_COLS, "gene"]},  # true gene: Markov by construction
    "recorded": {"cols": [*M_COLS, *GENE_REC]},
    "history": {"cols": M_COLS, "lags": 2},
    "revised": {"cols": [*M_COLS, *GENE_REC], "lags": 2},
    "random_noise": {"cols": [*M_COLS, "noise"]},
    "random_drop_m2": {"cols": ["m0", "m1"]},
}
GENE_CRITERIA = ["B.state_sufficiency", "B.reward_sufficiency", "B.action_alignment", "B.stationarity",
                 "D.behaviour_history", "D.overlap", "D.coverage", "D.ope_reliability"]


def _gene_frame(df: pd.DataFrame, rep: dict, rng: np.random.Generator) -> tuple[pd.DataFrame, list[str]]:
    """Data frame with the representation's state columns (lags padded at the start of each trajectory)."""
    d = df.assign(noise=rng.normal(size=len(df)))
    cols = list(rep["cols"])
    if rep.get("lags"):
        d = d.assign(a0=(d["action"] == 0).astype(float), a1=(d["action"] == 1).astype(float), a2=(d["action"] == 2).astype(float))
        d, lag_cols = P.add_lag_features(d, "id", [*M_COLS, "a0", "a1", "a2"], rep["lags"],
                                         pad={**{c: "first" for c in M_COLS}, "a0": 1.0, "a1": 0.0, "a2": 0.0})
        cols += lag_cols
    return d, cols


def _gene_obs_features(obs: dict, rep: dict, rng: np.random.Generator) -> np.ndarray:
    """The same features as ``_gene_frame`` built online inside a simulator rollout."""
    m = obs["m"]
    named = {"m0": m[:, 0], "m1": m[:, 1], "m2": m[:, 2], "gene": obs["gene"], "gene_tested": obs["gene_tested"],
             "gene_positive": obs["gene_positive"], "noise": rng.normal(size=len(m))}
    parts = [named[c] for c in rep["cols"]]
    for k in range(1, rep.get("lags", 0) + 1):
        if len(obs["m_hist"]) >= k:
            mk, ak = obs["m_hist"][-k], obs["a_hist"][-k]
        else:
            mk, ak = (obs["m_hist"][0] if obs["m_hist"] else m), np.zeros(len(m), dtype=int)
        parts += [mk[:, 0], mk[:, 1], mk[:, 2], (ak == 0).astype(float), (ak == 1).astype(float), (ak == 2).astype(float)]
    return np.column_stack(parts)


def _gene_transitions(d: pd.DataFrame, cols: list[str]):
    g = d.groupby("id", sort=False)
    nxt = g[cols].shift(-1)
    keep = d["action"].notna().to_numpy() & nxt.notna().all(axis=1).to_numpy()
    terminal = (g.cumcount(ascending=False) == 1).to_numpy()[keep]  # next row is the final observation
    first = (g.cumcount() == 0).to_numpy()[keep]
    return (d.loc[keep, cols].to_numpy(float), d.loc[keep, "action"].to_numpy(float), d.loc[keep, "reward"].to_numpy(float),
            nxt.loc[keep].to_numpy(float), terminal, first)


def _gene_schema(cols, **kw) -> TrajectorySchema:
    return TrajectorySchema(id_col="id", time_col="time", state_cols=list(cols), action_col="action",
                            reward_col="reward", action_levels=[0.0, 1.0, 2.0], **kw)


def _gene_job(job: dict) -> list[dict]:
    # One OpenMP thread per worker: multithreaded boosting on small data was ~100x slower here.
    with threadpool_limits(limits=1):
        return _gene_job_inner(job)


def _gene_job_inner(job: dict) -> list[dict]:
    cfg, scen, prev, sigma, seed = job["cfg"], job["scenario"], job["prevalence"], job["sigma"], job["seed"]
    env = GeneTreatmentEnv(prevalence=prev, sigma=sigma, horizon=cfg["horizon"], **cfg["scenarios"][scen])
    df = env.generate(cfg["n_traj"], seed=seed)
    gamma, model = cfg["gamma"], cfg["model"]
    base = {"scenario": scen, "prevalence": prev, "sigma": sigma, "seed": seed}

    # 1) The framework on the formulation a researcher would start with (measurables only).
    conf = AssessmentConfig(schema=_gene_schema(M_COLS).__dict__, lags=cfg["lags"], model=model, n_boot=cfg["n_boot"], seed=seed, gamma=gamma)
    report = FeasibilityAssessment(conf).evaluate(TrajectoryData(df, _gene_schema(M_COLS)))
    verdicts = {f"status[{c.id}]": c.status.value for c in report.criteria if c.id in GENE_CRITERIA}
    vals = {}
    for c in report.criteria:
        for m in c.metrics:
            if m.name in ("support_violation_rate", "support_violation_rate[history-aware]", "history_gain[action]") or m.name.startswith("is_ess_share[mix_uniform_0.25]"):
                vals[m.name] = m.value
                if m.name.startswith("is_ess_share"):
                    vals[m.name.replace("is_ess_share", "pareto_k")] = m.details.get("pareto_k")

    # Behaviour (clinician) value: on-policy Monte-Carlo return of the observed trajectories.
    tr_all = df[df["action"].notna()]
    v_behaviour = float((gamma ** tr_all["time"] * tr_all["reward"]).groupby(tr_all["id"]).sum().mean())

    rows = []
    rng = np.random.default_rng(seed)
    for name, rep in GENE_REPS.items():
        if name in ("recorded", "revised") and scen == "hidden":
            continue  # nothing recorded about the gene
        d, cols = _gene_frame(df, rep, rng)
        tr = TrajectoryData(d, _gene_schema(cols)).transitions(lags=cfg["lags"])
        gain = markov.history_gain(tr, "state", model=model, n_boot=cfg["n_boot"], seed=seed)
        test = markov.history_gcm(tr, "state", model=model, n_mult=cfg["n_mult"], seed=seed)
        status, _ = classify_gain(gain, test, 0.01, 0.05)
        row = base | {"representation": name, "gain": gain.value, "gain_lower": gain.lower, "gain_upper": gain.upper,
                      "gcm_p": test.p_value, "reject": test.p_value < 0.05, "status": status.value,
                      "v_behaviour": v_behaviour}
        if name in ("naive", "oracle"):
            row["gain_linear"] = markov.history_gain(tr, "state", model="linear", n_boot=0, seed=seed).value
        # 2) Train offline RL agents on this formulation and evaluate them in the true simulator.
        X, a, r, Xn, term, first = _gene_transitions(d, cols)
        for constrained in (False, True) if name in ("naive", "revised") else (False,):
            agent = fit_fqi(X, a, r, Xn, term, env.n_actions, gamma=gamma, iterations=cfg["fqi_iterations"], constrained=constrained, seed=seed)
            frng = np.random.default_rng(seed + 31)
            v_true = env.rollout(lambda o: agent.act(_gene_obs_features(o, rep, frng)), n_traj=cfg["mc"], seed=seed + 777, gamma=gamma)
            q0 = np.max(agent.q_values(X[first]), axis=1).mean()
            suffix = "_constrained" if constrained else ""
            row[f"v_agent{suffix}"] = v_true
            row[f"q_estimate{suffix}"] = float(q0)
        rows.append(row | (verdicts | vals if name == "naive" else {}))
    return rows


def exp_gene(cfg: dict) -> pd.DataFrame:
    jobs = [{"cfg": cfg, "scenario": s, "prevalence": p, "sigma": sd, "seed": k}
            for s in cfg["scenarios"] for p in cfg["prevalences"] for sd in cfg["sigmas"] for k in range(cfg["seeds"])]
    with ProcessPoolExecutor(max_workers=cfg.get("workers", 1)) as ex:
        rows = [r for res in ex.map(_gene_job, jobs) for r in res]
    return pd.DataFrame(rows)


# ===================================================== summaries
def summarise_overlap(df: pd.DataFrame) -> dict:
    diags = ["support_violation", "aoi", "entropy", "knn_deterministic", "propensity_p05", "esac", "occupancy_support", "ess_share", "pareto_k"]
    out = {"spearman_vs_wis_error": {}, "spearman_vs_true_support_violation": {}}
    for d in diags:
        out["spearman_vs_wis_error"][d] = spearman_with_ci(df[d], df["wis_err_est_pb"]).to_dict()
        out["spearman_vs_true_support_violation"][d] = spearman_with_ci(df[d], df["true_support_violation"]).to_dict()
    out["support_violation_bias"] = float((df["support_violation"] - df["true_support_violation"]).mean())
    return out


def summarise_state(df: pd.DataFrame, min_gain: float = 0.01) -> dict:
    out = {"rejection_rates": [], "rq2": {}, "rq3": {}}
    for (s, n, rep), g in df.groupby(["latent_strength", "n_traj", "representation"]):
        w = wilson_interval(int(g["reject"].sum()), len(g))
        out["rejection_rates"].append({"latent_strength": s, "n_traj": n, "representation": rep, "rate": w.value,
                                       "lower": w.lower, "upper": w.upper, "mean_gain": float(g["gain"].mean()),
                                       "mean_rel_value_error": float(g["rel_value_error"].mean())})
    part = df[df["representation"] == "partial"]
    out["rq2"]["spearman_gain_vs_rel_value_error"] = spearman_with_ci(part["gain"], part["rel_value_error"]).to_dict()
    # RQ3: recommended revision (history augmentation) vs random revisions, where the framework flags a concern.
    nmax = df["n_traj"].max()
    key = ["latent_strength", "seed"]
    sub = df[df["n_traj"] == nmax].pivot_table(index=key, columns="representation", values="rel_value_error")
    flags = df[(df["n_traj"] == nmax) & (df["representation"] == "partial")].set_index(key)
    flagged = sub[(flags["reject"] & (flags["gain_lower"] >= min_gain)).reindex(sub.index).fillna(False).astype(bool)]
    if len(flagged):
        rng = np.random.default_rng(0)
        randoms = ("random_noise_feature", "random_drop_x1")
        e_orig = flagged["partial"].to_numpy()
        e_rec = flagged["history"].to_numpy()
        e_rand = flagged[list(randoms)].to_numpy()  # (cases, 2)

        def ratio_rer(idx):
            rec = 1 - e_rec[idx].mean() / e_orig[idx].mean()
            rand = 1 - e_rand[idx].mean() / e_orig[idx].mean()
            return rec, rand

        rec, rand = ratio_rer(np.arange(len(e_orig)))
        boots = []
        for _ in range(2000):
            i = rng.integers(0, len(e_orig), len(e_orig))
            r1, r2 = ratio_rer(i)
            boots.append(r1 - r2)
        per_case_rec = (e_orig - e_rec) / e_orig
        per_case_rand = ((e_orig[:, None] - e_rand) / e_orig[:, None]).ravel()
        out["rq3"] = {
            "n_flagged_cases": int(len(flagged)),
            "rer_recommended": float(rec), "rer_random": float(rand),
            "difference": float(rec - rand), "difference_ci": np.quantile(boots, [0.025, 0.975]).tolist(),
            "median_per_case_rer_recommended": float(np.median(per_case_rec)),
            "median_per_case_rer_random": float(np.median(per_case_rand)),
            "by_latent_strength": flagged.assign(rer_rec=per_case_rec).groupby(level=0)["rer_rec"].median().to_dict(),
            "definition": "RER = 1 - mean(E_rev) / mean(E_orig) over flagged cases (ratio of means); bootstrap over cases.",
        }
    return out


def summarise_gene(df: pd.DataFrame) -> dict:
    out = {"framework_on_state": [], "agents": [], "verdicts_naive": [], "rq": {}}
    for (sc, p, sd, rep), g in df.groupby(["scenario", "prevalence", "sigma", "representation"]):
        w = wilson_interval(int(g["reject"].sum()), len(g))
        out["framework_on_state"].append({"scenario": sc, "prevalence": p, "sigma": sd, "representation": rep, "gcm_rejection": w.value,
                                          "ci": [w.lower, w.upper], "share_concern": float((g["status"] == Status.CONCERN.value).mean()),
                                          "mean_gain": float(g["gain"].mean())})
    agent_cols = [c for c in df.columns if c.startswith(("v_agent", "q_estimate"))]
    key = ["scenario", "prevalence", "sigma", "seed"]
    oracle = df[df["representation"] == "oracle"].set_index(key)["v_agent"]
    d = df.join(oracle.rename("v_oracle_agent"), on=key)
    d["regret"] = d["v_oracle_agent"] - d["v_agent"]
    for (sc, p, sd, rep), g in d.groupby(["scenario", "prevalence", "sigma", "representation"]):
        out["agents"].append({"scenario": sc, "prevalence": p, "sigma": sd, "representation": rep, "v_behaviour": float(g["v_behaviour"].mean()),
                              "regret_vs_oracle_agent": float(g["regret"].mean())} | {c: float(g[c].mean()) for c in agent_cols if g[c].notna().any()})
    naive = d[d["representation"] == "naive"]
    status_cols = [c for c in naive.columns if c.startswith("status[")]
    for (sc, p, sd), g in naive.groupby(["scenario", "prevalence", "sigma"]):
        out["verdicts_naive"].append({"scenario": sc, "prevalence": p, "sigma": sd} | {c: g[c].value_counts().to_dict() for c in status_cols})
    # Does the framework's verdict on the naive formulation anticipate the agent's shortfall?
    out["rq"]["spearman_naive_gain_vs_regret"] = spearman_with_ci(naive["gain"], naive["regret"]).to_dict()
    flagged = naive["status"] == Status.CONCERN.value
    out["rq"]["naive_regret_by_verdict"] = {"concern": float(naive.loc[flagged, "regret"].mean()) if flagged.any() else None,
                                            "no_concern_or_inconclusive": float(naive.loc[~flagged, "regret"].mean()) if (~flagged).any() else None,
                                            "n_concern": int(flagged.sum()), "n_other": int((~flagged).sum())}
    # Following the recommendation vs random revisions, among flagged datasets (value gained over the naive agent).
    piv = d.pivot_table(index=key, columns="representation", values="v_agent")
    fl = naive.set_index(key)["status"].eq(Status.CONCERN.value).reindex(piv.index).fillna(False).astype(bool)
    sub = piv[fl]
    if len(sub):
        rec = "revised" if sub["revised"].notna().all() else "history"
        rec_vals = sub["revised"].fillna(sub["history"])
        rand = sub[["random_noise", "random_drop_m2"]].mean(axis=1)
        gain_rec, gain_rand = rec_vals - sub["naive"], rand - sub["naive"]
        rng = np.random.default_rng(0)
        diff = (gain_rec - gain_rand).to_numpy()
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        out["rq"]["revision_vs_random"] = {"n_flagged": int(len(sub)), "recommended": rec,
                                           "mean_value_gain_recommended": float(gain_rec.mean()),
                                           "mean_value_gain_random": float(gain_rand.mean()),
                                           "difference": float(diff.mean()), "difference_ci": np.quantile(boots, [0.025, 0.975]).tolist(),
                                           "definition": "value gain = true value of the agent on the revised formulation minus the naive agent (same dataset); "
                                                         "recommended = 'revised' (recorded gene + 2 lags) where a test result is recorded, else 'history'."}
    # Recorded gene: unconstrained vs support-constrained agent (known carriers never received medication 1).
    rev = d[(d["representation"] == "revised") & (d["prevalence"] > 0) & (d["scenario"] == "partial")]
    if len(rev):
        rng = np.random.default_rng(1)
        out["rq"]["recorded_gene_agents"] = {}
        v_naive = naive.set_index(key)["v_agent"].reindex(rev.set_index(key).index).to_numpy()
        for col in ("v_agent", "v_agent_constrained"):
            diff = rev[col].to_numpy() - v_naive
            boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
            out["rq"]["recorded_gene_agents"][f"{col}_minus_naive"] = {"mean": float(diff.mean()), "ci": np.quantile(boots, [0.025, 0.975]).tolist(), "n": int(len(diff))}
    out["rq"]["share_naive_agent_worse_than_clinician"] = d[d["representation"] == "naive"].assign(w=lambda x: x["v_agent"] < x["v_behaviour"]).groupby("prevalence")["w"].mean().to_dict()
    # Optimism of the agents' own value estimates (Q at start states minus true value).
    out["rq"]["optimism"] = d.assign(opt=d["q_estimate"] - d["v_agent"]).groupby(["scenario", "representation"])["opt"].mean().reset_index().to_dict(orient="records")
    return out


def summarise_simple(df: pd.DataFrame, by: list[str]) -> list[dict]:
    num = df.select_dtypes("number").columns.difference(by + ["seed"])
    return df.groupby(by)[list(num)].mean().reset_index().to_dict(orient="records")


EXPERIMENTS = {"overlap": exp_overlap, "state": exp_state, "propofol": exp_propofol, "temporal": exp_temporal, "gene": exp_gene}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=[*EXPERIMENTS, "all"])
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiments.yaml"))
    ap.add_argument("--quick", action="store_true", help="use the 'quick' section of the config")
    args = ap.parse_args()
    conf = yaml.safe_load(Path(args.config).read_text())
    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    for name in names:
        cfg = conf[name] | (conf.get("quick", {}).get(name, {}) if args.quick else {})
        t0 = time.time()
        df = EXPERIMENTS[name](cfg)
        out = RESULTS / name
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "raw.csv", index=False)
        if name == "overlap":
            summary = summarise_overlap(df)
        elif name == "state":
            summary = summarise_state(df)
        elif name == "gene":
            summary = summarise_gene(df)
        elif name == "propofol":
            summary = {"by_condition": summarise_simple(df.assign(reject=df["reject"].astype(float)), ["decision_interval", "n_patients", "representation", "model"])}
        else:
            summary = {"by_condition": summarise_simple(df, ["perturbation", "level"])}
        summary["config"] = cfg
        summary["runtime_s"] = round(time.time() - t0, 1)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
        print(f"{name}: {len(df)} rows in {summary['runtime_s']} s -> {out}")


if __name__ == "__main__":
    main()
