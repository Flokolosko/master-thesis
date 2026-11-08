"""Learning curves and extracted policies of offline RL agents on the gene/medication baseline.

Supplements Exp 4 (``run.py gene``) for one representative setting (default:
scenario *partial*, prevalence 0.3, sigma 0.15). Two algorithms are trained on
each formulation: fitted Q iteration with a frozen extra-trees structure
(converges exactly; Ernst et al. 2005) and offline Double DQN with a
cosine-decayed learning rate. For each agent it records

* the true value (Monte-Carlo in the simulator), the agent's own Q-estimate
  and convergence diagnostics (change of Q, share of greedy actions that
  changed) after every FQI iteration / every 1000 DQN gradient steps,
* the final true value as a function of the number of training trajectories,
* the greedy action on a severity grid, by gene status (extracted policy), and
* the share of each medication given to carriers / non-carriers in rollouts.

Usage::

    uv run python -m mdp_diagnostics.experiments.gene_curves --out results/gene/curves.json

Every finished (dataset, agent) unit is checkpointed in ``--checkpoint-dir``;
after an interruption, rerunning the same command skips finished units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from mdp_diagnostics.experiments.agents import FQIAgent, fit_dqn, fit_fqi
from mdp_diagnostics.experiments.run import GENE_REPS, _gene_frame, _gene_obs_features, _gene_transitions
from mdp_diagnostics.experiments.synthetic import GeneTreatmentEnv

AGENTS = {  # label -> (representation, support-constrained, algorithm)
    "naive": ("naive", False, "fqi"),
    "naive_constrained": ("naive", True, "fqi"),
    "oracle": ("oracle", False, "fqi"),
    "recorded": ("recorded", False, "fqi"),
    "recorded_constrained": ("recorded", True, "fqi"),
    "dqn_naive": ("naive", False, "dqn"),
    "dqn_oracle": ("oracle", False, "dqn"),
    "dqn_recorded": ("recorded", False, "dqn"),
    "dqn_recorded_constrained": ("recorded", True, "dqn"),
}
SEVERITY = np.round(np.linspace(0.0, 2.5, 26), 2)


def rollout_tracked(env: GeneTreatmentEnv, policy, n_traj: int, seed: int, gamma: float):
    """Like ``env.rollout`` but also returns action counts by true gene status."""
    rng = np.random.default_rng(seed)
    g, tested, x = env._patients(n_traj, rng)
    recorded = tested & env.record_test
    obs = {"gene": g, "gene_tested": recorded.astype(float), "gene_positive": (recorded & (g == 1)).astype(float),
           "m_hist": [], "a_hist": []}
    ret = np.zeros(n_traj)
    counts = np.zeros((2, env.n_actions))
    for t in range(env.horizon):
        obs |= {"m": x, "t": t}
        p = policy(obs)
        a = np.minimum((p.cumsum(axis=1) < rng.random(n_traj)[:, None]).sum(axis=1), env.n_actions - 1)
        for gg in (0, 1):
            counts[gg] += np.bincount(a[g == gg], minlength=env.n_actions)
        x_next, r = env._step(x, a, g, rng)
        ret += gamma**t * r
        obs["m_hist"].append(x)
        obs["a_hist"].append(a)
        x = x_next
    share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    return float(ret.mean()), share


def train_with_trace(algo, X, a, r, Xn, term, first, n_actions, gamma, constrained, seed, evaluate, length,
                     eval_every: int = 1, **opts):
    """Train FQI (``length`` iterations) or DQN (``length`` gradient steps) with a convergence trace.

    At every checkpoint (each FQI iteration, every ``dqn_every`` DQN steps):
    RMS change of Q on the data since the last checkpoint (``delta_q``), share
    of data states whose greedy action changed (``policy_change``), the
    Q-estimate at the first step and, every ``eval_every`` checkpoints, the
    true value from ``evaluate``.
    """
    trace, prev = [], {}
    every = opts.pop("dqn_every", 1000)
    n_checkpoints = length if algo == "fqi" else length // every

    def cb(it, agent):
        k = len(trace) + 1
        qx = agent.q_values(X)
        qa = np.where(np.isfinite(qx), qx, np.nan)
        act = qx.argmax(axis=1)
        rec = {"iteration": it + 1 if algo == "fqi" else it, "q_estimate": float(qx[first].max(axis=1).mean()),
               "delta_q": float(np.sqrt(np.nanmean((qa - prev["q"]) ** 2))) if prev else None,
               "policy_change": float(np.mean(act != prev["a"])) if prev else None}
        if k % eval_every == 0 or k == n_checkpoints:
            rec["v_true"] = evaluate(agent)
        trace.append(rec)
        prev.update(q=qa, a=act)

    if algo == "fqi":
        fqi_opts = {k: opts[k] for k in ("model", "smooth", "max_iter", "leaves", "n_trees", "min_samples_leaf", "freeze_after") if k in opts}
        agent = fit_fqi(X, a, r, Xn, term, n_actions, gamma=gamma, iterations=length, constrained=constrained,
                        seed=seed, callback=cb, **fqi_opts)
    else:
        dqn_opts = {k: opts[k] for k in ("hidden", "lr", "batch", "target_update", "polyak", "lr_end") if k in opts}
        agent = fit_dqn(X, a, r, Xn, term, n_actions, gamma=gamma, steps=length, constrained=constrained, seed=seed,
                        callback=cb, callback_every=every, **dqn_opts)
    return agent, trace


def policy_grid(agent: FQIAgent, rep: str) -> dict:
    """Greedy action at t=0 (no history) for equal measurables m0=m1=m2=s, by gene information."""
    n = len(SEVERITY)
    m = np.repeat(SEVERITY[:, None], 3, axis=1)
    groups = {"naive": {"all": {}}, "oracle": {"non-carrier": {"gene": 0.0}, "carrier": {"gene": 1.0}},
              "recorded": {"untested": {"gene_tested": 0.0, "gene_positive": 0.0},
                           "tested negative": {"gene_tested": 1.0, "gene_positive": 0.0},
                           "tested positive": {"gene_tested": 1.0, "gene_positive": 1.0}}}[rep]
    out = {}
    for label, extra in groups.items():
        named = {"m0": m[:, 0], "m1": m[:, 1], "m2": m[:, 2], "gene": np.zeros(n), "gene_tested": np.zeros(n), "gene_positive": np.zeros(n)}
        named |= {k: np.full(n, v) for k, v in extra.items()}
        X = np.column_stack([named[c] for c in GENE_REPS[rep]["cols"]])
        out[label] = agent.q_values(X).argmax(axis=1).tolist()
    return out


def clinician_grid(env: GeneTreatmentEnv) -> dict:
    m = np.repeat(SEVERITY[:, None], 3, axis=1)
    n = len(SEVERITY)
    res = {}
    for label, g, tested in (("untested", 0, False), ("tested negative", 0, True), ("tested positive", 1, True)):
        p = env.behaviour_proba(m, np.full(n, g), np.full(n, tested), np.zeros(n))
        res[label] = p.round(3).tolist()
    return res


def job(args: dict) -> dict:
    """One unit of work: one agent (or the clinician summary, ``agent="_clinician"``) on one dataset."""
    with threadpool_limits(limits=1):
        c, seed, n_traj, full, label = args["cfg"], args["seed"], args["n_traj"], args["full"], args["agent"]
        env = GeneTreatmentEnv(prevalence=c["prevalence"], sigma=c["sigma"], horizon=c["horizon"], test_share=c["test_share"], record_test=True)
        df = env.generate(n_traj, seed=seed)
        tr_all = df[df["action"].notna()]
        if label == "_clinician":
            g = tr_all["gene"].to_numpy()
            out = {"v_behaviour": float((c["gamma"] ** tr_all["time"] * tr_all["reward"]).groupby(tr_all["id"]).sum().mean())}
            if full:
                out["clinician_share_data"] = {"noncarrier": np.bincount(tr_all["action"][g == 0].astype(int), minlength=3).tolist(),
                                               "carrier": np.bincount(tr_all["action"][g == 1].astype(int), minlength=3).tolist()}
                out["clinician_grid"] = clinician_grid(env)
            return out
        rep_name, constrained, algo = AGENTS[label]
        rep = GENE_REPS[rep_name]
        d, cols = _gene_frame(df, rep, np.random.default_rng(seed))
        X, a, r, Xn, term, first = _gene_transitions(d, cols)

        def evaluate(agent, mc=c["mc_trace"] if full else c["mc"]):
            frng = np.random.default_rng(seed + 31)
            return rollout_tracked(env, lambda o: agent.act(_gene_obs_features(o, rep, frng)), mc, seed + 777, c["gamma"])[0]

        opts = {k: c[k] for k in ("model", "smooth", "max_iter", "leaves", "n_trees", "min_samples_leaf", "freeze_after",
                                  "hidden", "lr", "batch", "target_update", "polyak", "lr_end", "dqn_every")}
        length = (c["iterations"] if full else c["fqi_iterations"]) if algo == "fqi" else (c["dqn_steps"] if full else c["dqn_steps_sizes"])
        agent, trace = train_with_trace(algo, X, a, r, Xn, term, first, env.n_actions, c["gamma"], constrained, seed,
                                        evaluate if full else (lambda _: np.nan), length,
                                        eval_every=c["eval_every"] if full else 10**6, **opts)
        frng = np.random.default_rng(seed + 31)
        v, share = rollout_tracked(env, lambda o: agent.act(_gene_obs_features(o, rep, frng)), c["mc"], seed + 777, c["gamma"])
        res = {"v_true": v, "share_noncarrier": share[0].tolist(), "share_carrier": share[1].tolist(), "final_trace": trace[-1]}
        if full:
            res |= {"trace": trace, "policy": policy_grid(agent, rep_name)}
        return res


def _job_key(j: dict) -> str:
    """File name of a unit's checkpoint; includes a hash of the settings so changed settings never reuse stale results."""
    c = {k: v for k, v in j["cfg"].items() if k not in ("agents", "sizes", "n_traj")}
    h = hashlib.sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()[:10]
    return f"{'full' if j['full'] else 'size'}_n{j['n_traj']}_s{j['seed']}_{j['agent']}_{h}"


def _fmt_time(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/gene/curves.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--n-traj", type=int, default=1000, help="training patients for the iteration curves and policies")
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--fqi-iterations", type=int, default=30, help="iterations for the data-size curve")
    ap.add_argument("--sizes", default="100,300,1000,3000,10000")
    ap.add_argument("--model", default="frozen_trees", choices=["gbm", "frozen_trees"])
    ap.add_argument("--smooth", type=float, default=1.0, help="target damping (1 = plain FQI)")
    ap.add_argument("--max-iter", type=int, default=100, help="gbm only")
    ap.add_argument("--leaves", type=int, default=15, help="gbm only")
    ap.add_argument("--n-trees", type=int, default=50, help="frozen_trees only")
    ap.add_argument("--min-samples-leaf", type=int, default=10, help="frozen_trees only")
    ap.add_argument("--freeze-after", type=int, default=10, help="frozen_trees only")
    ap.add_argument("--dqn-steps", type=int, default=30000)
    ap.add_argument("--dqn-every", type=int, default=1000, help="DQN checkpoint interval (gradient steps)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--target-update", type=int, default=500, help="hard target update interval (if --polyak 0)")
    ap.add_argument("--polyak", type=float, default=0.0)
    ap.add_argument("--lr-end", type=float, default=1e-5, help="cosine-decay the DQN learning rate to this value")
    ap.add_argument("--agents", default=",".join(AGENTS))
    ap.add_argument("--size-agents", default="naive,oracle,recorded,recorded_constrained,dqn_naive,dqn_oracle,dqn_recorded,dqn_recorded_constrained")
    ap.add_argument("--size-seeds", type=int, default=3)
    ap.add_argument("--no-sizes", action="store_true")
    ap.add_argument("--checkpoint-dir", default="results/gene/curves_checkpoints",
                    help="finished units are saved here; rerunning the same command resumes")
    a = ap.parse_args()
    cfg = {"prevalence": 0.3, "sigma": 0.15, "test_share": 0.5, "horizon": 12, "gamma": 0.95, "iterations": a.iterations,
           "fqi_iterations": a.fqi_iterations, "mc": 10000, "mc_trace": 10000, "eval_every": 1, "n_traj": a.n_traj,
           "sizes": [] if a.no_sizes else [int(x) for x in a.sizes.split(",")],
           "model": a.model, "smooth": a.smooth, "max_iter": a.max_iter, "leaves": a.leaves,
           "dqn_steps": a.dqn_steps, "dqn_steps_sizes": a.dqn_steps, "dqn_every": a.dqn_every, "hidden": a.hidden,
           "lr": a.lr, "batch": a.batch, "target_update": a.target_update, "polyak": a.polyak, "lr_end": a.lr_end, "n_trees": a.n_trees, "min_samples_leaf": a.min_samples_leaf, "freeze_after": a.freeze_after, "agents": a.agents.split(",")}
    cfg_sizes = cfg | {"agents": a.size_agents.split(",")}
    # Largest datasets first so the slowest jobs start early.
    jobs = [{"cfg": cfg_sizes, "seed": s, "n_traj": n, "full": False} for n in sorted(cfg["sizes"], reverse=True) for s in range(a.size_seeds)]
    jobs += [{"cfg": cfg, "seed": s, "n_traj": cfg["n_traj"], "full": True} for s in range(a.seeds)]
    # One unit per (dataset, agent) so work is balanced and every finished unit is checkpointed.
    units = [j | {"agent": ag} for j in jobs for ag in ["_clinician", *j["cfg"]["agents"]]]
    ckpt = Path(a.checkpoint_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    done = {}
    for u in units:
        f = ckpt / f"{_job_key(u)}.json"
        if f.exists():
            done[_job_key(u)] = json.loads(f.read_text())
    todo = [u for u in units if _job_key(u) not in done]
    print(f"{len(units)} units, {len(done)} already checkpointed in {ckpt}, {len(todo)} to run", flush=True)
    t0, n0 = time.time(), len(done)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futures = {ex.submit(job, u): u for u in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            u = futures[fut]
            res = fut.result()
            key = _job_key(u)
            tmp = ckpt / f"{key}.json.tmp"
            tmp.write_text(json.dumps(res))
            tmp.replace(ckpt / f"{key}.json")  # atomic: an interrupted write never leaves a broken checkpoint
            done[key] = res
            el = time.time() - t0
            eta = el / k * (len(todo) - k)
            width = 30
            filled = int(width * (n0 + k) / len(units))
            print(f"[{'#' * filled}{'.' * (width - filled)}] {n0 + k}/{len(units)}  elapsed {_fmt_time(el)}  eta {_fmt_time(eta)}  "
                  f"last: {'full' if u['full'] else 'size'} n={u['n_traj']} seed={u['seed']} {u['agent']}", flush=True)

    def assemble(j):
        r = done[_job_key(j | {"agent": "_clinician"})] | {"seed": j["seed"], "n_traj": j["n_traj"]}
        r["agents"] = {ag: done[_job_key(j | {"agent": ag})] for ag in j["cfg"]["agents"]}
        return r

    results = [assemble(j) for j in jobs]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"config": cfg, "severity_grid": SEVERITY.tolist(),
                                       "full": [r for r, j in zip(results, jobs) if j["full"]],
                                       "sizes": [r for r, j in zip(results, jobs) if not j["full"]]}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
