import numpy as np
from threadpoolctl import threadpool_limits

from mdp_diagnostics.experiments.agents import fit_fqi
from mdp_diagnostics.experiments.run import GENE_REPS, _gene_frame, _gene_obs_features, _gene_transitions
from mdp_diagnostics.experiments.synthetic import GeneTreatmentEnv


def test_gene_blocks_medication_1():
    env = GeneTreatmentEnv(prevalence=0.5, sigma=0.0)
    rng = np.random.default_rng(0)
    x = np.ones((2, 3))
    g = np.array([0.0, 1.0])
    x1, _ = env._step(x, np.array([1, 1]), g, rng)
    x0, _ = env._step(x, np.array([0, 0]), g, rng)
    assert np.allclose(x0[1], x1[1])  # carrier: medication 1 has no effect
    assert np.all(x1[0] < x0[0])


def test_known_carriers_never_get_medication_1():
    df = GeneTreatmentEnv(prevalence=0.5, test_share=1.0).generate(100, seed=0)
    assert (df.loc[df["gene"] == 1, "pb_1"] == 0).all()
    assert (df["gene_positive"] == df["gene"]).all()


def test_unrecorded_test_is_hidden_from_data():
    df = GeneTreatmentEnv(prevalence=0.5, test_share=1.0, record_test=False).generate(50, seed=0)
    assert (df["gene_tested"] == 0).all() and (df.loc[df["gene"] == 1, "pb_1"] == 0).all()


def test_online_features_match_offline_features():
    env = GeneTreatmentEnv(prevalence=0.3, test_share=0.5, horizon=4)
    df = env.generate(5, seed=1)
    rep = GENE_REPS["revised"]
    d, cols = _gene_frame(df, rep, np.random.default_rng(0))
    one = d[d["id"] == 0].reset_index(drop=True)
    obs = {"gene": np.array([one.loc[0, "gene"]]), "gene_tested": np.array([one.loc[0, "gene_tested"]]),
           "gene_positive": np.array([one.loc[0, "gene_positive"]]), "m_hist": [], "a_hist": []}
    for t in range(3):
        obs |= {"m": one.loc[[t], ["m0", "m1", "m2"]].to_numpy(), "t": t}
        assert np.allclose(_gene_obs_features(obs, rep, np.random.default_rng(0)), one.loc[[t], cols].to_numpy())
        obs["m_hist"].append(obs["m"])
        obs["a_hist"].append(np.array([int(one.loc[t, "action"])]))


def test_fqi_with_true_gene_beats_constant_policies():
    env = GeneTreatmentEnv(prevalence=0.5, test_share=0.0, sigma=0.2)
    rep = GENE_REPS["oracle"]
    d, cols = _gene_frame(env.generate(300, seed=0), rep, np.random.default_rng(0))
    with threadpool_limits(1):
        agent = fit_fqi(*_gene_transitions(d, cols)[:5], env.n_actions, iterations=10)
        rng = np.random.default_rng(0)
        v_agent = env.rollout(lambda o: agent.act(_gene_obs_features(o, rep, rng)), n_traj=2000, gamma=0.95)
    v_const = max(env.rollout(lambda o, k=k: np.tile(np.eye(3)[k], (len(o["m"]), 1)), n_traj=2000, gamma=0.95) for k in range(3))
    assert v_agent > v_const
