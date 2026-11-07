"""Synthetic decision processes with known ground truth.

1. ``LatentLinearEnv``: a linear-Gaussian process with an optional latent AR(1)
   factor. With the latent factor hidden, the observed state is not Markov;
   its history carries information about the latent factor. Behaviour policy
   overlap, support restrictions and confounding are controllable.
2. ``PropofolEnv``: compartmental pharmacokinetic-pharmacodynamic model of
   propofol anaesthesia (Schnider et al. 1998 PK, effect-site compartment,
   sigmoid Emax (Hill) BIS response). Hidden compartment concentrations make
   the BIS-only state non-Markov in a physiologically meaningful way.
3. ``GeneTreatmentEnv``: an arbitrary illness with three measurables, two
   medications and a (partially recorded) gene that makes medication 1
   ineffective. With the gene hidden, the measurables are not Markov; with the
   true gene in the state they are Markov by construction.

PK/PD parameter values follow the commonly implemented Schnider model; they
serve to create realistic *structure* (latent compartments, delays,
inter-patient variability) and must be checked against the original sources
(and e.g. the PAS simulator) before quantitative clinical statements are made.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.linalg import expm


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# =========================================================== latent linear
@dataclass
class LatentLinearEnv:
    """x_{t+1} = A x_t + e[a_t] + c * z_t * w + sigma * eps,  z_{t+1} = rho z_t + sqrt(1-rho^2) eta.

    Observed state: x (2-d). Latent: z. Actions push x0 down / keep / up.
    Behaviour policy: softmax(beta * (-x0 * dir_a) + conf * z * dir_a) with an
    optional support restriction (action ``forbidden_action`` impossible when
    x0 > ``forbid_above``). Reward: -(x0_{t+1})^2 - cost * |dir_a|.
    """

    latent_strength: float = 0.0
    rho: float = 0.8
    beta: float = 2.0
    confounding: float = 0.0
    forbid_above: float | None = None
    forbidden_action: int = 2
    sigma: float = 0.3
    horizon: int = 20
    cost: float = 0.05
    A: np.ndarray = field(default_factory=lambda: np.array([[0.85, 0.1], [0.0, 0.7]]))
    directions: np.ndarray = field(default_factory=lambda: np.array([-1.0, 0.0, 1.0]))
    effect_size: float = 0.5
    w: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.5]))

    @property
    def n_actions(self) -> int:
        return len(self.directions)

    def behaviour_proba(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        logits = self.beta * (-x[:, :1] * self.directions[None, :]) + self.confounding * z[:, None] * self.directions[None, :]
        p = _softmax(logits)
        if self.forbid_above is not None:
            mask = x[:, 0] > self.forbid_above
            p[mask, self.forbidden_action] = 0.0
            p[mask] /= p[mask].sum(axis=1, keepdims=True)
        return p

    def _step(self, x, z, a, rng):
        eff = self.effect_size * self.directions[a]
        x_next = x @ self.A.T + np.column_stack([eff, np.zeros_like(eff)]) + self.latent_strength * z[:, None] * self.w + self.sigma * rng.standard_normal(x.shape)
        z_next = self.rho * z + np.sqrt(1 - self.rho**2) * rng.standard_normal(len(z))
        r = -(x_next[:, 0] ** 2) - self.cost * np.abs(self.directions[a])
        return x_next, z_next, r

    def _init(self, n, rng):
        return rng.normal(0, 1.0, (n, 2)), rng.standard_normal(n)

    def generate(self, n_traj: int, seed: int = 0) -> pd.DataFrame:
        """Simulate observational data; true behaviour probabilities are in columns ``pb_<a>``.

        The last row of each trajectory is the terminal observation (no action/reward).
        """
        rng = np.random.default_rng(seed)
        x, z = self._init(n_traj, rng)
        rows = []
        for t in range(self.horizon + 1):
            p = self.behaviour_proba(x, z)
            u = rng.random(n_traj)[:, None]
            a = np.minimum((p.cumsum(axis=1) < u).sum(axis=1), self.n_actions - 1)
            x_next, z_next, r = self._step(x, z, a, rng)
            last = t == self.horizon
            frame = {"id": np.arange(n_traj), "time": float(t), "x0": x[:, 0], "x1": x[:, 1], "z": z,
                     "action": np.where(last, np.nan, a.astype(float)), "reward": np.where(last, np.nan, r)}
            frame |= {f"pb_{k}": p[:, k] for k in range(self.n_actions)}
            rows.append(pd.DataFrame(frame))
            x, z = x_next, z_next
        return pd.concat(rows, ignore_index=True).sort_values(["id", "time"], kind="mergesort").reset_index(drop=True)

    def policy_value(self, policy, n_traj: int = 20000, seed: int = 123, gamma: float = 1.0) -> float:
        """Monte-Carlo value of ``policy(x, z) -> probabilities`` in the true environment."""
        rng = np.random.default_rng(seed)
        x, z = self._init(n_traj, rng)
        ret = np.zeros(n_traj)
        for t in range(self.horizon):
            p = policy(x, z)
            u = rng.random(n_traj)[:, None]
            a = (p.cumsum(axis=1) < u).sum(axis=1)
            a = np.minimum(a, self.n_actions - 1)
            x, z, r = self._step(x, z, a, rng)
            ret += gamma**t * r
        return float(ret.mean())


# ================================================================ propofol
@dataclass
class Patient:
    age: float
    weight: float
    height: float
    male: bool
    c50: float
    gamma: float
    e0: float


def sample_patients(n: int, rng: np.random.Generator) -> list[Patient]:
    out = []
    for _ in range(n):
        male = bool(rng.random() < 0.5)
        height = rng.normal(177 if male else 165, 8)
        weight = float(np.clip(rng.normal(80 if male else 68, 12), 45, 140))
        out.append(Patient(age=float(np.clip(rng.normal(50, 15), 18, 85)), weight=weight, height=float(height), male=male,
                           c50=float(4.47 * np.exp(rng.normal(0, 0.2))), gamma=float(np.clip(rng.normal(2.0, 0.3), 1.2, 3.0)),
                           e0=float(rng.normal(97.4, 1.5))))
    return out


def schnider_rates(p: Patient) -> dict:
    """Schnider (1998) three-compartment propofol model; rates in 1/min, volumes in L."""
    lbm = (1.1 * p.weight - 128 * (p.weight / p.height) ** 2) if p.male else (1.07 * p.weight - 148 * (p.weight / p.height) ** 2)
    v1, v2, v3 = 4.27, 18.9 - 0.391 * (p.age - 53), 238.0
    return {
        "v1": v1,
        "k10": 0.443 + 0.0107 * (p.weight - 77) - 0.0159 * (lbm - 59) + 0.0062 * (p.height - 177),
        "k12": 0.302 - 0.0056 * (p.age - 53),
        "k13": 0.196,
        "k21": (1.29 - 0.024 * (p.age - 53)) / v2,
        "k31": 0.0035,
        "ke0": 0.456,
    }


@dataclass
class PropofolEnv:
    """Closed-loop propofol infusion with a simulated anaesthetist.

    State (latent): amounts A1..A3 [mg], effect-site concentration Ce [ug/ml].
    Observation: BIS (+ noise). Actions: infusion levels in mg/kg/h.
    Behaviour: noisy proportional titration towards BIS 50 on the *observed*
    BIS (optionally peeking at the true Ce, i.e. information absent from the
    data: ``unrecorded_info``). Reward: -|BIS_{t+1} - 50| / 50 - dose cost.
    """

    decision_interval: float = 2.0  # minutes between decisions
    duration: float = 60.0  # minutes per case
    substep: float = 1.0 / 6.0  # 10 s integration step
    levels: tuple[float, ...] = (0.0, 4.0, 8.0, 12.0)  # mg/kg/h
    bis_noise: float = 3.0
    target: float = 50.0
    beta: float = 1.5
    unrecorded_info: float = 0.0
    dose_cost: float = 0.005

    def _matrices(self, p: Patient):
        r = schnider_rates(p)
        M = np.array([
            [-(r["k10"] + r["k12"] + r["k13"]), r["k21"], r["k31"], 0.0],
            [r["k12"], -r["k21"], 0.0, 0.0],
            [r["k13"], 0.0, -r["k31"], 0.0],
            [r["ke0"] / r["v1"], 0.0, 0.0, -r["ke0"]],
        ])
        B = np.array([1.0, 0.0, 0.0, 0.0])
        aug = np.zeros((5, 5))
        aug[:4, :4] = M
        aug[:4, 4] = B
        E = expm(aug * self.substep)
        return E[:4, :4], E[:4, 4], r["v1"]

    def bis(self, ce: np.ndarray, p: Patient) -> np.ndarray:
        ce = np.maximum(ce, 0)
        return p.e0 * (1 - ce**p.gamma / (ce**p.gamma + p.c50**p.gamma))

    def behaviour_proba(self, bis_obs: float, ce: float, p: Patient) -> np.ndarray:
        err = bis_obs - self.target
        desired = 6.0 + 0.15 * err + self.unrecorded_info * (p.c50 - ce)
        lv = np.array(self.levels)
        return _softmax(-self.beta * ((lv - desired) / 4.0) ** 2)

    def simulate(self, patients: list[Patient], seed: int = 0, policy=None) -> pd.DataFrame:
        """Simulate cases; ``policy(bis_obs, ce, patient) -> probabilities`` overrides the clinician."""
        rng = np.random.default_rng(seed)
        n_dec = int(self.duration / self.decision_interval)
        n_sub = int(round(self.decision_interval / self.substep))
        rows = []
        for pid, p in enumerate(patients):
            Ad, Bd, v1 = self._matrices(p)
            x = np.zeros(4)
            pol = policy or self.behaviour_proba
            for k in range(n_dec + 1):
                state = (pid, k * self.decision_interval, float(self.bis(x[3], p)) + self.bis_noise * rng.standard_normal(),
                         x[3], x[0] / v1, x[1], x[2], p.age, p.weight)
                if k == n_dec:  # terminal observation, no action
                    rows.append((*state, np.nan, np.nan))
                    break
                pr = pol(state[2], x[3], p)
                a = int(rng.choice(len(self.levels), p=pr))
                u = self.levels[a] * p.weight / 60.0  # mg/min
                for _ in range(n_sub):
                    x = Ad @ x + Bd * u
                r = -abs(float(self.bis(x[3], p)) - self.target) / 50.0 - self.dose_cost * self.levels[a]
                rows.append((*state, float(a), r))
        return pd.DataFrame(rows, columns=["id", "time", "bis", "ce", "c1", "a2", "a3", "age", "weight", "action", "reward"])

    def policy_value(self, patients: list[Patient], policy, seed: int = 999) -> float:
        df = self.simulate(patients, seed=seed, policy=policy)
        return float(df.groupby("id")["reward"].sum().mean())


# ========================================================= gene / medication
@dataclass
class GeneTreatmentEnv:
    """Arbitrary illness, three measurables, two medications and a gene.

    Dynamics (x = measurables, 0 = healthy, positive = ill):
        x_{t+1} = A x_t + d - [a=1] (1 - g) e1 - [a=2] e2 + sigma * eps
    Actions: 0 = no treatment, 1 = medication 1, 2 = medication 2.
    Medication 1 is more effective and has fewer side effects, but has no
    effect in gene carriers (g = 1). Medication 2 works for everyone.
    Reward: -mean(x_{t+1}^2) - side-effect cost of the action.

    Gene status is known (tested) for a share ``test_share`` of patients. The
    simulated clinician uses a known result (carriers never receive
    medication 1) and otherwise starts with medication 1 and switches to
    medication 2 after non-response. ``record_test=False`` keeps known results
    out of the data: the clinician acts on information the data lack
    (unmeasured confounding).
    """

    prevalence: float = 0.2
    test_share: float = 0.0
    record_test: bool = True
    horizon: int = 12
    sigma: float = 0.3
    A: np.ndarray = field(default_factory=lambda: np.array([[0.8, 0.1, 0.0], [0.1, 0.75, 0.05], [0.0, 0.1, 0.85]]))
    drive: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.2, 0.15]))
    e1: np.ndarray = field(default_factory=lambda: np.array([0.45, 0.3, 0.2]))
    e2: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.25, 0.2]))
    costs: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.05, 0.3]))
    switch_strength: float = 1.5  # how strongly the clinician switches to medication 2 after non-response
    n_actions: int = 3

    def _patients(self, n, rng):
        g = (rng.random(n) < self.prevalence).astype(float)
        tested = rng.random(n) < self.test_share
        x = rng.normal(1.2, 0.4, (n, 3))
        return g, tested, x

    def _step(self, x, a, g, rng):
        eff = (a == 1)[:, None] * (1 - g)[:, None] * self.e1 + (a == 2)[:, None] * self.e2
        x_next = x @ self.A.T + self.drive - eff + self.sigma * rng.standard_normal(x.shape)
        r = -(x_next**2).mean(axis=1) - self.costs[a]
        return x_next, r

    def behaviour_proba(self, x, g, tested, nonresponse) -> np.ndarray:
        sev = x.mean(axis=1)
        known_carrier = tested & (g == 1)
        known_non = tested & (g == 0)
        logits = np.column_stack([
            2.0 * (0.5 - sev),
            0.8 + 1.0 * known_non,
            -0.5 + self.switch_strength * nonresponse + 2.5 * known_carrier,
        ])
        p = _softmax(logits)
        p[known_carrier, 1] = 0.0  # hard rule: no medication 1 for known carriers
        return p / p.sum(axis=1, keepdims=True)

    def generate(self, n_traj: int, seed: int = 0) -> pd.DataFrame:
        """Observational data with true gene, recorded test result and true behaviour probabilities ``pb_<a>``."""
        rng = np.random.default_rng(seed)
        g, tested, x = self._patients(n_traj, rng)
        nonresp = np.zeros(n_traj)
        recorded = tested & self.record_test
        rows = []
        for t in range(self.horizon + 1):
            p = self.behaviour_proba(x, g, tested, nonresp)
            a = np.minimum((p.cumsum(axis=1) < rng.random(n_traj)[:, None]).sum(axis=1), self.n_actions - 1)
            x_next, r = self._step(x, a, g, rng)
            last = t == self.horizon
            frame = {"id": np.arange(n_traj), "time": float(t), "m0": x[:, 0], "m1": x[:, 1], "m2": x[:, 2],
                     "gene": g, "gene_tested": recorded.astype(float), "gene_positive": (recorded & (g == 1)).astype(float),
                     "action": np.where(last, np.nan, a.astype(float)), "reward": np.where(last, np.nan, r)}
            frame |= {f"pb_{k}": p[:, k] for k in range(self.n_actions)}
            rows.append(pd.DataFrame(frame))
            improved = x_next.mean(axis=1) < x.mean(axis=1) - 0.05
            nonresp = np.where(a == 1, np.where(improved, 0.0, np.minimum(nonresp + 1, 2)), np.where(a == 2, nonresp, 0.0))
            x = x_next
        return pd.concat(rows, ignore_index=True).sort_values(["id", "time"], kind="mergesort").reset_index(drop=True)

    def rollout(self, policy, n_traj: int = 10000, seed: int = 123, gamma: float = 1.0) -> float:
        """Monte-Carlo value of ``policy(obs) -> probabilities`` in the true environment.

        ``obs`` is a dict with the current measurables ``m``, lists ``m_hist`` and
        ``a_hist`` of past measurables/actions (most recent last), the recorded
        gene columns ``gene_tested``/``gene_positive``, the true ``gene`` (for
        oracle policies only) and the step ``t``.
        """
        rng = np.random.default_rng(seed)
        g, tested, x = self._patients(n_traj, rng)
        recorded = tested & self.record_test
        obs = {"gene": g, "gene_tested": recorded.astype(float), "gene_positive": (recorded & (g == 1)).astype(float),
               "m_hist": [], "a_hist": []}
        ret = np.zeros(n_traj)
        for t in range(self.horizon):
            obs |= {"m": x, "t": t}
            p = policy(obs)
            a = np.minimum((p.cumsum(axis=1) < rng.random(n_traj)[:, None]).sum(axis=1), self.n_actions - 1)
            x_next, r = self._step(x, a, g, rng)
            ret += gamma**t * r
            obs["m_hist"].append(x)
            obs["a_hist"].append(a)
            x = x_next
        return float(ret.mean())
