"""End-to-end feasibility assessment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from mdp_diagnostics.data.preprocessing import PreprocessingLog, add_terminal_outcome_reward, discretize_actions, resample_to_grid
from mdp_diagnostics.data.schema import TrajectoryData, TrajectorySchema
from mdp_diagnostics.framework.criteria import CriterionResult, Status, classify_gain, gain_next_step, threshold_status, worst
from mdp_diagnostics.framework.report import Report
from mdp_diagnostics.metrics import coverage, markov, ope_reliability, overlap, temporal
from mdp_diagnostics.metrics.base import MetricResult


@dataclass
class Thresholds:
    """Decision thresholds. ``source`` documents how defensible each one is."""

    min_gain_nats: float = 0.01  # convention: ~1% of a nat per transition
    positivity_epsilon: float = 0.01  # convention, reported with sensitivity curve
    support_violation_concern: float = 0.10  # convention
    support_violation_serious: float = 0.50  # convention
    pareto_k_unreliable: float = 0.70  # Vehtari et al. (PSIS)
    ess_share_concern: float = 0.10  # convention
    reference_deviation: float = 0.25  # lambda of the reference target policy used for the status
    missingness_concern: float = 0.30  # convention
    informative_auc_concern: float = 0.60  # convention
    gap_trajectory_share_concern: float = 0.20  # convention
    gap_factor: float = 3.0
    epistemic_ratio_concern: float = 0.25  # convention
    within_step_change_concern: float = 0.20  # convention

    @staticmethod
    def sources() -> dict:
        return {"pareto_k_unreliable": "literature (Vehtari et al., PSIS)", "alpha": "statistical convention"} | {
            k: "convention (report with sensitivity analysis)" for k in Thresholds.__dataclass_fields__ if k != "pareto_k_unreliable"
        }


@dataclass
class AssessmentConfig:
    schema: dict
    dt: float | None = None  # decision grid width; None = use rows as decisions
    impute: str | None = None  # None or "locf" (explicit, recorded)
    action_agg: str = "mean"
    action_bins: int | None = None  # discretise continuous actions into this many levels
    terminal_reward: float | None = None  # reward = terminal_reward * outcome on the last transition
    lags: int = 3
    model: str = "linear"
    n_splits: int = 5
    n_boot: int = 300
    seed: int = 0
    alpha: float = 0.05
    gamma: float = 1.0
    thresholds: Thresholds = field(default_factory=Thresholds)
    clinical: dict = field(default_factory=dict)  # answers to Level A/E questions

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AssessmentConfig":
        d = yaml.safe_load(Path(path).read_text())
        d["thresholds"] = Thresholds(**d.get("thresholds", {}))
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


# Level A / E questions answered by clinical metadata, not by data.
CLINICAL_QUESTIONS = {
    "A": [
        ("sequential_decisions", "Are there repeated decision points whose choices affect later options?"),
        ("multiple_feasible_actions", "Are at least two clinically acceptable actions available at typical decision points?"),
        ("stochastic_response", "Is the patient response to treatment variable/uncertain?"),
        ("delayed_effects", "Do actions have delayed effects or long-term consequences?"),
        ("tradeoffs", "Is there a trade-off between immediate and long-term outcomes?"),
        ("reward_alignment", "Is the reward a clinically endorsed proxy of the treatment goal?"),
    ],
    "E": [
        ("constraints_definable", "Can hard clinical constraints (contraindications, dose limits) be specified?"),
        ("interpretability_required", "Are interpretability requirements for recommendations defined?"),
        ("fallback_policy", "Is there a defined fallback (e.g. standard of care) policy?"),
        ("ood_safeguards", "Are out-of-distribution safeguards planned?"),
        ("clinician_oversight", "Is clinician oversight of recommendations ensured?"),
    ],
}


class FeasibilityAssessment:
    """Run all diagnostics and map them onto the A-E criteria."""

    def __init__(self, config: AssessmentConfig):
        self.config = config
        self.th = config.thresholds

    # --------------------------------------------------------------- helpers
    def _kw(self) -> dict:
        c = self.config
        return {"model": c.model, "n_splits": c.n_splits, "seed": c.seed}

    @staticmethod
    def _safe(fn, name: str, *a, **kw) -> MetricResult:
        try:
            return fn(*a, **kw)
        except Exception as e:  # diagnostics must not crash the report
            return MetricResult.failed(name, f"{type(e).__name__}: {e}")

    def _clinical(self, level: str) -> list[CriterionResult]:
        out = []
        for key, question in CLINICAL_QUESTIONS[level]:
            ans = self.config.clinical.get(key)
            answer = ans.get("answer") if isinstance(ans, dict) else ans
            rationale = ans.get("rationale", "") if isinstance(ans, dict) else ""
            if answer is None:
                st, txt = Status.EXPERT_JUDGEMENT, "Not answered; requires clinical expert assessment."
            elif answer:
                st, txt = Status.NO_CONCERN_DETECTED, f"Affirmed by metadata. {rationale}".strip()
            else:
                st, txt = Status.SERIOUS_CONCERN if level == "A" else Status.CONCERN, f"Denied by metadata. {rationale}".strip()
            out.append(CriterionResult(f"{level}.meta.{key}", level, question, st, txt, evidence_type="metadata",
                                       next_step="Document the clinical rationale and its source." if answer is None else ""))
        return out

    # --------------------------------------------------------------- main
    def prepare(self, data: TrajectoryData) -> tuple[TrajectoryData, PreprocessingLog, object]:
        c = self.config
        log = PreprocessingLog()
        summary = None
        mdp = data
        if c.dt is not None:
            mdp, summary = resample_to_grid(mdp, c.dt, action_agg=c.action_agg, impute=c.impute, log=log)
        if c.action_bins and mdp.schema.action_type == "continuous":
            mdp = discretize_actions(mdp, c.action_bins, log=log)
        if c.terminal_reward is not None:
            mdp = add_terminal_outcome_reward(mdp, c.terminal_reward, log=log)
        return mdp, log, summary

    def evaluate(self, data: TrajectoryData) -> Report:
        c, th = self.config, self.th
        crit: list[CriterionResult] = []
        mdp, log, grid_summary = self.prepare(data)
        has_action = mdp.schema.action_col is not None

        # ---------------- Level C: temporal data quality on raw data
        miss = temporal.missingness(data)
        crit.append(CriterionResult("C.missingness", "C", "State completeness",
            threshold_status(miss.value, th.missingness_concern),
            f"{miss.value:.1%} of state values missing on decision rows.",
            "Consider a POMDP/belief-state formulation or explicit missingness indicators; never impute silently.", [miss]))
        si = temporal.sampling_intervals(data, th.gap_factor)
        gap_share = si.details.get("share_trajectories_with_gap", np.nan) if si.ok else np.nan
        crit.append(CriterionResult("C.sampling", "C", "Sampling regularity and fragmentation",
            threshold_status(gap_share, th.gap_trajectory_share_concern),
            f"Interval CV = {si.value:.2f}; {gap_share:.1%} of trajectories contain gaps > {th.gap_factor:g}x median interval." if si.ok else str(si.error),
            "Consider event-driven decision epochs (semi-MDP) or splitting trajectories at gaps.", [si]))
        heap = temporal.timestamp_heaping(data)
        crit.append(CriterionResult("C.timestamps", "C", "Timestamp precision (heaping)", Status.NO_CONCERN_DETECTED if heap.value < 0.5 else Status.CONCERN,
            f"{heap.value:.1%} of timestamps on multiples of the largest unit; interpret against the dataset's native grid.",
            "Do not choose a decision interval finer than the effective timestamp resolution.", [heap]))
        age = temporal.observation_age(data)
        crit.append(CriterionResult("C.alignment", "C", "State freshness at decision times",
            Status.NO_CONCERN_DETECTED if not miss.value else Status.INCONCLUSIVE,
            f"90th percentile of observation age: {age.value:.3g} time units.",
            "Compare observation age with the decision interval; stale states blur state-action ordering.",
            [age, temporal.action_without_fresh_observation(data)] if has_action else [age]))
        im_out = self._safe(temporal.informative_missingness_outcome, "informative_missingness_outcome_auc", data, seed=c.seed)
        im_metrics = [im_out]
        if has_action and data.schema.action_type == "discrete":
            im_metrics.append(self._safe(temporal.informative_missingness_action, "informative_missingness_action_gain", data, model=c.model, seed=c.seed, n_boot=c.n_boot))
        if im_out.ok:
            st = Status.CONCERN if (im_out.lower > 0.5 and im_out.value >= th.informative_auc_concern) else Status.NO_CONCERN_DETECTED
            txt = f"Outcome AUC from measurement patterns = {im_out.value:.2f} [{im_out.lower:.2f}, {im_out.upper:.2f}]."
        else:
            st, txt = Status.NOT_ASSESSABLE, f"Not assessable ({im_out.error})."
        crit.append(CriterionResult("C.informative_missingness", "C", "Informative observation process", st, txt,
            "If informative, include measurement indicators/times in the state and treat missingness as MNAR in sensitivity analyses.", im_metrics))
        tl = temporal.trajectory_length(data)
        crit.append(CriterionResult("C.length", "C", "Trajectory length", threshold_status(tl.details["share_shorter_than_min"], 0.25),
            f"Median {tl.value:.0f} rows; {tl.details['share_shorter_than_min']:.1%} shorter than {tl.details['min_length']}.", "", [tl]))
        if grid_summary is not None:
            ws = temporal.within_step_action_changes(grid_summary)
            crit.append(CriterionResult("B.decision_interval", "B", "Decision interval vs. decision frequency",
                threshold_status(ws.value, th.within_step_change_concern) if ws.ok else Status.NOT_ASSESSABLE,
                f"{ws.value:.1%} of grid steps aggregate more than one distinct raw action (dt = {c.dt:g})." if ws.ok else str(ws.error),
                "Use a finer grid or event-driven epochs if many decisions are aggregated; compare diagnostics across dt.", [ws]))

        if not has_action:
            crit.append(CriterionResult("A.actions", "A", "Actions recorded", Status.SERIOUS_CONCERN,
                "No action variable available: the decision process cannot be modelled as an MDP from these data.",
                "Identify a source of time-stamped treatment data before any MDP/RL modelling.", evidence_type="data"))
            crit += self._clinical("A") + self._clinical("E")
            return Report(crit, c, data, mdp, log)

        trans0 = mdp.transitions(lags=0)
        transL = mdp.transitions(lags=c.lags)
        kw = self._kw()

        # ---------------- Level A: data proxies
        levels = mdp.action_levels()
        if levels is not None:
            freq = mdp.df[mdp.schema.action_col].value_counts(normalize=True)
            n_used = int((freq >= 0.01).sum())
            changes = mdp.df.groupby(mdp.schema.id_col)[mdp.schema.action_col].apply(lambda a: a.dropna().nunique() > 1).mean()
            st = Status.SERIOUS_CONCERN if n_used < 2 else (Status.CONCERN if changes < 0.1 else Status.NO_CONCERN_DETECTED)
            crit.append(CriterionResult("A.action_multiplicity", "A", "Observed action multiplicity", st,
                f"{n_used} action levels used in >= 1% of decisions; {changes:.1%} of trajectories change action at least once.",
                "If treatment rarely changes within a trajectory, a single-step (bandit/causal) formulation may suffice.",
                [MetricResult("n_actions_used", n_used, unit="levels"), MetricResult("share_trajectories_with_action_change", float(changes))]))
        ae = self._safe(markov.action_effect_gain, "action_effect_gain", trans0, n_boot=c.n_boot, **kw)
        st, txt = classify_gain(ae, None, th.min_gain_nats, c.alpha)
        crit.append(CriterionResult("B.action_alignment", "B", "Recorded actions affect the next state (alignment)",
            Status.NO_CONCERN_DETECTED if st == Status.CONCERN else Status.CONCERN if st == Status.NO_CONCERN_DETECTED else Status.INCONCLUSIVE,
            ("Recorded actions are informative about the next state: " if st == Status.CONCERN else "Recorded actions carry little information about the next state: ") + txt,
            "If actions should act within one interval, check timestamp alignment of treatments vs. observations and the decision interval.", [ae]))
        dl = self._safe(markov.delayed_action_effect_gain, "delayed_action_effect_gain", transL, n_boot=c.n_boot, **kw)
        st, txt = classify_gain(dl, None, th.min_gain_nats, c.alpha)
        crit.append(CriterionResult("A.delayed_effects", "A", "Delayed treatment effects (data proxy)",
            Status.NO_CONCERN_DETECTED if st == Status.CONCERN else Status.INCONCLUSIVE,
            ("Past actions improve next-state prediction: " if st == Status.CONCERN else "No clear delayed effect of past actions: ") + txt,
            "Delayed effects support a sequential formulation but require them to be represented in the state (e.g. cumulative dose).", [dl]))

        # ---------------- Level B: formulation validity
        g_state = self._safe(markov.history_gain, "history_gain[state]", transL, "state", n_boot=c.n_boot, **kw)
        t_state = self._safe(markov.history_gcm, "history_gcm[state]", transL, "state", **kw)
        prof = [self._safe(markov.history_gain, f"history_gain[state,k={k}]", transL, "state", lags=k, n_boot=0, **kw) for k in range(1, c.lags + 1)]
        st, txt = classify_gain(g_state, t_state, th.min_gain_nats, c.alpha)
        crit.append(CriterionResult("B.state_sufficiency", "B", "State sufficiency (Markov property w.r.t. recorded history)", st, txt,
            gain_next_step(st,
                concern="Augment the state with the lags/aggregates that carry the gain (see lag profile), or use a belief state / recurrent representation; partial observability is likely.",
                inconclusive="Do not assume the state is sufficient: relevant history dependence is not excluded. Inspect the lag profile, "
                             "compare with an augmented formulation (state + lags) and, if possible, use more trajectories; a rejecting GCM test "
                             "with an uncertain effect size is a possible concern.",
                no_concern="Report as 'no relevant observable history dependence'; this does not exclude latent confounders."),
            [g_state, t_state, *prof]))
        if transL.reward is not None:
            g_r = self._safe(markov.history_gain, "history_gain[reward]", transL, "reward", n_boot=c.n_boot, **kw)
            t_r = self._safe(markov.history_gcm, "history_gcm[reward]", transL, "reward", **kw)
            st, txt = classify_gain(g_r, t_r, th.min_gain_nats, c.alpha)
            crit.append(CriterionResult("B.reward_sufficiency", "B", "Reward depends only on (s, a, s')", st, txt,
                gain_next_step(st,
                    concern="Include the history the reward depends on in the state or redefine the reward.",
                    inconclusive="Do not assume reward sufficiency yet: compare with the reward on an augmented state (state + lags) "
                                 "or use more trajectories."), [g_r, t_r]))
        else:
            crit.append(CriterionResult("B.reward_sufficiency", "B", "Reward depends only on (s, a, s')", Status.NOT_ASSESSABLE,
                "No reward column provided.", "Specify a reward to assess reward sufficiency.", evidence_type="none"))
        g_t = self._safe(markov.time_inhomogeneity_gain, "time_inhomogeneity_gain", trans0, n_boot=c.n_boot, **kw)
        st, txt = classify_gain(g_t, None, th.min_gain_nats, c.alpha)
        crit.append(CriterionResult("B.stationarity", "B", "Time-homogeneous dynamics", st, txt,
            gain_next_step(st,
                concern="Add time/phase of treatment to the state or model a finite-horizon, time-indexed MDP.",
                inconclusive="Do not assume time-homogeneity yet: compare with a time-indexed formulation (state + time/phase) "
                             "or use more trajectories."), [g_t]))
        stab = self._safe(coverage.transition_model_stability, "transition_epistemic_ratio", trans0, seed=c.seed)
        r2 = g_state.details.get("oof_r2_base") if g_state.ok else None
        crit.append(CriterionResult("B.transition_estimability", "B", "Transition dynamics estimable and stochastic",
            threshold_status(stab.value, th.epistemic_ratio_concern) if stab.ok else Status.INCONCLUSIVE,
            (f"Epistemic/aleatoric ratio = {stab.value:.3f}." if stab.ok else str(stab.error)) +
            (f" Out-of-fold R^2 of next state: {np.round(r2, 3).tolist()} (values near 1 = nearly deterministic)." if r2 else ""),
            "Coarsen the state/action space or collect more trajectories if dynamics are poorly determined.", [stab]))

        # ---------------- Level D: offline RL feasibility
        if levels is None:
            crit.append(CriterionResult("D.overlap", "D", "Action overlap / positivity", Status.NOT_ASSESSABLE,
                "Continuous actions: set `action_bins` to assess overlap.", "Discretise actions or use a conditional density model.", evidence_type="none"))
        else:
            # Behaviour-history dependence first: if clinicians act on history that the
            # state omits, overlap measured on the state alone is over-optimistic, so
            # overlap and IS diagnostics are repeated with a history-aware propensity model.
            g_a = self._safe(markov.history_gain, "history_gain[action]", transL, "action", n_boot=c.n_boot, **kw)
            st_a, txt_a = classify_gain(g_a, None, th.min_gain_nats, c.alpha)
            history_aware = st_a == Status.CONCERN
            bp = overlap.fit_behaviour_policy(trans0, model=c.model, n_splits=c.n_splits, seed=c.seed)
            svr = overlap.support_violation_rate(trans0, bp, th.positivity_epsilon, n_boot=c.n_boot, seed=c.seed)
            ent = overlap.action_entropy(trans0, bp, n_boot=c.n_boot, seed=c.seed)
            prop = overlap.observed_action_propensity(trans0, bp)
            aoi = overlap.action_overlap_index(trans0, bp)
            knn = self._safe(overlap.knn_local_action_diversity, "knn_locally_deterministic_share", trans0, seed=c.seed)
            cal = overlap.behaviour_calibration(trans0, bp)
            ov_metrics = [svr, ent, prop, cal, knn, aoi]
            st_ov = threshold_status(svr.value, th.support_violation_concern, th.support_violation_serious)
            txt_ov = (f"{svr.value:.1%} [{svr.lower:.1%}, {svr.upper:.1%}] of decisions have some action with pi_b < {th.positivity_epsilon:g}; "
                      f"mean normalised action entropy {ent.value:.2f}.")
            if cal.value > 0.05:  # convention
                txt_ov += f" Behaviour model calibration is poor (ECE = {cal.value:.3f}); propensity-based results are less reliable."
                st_ov = Status.INCONCLUSIVE if st_ov == Status.NO_CONCERN_DETECTED else st_ov
            if history_aware:
                bp_h = overlap.fit_behaviour_policy(transL, features=np.hstack([transL.state, transL.history]), model=c.model, n_splits=c.n_splits, seed=c.seed)
                svr_h = overlap.support_violation_rate(transL, bp_h, th.positivity_epsilon, n_boot=c.n_boot, seed=c.seed)
                ent_h = overlap.action_entropy(transL, bp_h, n_boot=c.n_boot, seed=c.seed)
                for m in (svr_h, ent_h):
                    m.name += "[history-aware]"
                ov_metrics += [svr_h, ent_h]
                st_ov = worst(st_ov, threshold_status(svr_h.value, th.support_violation_concern, th.support_violation_serious))
                txt_ov += (f" With a history-aware behaviour model (state + {c.lags} lags): {svr_h.value:.1%} of decisions violate support, "
                           f"entropy {ent_h.value:.2f}; the state-only figures are over-optimistic if these differ.")
            crit.append(CriterionResult("D.overlap", "D", "Action overlap / positivity", st_ov, txt_ov,
                "Restrict candidate policies to supported actions, coarsen actions, or use pessimistic/constrained offline RL.", ov_metrics))
            occ = coverage.occupancy_weighted_support(trans0, seed=c.seed)
            es = coverage.esac(trans0, seed=c.seed)
            crit.append(CriterionResult("D.coverage", "D", "State-action coverage",
                threshold_status(occ.details["share_decisions_in_unsupported_cell"], 0.1, 0.3),
                f"Occupancy-weighted share of supported actions = {occ.value:.2f}; "
                f"{occ.details['share_decisions_in_unsupported_cell']:.1%} of decisions lie in under-supported (cluster, action) cells.",
                "Check sensitivity to n_clusters/n_min; consider coarser state abstraction.", [occ, es]))

            def ope_status(trans, bpol, label):
                curve = ope_reliability.ess_deviation_curve(trans, bpol, returns=ope_reliability.trajectory_returns(trans, c.gamma), seed=c.seed)
                for m in curve:
                    m.name += label
                ref = min(curve, key=lambda m: abs(float(m.details["target"].split("_")[-1]) - th.reference_deviation))
                k = ref.details["pareto_k"]
                if k is not None and np.isfinite(k) and k > th.pareto_k_unreliable:
                    st = Status.SERIOUS_CONCERN
                else:
                    st = threshold_status(ref.value, th.ess_share_concern, higher_is_worse=False)
                return st, curve, f"ESS share = {ref.value:.3f}, Pareto-k = {k:.2f}"

            st_ope, curve, txt_ope = ope_status(trans0, bp, "")
            txt_ope = f"For a reference policy deviating by lambda={th.reference_deviation:g} from practice: {txt_ope}."
            if history_aware:
                st_h, curve_h, txt_h = ope_status(transL, bp_h, "[history-aware]")
                st_ope, curve = worst(st_ope, st_h), curve + curve_h
                txt_ope += f" History-aware behaviour model: {txt_h}."
            crit.append(CriterionResult("D.ope_reliability", "D", "Importance-sampling OPE reliability (pre-training)", st_ope, txt_ope,
                "Prefer model-based/FQE estimates with uncertainty, restrict deviation from practice, or shorten horizons.", curve))
            crit.append(CriterionResult("D.behaviour_history", "D", "Behaviour policy depends on unrecorded/historical information (symptom)",
                Status.CONCERN if history_aware else st_a,
                txt_a + " A positive gain means clinicians use information beyond the current state, a symptom (not proof) of potential confounding.",
                "Include the predictive history in the state used for propensity models; plan confounding sensitivity analyses.", [g_a]))
        crit.append(CriterionResult("D.unmeasured_confounding", "D", "No unmeasured confounding (sequential exchangeability)", Status.NOT_ASSESSABLE,
            "Not testable from observational data alone.",
            "Use clinical knowledge to list plausible unrecorded confounders; run sensitivity analyses (e.g. marginal sensitivity models) and negative-control checks.",
            evidence_type="none"))

        crit += self._clinical("A") + self._clinical("E")
        return Report(crit, c, data, mdp, log)
