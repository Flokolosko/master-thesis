"""Criterion results and decision rules of the hierarchical framework.

Statuses are deliberately conservative. The framework never returns "safe" or
"identified"; the strongest positive statement is NO_CONCERN_DETECTED, which
means that the data provided no evidence of a problem *with the given power and
model class*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from mdp_diagnostics.metrics.base import MetricResult, _clean


class Status(str, Enum):
    NO_CONCERN_DETECTED = "no_concern_detected"
    CONCERN = "concern"
    SERIOUS_CONCERN = "serious_concern"
    INCONCLUSIVE = "inconclusive"
    NOT_ASSESSABLE = "not_assessable"
    EXPERT_JUDGEMENT = "expert_judgement_required"


LEVELS = {
    "A": "Clinical problem suitability",
    "B": "MDP formulation validity",
    "C": "Dataset suitability (temporal data quality)",
    "D": "Offline RL feasibility",
    "E": "Safety and deployment readiness (qualitative)",
}

QUESTIONS = {
    "A": "Question A: Is the problem/formulation appropriate for MDP modelling?",
    "B": "Question A: Is the problem/formulation appropriate for MDP modelling?",
    "C": "Question B: Does the dataset support offline RL?",
    "D": "Question B: Does the dataset support offline RL?",
    "E": "Deployment prerequisites (outside the scope of data diagnostics)",
}


@dataclass
class CriterionResult:
    id: str
    level: str
    title: str
    status: Status
    interpretation: str
    next_step: str = ""
    metrics: list[MetricResult] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    evidence_type: str = "data"  # "data", "metadata" or "none"

    @property
    def question(self) -> str:
        return QUESTIONS[self.level]

    def to_dict(self) -> dict:
        return _clean({
            "id": self.id, "level": self.level, "level_name": LEVELS[self.level],
            "question": self.question, "title": self.title, "status": self.status.value,
            "interpretation": self.interpretation, "next_step": self.next_step,
            "evidence_type": self.evidence_type,
            "assumptions": sorted(set(self.assumptions + [a for m in self.metrics for a in m.assumptions])),
            "metrics": [m.to_dict() for m in self.metrics],
        })


def classify_gain(gain: MetricResult, test: MetricResult | None, min_gain: float, alpha: float) -> tuple[Status, str]:
    """Equivalence-style rule for information-gain diagnostics.

    * relevant dependence: test rejects and the CI lower bound >= ``min_gain``
    * absence of *relevant* dependence: CI upper bound < ``min_gain``
      (regardless of statistical significance of tiny effects)
    * otherwise inconclusive or a (milder) concern.
    """
    if not gain.ok:
        return Status.INCONCLUSIVE, f"Could not be estimated ({gain.error})."
    rejected = test is not None and test.ok and test.p_value is not None and test.p_value < alpha
    up = gain.upper if np.isfinite(gain.upper) else gain.value
    lo = gain.lower if np.isfinite(gain.lower) else gain.value
    txt = f"gain = {gain.value:.4f} nats [{lo:.4f}, {up:.4f}]"
    if test is not None and test.ok:
        txt += f", GCM p = {test.p_value:.3g}"
    if up < min_gain:
        extra = " Dependence is statistically detectable but below the relevance threshold." if rejected else ""
        return Status.NO_CONCERN_DETECTED, f"No relevant dependence detected ({txt}; threshold {min_gain:g}).{extra}"
    if lo >= min_gain and (rejected or test is None):
        return Status.CONCERN, f"Relevant dependence detected ({txt}; threshold {min_gain:g})."
    if rejected and gain.value >= min_gain:
        return Status.CONCERN, f"Dependence detected; effect size uncertain ({txt})."
    return Status.INCONCLUSIVE, f"Relevant dependence can neither be confirmed nor excluded ({txt}; threshold {min_gain:g})."


def threshold_status(value: float, concern: float, serious: float | None = None, higher_is_worse: bool = True) -> Status:
    if value is None or not np.isfinite(value):
        return Status.INCONCLUSIVE
    worse = (lambda v, t: v >= t) if higher_is_worse else (lambda v, t: v <= t)
    if serious is not None and worse(value, serious):
        return Status.SERIOUS_CONCERN
    if worse(value, concern):
        return Status.CONCERN
    return Status.NO_CONCERN_DETECTED


_SEVERITY = [Status.NO_CONCERN_DETECTED, Status.INCONCLUSIVE, Status.CONCERN, Status.SERIOUS_CONCERN]


def worst(*statuses: Status) -> Status:
    """Most severe of several data-based statuses."""
    ranked = [s for s in statuses if s in _SEVERITY]
    return max(ranked, key=_SEVERITY.index) if ranked else statuses[0]
