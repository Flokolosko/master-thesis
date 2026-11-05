"""Common result type of all diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mdp_diagnostics.utils.statistics import Estimate


def _clean(obj):
    """Make nested results JSON-serialisable."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, Estimate):
        return _clean(obj.to_dict())
    return obj


@dataclass
class MetricResult:
    """Outcome of one diagnostic.

    ``value`` is the headline quantity with its unit; ``p_value`` is set for
    hypothesis tests; ``details`` holds secondary numbers. ``assumptions`` lists
    what must hold for the number to mean what its name says.
    """

    name: str
    value: float
    unit: str = ""
    lower: float = float("nan")
    upper: float = float("nan")
    p_value: float | None = None
    n_transitions: int | None = None
    n_trajectories: int | None = None
    details: dict = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    error: str | None = None

    @classmethod
    def from_estimate(cls, name: str, est: Estimate, **kw) -> "MetricResult":
        return cls(name=name, value=est.value, lower=est.lower, upper=est.upper, **kw)

    @classmethod
    def failed(cls, name: str, error: str) -> "MetricResult":
        return cls(name=name, value=float("nan"), error=error)

    @property
    def ok(self) -> bool:
        return self.error is None and np.isfinite(self.value)

    def to_dict(self) -> dict:
        return _clean(self.__dict__.copy())
