import json

import numpy as np

from mdp_diagnostics import AssessmentConfig, FeasibilityAssessment, Status, TrajectoryData, TrajectorySchema
from tests.conftest import SCHEMA, make_ar_data


def _assess(df, **kw):
    cfg = AssessmentConfig(schema=SCHEMA, lags=2, n_boot=50, **kw)
    return FeasibilityAssessment(cfg).evaluate(TrajectoryData(df, TrajectorySchema(**SCHEMA)))


def test_markov_vs_second_order_status():
    assert _assess(make_ar_data()).by_id("B.state_sufficiency").status == Status.NO_CONCERN_DETECTED
    assert _assess(make_ar_data(phi2=0.35)).by_id("B.state_sufficiency").status == Status.CONCERN


def test_confounding_never_claimed_resolved():
    rep = _assess(make_ar_data())
    assert rep.by_id("D.unmeasured_confounding").status == Status.NOT_ASSESSABLE
    assert all(c.status.value != "safe" for c in rep.criteria)


def test_no_action_column():
    df = make_ar_data().drop(columns=["action"])
    schema = {k: v for k, v in SCHEMA.items() if k != "action_col"}
    cfg = AssessmentConfig(schema=schema, n_boot=20)
    rep = FeasibilityAssessment(cfg).evaluate(TrajectoryData(df, TrajectorySchema(**schema)))
    assert rep.by_id("A.actions").status == Status.SERIOUS_CONCERN


def test_reports_serialise(tmp_path):
    rep = _assess(make_ar_data(n_traj=60))
    rep.save_json(tmp_path / "r.json")
    rep.save_html(tmp_path / "r.html")
    d = json.loads((tmp_path / "r.json").read_text())
    assert d["disclaimer"] and len(d["criteria"]) > 10
    assert "not establish causal" in (tmp_path / "r.html").read_text()


def test_reproducible_with_seed():
    a = _assess(make_ar_data(n_traj=80)).to_dict()["criteria"]
    b = _assess(make_ar_data(n_traj=80)).to_dict()["criteria"]
    va = [m["value"] for c in a for m in c["metrics"]]
    vb = [m["value"] for c in b for m in c["metrics"]]
    assert va == vb


def test_clinical_metadata():
    rep = _assess(make_ar_data(n_traj=60), clinical={"sequential_decisions": {"answer": True, "rationale": "titration"},
                                                       "fallback_policy": False})
    assert rep.by_id("A.meta.sequential_decisions").status == Status.NO_CONCERN_DETECTED
    assert rep.by_id("E.meta.fallback_policy").status == Status.CONCERN
    assert rep.by_id("A.meta.tradeoffs").status == Status.EXPERT_JUDGEMENT


def test_history_aware_overlap_downgrades_state_only_optimism():
    # Clinicians keep the previous dose most of the time: given the state alone the
    # action looks random (good overlap), given history it is nearly deterministic.
    rng = np.random.default_rng(0)
    rows = []
    for i in range(150):
        a, s = int(rng.integers(0, 3)), rng.normal()
        for t in range(25):
            if rng.random() < 0.1:
                a = int(rng.integers(0, 3))
            rows.append((i, float(t), s, float(a), -s**2))
            s = 0.7 * s + 0.2 * (a - 1) + 0.5 * rng.normal()
        rows.append((i, 25.0, s, np.nan, np.nan))
    import pandas as pd
    df = pd.DataFrame(rows, columns=["id", "time", "s", "action", "reward"])
    rep = _assess(df)
    assert rep.by_id("D.behaviour_history").status == Status.CONCERN
    metrics = {m.name: m.value for m in rep.by_id("D.overlap").metrics}
    # Switching keeps ~3% probability per alternative action, so positivity at
    # epsilon = 0.01 holds, but the history-aware policy is far less diverse ...
    assert metrics["normalised_action_entropy[history-aware]"] < metrics["normalised_action_entropy"] - 0.3
    # ... and importance sampling becomes unreliable.
    assert rep.by_id("D.ope_reliability").status == Status.SERIOUS_CONCERN
