"""Apply the feasibility framework to the eICU sepsis cohort (and a boundary case).

Usage:  uv run python examples/case_study_icu.py [--config configs/case_eicu_sepsis.yaml]

Writes JSON/HTML reports and a formulation comparison table to results/case_study/.
Only aggregate statistics are written; no patient-level data leave the data directory.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import pandas as pd
import yaml

from mdp_diagnostics import AssessmentConfig, FeasibilityAssessment, Thresholds, TrajectoryData, TrajectorySchema

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "case_study"

# Headline metric per criterion for the comparison table.
KEY_METRICS = {
    "B.state_sufficiency": "history_gain[state]",
    "B.reward_sufficiency": "history_gain[reward]",
    "B.action_alignment": "action_effect_gain",
    "B.stationarity": "time_inhomogeneity_gain",
    "B.decision_interval": "within_step_action_changes",
    "B.transition_estimability": "transition_epistemic_ratio",
    "C.missingness": "state_missingness",
    "C.sampling": "sampling_interval_cv",
    "C.informative_missingness": "informative_missingness_outcome_auc",
    "D.overlap": "support_violation_rate",
    "D.coverage": "occupancy_weighted_support",
    "D.ope_reliability": "is_ess_share[mix_uniform_0.25]",
    "D.behaviour_history": "history_gain[action]",
}


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def make_config(d: dict) -> AssessmentConfig:
    d = copy.deepcopy(d)
    d["thresholds"] = Thresholds(**d.get("thresholds", {}))
    return AssessmentConfig(**d)


def run(name: str, df: pd.DataFrame, conf: dict) -> dict:
    cfg = make_config(conf)
    data = TrajectoryData(df, TrajectorySchema(**cfg.schema))
    rep = FeasibilityAssessment(cfg).evaluate(data)
    rep.save_json(OUT / f"{name}.json")
    rep.save_html(OUT / f"{name}.html")
    row = {"formulation": name}
    for c in rep.criteria:
        if c.id in KEY_METRICS:
            m = next((m for m in c.metrics if m.name == KEY_METRICS[c.id]), None)
            row[c.id] = f"{c.status.value}" + (f" ({m.value:.3g})" if m is not None and m.ok else "")
    print(f"--- {name}\n{rep.summary()}\n")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "case_eicu_sepsis.yaml"))
    args = ap.parse_args()
    conf = yaml.safe_load(Path(args.config).read_text())
    data_dir = Path(os.environ.get("MDP_DIAG_DATA_DIR", conf["data_dir"])).expanduser()
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_dir / conf["file"])
    rows = [run("sepsis_primary", df, conf["assessment"])]
    for name, upd in conf.get("variants", {}).items():
        rows.append(run(f"sepsis_{name}", df, deep_update(conf["assessment"], upd)))
    table = pd.DataFrame(rows).set_index("formulation").T
    (OUT / "formulation_comparison.md").write_text(table.to_markdown())

    bc = conf.get("boundary_case")
    if bc:
        cdf = pd.read_csv(data_dir / bc["file"])
        ids = cdf["HADM_ID"].drop_duplicates().sample(n=min(bc["max_trajectories"], cdf["HADM_ID"].nunique()), random_state=0)
        cdf = cdf[cdf["HADM_ID"].isin(ids)]
        run("cardiac_arrest_boundary", cdf, {"schema": bc["schema"], "n_boot": 100})


if __name__ == "__main__":
    main()
