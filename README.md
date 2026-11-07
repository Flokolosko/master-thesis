# MDP feasibility diagnostics for healthcare (Master's thesis, TU Wien)

Research code for *A Quantitative Framework for Assessing the Empirical Adequacy of Markov Decision
Process Models in Healthcare* (proposal: `proposal/proposal.tex`).

The package `mdp_diagnostics` assesses, **before any RL training**, whether observational clinical
trajectory data support

- **Question A** — an MDP formulation (state sufficiency, reward sufficiency, decision interval, action alignment, stationarity, transition estimability), and
- **Question B** — offline RL / off-policy evaluation (temporal data quality, overlap, coverage, importance-sampling reliability; confounding flagged as not assessable),

and reports each criterion with a value, uncertainty, assumptions, a conservative status and a next step.
It never produces an overall score and never claims causal identifiability or safety.

## Quick start
```bash
uv sync
uv run pytest
```
```python
from mdp_diagnostics import AssessmentConfig, FeasibilityAssessment, TrajectoryData, TrajectorySchema

schema = dict(id_col="id", time_col="time", state_cols=["x0", "x1"], action_col="action", reward_col="reward")
data = TrajectoryData(df, TrajectorySchema(**schema))
report = FeasibilityAssessment(AssessmentConfig(schema=schema, lags=3)).evaluate(data)
report.save_json("feasibility.json"); report.save_html("feasibility.html")
print(report.summary())
```
Configs can be loaded with `AssessmentConfig.from_yaml(...)`; see `configs/case_eicu_sepsis.yaml`.

## Layout
| Path | Content |
|---|---|
| `src/mdp_diagnostics/data` | schema, validation, explicit preprocessing (grid, discretisation, terminal reward) |
| `src/mdp_diagnostics/metrics` | Markov/information-gain + GCM, overlap, coverage, IS reliability, temporal indicators |
| `src/mdp_diagnostics/framework` | criteria A–E, assessment logic, JSON/HTML reports |
| `src/mdp_diagnostics/experiments` | synthetic environments (latent linear, PK/PD propofol, gene/medication), perturbations, offline RL agents (FQI) for downstream checks, runners |
| `configs/` | experiment and case-study configurations |
| `examples/case_study_icu.py` | eICU sepsis case study (data not included) |
| `research/` | literature inventory, metric comparison, framework spec, validation plan, gap, open questions, results |
| `results/` | experiment outputs (aggregates only) |

## Reproducing experiments
```bash
uv run python -m mdp_diagnostics.experiments.run all          # full grid (slow)
uv run python -m mdp_diagnostics.experiments.run state --quick
uv run python examples/case_study_icu.py                        # needs PhysioNet data, see config
```
