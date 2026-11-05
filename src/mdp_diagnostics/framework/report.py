"""Machine- and human-readable feasibility reports."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Template

from mdp_diagnostics.framework.criteria import LEVELS, CriterionResult
from mdp_diagnostics.metrics.base import _clean

if TYPE_CHECKING:
    from mdp_diagnostics.data.preprocessing import PreprocessingLog
    from mdp_diagnostics.data.schema import TrajectoryData

DISCLAIMER = (
    "This report summarises statistical diagnostics of observational data. It does not establish causal "
    "identifiability, clinical validity or safety of any model or policy. 'No concern detected' means only that "
    "the diagnostics found no evidence of a problem with the given data, model class and power."
)

_HTML = Template("""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>MDP feasibility report</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px auto;max-width:1100px;padding:0 16px;color:#1d2433;background:#fff}
h1{font-size:1.5rem}h2{font-size:1.2rem;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:4px}
table{border-collapse:collapse;width:100%;font-size:.9rem}td,th{border-bottom:1px solid #eee;padding:6px;vertical-align:top;text-align:left}
.s{font-weight:600;padding:2px 6px;border-radius:4px;white-space:nowrap}
.no_concern_detected{background:#e3f4e8}.concern{background:#fff1cc}.serious_concern{background:#fbd5d5}
.inconclusive{background:#e8e8f4}.not_assessable{background:#eee}.expert_judgement_required{background:#e0eefc}
.disc{background:#f6f6f6;padding:10px;border-left:4px solid #999;font-size:.9rem}details{font-size:.85rem}
pre{white-space:pre-wrap}
</style></head><body>
<h1>MDP / offline-RL feasibility report</h1>
<p class="disc">{{ disclaimer }}</p>
<p>Generated {{ d.generated }} &middot; {{ d.data.n_trajectories }} trajectories, {{ d.data.n_rows }} rows &middot;
status counts: {% for k, v in d.status_counts.items() %}<span class="s {{k}}">{{k}}: {{v}}</span> {% endfor %}</p>
{% for level, name in levels.items() %}{% set rows = d.criteria | selectattr("level", "equalto", level) | list %}{% if rows %}
<h2>Level {{ level }} &mdash; {{ name }}</h2><p><em>{{ rows[0].question }}</em></p>
<table><tr><th>Criterion</th><th>Status</th><th>Interpretation</th><th>Next step</th></tr>
{% for c in rows %}<tr><td>{{ c.title }}<br><small>{{ c.id }} &middot; evidence: {{ c.evidence_type }}</small></td>
<td><span class="s {{ c.status }}">{{ c.status }}</span></td><td>{{ c.interpretation }}
{% if c.metrics %}<details><summary>metrics</summary><pre>{% for m in c.metrics %}{{ m.name }}: {{ m.value }} {{ m.unit }}{% if m.lower is not none %} [{{ m.lower }}, {{ m.upper }}]{% endif %}{% if m.p_value is not none %} p={{ m.p_value }}{% endif %}{% if m.error %} ERROR: {{ m.error }}{% endif %}
{% endfor %}</pre></details>{% endif %}
{% if c.assumptions %}<details><summary>assumptions</summary><ul>{% for a in c.assumptions %}<li>{{ a }}</li>{% endfor %}</ul></details>{% endif %}
</td><td>{{ c.next_step }}</td></tr>{% endfor %}</table>{% endif %}{% endfor %}
<h2>Data and preprocessing</h2><pre>{{ d.preprocessing | tojson(indent=2) }}</pre>
<h2>Validation issues</h2><ul>{% for i in d.validation_issues %}<li>{{ i.severity }}: {{ i.message }}</li>{% endfor %}</ul>
</body></html>""")


class Report:
    def __init__(self, criteria: list[CriterionResult], config, raw: "TrajectoryData", mdp: "TrajectoryData", log: "PreprocessingLog"):
        self.criteria = criteria
        self.config = config
        self.raw = raw
        self.mdp = mdp
        self.log = log

    def by_id(self, cid: str) -> CriterionResult:
        return next(c for c in self.criteria if c.id == cid)

    def to_dict(self) -> dict:
        crit = [c.to_dict() for c in self.criteria]
        return _clean({
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "disclaimer": DISCLAIMER,
            "data": {"n_trajectories": self.raw.n_trajectories, "n_rows": len(self.raw.df),
                     "schema": self.raw.schema.__dict__, "mdp_schema": self.mdp.schema.__dict__},
            "config": self.config.to_dict(),
            "preprocessing": self.log.steps,
            "validation_issues": [i.to_dict() for i in self.raw.issues],
            "status_counts": dict(Counter(c["status"] for c in crit)),
            "criteria": crit,
        })

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def save_html(self, path: str | Path) -> None:
        Path(path).write_text(_HTML.render(d=self.to_dict(), levels=LEVELS, disclaimer=DISCLAIMER))

    def summary(self) -> str:
        lines = [DISCLAIMER, ""]
        for level, name in LEVELS.items():
            rows = [c for c in self.criteria if c.level == level]
            if not rows:
                continue
            lines.append(f"Level {level} - {name}")
            for c in rows:
                lines.append(f"  [{c.status.value:>25}] {c.title}: {c.interpretation}")
        return "\n".join(lines)
