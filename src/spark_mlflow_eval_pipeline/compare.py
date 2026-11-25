"""Compare two evaluation run reports.

``compare`` answers the question a model team asks after every
change: did the new model get better, and where did it regress?
Deltas are computed on the overall metrics and on every shared
category and slice, and each change is labelled an improvement, a
regression, or unchanged (a delta smaller than 0.005 is noise at
three-decimal precision and counts as unchanged).

Win rate and kappa are recomputed from the scored items in the two
reports, so the comparison shows not just "the mean went up" but
"the new model wins 62% of head-to-head items, and the two models
agree on pass/fail with kappa 0.41".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .aggregation import (
    RunReport,
    cohens_kappa,
    pairwise_win_rate,
)

_NOISE = 0.005

_OVERALL_METRICS = ("mean_score", "factual_accuracy", "pass_rate", "refusal_rate")


@dataclass
class MetricDelta:
    metric: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    verdict: str  # "improved" | "regressed" | "unchanged" | "n/a"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "delta": self.delta,
            "verdict": self.verdict,
        }


def _delta(
    metric: str,
    baseline: float | None,
    candidate: float | None,
    higher_is_better: bool = True,
) -> MetricDelta:
    if baseline is None or candidate is None:
        return MetricDelta(metric, baseline, candidate, None, "n/a")
    change = round(candidate - baseline, 3)
    if abs(change) < _NOISE:
        verdict = "unchanged"
    elif (change > 0) == higher_is_better:
        verdict = "improved"
    else:
        verdict = "regressed"
    return MetricDelta(metric, baseline, candidate, change, verdict)


@dataclass
class Comparison:
    baseline_model: str
    candidate_model: str
    overall: list[MetricDelta] = field(default_factory=list)
    by_category: dict[str, list[MetricDelta]] = field(default_factory=dict)
    by_slice: dict[str, list[MetricDelta]] = field(default_factory=dict)
    win_rate: dict[str, Any] = field(default_factory=dict)
    kappa: dict[str, Any] = field(default_factory=dict)

    @property
    def improvements(self) -> list[str]:
        return [
            f"overall {d.metric}" for d in self.overall if d.verdict == "improved"
        ] + [
            f"{cat} {d.metric}"
            for cat, deltas in self.by_category.items()
            for d in deltas
            if d.verdict == "improved"
        ]

    @property
    def regressions(self) -> list[str]:
        return [
            f"overall {d.metric}" for d in self.overall if d.verdict == "regressed"
        ] + [
            f"{cat} {d.metric}"
            for cat, deltas in self.by_category.items()
            for d in deltas
            if d.verdict == "regressed"
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_model": self.baseline_model,
            "candidate_model": self.candidate_model,
            "overall": [d.to_dict() for d in self.overall],
            "by_category": {
                k: [d.to_dict() for d in v] for k, v in self.by_category.items()
            },
            "by_slice": {
                k: [d.to_dict() for d in v] for k, v in self.by_slice.items()
            },
            "win_rate": self.win_rate,
            "kappa": self.kappa,
            "improvements": self.improvements,
            "regressions": self.regressions,
        }


def compare_reports(baseline: RunReport, candidate: RunReport) -> Comparison:
    """Compare a candidate run report against a baseline report."""
    overall = [
        _delta(
            metric,
            getattr(baseline.overall, metric),
            getattr(candidate.overall, metric),
            higher_is_better=(metric != "refusal_rate"),
        )
        for metric in _OVERALL_METRICS
    ]
    ece_delta = _delta("ece", baseline.overall.ece, candidate.overall.ece,
                       higher_is_better=False)
    overall.append(ece_delta)

    by_category: dict[str, list[MetricDelta]] = {}
    for category in sorted(set(baseline.by_category) | set(candidate.by_category)):
        base = baseline.by_category.get(category)
        cand = candidate.by_category.get(category)
        by_category[category] = [
            _delta(
                metric,
                getattr(base, metric) if base else None,
                getattr(cand, metric) if cand else None,
                higher_is_better=(metric != "refusal_rate"),
            )
            for metric in _OVERALL_METRICS
        ]

    by_slice: dict[str, list[MetricDelta]] = {}
    for slice_name in sorted(set(baseline.by_slice) | set(candidate.by_slice)):
        base = baseline.by_slice.get(slice_name)
        cand = candidate.by_slice.get(slice_name)
        by_slice[slice_name] = [
            _delta(
                metric,
                getattr(base, metric) if base else None,
                getattr(cand, metric) if cand else None,
                higher_is_better=(metric != "refusal_rate"),
            )
            for metric in _OVERALL_METRICS
        ]

    return Comparison(
        baseline_model=baseline.model,
        candidate_model=candidate.model,
        overall=overall,
        by_category=by_category,
        by_slice=by_slice,
        win_rate=pairwise_win_rate(baseline.scored_items, candidate.scored_items),
        kappa=cohens_kappa(baseline.scored_items, candidate.scored_items),
    )
