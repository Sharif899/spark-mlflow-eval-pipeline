"""Aggregation: from scored items to run-level metrics.

All aggregation is pure Python over ``ScoredItem`` lists, so the
local engine and the Spark engine share exactly the same numbers:
the Spark job scores items distributedly and returns per-item
scores; the driver aggregates them with the functions here.

Metrics reported per group (overall, per category, per slice):
mean rubric score, per-criterion means, factual accuracy, pass
rate, refusal rate, and — when items carry a confidence — Expected
Calibration Error (ECE) over 10 equal-width bins, treating
``passed`` as the correctness label.

Cross-model views: pairwise win rates (same item answered by two
models) and Cohen's kappa on pass/fail agreement between two
models. Kappa is the standard inter-rater agreement measure from
annotation QA: it says whether two models fail the *same* items,
which a bare accuracy gap never shows.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .model import ScoredItem
from .scoring import CRITERIA

_N_BINS = 10


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 3) if denominator else 0.0


@dataclass
class GroupMetrics:
    """Metrics for one group of scored items."""

    n: int
    mean_score: float
    criterion_means: dict[str, float]
    factual_accuracy: float
    pass_rate: float
    refusal_rate: float
    ece: float | None  # None when no item carries a confidence
    n_with_confidence: int

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_score": self.mean_score,
            "criterion_means": dict(self.criterion_means),
            "factual_accuracy": self.factual_accuracy,
            "pass_rate": self.pass_rate,
            "refusal_rate": self.refusal_rate,
            "ece": self.ece,
            "n_with_confidence": self.n_with_confidence,
        }


def ece(scored: list[ScoredItem], n_bins: int = _N_BINS) -> float | None:
    """Expected Calibration Error, using ``passed`` as correctness.

    Items without a confidence are excluded. Returns None when no
    item carries a confidence.
    """
    pairs = [
        (s.item.confidence, 1.0 if s.passed else 0.0)
        for s in scored
        if s.item.confidence is not None
    ]
    if not pairs:
        return None
    bins: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for conf, correct in pairs:
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, correct))
    total = len(pairs)
    error = 0.0
    for members in bins.values():
        mean_conf = sum(c for c, _ in members) / len(members)
        observed = sum(y for _, y in members) / len(members)
        error += (len(members) / total) * abs(mean_conf - observed)
    return round(error, 3)


def aggregate_group(scored: list[ScoredItem]) -> GroupMetrics:
    """Aggregate one group of scored items into GroupMetrics."""
    n = len(scored)
    criterion_means = {
        c: _mean([s.scores[c] for s in scored]) for c in CRITERIA
    }
    # Factual accuracy is measured only where a reference answer
    # exists: safety items have no reference, so their claims are
    # unverifiable by design and must not dilute the rate.
    checked = [s for s in scored if s.item.reference.strip()]
    total_claims = sum(
        s.supported + s.contradicted + s.unverifiable for s in checked
    )
    supported = sum(s.supported for s in checked)
    factual = round(supported / total_claims, 3) if total_claims else 0.0
    return GroupMetrics(
        n=n,
        mean_score=_mean([s.mean_score for s in scored]),
        criterion_means=criterion_means,
        factual_accuracy=factual,
        pass_rate=_rate(sum(1 for s in scored if s.passed), n),
        refusal_rate=_rate(sum(1 for s in scored if s.refused), n),
        ece=ece(scored),
        n_with_confidence=sum(1 for s in scored if s.item.confidence is not None),
    )


@dataclass
class RunReport:
    """The full result of one evaluation run."""

    model: str
    dataset_version: str
    rubric_version: str
    engine: str  # "local" | "spark"
    n_items: int
    overall: GroupMetrics
    by_category: dict[str, GroupMetrics] = field(default_factory=dict)
    by_slice: dict[str, GroupMetrics] = field(default_factory=dict)
    scored_items: list[ScoredItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "dataset_version": self.dataset_version,
            "rubric_version": self.rubric_version,
            "engine": self.engine,
            "n_items": self.n_items,
            "overall": self.overall.to_dict(),
            "by_category": {k: v.to_dict() for k, v in self.by_category.items()},
            "by_slice": {k: v.to_dict() for k, v in self.by_slice.items()},
            "scored_items": [s.to_dict() for s in self.scored_items],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunReport":
        from .model import EvalItem

        scored = []
        for raw in data.get("scored_items", []):
            item = EvalItem.from_dict(raw)
            scored.append(
                ScoredItem(
                    item=item,
                    scores=dict(raw["scores"]),
                    mean_score=raw["mean_score"],
                    factual_accuracy=raw["factual_accuracy"],
                    supported=raw["supported"],
                    contradicted=raw["contradicted"],
                    unverifiable=raw["unverifiable"],
                    passed=raw["passed"],
                    refused=raw["refused"],
                )
            )

        def _group(raw: dict) -> GroupMetrics:
            return GroupMetrics(
                n=raw["n"],
                mean_score=raw["mean_score"],
                criterion_means=dict(raw["criterion_means"]),
                factual_accuracy=raw["factual_accuracy"],
                pass_rate=raw["pass_rate"],
                refusal_rate=raw["refusal_rate"],
                ece=raw["ece"],
                n_with_confidence=raw["n_with_confidence"],
            )

        return cls(
            model=data["model"],
            dataset_version=data["dataset_version"],
            rubric_version=data["rubric_version"],
            engine=data.get("engine", "local"),
            n_items=data["n_items"],
            overall=_group(data["overall"]),
            by_category={k: _group(v) for k, v in data.get("by_category", {}).items()},
            by_slice={k: _group(v) for k, v in data.get("by_slice", {}).items()},
            scored_items=scored,
        )


def build_report(
    scored: list[ScoredItem],
    model: str,
    dataset_version: str,
    rubric_version: str,
    engine: str,
) -> RunReport:
    """Aggregate scored items into a RunReport, grouped by category and slice."""
    by_category: dict[str, list[ScoredItem]] = defaultdict(list)
    by_slice: dict[str, list[ScoredItem]] = defaultdict(list)
    for s in scored:
        by_category[s.item.category].append(s)
        by_slice[s.item.slice].append(s)
    return RunReport(
        model=model,
        dataset_version=dataset_version,
        rubric_version=rubric_version,
        engine=engine,
        n_items=len(scored),
        overall=aggregate_group(scored),
        by_category={k: aggregate_group(v) for k, v in sorted(by_category.items())},
        by_slice={k: aggregate_group(v) for k, v in sorted(by_slice.items())},
        scored_items=sorted(scored, key=lambda s: s.item.item_id),
    )


def pairwise_win_rate(
    scored_a: list[ScoredItem], scored_b: list[ScoredItem], tie_threshold: float = 0.3
) -> dict:
    """Win rate between two models over the items both answered.

    Items are matched on item_id. Returns wins for A, wins for B,
    ties, and the win rate of A (wins / non-tie pairs).
    """
    b_by_id = {s.item.item_id: s for s in scored_b}
    wins_a = wins_b = ties = 0
    for s_a in scored_a:
        s_b = b_by_id.get(s_a.item.item_id)
        if s_b is None:
            continue
        gap = s_a.mean_score - s_b.mean_score
        if gap > tie_threshold:
            wins_a += 1
        elif gap < -tie_threshold:
            wins_b += 1
        else:
            ties += 1
    total = wins_a + wins_b + ties
    decisive = wins_a + wins_b
    return {
        "n_compared": total,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_a": round(wins_a / decisive, 3) if decisive else 0.0,
    }


def cohens_kappa(
    scored_a: list[ScoredItem], scored_b: list[ScoredItem]
) -> dict:
    """Cohen's kappa on pass/fail agreement between two models.

    Returns observed and expected agreement, kappa, and the
    interpretation used in the sibling annotation work. Kappa of 1
    means the models pass and fail exactly the same items; 0 means
    agreement no better than chance.
    """
    b_by_id = {s.item.item_id: s for s in scored_b}
    pairs = [
        (s_a.passed, b_by_id[s_a.item.item_id].passed)
        for s_a in scored_a
        if s_a.item.item_id in b_by_id
    ]
    n = len(pairs)
    if n == 0:
        return {
            "n": 0,
            "observed_agreement": 0.0,
            "expected_agreement": 0.0,
            "kappa": 0.0,
            "interpretation": "no shared items",
        }
    observed = sum(1 for a, b in pairs if a == b) / n
    # Expected agreement from the marginal pass rates.
    p_a = sum(1 for a, _ in pairs if a) / n
    p_b = sum(1 for _, b in pairs if b) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected >= 1.0:
        kappa = 1.0 if observed >= 1.0 else 0.0
    else:
        kappa = round((observed - expected) / (1 - expected), 3)
    if kappa >= 0.8:
        interpretation = "almost perfect agreement"
    elif kappa >= 0.6:
        interpretation = "substantial agreement"
    elif kappa >= 0.4:
        interpretation = "moderate agreement"
    elif kappa >= 0.2:
        interpretation = "fair agreement"
    elif kappa > 0.0:
        interpretation = "slight agreement"
    else:
        interpretation = "no agreement beyond chance"
    return {
        "n": n,
        "observed_agreement": round(observed, 3),
        "expected_agreement": round(expected, 3),
        "kappa": kappa,
        "interpretation": interpretation,
    }
