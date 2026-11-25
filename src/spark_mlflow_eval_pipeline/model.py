"""Data model: evaluation items, scored items, and dataset loading.

An evaluation item is one prompt answered by one model. A dataset
file is one JSONL file per model run: every line is one item with a
prompt, an optional reference answer, a category, a slice label
(e.g. language or difficulty), the model's response and an optional
self-reported confidence in [0, 1].

The model is deliberately boring: plain dataclasses with ``to_dict``
methods, so a scored item can go into a JSON report, a Spark Row,
an MLflow artifact or a pandas DataFrame without translation code
in three places.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CATEGORIES = ("factual_qa", "maths", "instruction_following", "safety")

RUBRIC_VERSION = "rubric-v1"
DATASET_VERSION = "eval-items-v1"


class DatasetError(ValueError):
    """A dataset file is malformed or fails validation."""


@dataclass
class EvalItem:
    item_id: str
    prompt: str
    reference: str = ""
    category: str = "factual_qa"
    slice: str = "all"
    model: str = ""
    response: str = ""
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "prompt": self.prompt,
            "reference": self.reference,
            "category": self.category,
            "slice": self.slice,
            "model": self.model,
            "response": self.response,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str = "") -> "EvalItem":
        if not isinstance(data, dict):
            raise DatasetError(f"{source}: each line must be a JSON object")
        item_id = data.get("item_id")
        if not item_id:
            raise DatasetError(f"{source}: item is missing 'item_id'")
        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise DatasetError(f"{source}: item {item_id!r} is missing 'prompt'")
        category = data.get("category", "factual_qa")
        if category not in CATEGORIES:
            raise DatasetError(
                f"{source}: item {item_id!r} has unknown category {category!r}; "
                f"expected one of {', '.join(CATEGORIES)}"
            )
        confidence = data.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                raise DatasetError(
                    f"{source}: item {item_id!r} has non-numeric confidence"
                ) from None
            if not 0.0 <= confidence <= 1.0:
                raise DatasetError(
                    f"{source}: item {item_id!r} confidence {confidence} is not in [0, 1]"
                )
        return cls(
            item_id=str(item_id),
            prompt=prompt,
            reference=data.get("reference") or "",
            category=category,
            slice=str(data.get("slice") or "all"),
            model=str(data.get("model") or ""),
            response=data.get("response") or "",
            confidence=confidence,
        )


@dataclass
class ScoredItem:
    """One item after scoring, with per-criterion rubric scores."""

    item: EvalItem
    scores: dict[str, int]  # criterion -> 1..5
    mean_score: float
    factual_accuracy: float  # supported claims / all claims, 0..1
    supported: int
    contradicted: int
    unverifiable: int
    passed: bool
    refused: bool

    def to_dict(self) -> dict[str, Any]:
        d = self.item.to_dict()
        d.update(
            {
                "scores": dict(self.scores),
                "mean_score": self.mean_score,
                "factual_accuracy": self.factual_accuracy,
                "supported": self.supported,
                "contradicted": self.contradicted,
                "unverifiable": self.unverifiable,
                "passed": self.passed,
                "refused": self.refused,
            }
        )
        return d


def load_jsonl(path: str | Path) -> list[EvalItem]:
    """Load a JSONL dataset file; blank lines are ignored."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"dataset file not found: {path}")
    items: list[EvalItem] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            item = EvalItem.from_dict(data, source=f"{path}:{line_no}")
            if item.item_id in seen:
                raise DatasetError(
                    f"{path}:{line_no}: duplicate item_id {item.item_id!r}"
                )
            seen.add(item.item_id)
            items.append(item)
    if not items:
        raise DatasetError(f"{path}: no evaluation items found")
    return items


def load_dataset(path: str | Path) -> list[EvalItem]:
    """Load a dataset from JSONL (or CSV with the same columns)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    return load_jsonl(path)


def _load_csv(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for line_no, row in enumerate(reader, start=2):
            if not any((v or "").strip() for v in row.values()):
                continue
            row = {k: (v if v != "" else None) for k, v in row.items()}
            if row.get("confidence") is not None:
                row["confidence"] = float(row["confidence"])
            item = EvalItem.from_dict(row, source=f"{path}:{line_no}")
            if item.item_id in seen:
                raise DatasetError(
                    f"{path}:{line_no}: duplicate item_id {item.item_id!r}"
                )
            seen.add(item.item_id)
            items.append(item)
    if not items:
        raise DatasetError(f"{path}: no evaluation items found")
    return items


def items_to_rows(items: list[ScoredItem]) -> list[dict[str, Any]]:
    """Flatten scored items into plain rows for CSV artifacts."""
    rows = []
    for scored in items:
        row = scored.to_dict()
        for criterion, value in scored.scores.items():
            row[f"score_{criterion}"] = value
        del row["scores"]
        rows.append(row)
    return rows
