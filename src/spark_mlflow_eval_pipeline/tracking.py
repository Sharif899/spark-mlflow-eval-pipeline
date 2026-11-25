"""MLflow layer: log an evaluation run where it can be compared.

``mlflow`` is an optional dependency. This module imports cleanly
when mlflow is absent: the import is guarded, and a clear
``MlflowUnavailable`` error is raised only when logging is actually
attempted. The payload-building logic — what params, metrics,
artifacts and tags a run produces — is pure Python in
:func:`build_tracking_payload` and is fully tested without any
mlflow client, so the tracked content is verified even where
mlflow cannot run.

Logged per run:
- params: model, dataset version, rubric version, engine, item count
- metrics: overall mean score, pass rate, factual accuracy,
  refusal rate, ECE (when confidences exist), and per-category mean
  scores as ``category_<name>_mean``
- tags: engine, dataset version
- artifacts: the JSON report and a per-item scores CSV
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
from pathlib import Path
from typing import Any

from .aggregation import RunReport
from .model import items_to_rows

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the guard test
    mlflow = None  # type: ignore[assignment]
    _MLFLOW_AVAILABLE = False


class MlflowUnavailable(RuntimeError):
    """Raised when MLflow logging is attempted but mlflow is not installed."""


def is_mlflow_available() -> bool:
    return _MLFLOW_AVAILABLE


def require_mlflow() -> None:
    if not _MLFLOW_AVAILABLE:
        raise MlflowUnavailable(
            "mlflow is not installed. Install the [mlflow] extra "
            "(pip install -e '.[mlflow]') to log runs to MLflow."
        )


def build_tracking_payload(report: RunReport) -> dict[str, Any]:
    """Build the exact params/metrics/tags/artifacts for a run.

    Pure function — no mlflow client involved — so tests can assert
    on the tracked content without a tracking server.
    """
    overall = report.overall
    metrics: dict[str, float] = {
        "overall_mean": overall.mean_score,
        "pass_rate": overall.pass_rate,
        "factual_accuracy": overall.factual_accuracy,
        "refusal_rate": overall.refusal_rate,
    }
    if overall.ece is not None:
        metrics["ece"] = overall.ece
    for category, group in sorted(report.by_category.items()):
        metrics[f"category_{category}_mean"] = group.mean_score
        metrics[f"category_{category}_pass_rate"] = group.pass_rate
    for criterion, value in sorted(overall.criterion_means.items()):
        metrics[f"criterion_{criterion}_mean"] = value
    return {
        "params": {
            "model": report.model,
            "dataset_version": report.dataset_version,
            "rubric_version": report.rubric_version,
            "engine": report.engine,
            "n_items": str(report.n_items),
        },
        "metrics": metrics,
        "tags": {
            "engine": report.engine,
            "dataset_version": report.dataset_version,
        },
    }


def report_to_csv(report: RunReport) -> str:
    """Render the per-item scored rows as CSV text (MLflow artifact)."""
    rows = items_to_rows(report.scored_items)
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def log_evaluation_run(
    report: RunReport,
    tracking_uri: str,
    experiment: str = "llm-evaluation",
    run_name: str | None = None,
    report_path: str | Path | None = None,
) -> str:
    """Log one evaluation report to MLflow; returns the run id."""
    require_mlflow()
    payload = build_tracking_payload(report)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(payload["params"])
        mlflow.log_metrics(payload["metrics"])
        mlflow.set_tags(payload["tags"])
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            json_path = tmp_path / "report.json"
            json_path.write_text(
                json.dumps(report.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            csv_path = tmp_path / "scored_items.csv"
            csv_path.write_text(report_to_csv(report), encoding="utf-8")
            mlflow.log_artifact(str(json_path))
            mlflow.log_artifact(str(csv_path))
            if report_path is not None and Path(report_path).exists():
                mlflow.log_artifact(str(report_path))
    return run.info.run_id
