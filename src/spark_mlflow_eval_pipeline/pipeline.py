"""Pipeline orchestration: one evaluation run, end to end.

A run is: load a dataset -> score every item (locally with the
pure-Python scorer, or distributed with the Spark job) -> aggregate
-> write a JSON report -> optionally log to MLflow.

The engine choice changes *where* scoring happens, never *what*
the scores are: the Spark job calls the same ``score_item``
function, and aggregation always runs through
:func:`aggregation.build_report`. A local run and a Spark run over
the same data must produce the same report; there is a test that
checks exactly that when Spark is available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .aggregation import RunReport, build_report
from .model import DATASET_VERSION, RUBRIC_VERSION, load_dataset
from .scoring import score_items


@dataclass
class RunConfig:
    dataset_path: str
    model: str
    dataset_version: str = DATASET_VERSION
    rubric_version: str = RUBRIC_VERSION
    engine: str = "local"  # "local" | "spark"
    tracking_uri: str | None = None
    experiment: str = "llm-evaluation"
    run_name: str | None = None

    def validate(self) -> None:
        if self.engine not in ("local", "spark"):
            raise ValueError(
                f"unknown engine {self.engine!r}; expected 'local' or 'spark'"
            )
        if not self.model.strip():
            raise ValueError("model name must not be empty")


def run_evaluation(config: RunConfig) -> RunReport:
    """Run one evaluation and return the report (no side effects)."""
    config.validate()
    items = load_dataset(config.dataset_path)
    model = config.model or (items[0].model if items and items[0].model else "unknown")
    if config.engine == "spark":
        from .spark_jobs import run_spark_evaluation

        scored = run_spark_evaluation(config.dataset_path)
    else:
        scored = score_items(items)
    return build_report(
        scored,
        model=model,
        dataset_version=config.dataset_version,
        rubric_version=config.rubric_version,
        engine=config.engine,
    )


def write_report(report: RunReport, path: str | Path) -> Path:
    """Write a run report as indented, deterministically ordered JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def load_report(path: str | Path) -> RunReport:
    """Load a previously written run report."""
    with Path(path).open(encoding="utf-8") as fh:
        return RunReport.from_dict(json.load(fh))


def run_and_log(config: RunConfig, report_path: str | Path) -> dict[str, Any]:
    """Run an evaluation, write the report, log to MLflow if asked.

    Returns a summary dict with the report path and, when tracking
    was requested, the MLflow run id. MLflow logging is skipped
    (not faked) when no tracking URI is given.
    """
    report = run_evaluation(config)
    write_report(report, report_path)
    summary: dict[str, Any] = {
        "report_path": str(report_path),
        "model": report.model,
        "engine": report.engine,
        "n_items": report.n_items,
        "overall_mean": report.overall.mean_score,
        "pass_rate": report.overall.pass_rate,
        "mlflow_run_id": None,
    }
    if config.tracking_uri:
        from .tracking import log_evaluation_run

        run_id = log_evaluation_run(
            report,
            tracking_uri=config.tracking_uri,
            experiment=config.experiment,
            run_name=config.run_name or f"{report.model}-{report.engine}",
            report_path=Path(report_path),
        )
        summary["mlflow_run_id"] = run_id
    return summary
