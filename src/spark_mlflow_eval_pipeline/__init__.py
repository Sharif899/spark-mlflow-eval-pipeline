"""Spark + MLflow evaluation pipeline for LLM/model outputs.

Score model outputs at any scale — locally with the pure-Python
scorer, or distributed with the Spark engine — aggregate the
results, track runs in MLflow, and compare models head to head.
"""

from .aggregation import RunReport, aggregate_group, build_report
from .model import DATASET_VERSION, RUBRIC_VERSION, EvalItem, ScoredItem
from .pipeline import RunConfig, load_report, run_and_log, run_evaluation, write_report
from .scoring import CRITERIA, score_item, score_items

__all__ = [
    "CRITERIA",
    "DATASET_VERSION",
    "RUBRIC_VERSION",
    "EvalItem",
    "RunConfig",
    "RunReport",
    "ScoredItem",
    "aggregate_group",
    "build_report",
    "load_report",
    "run_and_log",
    "run_evaluation",
    "score_item",
    "score_items",
    "write_report",
]

__version__ = "0.1.0"
