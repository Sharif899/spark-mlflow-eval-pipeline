"""Tests for pipeline orchestration and report comparison."""

import json

import pytest

from spark_mlflow_eval_pipeline.aggregation import build_report
from spark_mlflow_eval_pipeline.compare import compare_reports
from spark_mlflow_eval_pipeline.model import load_dataset
from spark_mlflow_eval_pipeline.pipeline import (
    RunConfig,
    load_report,
    run_and_log,
    run_evaluation,
    write_report,
)
from spark_mlflow_eval_pipeline.scoring import score_items


class TestRunConfig:
    def test_unknown_engine_rejected(self):
        with pytest.raises(ValueError):
            RunConfig(dataset_path="x.jsonl", model="m", engine="dask").validate()

    def test_empty_model_rejected(self):
        with pytest.raises(ValueError):
            RunConfig(dataset_path="x.jsonl", model="  ").validate()


class TestRunEvaluation:
    def test_local_run(self, tiny_dataset):
        report = run_evaluation(RunConfig(dataset_path=str(tiny_dataset), model="model-a"))
        assert report.n_items == 4
        assert report.model == "model-a"
        assert report.engine == "local"
        assert report.dataset_version == "eval-items-v1"

    def test_missing_dataset(self):
        with pytest.raises(Exception):
            run_evaluation(RunConfig(dataset_path="/tmp/nope.jsonl", model="m"))

    def test_write_and_load_report(self, tiny_dataset, tmp_path):
        report = run_evaluation(RunConfig(dataset_path=str(tiny_dataset), model="model-a"))
        path = tmp_path / "report.json"
        write_report(report, path)
        restored = load_report(path)
        assert restored.overall.mean_score == report.overall.mean_score
        assert restored.n_items == 4

    def test_run_and_log_without_tracking(self, tiny_dataset, tmp_path):
        summary = run_and_log(
            RunConfig(dataset_path=str(tiny_dataset), model="model-a"),
            tmp_path / "report.json",
        )
        assert summary["mlflow_run_id"] is None
        assert summary["n_items"] == 4
        assert summary["overall_mean"] > 0

    def test_report_json_is_sorted(self, tiny_dataset, tmp_path):
        report = run_evaluation(RunConfig(dataset_path=str(tiny_dataset), model="model-a"))
        path = tmp_path / "report.json"
        write_report(report, path)
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        assert list(data.keys()) == sorted(data.keys())


class TestCompare:
    def _reports(self, example_a, example_b):
        report_a = build_report(
            score_items(load_dataset(example_a)), model="model-a",
            dataset_version="eval-items-v1", rubric_version="rubric-v1", engine="local",
        )
        report_b = build_report(
            score_items(load_dataset(example_b)), model="model-b",
            dataset_version="eval-items-v1", rubric_version="rubric-v1", engine="local",
        )
        return report_a, report_b

    def test_overall_deltas(self, example_a, example_b):
        baseline, candidate = self._reports(example_a, example_b)
        comparison = compare_reports(baseline, candidate)
        by_metric = {d.metric: d for d in comparison.overall}
        assert by_metric["mean_score"].delta == -0.365
        assert by_metric["mean_score"].verdict == "regressed"
        assert by_metric["factual_accuracy"].delta == -0.385
        assert by_metric["pass_rate"].delta == -0.225
        # ECE going up is a regression (lower is better).
        assert by_metric["ece"].delta == 0.31
        assert by_metric["ece"].verdict == "regressed"

    def test_category_deltas(self, example_a, example_b):
        baseline, candidate = self._reports(example_a, example_b)
        comparison = compare_reports(baseline, candidate)
        maths = {d.metric: d for d in comparison.by_category["maths"]}
        assert maths["mean_score"].delta == -1.42
        inst = {d.metric: d for d in comparison.by_category["instruction_following"]}
        assert inst["mean_score"].delta == 1.06
        assert inst["mean_score"].verdict == "improved"

    def test_head_to_head_in_comparison(self, example_a, example_b):
        baseline, candidate = self._reports(example_a, example_b)
        comparison = compare_reports(baseline, candidate)
        assert comparison.win_rate["wins_a"] == 21
        assert comparison.win_rate["wins_b"] == 9
        assert comparison.kappa["kappa"] == 0.0

    def test_improvements_and_regressions_listed(self, example_a, example_b):
        baseline, candidate = self._reports(example_a, example_b)
        comparison = compare_reports(baseline, candidate)
        assert "instruction_following mean_score" in comparison.improvements
        assert "overall mean_score" in comparison.regressions
        assert "maths mean_score" in comparison.regressions

    def test_identical_reports_are_unchanged(self, example_a):
        report = build_report(
            score_items(load_dataset(example_a)), model="model-a",
            dataset_version="eval-items-v1", rubric_version="rubric-v1", engine="local",
        )
        comparison = compare_reports(report, report)
        assert all(d.verdict == "unchanged" for d in comparison.overall)
        assert comparison.win_rate["ties"] == 40
