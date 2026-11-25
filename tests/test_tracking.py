"""Tests for the MLflow tracking layer."""

import csv
import io

import pytest

from spark_mlflow_eval_pipeline.aggregation import build_report
from spark_mlflow_eval_pipeline.model import load_dataset
from spark_mlflow_eval_pipeline.scoring import score_items
from spark_mlflow_eval_pipeline.tracking import (
    MlflowUnavailable,
    build_tracking_payload,
    is_mlflow_available,
    log_evaluation_run,
    report_to_csv,
)


def _report(example_a):
    return build_report(
        score_items(load_dataset(example_a)), model="model-a",
        dataset_version="eval-items-v1", rubric_version="rubric-v1", engine="local",
    )


class TestPayload:
    def test_params(self, example_a):
        payload = build_tracking_payload(_report(example_a))
        assert payload["params"] == {
            "model": "model-a",
            "dataset_version": "eval-items-v1",
            "rubric_version": "rubric-v1",
            "engine": "local",
            "n_items": "40",
        }

    def test_metrics(self, example_a):
        payload = build_tracking_payload(_report(example_a))
        metrics = payload["metrics"]
        assert metrics["overall_mean"] == 3.885
        assert metrics["pass_rate"] == 0.625
        assert metrics["factual_accuracy"] == 0.73
        assert metrics["ece"] == 0.112
        assert metrics["category_maths_mean"] == 4.56
        assert metrics["category_instruction_following_mean"] == 3.2
        assert metrics["criterion_safety_mean"] == 5.0

    def test_no_ece_metric_without_confidence(self, tiny_dataset):
        from spark_mlflow_eval_pipeline.model import EvalItem
        from spark_mlflow_eval_pipeline.scoring import score_items as _score

        items = load_dataset(tiny_dataset)
        items = [EvalItem(**{**i.to_dict(), "confidence": None}) for i in items]
        report = build_report(
            _score(items), model="model-a",
            dataset_version="test", rubric_version="rubric-v1", engine="local",
        )
        payload = build_tracking_payload(report)
        assert "ece" not in payload["metrics"]

    def test_tags(self, example_a):
        payload = build_tracking_payload(_report(example_a))
        assert payload["tags"] == {"engine": "local", "dataset_version": "eval-items-v1"}

    def test_csv_artifact(self, example_a):
        text = report_to_csv(_report(example_a))
        rows = list(csv.DictReader(io.StringIO(text)))
        assert len(rows) == 40
        assert rows[0]["item_id"] == "fact-01"
        assert "score_factual_accuracy" in rows[0]
        assert rows[0]["model"] == "model-a"


class TestMlflowLogging:
    @pytest.mark.skipif(not is_mlflow_available(), reason="mlflow not installed")
    def test_log_to_local_file_store(self, example_a, tmp_path):
        report = _report(example_a)
        tracking_uri = f"file://{tmp_path / 'mlruns'}"
        run_id = log_evaluation_run(
            report, tracking_uri=tracking_uri, experiment="test-exp",
            run_name="test-run", report_path=None,
        )
        assert run_id

        import mlflow

        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        run = client.get_run(run_id)
        assert run.data.params["model"] == "model-a"
        assert run.data.params["n_items"] == "40"
        assert run.data.metrics["overall_mean"] == pytest.approx(3.885)
        assert run.data.metrics["category_maths_mean"] == pytest.approx(4.56)
        artifacts = [a.path for a in client.list_artifacts(run_id)]
        assert "report.json" in artifacts
        assert "scored_items.csv" in artifacts

    def test_guard_raises_when_mlflow_absent(self, example_a, monkeypatch):
        import spark_mlflow_eval_pipeline.tracking as tracking

        monkeypatch.setattr(tracking, "_MLFLOW_AVAILABLE", False)
        with pytest.raises(MlflowUnavailable):
            log_evaluation_run(_report(example_a), tracking_uri="file:///tmp/x")
