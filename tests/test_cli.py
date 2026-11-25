"""Tests for the evpipe CLI."""

import json

import pytest

from spark_mlflow_eval_pipeline.cli import main
from spark_mlflow_eval_pipeline.tracking import is_mlflow_available


def test_run_writes_report(tiny_dataset, tmp_path, capsys):
    out = tmp_path / "report.json"
    assert main(["run", "--dataset", str(tiny_dataset), "--model", "model-a",
                 "--out", str(out)]) == 0
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n_items"] == 4
    assert data["model"] == "model-a"
    captured = capsys.readouterr()
    assert "Report written" in captured.out

    assert main(["report", "--report", str(out)]) == 0
    captured = capsys.readouterr()
    assert "model-a" in captured.out
    assert "Overall mean" in captured.out


def test_run_json_output(tiny_dataset, tmp_path, capsys):
    out = tmp_path / "report.json"
    assert main(["run", "--dataset", str(tiny_dataset), "--model", "model-a",
                 "--out", str(out), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["n_items"] == 4
    assert summary["mlflow_run_id"] is None


@pytest.mark.skipif(not is_mlflow_available(), reason="mlflow not installed")
def test_run_with_mlflow_logging(tiny_dataset, tmp_path):
    out = tmp_path / "report.json"
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    assert main(["run", "--dataset", str(tiny_dataset), "--model", "model-a",
                 "--out", str(out), "--tracking-uri", tracking_uri,
                 "--experiment", "cli-test"]) == 0
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name("cli-test")
    runs = client.search_runs(exp.experiment_id)
    assert len(runs) == 1
    assert runs[0].data.params["model"] == "model-a"


def test_compare(example_a, example_b, tmp_path, capsys):
    run_a = tmp_path / "a.json"
    run_b = tmp_path / "b.json"
    assert main(["run", "--dataset", str(example_a), "--model", "model-a",
                 "--out", str(run_a)]) == 0
    assert main(["run", "--dataset", str(example_b), "--model", "model-b",
                 "--out", str(run_b)]) == 0
    assert main(["compare", "--baseline", str(run_a), "--candidate", str(run_b)]) == 0
    captured = capsys.readouterr()
    assert "model-a" in captured.out
    assert "model-b" in captured.out
    assert "regressed" in captured.out
    assert "instruction_following" in captured.out

    assert main(["compare", "--baseline", str(run_a), "--candidate", str(run_b),
                 "--json"]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["baseline_model"] == "model-a"
    assert comparison["candidate_model"] == "model-b"
    assert comparison["win_rate"]["wins_a"] == 21


def test_missing_dataset_exits_two(tmp_path, capsys):
    assert main(["run", "--dataset", "/tmp/nope.jsonl", "--model", "m",
                 "--out", str(tmp_path / "r.json")]) == 2
    assert "error" in capsys.readouterr().err


def test_invalid_engine_rejected(tiny_dataset, tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["run", "--dataset", str(tiny_dataset), "--model", "m",
              "--out", str(tmp_path / "r.json"), "--engine", "bogus"])
    assert exc.value.code == 2
