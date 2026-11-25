"""Command line interface: evpipe run | compare | report.

- ``run`` scores a dataset with the local engine (default) or the
  Spark engine (``--engine spark``), writes a JSON report, and
  optionally logs the run to MLflow (``--tracking-uri``).
- ``compare`` shows the per-metric delta between two run reports.
- ``report`` prints a human-readable summary of one run report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .aggregation import RunReport
from .compare import compare_reports
from .pipeline import load_report, run_and_log, RunConfig


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _print_report(report: RunReport) -> None:
    overall = report.overall
    print(f"Evaluation report — {report.model} (engine: {report.engine})")
    print("=" * 64)
    print(f"Items: {report.n_items}   Dataset: {report.dataset_version}   Rubric: {report.rubric_version}")
    print(f"Overall mean: {overall.mean_score}   Pass rate: {overall.pass_rate}")
    print(f"Factual accuracy: {overall.factual_accuracy}   Refusal rate: {overall.refusal_rate}")
    if overall.ece is not None:
        print(f"ECE: {overall.ece} (n with confidence: {overall.n_with_confidence})")
    print()
    print("Per criterion:")
    for criterion, value in sorted(overall.criterion_means.items()):
        print(f"  {criterion:<24} {value}")
    print()
    print("Per category:")
    for category, group in sorted(report.by_category.items()):
        print(
            f"  {category:<22} n={group.n:<3} mean={group.mean_score} "
            f"pass={group.pass_rate} factual={group.factual_accuracy}"
        )
    print()
    print("Per slice:")
    for slice_name, group in sorted(report.by_slice.items()):
        print(
            f"  {slice_name:<22} n={group.n:<3} mean={group.mean_score} "
            f"pass={group.pass_rate} factual={group.factual_accuracy}"
        )


def _print_comparison(comparison) -> None:
    print(
        f"Comparison — baseline {comparison.baseline_model} "
        f"vs candidate {comparison.candidate_model}"
    )
    print("=" * 64)
    for delta in comparison.overall:
        base = "-" if delta.baseline is None else delta.baseline
        cand = "-" if delta.candidate is None else delta.candidate
        change = "-" if delta.delta is None else f"{delta.delta:+.3f}"
        print(f"  overall {delta.metric:<18} {base} -> {cand}  ({change})  {delta.verdict}")
    print()
    for category, deltas in sorted(comparison.by_category.items()):
        for delta in deltas:
            if delta.metric == "mean_score":
                base = "-" if delta.baseline is None else delta.baseline
                cand = "-" if delta.candidate is None else delta.candidate
                change = "-" if delta.delta is None else f"{delta.delta:+.3f}"
                print(f"  {category:<22} mean {base} -> {cand}  ({change})  {delta.verdict}")
    print()
    win = comparison.win_rate
    decisive = win["wins_a"] + win["wins_b"]
    candidate_win_rate = round(win["wins_b"] / decisive, 3) if decisive else 0.0
    print(
        f"Head-to-head: {win['n_compared']} shared items, "
        f"baseline wins {win['wins_a']}, candidate wins {win['wins_b']}, "
        f"ties {win['ties']} (candidate win rate {candidate_win_rate} "
        f"on decisive pairs)"
    )
    kappa = comparison.kappa
    print(
        f"Pass/fail agreement: kappa {kappa['kappa']} "
        f"({kappa['interpretation']}, n={kappa['n']})"
    )
    if comparison.improvements:
        print("Improvements: " + ", ".join(comparison.improvements))
    if comparison.regressions:
        print("Regressions: " + ", ".join(comparison.regressions))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evpipe",
        description="Spark + MLflow evaluation pipeline for LLM/model outputs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Score a dataset and write a run report.")
    run.add_argument("--dataset", required=True, help="JSONL or CSV dataset file.")
    run.add_argument("--model", required=True, help="Model name for this run.")
    run.add_argument("--out", required=True, help="Path for the JSON report.")
    run.add_argument("--engine", choices=["local", "spark"], default="local")
    run.add_argument("--dataset-version", default="eval-items-v1")
    run.add_argument("--rubric-version", default="rubric-v1")
    run.add_argument("--tracking-uri", default=None,
                     help="MLflow tracking URI; omit to skip MLflow logging.")
    run.add_argument("--experiment", default="llm-evaluation")
    run.add_argument("--run-name", default=None)
    run.add_argument("--json", action="store_true", help="Print the summary as JSON.")

    compare = sub.add_parser("compare", help="Compare two run reports.")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--json", action="store_true")

    report = sub.add_parser("report", help="Print a human-readable run report.")
    report.add_argument("--report", required=True)
    report.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            config = RunConfig(
                dataset_path=args.dataset,
                model=args.model,
                dataset_version=args.dataset_version,
                rubric_version=args.rubric_version,
                engine=args.engine,
                tracking_uri=args.tracking_uri,
                experiment=args.experiment,
                run_name=args.run_name,
            )
            summary = run_and_log(config, args.out)
            if args.json:
                _print_json(summary)
            else:
                print(f"Report written to {summary['report_path']}")
                print(
                    f"Model {summary['model']} ({summary['engine']}): "
                    f"n={summary['n_items']}, mean={summary['overall_mean']}, "
                    f"pass rate={summary['pass_rate']}"
                )
                if summary["mlflow_run_id"]:
                    print(f"MLflow run: {summary['mlflow_run_id']}")
            return 0
        if args.command == "compare":
            baseline = load_report(args.baseline)
            candidate = load_report(args.candidate)
            comparison = compare_reports(baseline, candidate)
            if args.json:
                _print_json(comparison.to_dict())
            else:
                _print_comparison(comparison)
            return 0
        if args.command == "report":
            report = load_report(args.report)
            if args.json:
                _print_json(report.to_dict())
            else:
                _print_report(report)
            return 0
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"evpipe: error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # Spark/MLflow failures surface here, not as a traceback
        print(f"evpipe: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
