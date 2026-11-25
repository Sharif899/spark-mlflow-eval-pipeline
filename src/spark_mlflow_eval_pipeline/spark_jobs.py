"""Spark layer: the same scoring, distributed.

The Spark job reads a JSONL dataset, scores every item with the
pure-Python ``score_item`` function via ``mapPartitions`` (so the
scoring logic is identical to the local engine — there is only one
implementation), aggregates per (model, category) with a real
``groupBy`` aggregation, and returns the full per-item scored rows
to the driver, where the standard :func:`aggregation.build_report`
builds the report.

``pyspark`` is an optional dependency. This module imports cleanly
when pyspark is absent: the import is guarded, and a clear
``SparkUnavailable`` error is raised only when a Spark function is
actually called. Nothing here pretends Spark ran when it did not.

SparkSession is created with ``local[*]`` by default and a small
driver memory, so the job can run on a laptop or in CI. On this
project's development VM the local-mode test runs with
``local[1]`` and a 512 MB driver heap; the README documents the
environment honestly.
"""

from __future__ import annotations

import sys
from typing import Any

try:
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        BooleanType,
        FloatType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    _PYSPARK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the guard test
    DataFrame = None  # type: ignore[assignment]
    SparkSession = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _PYSPARK_AVAILABLE = False


class SparkUnavailable(RuntimeError):
    """Raised when a Spark function is called but pyspark is not installed."""


def is_spark_available() -> bool:
    return _PYSPARK_AVAILABLE


def require_spark() -> None:
    if not _PYSPARK_AVAILABLE:
        raise SparkUnavailable(
            "pyspark is not installed. Install the [spark] extra "
            "(pip install -e '.[spark]') to run the Spark engine."
        )


SCORED_SCHEMA = StructType(
    [
        StructField("item_id", StringType(), False),
        StructField("model", StringType(), True),
        StructField("category", StringType(), False),
        StructField("slice", StringType(), False),
        StructField("score_factual_accuracy", IntegerType(), False),
        StructField("score_reasoning_quality", IntegerType(), False),
        StructField("score_instruction_adherence", IntegerType(), False),
        StructField("score_helpfulness", IntegerType(), False),
        StructField("score_safety", IntegerType(), False),
        StructField("mean_score", FloatType(), False),
        StructField("factual_accuracy", FloatType(), False),
        StructField("supported", IntegerType(), False),
        StructField("contradicted", IntegerType(), False),
        StructField("unverifiable", IntegerType(), False),
        StructField("passed", BooleanType(), False),
        StructField("refused", BooleanType(), False),
    ]
)


def build_spark_session(
    app_name: str = "evpipe-evaluation",
    master: str = "local[*]",
    driver_memory: str = "1g",
) -> "SparkSession":
    """Create a SparkSession for evaluation jobs.

    The worker Python is set to the interpreter running this code
    (``sys.executable``): without that, Spark workers spawn the
    system ``python``, which has neither pyspark nor this package,
    and the job dies with a broken pipe. This is the standard fix
    for running PySpark from a virtualenv.
    """
    require_spark()
    return (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )


def _score_partition(rows: Any) -> Any:
    """Score one partition of dataset rows (runs inside Spark workers).

    Imported lazily inside the worker so the driver never needs the
    scoring module on its own classpath beyond the local package.
    Every field of the ScoredItem is emitted, so the driver can
    rebuild the report from Spark's output alone — no local
    re-scoring happens afterwards.
    """
    from .model import EvalItem
    from .scoring import score_item

    for row in rows:
        data = row.asDict(recursive=True)
        item = EvalItem.from_dict(data)
        scored = score_item(item)
        yield {
            "item_id": scored.item.item_id,
            "model": scored.item.model,
            "category": scored.item.category,
            "slice": scored.item.slice,
            "score_factual_accuracy": scored.scores["factual_accuracy"],
            "score_reasoning_quality": scored.scores["reasoning_quality"],
            "score_instruction_adherence": scored.scores["instruction_adherence"],
            "score_helpfulness": scored.scores["helpfulness"],
            "score_safety": scored.scores["safety"],
            "mean_score": float(scored.mean_score),
            "factual_accuracy": float(scored.factual_accuracy),
            "supported": scored.supported,
            "contradicted": scored.contradicted,
            "unverifiable": scored.unverifiable,
            "passed": bool(scored.passed),
            "refused": bool(scored.refused),
        }


def score_with_spark(
    spark: "SparkSession", dataset_path: str
) -> "DataFrame":
    """Score a JSONL dataset in Spark; returns a DataFrame of per-item scores."""
    require_spark()
    raw = spark.read.json(dataset_path)
    # Keep only the columns the scorer needs; missing optional
    # columns are filled so from_dict always sees a full record.
    for column, default in (
        ("reference", ""),
        ("category", "factual_qa"),
        ("slice", "all"),
        ("model", ""),
        ("response", ""),
        ("confidence", None),
    ):
        if column not in raw.columns:
            raw = raw.withColumn(column, F.lit(default))
    rdd = raw.rdd.mapPartitions(_score_partition)
    return spark.createDataFrame(rdd, schema=SCORED_SCHEMA)


def aggregate_in_spark(scored_df: "DataFrame") -> "DataFrame":
    """Aggregate per-item scores per (model, category) inside Spark.

    Returns one row per (model, category) with item count, mean
    score, mean factual accuracy, pass rate and refusal rate — the
    grouped aggregation a Databricks job would hand to a dashboard.
    """
    require_spark()
    return (
        scored_df.groupBy("model", "category")
        .agg(
            F.count("*").alias("n"),
            F.round(F.avg("mean_score"), 3).alias("mean_score"),
            F.round(F.avg("factual_accuracy"), 3).alias("factual_accuracy"),
            F.round(F.avg(F.col("passed").cast("double")), 3).alias("pass_rate"),
            F.round(F.avg(F.col("refused").cast("double")), 3).alias("refusal_rate"),
        )
        .orderBy("model", "category")
    )


def run_spark_evaluation(
    dataset_path: str,
    spark: "SparkSession | None" = None,
    master: str = "local[*]",
    driver_memory: str = "1g",
) -> list:
    """Score a dataset with Spark and return ScoredItem objects.

    The per-item scores come back from Spark and are rebuilt into
    full ScoredItem objects; the item records themselves are
    re-attached from the local file. The caller owns the SparkSession
    when one is passed in; otherwise a session is created and stopped
    here.
    """
    require_spark()
    from .model import EvalItem, ScoredItem, load_dataset
    from .scoring import CRITERIA

    own_session = spark is None
    if own_session:
        spark = build_spark_session(master=master, driver_memory=driver_memory)
    try:
        scored_df = score_with_spark(spark, dataset_path)
        rows = {row["item_id"]: row for row in scored_df.collect()}
        items = load_dataset(dataset_path)
        scored: list[ScoredItem] = []
        for item in items:
            row = rows[item.item_id]
            scored.append(
                ScoredItem(
                    item=item,
                    scores={
                        c: int(row[f"score_{c}"]) for c in CRITERIA
                    },
                    mean_score=float(row["mean_score"]),
                    factual_accuracy=float(row["factual_accuracy"]),
                    supported=int(row["supported"]),
                    contradicted=int(row["contradicted"]),
                    unverifiable=int(row["unverifiable"]),
                    passed=bool(row["passed"]),
                    refused=bool(row["refused"]),
                )
            )
        return scored
    finally:
        if own_session:
            spark.stop()
