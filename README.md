# spark-mlflow-eval-pipeline

**Score LLM/model outputs at any scale, track the runs, and compare models head to head — with Spark and MLflow.**

This is the evaluation pipeline layer of the Databricks-style
workflow: a dataset of model outputs goes in, a scored, aggregated,
MLflow-tracked run report comes out, and `evpipe compare` tells you
whether the new model actually got better — and where it regressed.
It is the fifth repo in a set built from real evaluation work; the
scoring ideas come from the sibling repos
([llm-eval-toolkit](https://github.com/Sharif899/llm-eval-toolkit),
[ai-safety-red-teaming](https://github.com/Sharif899/ai-safety-red-teaming),
[annotation-qa-toolkit](https://github.com/Sharif899/annotation-qa-toolkit),
[ml-evaluation-toolkit](https://github.com/Sharif899/ml-evaluation-toolkit))
but everything here is implemented from scratch, so this repo stands
alone.

## What it does

- **Scores every response 1–5 on five criteria** — factual
  accuracy, reasoning quality, instruction adherence, helpfulness,
  safety — with deterministic, auditable rules (see
  `src/spark_mlflow_eval_pipeline/scoring.py`). Factual claims are
  checked against the reference answer: a wrong number, a swapped
  entity (Saturn for Jupiter, Lagos for Abuja) or a wrong Yoruba
  number word is a contradiction, not a near-miss.
- **Aggregates** per model, per category (factual QA, maths,
  instruction following, safety) and per slice (language: English /
  Yoruba), with pass rates, refusal rates, factual accuracy and
  Expected Calibration Error from the models' self-reported
  confidence.
- **Runs locally or on Spark.** The local engine scores in-process;
  the Spark engine (`--engine spark`) scores the same items
  distributed via `mapPartitions` — one scoring implementation, so
  a local run and a Spark run over the same data produce the same
  report — and aggregates per (model, category) with a real
  `groupBy`.
- **Tracks runs in MLflow**: params (model, dataset and rubric
  versions, engine, item count), metrics (overall and per-category
  means, pass rates, factual accuracy, ECE), tags, and artifacts
  (the JSON report and a per-item scores CSV).
- **Compares two runs**: per-metric deltas labelled improved /
  regressed / unchanged, head-to-head win rates over shared items,
  and Cohen's kappa on pass/fail agreement — which shows whether
  two models fail the *same* items, something a bare accuracy gap
  never shows.

## Install

```bash
pip install -e .            # core only: pandas, no heavy deps
pip install -e ".[spark]"   # + PySpark, for --engine spark
pip install -e ".[mlflow]"  # + MLflow, for --tracking-uri
pip install -e ".[spark,mlflow]"  # everything
```

Python 3.10+. Spark and MLflow are optional extras on purpose:
the core pipeline, the scoring and the CLI work with zero heavy
dependencies.

## Usage

```bash
# Score one model's outputs and write a run report
evpipe run --dataset data/examples/model_a.jsonl --model model-a --out runs/a.json

# Same, distributed on Spark and logged to MLflow
evpipe run --dataset data/examples/model_a.jsonl --model model-a --out runs/a.json \
  --engine spark --tracking-uri file:///home/sharif/mlruns --experiment llm-evaluation

# Compare two runs
evpipe compare --baseline runs/a.json --candidate runs/b.json

# Print a human-readable report (or --json for the raw report)
evpipe report --report runs/a.json
```

Dataset format: one JSONL file per model run. Every line is one
item:

```json
{"item_id": "fact-01", "prompt": "How many states does Nigeria have?",
 "reference": "Nigeria has 36 states and a Federal Capital Territory, Abuja.",
 "category": "factual_qa", "slice": "en", "model": "model-a",
 "response": "Nigeria has 36 states, plus the Federal Capital Territory in Abuja.",
 "confidence": 0.9}
```

`reference` is optional (safety items have none — their claims are
unverifiable by design, and factual accuracy is measured only where
a reference exists). `confidence` is optional; without it, ECE is
not reported. CSV files with the same columns also load.

## Example output

Real output from `evpipe run` and `evpipe compare` on the example
data (two models, 40 items each, 4 categories):

```
$ evpipe run --dataset data/examples/model_a.jsonl --model model-a --out runs/a.json
Report written to runs/a.json
Model model-a (local): n=40, mean=3.885, pass rate=0.625

$ evpipe run --dataset data/examples/model_b.jsonl --model model-b --out runs/b.json
Report written to runs/b.json
Model model-b (local): n=40, mean=3.52, pass rate=0.4

$ evpipe compare --baseline runs/a.json --candidate runs/b.json
Comparison — baseline model-a vs candidate model-b
================================================================
  overall mean_score         3.885 -> 3.52  (-0.365)  regressed
  overall factual_accuracy   0.73 -> 0.345  (-0.385)  regressed
  overall pass_rate          0.625 -> 0.4  (-0.225)  regressed
  overall refusal_rate       0.225 -> 0.075  (-0.150)  improved
  overall ece                0.112 -> 0.422  (+0.310)  regressed

  factual_qa             mean 4.46 -> 3.44  (-1.020)  regressed
  instruction_following  mean 3.2 -> 4.26  (+1.060)  improved
  maths                  mean 4.56 -> 3.14  (-1.420)  regressed
  safety                 mean 3.32 -> 3.24  (-0.080)  regressed

Head-to-head: 40 shared items, baseline wins 21, candidate wins 9, ties 10 (candidate win rate 0.3 on decisive pairs)
Pass/fail agreement: kappa 0.0 (no agreement beyond chance, n=40)
Improvements: overall refusal_rate, instruction_following mean_score, instruction_following factual_accuracy, instruction_following pass_rate, safety refusal_rate
Regressions: overall mean_score, overall factual_accuracy, overall pass_rate, overall ece, factual_qa mean_score, factual_qa factual_accuracy, factual_qa pass_rate, maths mean_score, maths factual_accuracy, maths pass_rate, safety mean_score
```

The comparison is the point: model-b is not just "worse" — it is
worse on facts (0.73 → 0.345 factual accuracy, 10 contradicted
claims to model-a's 0), worse on maths, worse calibrated
(ECE 0.112 → 0.422: model-b is overconfident where it is wrong),
but *better* at following format instructions (3.2 → 4.26:
model-a rambles past "answer in one sentence" while model-b
complies). A single overall number would have hidden all of that.

Model-a's own report (`evpipe report`):

```
Evaluation report — model-a (engine: local)
================================================================
Items: 40   Dataset: eval-items-v1   Rubric: rubric-v1
Overall mean: 3.885   Pass rate: 0.625
Factual accuracy: 0.73   Refusal rate: 0.225
ECE: 0.112 (n with confidence: 40)

Per criterion:
  factual_accuracy         3.475
  helpfulness              4.125
  instruction_adherence    3.55
  reasoning_quality        3.275
  safety                   5.0

Per category:
  factual_qa             n=10  mean=4.46 pass=1.0 factual=1.0
  instruction_following  n=10  mean=3.2 pass=0.4 factual=0.231
  maths                  n=10  mean=4.56 pass=1.0 factual=1.0
  safety                 n=10  mean=3.32 pass=0.1 factual=1.0

Per slice:
  en                     n=29  mean=3.91 pass=0.621 factual=0.75
  yo                     n=11  mean=3.818 pass=0.636 factual=0.667
```

## Project layout

```
src/spark_mlflow_eval_pipeline/
  _text.py       tokenising, overlap, refusal / harm markers (incl. Yoruba)
  model.py       EvalItem / ScoredItem, JSONL + CSV loading and validation
  scoring.py     rubric scoring, factual claim checks, pairwise comparison
  aggregation.py group metrics, ECE, Cohen's kappa, win rates, RunReport
  pipeline.py    run orchestration: load -> score -> aggregate -> report
  spark_jobs.py  PySpark job (mapPartitions scoring, groupBy aggregation)
  tracking.py    MLflow payload building and run logging
  compare.py     run-to-run deltas, improvements and regressions
  cli.py         evpipe run | compare | report
data/examples/   model_a.jsonl, model_b.jsonl (40 items each, 4 categories)
tests/           87 tests, 2 Spark local-mode tests (see below)
```

## Tests

```bash
pip install -e ".[test]"
python -m pytest -q
```

87 tests pass. Two further tests (marked `spark`) run the job
end-to-end in Spark local mode; they run on a machine where Spark
Python workers work, and skip cleanly where they cannot.

**Honest note on validation.** On this project's development VM,
PySpark 3.5.3 installs and imports fine, and MLflow 2.22.5 runs
fully against a local file store (the MLflow tests log a real run
and read it back). But the sandbox blocks the JVM's socket to the
Python worker daemon, so even a trivial `rdd.map` fails there with
a broken pipe — the Spark *job* is therefore validated here by
unit-testing its partition function directly (feeding it rows and
asserting the scored output matches local scoring exactly), by
schema checks, and by the shared pure-Python scorer it calls; the
two local-mode Spark tests exist to run for real in CI or on a
normal machine. `build_spark_session` sets the worker Python to
the current interpreter, which is the standard fix for running
PySpark from a virtualenv — the failure on this VM is the sandbox,
not the code, and the README says so rather than pretending
otherwise.

## What this is not

- **Not a training framework.** It evaluates saved predictions; it
  never trains or fine-tunes a model.
- **Not a judge LLM.** Scores are deterministic heuristics
  (token overlap, number and entity checks, refusal markers,
  format constraints). They catch the common, checkable failures
  and they are auditable by hand — but they cannot judge nuance,
  and a high overlap score is not the same as a correct answer.
  The README of the example data documents every deliberate flaw
  so the numbers can be reproduced and challenged.
- **Not a production Databricks job.** The Spark layer is a real
  PySpark job, but it is written for a single JSONL dataset and a
  local or small-cluster session, not for a Delta Lake / Unity
  Catalog deployment. The MLflow layer logs to any tracking URI,
  including a Databricks workspace, but it does not manage one.
- **Factual accuracy is only as good as the reference.** Where
  there is no reference (safety items), claims are unverifiable by
  design, and the metric excludes them rather than guessing.
