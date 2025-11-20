# -*- coding: utf-8 -*-
"""
RM-Gallery Core Module

This module contains the core abstractions for building evaluation systems:
- Schema: Standard data structures (EvalCase, EvaluationResult, EvaluationReport, MetricResult)
- Runner: Execute evaluation logic (BaseRunner, EvaluationRunner)
- Metric: Compute metrics from results (BaseMetric, AccuracyMetric, ConflictMetric)

Example:
    ```python
    from rm_gallery.core import (
        EvaluationRunner,
        AccuracyMetric,
        ConflictMetric,
        EvalCase,
    )

    # Define your runner by inheriting from EvaluationRunner
    class MyRunner(EvaluationRunner):
        async def _execute_evaluation(self, eval_cases, **kwargs):
            # Implement evaluation logic
            return {"model": "my_model", "results": [...]}

    # Create runner with metrics
    runner = MyRunner(metrics=[AccuracyMetric(), ConflictMetric()])

    # Run evaluation
    report = await runner(eval_cases)
    print(f"Accuracy: {report.metrics['accuracy'].value}")
    ```
"""

# Runner
from rm_gallery.core.runner import BaseRunner

# Evaluation framework
from rm_gallery.core.runner.evaluation import (
    AccuracyMetric,
    BaseMetric,
    ConflictMetric,
    EvaluationReport,
    EvaluationResult,
    EvaluationRunner,
    MetricResult,
)

# Core schema
from rm_gallery.core.schema import EvalCase

__all__ = [
    # Schema
    "EvalCase",
    "EvaluationResult",
    "EvaluationReport",
    "MetricResult",
    # Runner
    "BaseRunner",
    "EvaluationRunner",
    # Metrics
    "BaseMetric",
    "AccuracyMetric",
    "ConflictMetric",
]
