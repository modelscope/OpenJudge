# -*- coding: utf-8 -*-
"""
RM-Gallery Core Module

This module contains the core abstractions for building evaluation systems:
- Schema: Standard data structures (DataSample, EvaluationResult, EvaluationReport, MetricResult)
- Runner: Execute evaluation logic (BaseRunner, EvaluationRunner)
- Metric: Compute metrics from results (BaseMetric, AccuracyMetric, ConflictMetric)

Example:
    ```python
    from rm_gallery.core import (
        EvaluationRunner,
        AccuracyMetric,
        ConflictMetric,
        DataSample,
    )

    # Define your runner by inheriting from EvaluationRunner
    class MyRunner(EvaluationRunner):
        async def _execute_evaluation(self, data_samples, **kwargs):
            # Implement evaluation logic
            return {"model": "my_model", "results": [...]}

    # Create runner with metrics
    runner = MyRunner(metrics=[AccuracyMetric(), ConflictMetric()])

    # Run evaluation
    report = await runner(data_samples)
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
from rm_gallery.core.schema import DataSample

__all__ = [
    # Schema
    "DataSample",
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
