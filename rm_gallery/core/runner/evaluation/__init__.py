# -*- coding: utf-8 -*-
"""
Evaluation module for RM-Gallery.

This module provides a framework for building evaluation systems:
- Schema: Standard data structures (EvaluationResult, MetricResult, EvaluationReport)
- Runner: Evaluation execution with metrics (EvaluationRunner)
- Metric: Metric computation (accuracy, conflict rate, etc.)

Example:
    ```python
    from rm_gallery.core.runner.evaluation import (
        EvaluationRunner,
        AccuracyMetric,
        ConflictMetric,
    )

    # Create evaluation runner with metrics
    eval_runner = EvaluationRunner(
        runner=my_runner,
        metrics=[AccuracyMetric(), ConflictMetric()]
    )

    # Run evaluation
    report = await eval_runner(data_samples)
    print(report.summary())
    ```
"""

from rm_gallery.core.runner.evaluation.metric import (
    AccuracyMetric,
    BaseMetric,
    ConflictMetric,
)
from rm_gallery.core.runner.evaluation.runner import EvaluationRunner
from rm_gallery.core.runner.evaluation.schema import (
    EvaluationReport,
    EvaluationResult,
    MetricResult,
)

__all__ = [
    # Schema
    "EvaluationResult",
    "EvaluationReport",
    "MetricResult",
    # Runner
    "EvaluationRunner",
    # Metrics
    "BaseMetric",
    "AccuracyMetric",
    "ConflictMetric",
]
