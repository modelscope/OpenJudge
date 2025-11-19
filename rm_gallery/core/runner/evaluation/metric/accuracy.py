# -*- coding: utf-8 -*-
"""
Accuracy metric for evaluation.
"""
from typing import List

from rm_gallery.core.runner.evaluation.metric.base import BaseMetric
from rm_gallery.core.runner.evaluation.schema import EvaluationResult, MetricResult


class AccuracyMetric(BaseMetric):
    """
    Accuracy metric that computes the percentage of correct predictions.

    Supports multiple accuracy calculation modes:
    - Simple binary accuracy: correct_count / total_count
    - Multi-class accuracy: checks if predicted_index matches ground_truth_index
    - Score-based accuracy: checks if ground_truth sample has the highest score
    """

    def __init__(self, name: str = "accuracy") -> None:
        """
        Initialize accuracy metric.

        Args:
            name: Name of the metric (default: "accuracy")
        """
        super().__init__(name)

    def compute(self, results: List[EvaluationResult]) -> MetricResult:
        """
        Compute accuracy from evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            MetricResult with accuracy value and detailed statistics
        """
        if not results:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                details={
                    "correct_count": 0,
                    "total_count": 0,
                    "valid_count": 0,
                },
            )

        # Filter valid results
        valid_results = [r for r in results if r.is_valid]

        if not valid_results:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                details={
                    "correct_count": 0,
                    "total_count": len(results),
                    "valid_count": 0,
                    "error_count": len(results),
                },
            )

        # Count correct predictions
        correct_count = sum(1 for r in valid_results if r.is_correct)
        valid_count = len(valid_results)
        accuracy = correct_count / valid_count if valid_count > 0 else 0.0

        return MetricResult(
            metric_name=self.name,
            value=float(accuracy),
            details={
                "correct_count": correct_count,
                "total_count": len(results),
                "valid_count": valid_count,
                "error_count": len(results) - valid_count,
            },
        )
