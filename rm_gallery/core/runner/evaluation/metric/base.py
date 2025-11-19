# -*- coding: utf-8 -*-
"""
Base metric class for evaluation.
"""
from abc import ABC, abstractmethod
from typing import List

from rm_gallery.core.runner.evaluation.schema import EvaluationResult, MetricResult


class BaseMetric(ABC):
    """
    Base class for evaluation metrics.

    A metric computes a score based on evaluation results.
    Subclasses should implement the compute method.

    Attributes:
        name: Name of the metric
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the metric.

        Args:
            name: Name of the metric
        """
        self.name = name

    @abstractmethod
    def compute(self, results: List[EvaluationResult]) -> MetricResult:
        """
        Compute the metric from evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            MetricResult containing the computed metric value and details
        """
        ...  # pylint: disable=unnecessary-ellipsis

    def __call__(self, results: List[EvaluationResult]) -> MetricResult:
        """
        Allow metric to be called directly.

        Args:
            results: List of evaluation results

        Returns:
            MetricResult containing the computed metric value
        """
        return self.compute(results)
