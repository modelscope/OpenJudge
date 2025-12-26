# -*- coding: utf-8 -*-
"""Base class for validation analyzers that compare grader results with ground truth.

This module defines the base class for validation analyzers, which evaluate
grader performance by comparing their results against ground truth data.
"""

from abc import abstractmethod
from typing import Any, Dict, List

from openjudge.analyzer.base_analyzer import AnalysisResult, BaseAnalyzer
from openjudge.graders.schema import GraderResult
from openjudge.utils.mapping import get_value_by_path


class BaseValidationAnalyzer(BaseAnalyzer):
    """Abstract base class for validation analyzers.

    Validation analyzers evaluate grader performance by comparing their results
    against ground truth data. These analyzers compute metrics like accuracy,
    precision, recall, etc., which require labeled data for comparison.

    Subclasses should implement the analyze method to compare grader results
    with ground truth values from the data samples.

    Attributes:
        name (str): The name of the analyzer. Defaults to "Validation Analysis".

    Example:
        >>> from openjudge.graders.schema import GraderResult, GraderScore
        >>> class AccuracyAnalyzer(BaseValidationAnalyzer ):
        ...     name = "Accuracy Analyzer"
        ...     def analyze(self, data, grader_results, label_path="label", **kwargs):
        ...         # Compare grader_results with ground truth from data[label_path]
        ...         # Return computed accuracy metrics
        ...         pass
        >>> analyzer = AccuracyAnalyzer()
        >>> print(analyzer.name)
        Accuracy Analyzer
    """

    name: str = "Validation Analysis"

    def _extract(self, sample: Dict[str, Any], label_path: str) -> Any:
        """Extract label from a data sample.

        Args:
            sample: A data sample dictionary containing ground truth information.
            label_path: The key or path to extract the label from the sample.
                      Can be a simple key like "label" or a nested path like "labels.correct_answer".

        Returns:
            Any: The extracted label value.

        Example:
            >>> sample = {"labels": {"correct_answer": "A"}}
            >>> analyzer = ComparativeAnalyzer()
            >>> label = analyzer._extract(sample, "labels.correct_answer")
            >>> print(label)
            A
        """
        return get_value_by_path(sample, label_path)

    @abstractmethod
    def analyze(
        self,
        dataset: List[dict],
        grader_results: List[GraderResult],
        label_path: str = "label",
        **kwargs: Any,
    ) -> AnalysisResult:
        """Compare grader results with ground truth data.

        Args:
            dataset: The data samples that were evaluated, including ground truth labels.
                 Each dictionary should contain the input data and expected output.
            grader_results: The evaluation results from a single grader.
                          Contains the grader's evaluation of each sample.
            label_path: The key or path to extract the ground truth label from each sample.
                       Defaults to "label". Can be a nested path like "labels.correct_answer".
            **kwargs: Additional keyword arguments for analyzer configuration.

        Returns:
            AnalysisResult: The computed comparative analysis result containing metrics
                         comparing grader performance to ground truth.

        Example:
            >>> from openjudge.graders.schema import GraderResult, GraderScore
            >>> # This is an abstract method that must be implemented by subclasses
            >>> # Implementation would compare grader_results with ground truth from data
            >>> # and return a AnalysisResult with computed metrics
        """
