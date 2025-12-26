# -*- coding: utf-8 -*-
"""Base classes for analyzers that compute aggregated results from evaluator outputs.

This module defines the base classes for analyzers, which are responsible for
computing aggregated results from individual evaluator outputs. Analyzers take
the detailed results from a single evaluator and compute summary statistics.

Classes:
    AnalysisResult: Base class for analyzer results.
    BaseAnalyzer: Abstract base class for analyzers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from openjudge.graders.schema import GraderResult


class AnalysisResult(BaseModel):
    """Base class for storing analyzer computation results.

    This class serves as the foundation for all analyzer results. It contains essential
    metadata about the computation process and can be extended to include specific
    result fields for different types of analyzers.

    Attributes:
        name (str): The unique identifier name of the analyzer.
        metadata (Dict[str, Any]): Additional information about the analyzer result,
            including computation details, timestamps, input parameters, and other
            relevant contextual data.

    Examples:
        Create a basic analysis result:
        >>> result = AnalysisResult(name="test_analysis")
        >>> print(result.name)
        test_analysis

        Create an analysis with metadata:
        >>> result = AnalysisResult(
        ...     name="accuracy_analyzer",
        ...     metadata={"computed_at": "2023-01-01", "version": "1.0"}
        ... )
        >>> print(result.metadata)
        {'computed_at': '2023-01-01', 'version': '1.0'}
    """

    name: str = Field(default=..., description="The name of the analyzer")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="The metadata of the analyzer result",
    )


class BaseAnalyzer(ABC):
    """Abstract base class for analyzers that compute aggregated evaluation results.

    Analyzers process detailed evaluation results from individual evaluators and
    compute aggregated metrics, statistics, and summary insights. They are
    primarily used to transform granular evaluation data into meaningful,
    high-level reports and analytics.

    As an abstract base class, this defines the standard interface that all
    concrete analyzer implementations must follow. Subclasses are required to
    implement the `analyze` method to provide specific computational logic.

    Attributes:
        name (str): The human-readable name of the analyzer, used for identification
            and reporting purposes.

    Examples:
        Creating a custom analyzer implementation:
        >>> from openjudge.graders.schema import GraderResult
        >>> class AccuracyAnalyzer(BaseAnalyzer):
        ...     name = "Accuracy Analyzer"
        ...     def analyze(self, data, grader_results, **kwargs):
        ...         # Calculate accuracy metrics from results
        ...         return AnalysisResult(
        ...             name=self.name,
        ...             metadata={
        ...                 "sample_count": len(data),
        ...                 "analysis_type": "accuracy"
        ...             }
        ...         )
        >>> analyzer = AccuracyAnalyzer()
        >>> print(analyzer.name)
        Accuracy Analyzer
    """

    name: str = "Base Analysis"

    @abstractmethod
    def analyze(
        self,
        dataset: List[dict],
        grader_results: List[GraderResult],
        **kwargs: Any,
    ) -> AnalysisResult:
        """Execute the analysis computation on provided evaluation data.

        This abstract method must be implemented by all subclasses to define
        the specific logic for analyzing evaluation results. The method processes
        the input data and grader results to produce aggregated metrics and insights.

        Args:
            dataset (List[dict]): The collection of data samples that were evaluated.
                Each dictionary represents a single sample containing its input
                parameters and expected outputs.
            grader_results (List[GraderResult]): The detailed evaluation results
                from a single evaluator, structured as a list of GraderResult
                objects (one per sample).
            **kwargs: Optional configuration parameters for the analysis process.
                These may include filtering criteria, threshold values, or other
                algorithm-specific settings.

        Returns:
            AnalysisResult: An instance containing the computed analysis results,
            including aggregated metrics and relevant metadata about the computation.

        Examples:
            Implementing a concrete analyzer:
            >>> from openjudge.graders.schema import GraderResult
            >>> class PerformanceAnalyzer(BaseAnalyzer):
            ...     name = "Performance Analyzer"
            ...     def analyze(self, data, grader_results, **kwargs):
            ...         # Process the evaluation results to compute performance metrics
            ...         successful_count = sum(1 for r in grader_results if r.success)
            ...         return AnalysisResult(
            ...             name=self.name,
            ...             metadata={
            ...                 "total_samples": len(data),
            ...                 "successful_completions": successful_count,
            ...                 "success_rate": successful_count / len(data) if data else 0
            ...             }
            ...         )
            >>> analyzer = PerformanceAnalyzer()
            >>> # Note: This is a simplified example - actual implementation would
            >>> # process the grader_results more thoroughly
        """
