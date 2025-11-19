# -*- coding: utf-8 -*-
"""
Evaluation result schema definitions.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """
    Standard evaluation result format.

    This is the unified output format that all runners should return.
    Metrics will compute scores based on this format.
    """

    unique_id: str = Field(default="", description="Unique identifier for the sample")
    scores: Optional[List[float]] = Field(
        default=None,
        description="Score for each sample (used for accuracy calculation)",
    )
    predicted_index: Optional[int] = Field(
        default=None,
        description="Predicted best answer index (for classification tasks)",
    )
    ground_truth_index: Optional[int] = Field(
        default=None,
        description="Ground truth best answer index",
    )
    comparison_matrix: Optional[List[List[Optional[float]]]] = Field(
        default=None,
        description="Pairwise comparison matrix (for conflict detection)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if evaluation failed",
    )

    @property
    def is_valid(self) -> bool:
        """Check if this result is valid (no error)."""
        return self.error is None

    @property
    def is_correct(self) -> bool:
        """
        Check if prediction is correct.

        Returns True if:
        - For classification: predicted_index == ground_truth_index
        - For scoring: score of ground_truth sample is highest
        """
        if not self.is_valid:
            return False

        # Classification-based correctness
        if self.predicted_index is not None and self.ground_truth_index is not None:
            return self.predicted_index == self.ground_truth_index

        # Score-based correctness
        if self.scores and self.ground_truth_index is not None:
            max_score = max(self.scores)
            return self.scores[self.ground_truth_index] == max_score

        return False


class MetricResult(BaseModel):
    """Result of a metric computation."""

    metric_name: str = Field(description="Name of the metric")
    value: float = Field(description="Metric value")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if metric computation failed",
    )

    @property
    def is_valid(self) -> bool:
        """Check if metric computation was successful."""
        return self.error is None


class EvaluationReport(BaseModel):
    """Complete evaluation report including results and metrics."""

    model_name: str = Field(default="", description="Name of the evaluated model")
    total_samples: int = Field(description="Total number of samples")
    valid_samples: int = Field(description="Number of valid samples")
    results: List[EvaluationResult] = Field(
        default_factory=list,
        description="Individual evaluation results",
    )
    metrics: Dict[str, MetricResult] = Field(
        default_factory=dict,
        description="Computed metrics",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Errors encountered during evaluation",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    @property
    def error_rate(self) -> float:
        """Calculate the error rate."""
        if self.total_samples == 0:
            return 0.0
        invalid_samples = self.total_samples - self.valid_samples
        return invalid_samples / self.total_samples

    @property
    def success_rate(self) -> float:
        """Calculate the success rate."""
        if self.total_samples == 0:
            return 0.0
        return self.valid_samples / self.total_samples

    def get_metric_value(self, metric_name: str, default: float = 0.0) -> float:
        """
        Get metric value by name with fallback.

        Args:
            metric_name: Name of the metric
            default: Default value if metric not found or invalid

        Returns:
            Metric value or default
        """
        metric = self.metrics.get(metric_name)
        if metric and metric.is_valid:
            return metric.value
        return default

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the evaluation report.

        Returns:
            Dictionary with key statistics
        """
        return {
            "model_name": self.model_name,
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "error_rate": self.error_rate,
            "success_rate": self.success_rate,
            "metrics": {
                name: metric.value
                for name, metric in self.metrics.items()
                if metric.is_valid
            },
            "errors_count": len(self.errors),
        }
