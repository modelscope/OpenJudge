# -*- coding: utf-8 -*-
"""
Evaluation runner that integrates runner execution with metric computation.
"""
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from loguru import logger

from rm_gallery.core.runner.base import BaseRunner
from rm_gallery.core.runner.evaluation.metric.base import BaseMetric
from rm_gallery.core.runner.evaluation.schema import (
    EvaluationReport,
    EvaluationResult,
    MetricResult,
)
from rm_gallery.core.schema.data import EvalCase


class EvaluationRunner(BaseRunner):
    """
    Base runner that integrates evaluation execution with metric computation.

    Subclasses should implement _execute_evaluation() to define specific evaluation logic.
    This runner automatically computes metrics after execution and provides evaluation reports.

    Example:
        ```python
        from rm_gallery.core import EvaluationRunner, AccuracyMetric

        class MyCustomRunner(EvaluationRunner):
            async def _execute_evaluation(self, eval_cases, *args, **kwargs):
                # Implement custom evaluation logic
                results = []
                for sample in eval_cases:
                    result = await self._evaluate_single_sample(sample)
                    results.append(result)
                return {
                    "model": self.model.model_name,
                    "total_samples": len(eval_cases),
                    "results": [r.model_dump() for r in results],
                }

        # Create runner with metrics
        runner = MyCustomRunner(model=my_model, metrics=[AccuracyMetric()])

        # Run evaluation (returns EvaluationReport)
        report = await runner(eval_cases)
        print(f"Accuracy: {report.get_metric_value('accuracy')}")
        ```
    """

    def __init__(
        self,
        metrics: Optional[List[BaseMetric]] = None,
    ) -> None:
        """
        Initialize evaluation runner.

        Args:
            metrics: List of metrics to compute (optional)
        """
        self.metrics = metrics or []
        self.errors: List[str] = []

    @abstractmethod
    async def _execute_evaluation(
        self,
        eval_cases: List[EvalCase],
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        """
        Execute evaluation on data samples.

        Subclasses must implement this method to define their specific evaluation logic.

        Args:
            eval_cases: List of data samples to evaluate
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary containing:
                - model: Model name (string)
                - total_samples: Total number of samples (int)
                - results: List of evaluation results (list of dicts or EvaluationResult objects)
                - metadata: Optional metadata (dict)
        """
        raise NotImplementedError("Subclasses must implement _execute_evaluation()")

    def add_metric(self, metric: BaseMetric) -> "EvaluationRunner":
        """
        Add a metric to the evaluation runner.

        Args:
            metric: Metric to add

        Returns:
            Self for chaining
        """
        self.metrics.append(metric)
        return self

    async def aevaluate_batch(
        self,
        eval_cases: List[EvalCase],
        *args: Any,
        **kwargs: Any,
    ) -> EvaluationReport:
        """
        Execute the evaluation and compute metrics.

        Args:
            eval_cases: List of data samples to evaluate
            *args: Additional positional arguments for evaluation
            **kwargs: Additional keyword arguments for evaluation

        Returns:
            EvaluationReport with results and computed metrics
        """
        self.errors = []  # Reset errors for new run

        logger.info(f"Starting evaluation with {len(eval_cases)} samples")

        # Step 1: Execute evaluation
        logger.info("Running evaluation...")
        try:
            runner_output = await self._execute_evaluation(eval_cases, *args, **kwargs)
        except Exception as e:
            error_msg = f"Evaluation execution failed: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return self._build_error_report(eval_cases, error_msg)

        # Step 2: Parse runner output into EvaluationResult list
        logger.info("Parsing evaluation results...")
        try:
            results = self._parse_runner_output(runner_output)
        except Exception as e:
            error_msg = f"Failed to parse runner output: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return self._build_error_report(eval_cases, error_msg)

        # Step 3: Compute metrics
        logger.info(f"Computing {len(self.metrics)} metrics...")
        metric_results = {}
        for metric in self.metrics:
            try:
                metric_result = metric.compute(results)
                metric_results[metric.name] = metric_result
                if metric_result.is_valid:
                    logger.info(f"  {metric.name}: {metric_result.value:.4f}")
                else:
                    logger.warning(f"  {metric.name}: Failed - {metric_result.error}")
            except Exception as e:
                error_msg = f"Failed to compute metric {metric.name}: {str(e)}"
                logger.error(error_msg)
                self.errors.append(error_msg)
                # Add failed metric result
                metric_results[metric.name] = MetricResult(
                    metric_name=metric.name,
                    value=0.0,
                    error=str(e),
                )

        # Step 4: Build report
        valid_results = [r for r in results if r.is_valid]
        model_name = runner_output.get("model", "unknown")

        report = EvaluationReport(
            model_name=model_name,
            total_samples=len(results),
            valid_samples=len(valid_results),
            results=results,
            metrics=metric_results,
            errors=self.errors,
            metadata=runner_output.get("metadata", {}),
        )

        logger.info(
            f"Evaluation completed: {len(valid_results)}/{len(results)} valid samples",
        )
        if self.errors:
            logger.warning(f"Encountered {len(self.errors)} errors during evaluation")

        return report

    def _parse_runner_output(
        self,
        runner_output: Dict[str, Any],
    ) -> List[EvaluationResult]:
        """
        Parse runner output into standard EvaluationResult format.

        Args:
            runner_output: Raw output from runner

        Returns:
            List of EvaluationResult objects
        """
        results = []
        raw_results = runner_output.get("results", [])

        for idx, raw_result in enumerate(raw_results):
            try:
                # Try to parse as EvaluationResult
                if isinstance(raw_result, EvaluationResult):
                    results.append(raw_result)
                elif isinstance(raw_result, dict):
                    # Convert dict to EvaluationResult
                    eval_result = EvaluationResult(
                        unique_id=raw_result.get("unique_id", f"sample_{idx}"),
                        scores=raw_result.get("scores"),
                        predicted_index=raw_result.get("predicted_index"),
                        ground_truth_index=raw_result.get("ground_truth_index"),
                        comparison_matrix=raw_result.get("comparison_matrix"),
                        metadata=raw_result.get("metadata", {}),
                        error=raw_result.get("error"),
                    )
                    results.append(eval_result)
                else:
                    error_msg = (
                        f"Unrecognized result format at index {idx}: {type(raw_result)}"
                    )
                    logger.warning(error_msg)
                    # Create error result
                    results.append(
                        EvaluationResult(
                            unique_id=f"sample_{idx}",
                            error=error_msg,
                        ),
                    )
            except Exception as e:
                error_msg = f"Failed to parse result at index {idx}: {str(e)}"
                logger.error(error_msg)
                self.errors.append(error_msg)
                results.append(
                    EvaluationResult(
                        unique_id=f"sample_{idx}",
                        error=error_msg,
                    ),
                )

        return results

    def _build_error_report(
        self,
        eval_cases: List[EvalCase],
        error_message: str,
    ) -> EvaluationReport:
        """
        Build an error report when evaluation fails early.

        Args:
            eval_cases: Original data samples
            error_message: Error message describing the failure

        Returns:
            EvaluationReport with error information
        """
        return EvaluationReport(
            model_name="unknown",
            total_samples=len(eval_cases),
            valid_samples=0,
            results=[],
            metrics={},
            errors=[error_message],
            metadata={"evaluation_failed": True},
        )
