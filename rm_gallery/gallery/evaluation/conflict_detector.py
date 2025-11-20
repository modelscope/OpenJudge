# -*- coding: utf-8 -*-
"""
Conflict Detector - Refactored using the new metric/pipeline architecture

This is an example of how to use the new evaluation framework.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.runner.evaluation import (
    AccuracyMetric,
    BaseMetric,
    ConflictMetric,
    EvaluationRunner,
)
from rm_gallery.core.runner.evaluation.schema import EvaluationResult
from rm_gallery.core.schema.data import EvalCase


class PairwiseComparisonRunner(EvaluationRunner):
    """
    Runner that performs pairwise comparisons between all responses.

    This runner:
    1. Takes EvalCase with multiple answers
    2. Compares all pairs of answers
    3. Builds a comparison matrix
    4. Returns standardized EvaluationResult
    """

    def __init__(
        self,
        model: OpenAIChatModel,
        max_workers: int = 8,
        metrics: Optional[List[BaseMetric]] = None,
    ):
        super().__init__(metrics=metrics)
        self.model = model
        self.max_workers = max_workers

    async def _compare_pair(self, query: str, answer_a: str, answer_b: str) -> int:
        """Compare a pair of answers and return comparison result."""
        try:
            prompt = f"""Please compare two responses to: {query}

Response A: {answer_a}

Response B: {answer_b}

Which is better? Reply with [[BEST: A]], [[BEST: B]], or [[TIE]].
"""

            response = await self.model.achat(
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract text from ChatResponse
            response_text = ""
            for block in response.content:
                if block.type == "text":
                    response_text += block.text

            if "[[BEST: A]]" in response_text.upper():
                return 1
            elif "[[BEST: B]]" in response_text.upper():
                return -1
            else:
                return 0

        except Exception as e:
            logger.error(f"Comparison failed: {str(e)}")
            return 0

    def _build_comparison_matrix(
        self,
        num_responses: int,
        comparison_results: Dict[tuple, int],
    ) -> List[List[float]]:
        """Build comparison matrix from pairwise comparison results."""
        matrix = np.zeros((num_responses, num_responses), dtype=float)
        for (i, j), result in comparison_results.items():
            matrix[i][j] = float(result)
            matrix[j][i] = float(-result)
        return matrix.tolist()

    async def _evaluate_single_sample(
        self,
        eval_case: EvalCase,
    ) -> EvaluationResult:
        """Evaluate a single sample and build comparison matrix."""
        try:
            query = eval_case.input.get("query", "")
            answers = [sample.get("answer", "") for sample in eval_case.outputs]

            if len(answers) < 2:
                return EvaluationResult(
                    unique_id=eval_case.input.get("unique_id", ""),
                    error="Insufficient answers",
                )

            # Perform pairwise comparisons
            comparison_pairs = [
                (i, j) for i in range(len(answers)) for j in range(i + 1, len(answers))
            ]
            tasks = [
                self._compare_pair(query, answers[i], answers[j])
                for i, j in comparison_pairs
            ]
            results = await asyncio.gather(*tasks)

            comparison_results = dict(zip(comparison_pairs, results))
            comparison_matrix = self._build_comparison_matrix(
                len(answers),
                comparison_results,
            )

            # Find ground truth (chosen) index for accuracy calculation
            ground_truth_index = None
            for i, sample in enumerate(eval_case.outputs):
                if sample.get("preference") == "chosen":
                    ground_truth_index = i
                    break

            # Calculate predicted index (highest score)
            row_sums = [sum(row) for row in comparison_matrix]
            predicted_index = row_sums.index(max(row_sums)) if row_sums else None

            return EvaluationResult(
                unique_id=eval_case.input.get("unique_id", ""),
                comparison_matrix=comparison_matrix,
                predicted_index=predicted_index,
                ground_truth_index=ground_truth_index,
                metadata={
                    "num_answers": len(answers),
                    "num_comparisons": len(comparison_pairs),
                },
            )

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return EvaluationResult(
                unique_id=eval_case.input.get("unique_id", ""),
                error=str(e),
            )

    async def _execute_evaluation(
        self,
        eval_cases: List[EvalCase],
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        """Execute pairwise comparison evaluation."""
        if not eval_cases:
            return {"error": "No samples to evaluate"}

        logger.info(f"Processing {len(eval_cases)} samples")

        tasks = [self._evaluate_single_sample(sample) for sample in eval_cases]
        results = await asyncio.gather(*tasks)

        return {
            "model": self.model.model_name,
            "total_samples": len(eval_cases),
            "results": [r.model_dump() for r in results],
        }


def load_eval_cases(file_path: str, max_samples: int = -1) -> List[EvalCase]:
    """Load data samples from JSONL file."""
    import json

    logger.info(f"Loading data from {file_path}")
    eval_cases = []

    if not os.path.exists(file_path):
        logger.warning(f"Data file not found: {file_path}")
        return eval_cases

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # pylint: disable=chained-comparison
            if max_samples > 0 and i >= max_samples:
                break
            try:
                data = json.loads(line.strip())
                sample = EvalCase(**data)
                eval_cases.append(sample)
            except Exception as e:
                logger.error(f"Failed to parse line {i+1}: {e}")
                continue

    logger.info(f"Successfully loaded {len(eval_cases)} data samples")
    return eval_cases


async def evaluate_async(
    eval_cases: List[EvalCase],
    model: OpenAIChatModel,
    metrics: List[str] | None = None,
) -> dict:
    """
    Run evaluation with specified metrics.

    Args:
        eval_cases: Data to evaluate
        model: Model to use for evaluation
        metrics: List of metric names to compute (e.g., ["accuracy", "conflict_rate"])
                If None, computes both accuracy and conflict rate

    Returns:
        Dictionary with evaluation report
    """
    # Create metrics
    metric_objects: List[BaseMetric] = []
    if metrics is None:
        metrics = ["accuracy", "conflict_rate"]

    for metric_name in metrics:
        if metric_name == "accuracy":
            metric_objects.append(AccuracyMetric())
        elif metric_name == "conflict_rate":
            metric_objects.append(ConflictMetric())
        else:
            logger.warning(f"Unknown metric: {metric_name}")

    # Create runner with metrics
    runner = PairwiseComparisonRunner(model=model, metrics=metric_objects)

    # Run evaluation
    report = await runner(eval_cases)

    return report.model_dump()
