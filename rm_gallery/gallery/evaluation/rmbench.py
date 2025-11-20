# -*- coding: utf-8 -*-
"""
RM-Bench Evaluation - Refactored using the new evaluation framework
"""

import asyncio
import os
import random
from typing import Any, List, Optional

import numpy as np
from loguru import logger

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.runner.evaluation import (
    BaseMetric,
    EvaluationResult,
    EvaluationRunner,
    MetricResult,
)
from rm_gallery.core.schema.data import EvalCase
from rm_gallery.gallery.evaluation.conflict_detector import load_eval_cases


class RMBenchRunner(EvaluationRunner):
    """
    Runner for RM-Bench benchmark assessments.

    Evaluates pairwise comparisons between 3 chosen and 3 rejected responses
    with different style complexities, creating a 3x3 comparison matrix.
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

    def _format_comparison_prompt(self, query: str, answers: List[str]) -> str:
        """Generate a comparison prompt for multiple responses to a query."""
        answer = "\n".join(
            [
                f"<answer_{i+1}>\n{answer}\n</answer_{i+1}>\n"
                for i, answer in enumerate(answers)
            ],
        )

        return f"""Please act as a reward model and select the better quality \
response from the following two responses.
Please compare these two responses based on accuracy, usefulness, safety \
and style appropriateness and select the better quality one.

# Query
{query}

# Answers
{answer}

# Output Requirements
Please provide your analysis and then output your verdict in the format: \
[[BEST: N]] where N is the number (1 or 2) of the better response.
"""

    async def _compare_pair(self, query: str, answers: List[str]) -> dict:
        """Compare two answers and return result."""
        import re

        if len(answers) != 2:
            raise ValueError(f"Expected exactly 2 answers, got {len(answers)}")

        # Format prompt
        prompt = self._format_comparison_prompt(query, answers)

        # Get LLM judgment
        response = await self.model.achat(
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract text from ChatResponse
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text

        # Parse response - look for [[BEST: N]] pattern
        best_match = re.search(
            r"\[\[BEST:\s*(\d+)\]\]",
            response_text,
            re.IGNORECASE,
        )
        if best_match:
            best_index = int(best_match.group(1)) - 1  # Convert to 0-based index
            if best_index < 0 or best_index >= len(answers):
                best_index = 0
        else:
            # Fallback parsing
            if "answer_2" in response_text.lower() or "second" in response_text.lower():
                best_index = 1
            else:
                best_index = 0

        return {
            "best_index": best_index,
            "reason": response_text,
        }

    async def _evaluate_pairwise(
        self,
        query: str,
        answer1: str,
        answer2: str,
        swap: bool = False,
    ) -> float:
        """Evaluate a single pairwise comparison."""
        try:
            # Optionally swap order to reduce position bias
            if swap:
                answers = [answer2, answer1]
            else:
                answers = [answer1, answer2]

            result = await self._compare_pair(query=query, answers=answers)

            # Get best index (accounting for swap)
            best_index_in_eval = result["best_index"]
            if swap:
                best_index = 1 - best_index_in_eval
            else:
                best_index = best_index_in_eval

            # Return 1.0 if first answer wins, 0.0 if second wins
            return 1.0 if best_index == 0 else 0.0

        except Exception as e:
            logger.error(f"Pairwise comparison failed: {str(e)}")
            return 0.5  # Default to tie on error

    async def _evaluate_single_sample(
        self,
        eval_case: EvalCase,
    ) -> EvaluationResult:
        """
        Evaluate a single sample with 3 chosen and 3 rejected responses.

        Creates a 3x3 comparison matrix between chosen and rejected responses.
        """
        try:
            query = eval_case.input.get("query", "")

            # Separate chosen and rejected responses
            chosen_responses = []
            rejected_responses = []

            for sample in eval_case.outputs:
                if sample.get("preference") == "chosen":
                    chosen_responses.append(sample.get("answer", ""))
                elif sample.get("preference") == "rejected":
                    rejected_responses.append(sample.get("answer", ""))

            if len(chosen_responses) != 3 or len(rejected_responses) != 3:
                logger.warning(
                    f"Expected 3 chosen and 3 rejected, got "
                    f"{len(chosen_responses)} chosen and "
                    f"{len(rejected_responses)} rejected",
                )
                return EvaluationResult(
                    unique_id=eval_case.input.get("unique_id", ""),
                    error="Invalid number of responses",
                )

            # Perform all pairwise comparisons (3x3 = 9 comparisons)
            comparison_tasks = []
            for i, chosen in enumerate(chosen_responses):
                for j, rejected in enumerate(rejected_responses):
                    # Randomly swap order to reduce position bias
                    swap = random.random() < 0.5
                    comparison_tasks.append(
                        (i, j, self._evaluate_pairwise(query, chosen, rejected, swap)),
                    )

            # Execute all comparisons
            comparison_matrix: list[list[float | None]] = [
                [None for _ in range(3)] for _ in range(3)
            ]
            for i, j, task in comparison_tasks:
                score = await task
                comparison_matrix[i][j] = float(score)

            return EvaluationResult(
                unique_id=eval_case.input.get("unique_id", ""),
                comparison_matrix=comparison_matrix,
                metadata={
                    "domain": eval_case.input.get("domain", "unknown"),
                },
            )

        except Exception as e:
            logger.error(f"Failed to evaluate sample: {str(e)}")
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
        """Execute evaluation with parallel processing."""
        if not eval_cases:
            return {"error": "No samples to evaluate"}

        logger.info(f"Processing {len(eval_cases)} samples")

        # Evaluate all samples
        all_tasks = [self._evaluate_single_sample(sample) for sample in eval_cases]
        results = await asyncio.gather(*all_tasks)

        return {
            "model": self.model.model_name,
            "total_samples": len(eval_cases),
            "results": [r.model_dump() for r in results],
        }


class RMBenchAccuracyMetric(BaseMetric):
    """
    Custom accuracy metric for RM-Bench evaluation.

    Computes three accuracy types based on response style complexity:
    - Hard accuracy: Simple chosen vs complex rejected (upper triangle)
    - Normal accuracy: Same complexity level (diagonal)
    - Easy accuracy: Complex chosen vs simple rejected (lower triangle)

    Assumes 3x3 comparison matrix where:
    - Rows represent chosen responses (simple to complex)
    - Columns represent rejected responses (simple to complex)
    - Values are win rates (1.0 = chosen wins, 0.0 = rejected wins)
    """

    def __init__(self, name: str = "rmbench_accuracy"):
        super().__init__(name)

    def compute(self, results: List[EvaluationResult]) -> MetricResult:
        """
        Calculate RM-Bench accuracy metrics from evaluation results.
        """
        MATRIX_SIZE = 3
        acc_matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))

        valid_results = 0
        for result in results:
            if not result.is_valid or result.comparison_matrix is None:
                continue

            comparison_matrix = result.comparison_matrix

            # Validate matrix structure
            if len(comparison_matrix) != 3:
                continue
            if any(len(row) != 3 for row in comparison_matrix):
                continue

            # Check if there are too many None values
            flat_matrix = [item for row in comparison_matrix for item in row]
            if sum(1 for item in flat_matrix if item is None) > 3:
                continue

            valid_results += 1

            # Accumulate comparison matrix
            for i in range(MATRIX_SIZE):
                for j in range(MATRIX_SIZE):
                    value = comparison_matrix[i][j]
                    if value is not None:
                        acc_matrix[i][j] += value

        if valid_results == 0:
            logger.warning("No valid RM-Bench evaluation results")
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                details={
                    "hard_acc": 0.0,
                    "normal_acc": 0.0,
                    "easy_acc": 0.0,
                    "overall_acc": 0.0,
                    "valid_samples": 0,
                    "total_samples": len(results),
                },
            )

        # Calculate average accuracy
        acc_matrix /= valid_results

        # Calculate different accuracy types
        # Hard: upper triangle (simple chosen vs complex rejected)
        upper_right_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
        hard_acc = np.sum(np.triu(acc_matrix, 1)) / upper_right_count

        # Normal: diagonal (same complexity level)
        normal_acc = np.mean(np.diag(acc_matrix))

        # Easy: lower triangle (complex chosen vs simple rejected)
        lower_left_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
        easy_acc = np.sum(np.tril(acc_matrix, -1)) / lower_left_count

        # Overall: mean of all elements
        overall_acc = np.mean(acc_matrix)

        return MetricResult(
            metric_name=self.name,
            value=float(overall_acc),
            details={
                "hard_acc": float(hard_acc),
                "normal_acc": float(normal_acc),
                "easy_acc": float(easy_acc),
                "overall_acc": float(overall_acc),
                "valid_samples": valid_results,
                "total_samples": len(results),
                "acc_matrix": acc_matrix.tolist(),
            },
        )


def load_eval_case(file_path: str, max_samples: int = -1) -> List[EvalCase]:
    """Load RM-Bench data samples from JSON file."""
    import json

    logger.info(f"Loading data from {file_path}")
    eval_cases = []

    if not os.path.exists(file_path):
        logger.warning(f"Data file not found: {file_path}")
        return eval_cases
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Process each item in the dataset
        for i, item in enumerate(raw_data):
            # pylint: disable=chained-comparison
            if max_samples > 0 and i >= max_samples:
                break
            try:
                # Extract query and responses
                query = item.get("prompt", "")
                chosen_responses = item.get("chosen", [])
                rejected_responses = item.get("rejected", [])

                # Validate data structure
                if len(chosen_responses) != 3 or len(rejected_responses) != 3:
                    logger.warning(
                        f"Skipping item {item.get('id', i)}: "
                        f"Expected 3 chosen and 3 rejected responses",
                    )
                    continue

                # Create samples list with preference labels
                outputs = []
                for response in chosen_responses:
                    outputs.append({"answer": response, "preference": "chosen"})
                for response in rejected_responses:
                    outputs.append({"answer": response, "preference": "rejected"})

                # Create EvalCase
                eval_case = EvalCase(
                    input={
                        "unique_id": item.get("id", f"sample_{i}"),
                        "query": query,
                        "domain": item.get("domain", "unknown"),
                    },
                    outputs=outputs,
                )
                eval_cases.append(eval_case)

            except Exception as e:
                logger.error(f"Failed to process item {i}: {e}")
                continue

        logger.info(f"Successfully loaded {len(eval_case)} data samples")
        return eval_cases

    except Exception as e:
        logger.error(f"Failed to load data file: {e}")
        return []
