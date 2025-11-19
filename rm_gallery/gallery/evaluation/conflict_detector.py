# -*- coding: utf-8 -*-
"""
Conflict Detector - Refactored using the new metric/pipeline architecture

This is an example of how to use the new evaluation framework.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

import fire
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
from rm_gallery.core.schema.data import DataSample


class PairwiseComparisonRunner(EvaluationRunner):
    """
    Runner that performs pairwise comparisons between all responses.

    This runner:
    1. Takes DataSample with multiple answers
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

            response = await self.model(messages=[{"role": "user", "content": prompt}])

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
        data_sample: DataSample,
    ) -> EvaluationResult:
        """Evaluate a single sample and build comparison matrix."""
        try:
            query = data_sample.data.get("query", "")
            answers = [sample.get("answer", "") for sample in data_sample.samples]

            if len(answers) < 2:
                return EvaluationResult(
                    unique_id=data_sample.data.get("unique_id", ""),
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
            for i, sample in enumerate(data_sample.samples):
                if sample.get("preference") == "chosen":
                    ground_truth_index = i
                    break

            # Calculate predicted index (highest score)
            row_sums = [sum(row) for row in comparison_matrix]
            predicted_index = row_sums.index(max(row_sums)) if row_sums else None

            return EvaluationResult(
                unique_id=data_sample.data.get("unique_id", ""),
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
                unique_id=data_sample.data.get("unique_id", ""),
                error=str(e),
            )

    async def _execute_evaluation(
        self,
        data_samples: List[DataSample],
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        """Execute pairwise comparison evaluation."""
        if not data_samples:
            return {"error": "No samples to evaluate"}

        logger.info(f"Processing {len(data_samples)} samples")

        tasks = [self._evaluate_single_sample(sample) for sample in data_samples]
        results = await asyncio.gather(*tasks)

        return {
            "model": self.model.model_name,
            "total_samples": len(data_samples),
            "results": [r.model_dump() for r in results],
        }


def load_data_samples(file_path: str, max_samples: int = -1) -> List[DataSample]:
    """Load data samples from JSONL file."""
    import json

    logger.info(f"Loading data from {file_path}")
    data_samples = []

    if not os.path.exists(file_path):
        logger.warning(f"Data file not found: {file_path}")
        return data_samples

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # pylint: disable=chained-comparison
            if max_samples > 0 and i >= max_samples:
                break
            try:
                data = json.loads(line.strip())
                sample = DataSample(**data)
                data_samples.append(sample)
            except Exception as e:
                logger.error(f"Failed to parse line {i+1}: {e}")
                continue

    logger.info(f"Successfully loaded {len(data_samples)} data samples")
    return data_samples


async def evaluate_async(
    data_samples: List[DataSample],
    model: OpenAIChatModel,
    metrics: List[str] | None = None,
) -> dict:
    """
    Run evaluation with specified metrics.

    Args:
        data_samples: Data to evaluate
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
    report = await runner(data_samples)

    return report.model_dump()


def main(
    data_path: str,
    result_path: str = "data/results/conflict_detector.json",
    model_name: str = "gpt-4o",
    api_key: str | None = None,
    base_url: str | None = None,
    max_samples: int = 10,
    metrics: str = "accuracy,conflict_rate",
) -> None:
    """
    Main execution function.

    Args:
        data_path: Path to JSONL data file
        result_path: Path to save results
        model_name: Name of the model to use
        api_key: OpenAI API key (optional, reads from env if not provided)
        base_url: Base URL for API (optional)
        max_samples: Maximum number of samples to evaluate (-1 for all)
        metrics: Comma-separated list of metrics to compute
    """
    import json

    try:
        print(f"Loading data from: {data_path}")
        data_samples = load_data_samples(file_path=data_path, max_samples=max_samples)

        if not data_samples:
            print(f"No data samples loaded. Please check the data path: {data_path}")
            return

        print(f"Loaded {len(data_samples)} samples")

        # Initialize model
        model = OpenAIChatModel(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            generate_kwargs={"temperature": 0.1},
        )

        # Parse metrics
        metric_list = [m.strip() for m in metrics.split(",") if m.strip()]

        # Run evaluation
        report = asyncio.run(evaluate_async(data_samples, model, metric_list))

        # Print results
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS (using new framework)")
        print("=" * 80)
        print(f"\nModel: {report.get('model_name', 'Unknown')}")
        print(f"Total samples: {report.get('total_samples', 0)}")
        print(f"Valid samples: {report.get('valid_samples', 0)}")

        for metric_name, metric_result in report.get("metrics", {}).items():
            print(f"\n{metric_name}: {metric_result.get('value', 0):.4f}")
            print(f"  Details: {metric_result.get('details', {})}")

        # Save results
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"\nResults saved to: {result_path}")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    fire.Fire(main)
