"""
RewardBench2 Benchmark Test

Test script for RewardBench2 evaluation with concurrent model calls.
This test validates the grader implementation against RewardBench2 dataset.

Concurrency Implementation:
- Leverages GradingRunner's built-in ConcurrencyManager
- Uses singleton pattern with asyncio.Semaphore for global concurrency control
- Configurable max_concurrency parameter (default: 8)

Usage:
    # Run with default settings (8 concurrent requests)
    python tests/benchmarks/test_rewardbench2.py

    # Run with custom concurrency
    python tests/benchmarks/test_rewardbench2.py --max_concurrency=16

    # Run with limited samples for quick testing
    python tests/benchmarks/test_rewardbench2.py --max_samples=10 --max_concurrency=4

    # Run with custom model
    python tests/benchmarks/test_rewardbench2.py --model_name=gpt-4 --max_concurrency=4
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import fire
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


from cookbooks.grader_validation.rewardbench2 import (
    RewardBench2Analyzer,
    RewardBench2Grader,
    load_rewardbench2_data,
)
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.runner.grading_runner import GradingRunner


async def run_rewardbench2_test(
    data_path: str = "data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet",
    result_path: str = "test_results/rewardbench2_concurrent.json",
    max_samples: int = -1,
    model_name: str = "qwen2.5-72b-instruct",
    max_concurrency: int = 8,
):
    """
    Run RewardBench2 evaluation test with concurrent model calls.

    Args:
        data_path: Path to the RewardBench2 parquet data file
        result_path: Path to save evaluation results
        max_samples: Maximum number of samples to evaluate (-1 for all)
        model_name: Name of the model to use for evaluation
        max_concurrency: Maximum number of concurrent API requests
    """
    logger.info("=" * 80)
    logger.info("RewardBench2 Concurrent Evaluation Test")
    logger.info("=" * 80)

    try:
        # Resolve paths relative to project root
        if not os.path.isabs(data_path):
            data_path = os.path.join(project_root, data_path)
        if not os.path.isabs(result_path):
            result_path = os.path.join(project_root, result_path)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        logger.info(f"Configuration:")
        logger.info(f"  Data path: {data_path}")
        logger.info(f"  Result path: {result_path}")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Max samples: {max_samples if max_samples > 0 else 'ALL'}")
        logger.info(f"  Max concurrency: {max_concurrency}")
        logger.info("")

        # Load data
        logger.info("Loading dataset...")
        dataset = load_rewardbench2_data(data_path, max_samples)
        logger.info(f"Loaded {len(dataset)} samples for evaluation")

        if not dataset:
            logger.error("No samples loaded. Check data file.")
            return

        # Initialize model
        logger.info(f"Initializing model: {model_name}")

        chat_model = OpenAIChatModel(model=model_name)

        # Initialize grader
        grader = RewardBench2Grader(
            model=chat_model,
            name="rewardbench2",
            description="RewardBench2 evaluation grader",
        )

        # Initialize runner with concurrency control
        # GradingRunner's ConcurrencyManager handles concurrent API requests
        runner = GradingRunner(
            grader_configs={"rewardbench2": grader},
            max_concurrency=max_concurrency,
        )

        # Run evaluation
        logger.info(f"Running evaluation with {max_concurrency} concurrent requests...")
        logger.info("(Using GradingRunner's built-in ConcurrencyManager)")
        logger.info("")
        runner_result = await runner.arun(dataset)

        grader_results = runner_result.get("rewardbench2", [])

        # Analyze results
        analyzer = RewardBench2Analyzer()
        analysis_result = analyzer.analyze(dataset, grader_results)

        # Print results
        logger.info("")
        logger.info("=" * 80)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 80)

        logger.info(f"\nModel: {model_name}")

        metadata = analysis_result.metadata
        logger.info("\nOverall Performance:")
        logger.info(f"  Accuracy: {metadata.get('accuracy', 0):.4f} ({metadata.get('accuracy', 0)*100:.2f}%)")
        logger.info(f"  Correct: {metadata.get('correct_count', 0)}/{metadata.get('valid_samples', 0)}")
        logger.info(f"  Total samples: {metadata.get('total_samples', 0)}")

        subset_acc = metadata.get("subset_accuracy", {})
        if subset_acc:
            logger.info("\nSubset Performance:")
            for subset, metrics in subset_acc.items():
                accuracy = metrics.get("accuracy", 0)
                correct = metrics.get("correct_count", 0)
                total = metrics.get("total_samples", 0)
                logger.info(
                    f"  {subset:15s}: {accuracy:.4f} ({accuracy*100:5.2f}%) - " f"{correct:2d}/{total:2d} correct",
                )

        logger.info("\n" + "=" * 80)

        # Save results
        result_dir = os.path.dirname(result_path)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)

        with open(result_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Results saved to: {result_path}")
        logger.success("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def main(**kwargs):
    """Wrapper to run async test function."""
    asyncio.run(run_rewardbench2_test(**kwargs))


if __name__ == "__main__":
    fire.Fire(main)
