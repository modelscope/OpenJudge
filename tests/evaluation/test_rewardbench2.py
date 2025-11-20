# -*- coding: utf-8 -*-
"""
Example usage of RewardBench2 evaluation
"""
import asyncio
import os

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.runner.evaluation import AccuracyMetric
from rm_gallery.gallery.evaluation.rewardbench2 import (
    RewardBench2Runner,
    TiesAccuracyMetric,
    load_rewardbench2_data,
)


async def test_rewardbench2_basic():
    """Test basic RewardBench2 evaluation."""
    # Setup paths
    data_path = "data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet"
    result_path = "data/results/rewardbench2_test.json"

    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please download the RewardBench2 dataset first.")
        print("Dataset: https://huggingface.co/datasets/allenai/reward-bench-2")
        return

    # Load data (limit to 10 samples for quick testing)
    print(f"Loading data from: {data_path}")
    eval_cases = load_rewardbench2_data(data_path, max_samples=10)

    if not eval_cases:
        print("No data samples loaded.")
        return

    print(f"Loaded {len(eval_cases)} samples")

    # Initialize model
    print("Initializing model...")
    model = OpenAIChatModel(
        model_name="gpt-4o-mini",
        generate_kwargs={"temperature": 0.1},
    )

    # Create runner with metrics
    runner = RewardBench2Runner(
        model=model,
        max_workers=8,
        metrics=[
            AccuracyMetric(name="overall_accuracy"),
            TiesAccuracyMetric(name="ties_accuracy"),
        ],
    )

    # Execute evaluation
    print("\nRunning evaluation...")
    report = await runner(eval_cases)

    # Print results
    print("\n" + "=" * 80)
    print("REWARDBENCH2 EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nModel: {report.model_name}")
    print(f"Total samples: {report.total_samples}")
    print(f"Valid samples: {report.valid_samples}")

    for metric_name, metric_result in report.metrics.items():
        print(f"\n{metric_name}: {metric_result.value:.4f}")
        if metric_result.details:
            print(f"  Details: {metric_result.details}")

    # Save results
    import json

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)

    print(f"\nResults saved to: {result_path}")


async def test_rewardbench2_subset():
    """Test RewardBench2 evaluation with specific subset filtering."""
    data_path = "data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet"

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    # Load data
    all_eval_cases = load_rewardbench2_data(data_path, max_samples=50)

    if not all_eval_cases:
        print("No data samples loaded.")
        return

    # Filter for Ties subset
    ties_cases = [
        case
        for case in all_eval_cases
        if case.input.get("subset", "").lower() == "ties"
    ]

    print(f"\nTotal cases: {len(all_eval_cases)}")
    print(f"Ties cases: {len(ties_cases)}")

    if not ties_cases:
        print("No Ties subset cases found in the sample.")
        return

    # Initialize model
    model = OpenAIChatModel(
        model_name="gpt-4o-mini",
        generate_kwargs={"temperature": 0.1},
    )

    # Create runner with Ties-specific metric
    runner = RewardBench2Runner(
        model=model,
        max_workers=4,
        metrics=[TiesAccuracyMetric()],
    )

    # Execute evaluation on Ties subset only
    print("\nEvaluating Ties subset...")
    report = await runner(ties_cases)

    # Print results
    print("\n" + "=" * 80)
    print("TIES SUBSET EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nModel: {report.model_name}")
    print(f"Total samples: {report.total_samples}")
    print(f"Valid samples: {report.valid_samples}")

    for metric_name, metric_result in report.metrics.items():
        print(f"\n{metric_name}: {metric_result.value:.4f}")
        if metric_result.details:
            for key, value in metric_result.details.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")


def main():
    """Main function to run tests."""
    print("Testing RewardBench2 Evaluation\n")

    print("=" * 80)
    print("Test 1: Basic evaluation with all subsets")
    print("=" * 80)
    asyncio.run(test_rewardbench2_basic())

    print("\n\n")
    print("=" * 80)
    print("Test 2: Ties subset evaluation")
    print("=" * 80)
    asyncio.run(test_rewardbench2_subset())


if __name__ == "__main__":
    main()

