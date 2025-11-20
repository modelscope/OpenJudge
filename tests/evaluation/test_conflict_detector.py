# -*- coding: utf-8 -*-
"""
Example usage of ConflictDetector evaluation
"""
import asyncio
import os

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.runner.evaluation import AccuracyMetric, ConflictMetric
from rm_gallery.gallery.evaluation.conflict_detector import (
    PairwiseComparisonRunner,
    evaluate_async,
    load_eval_cases,
)


async def test_conflict_detector_basic():
    """Test basic conflict detector functionality."""
    # Setup paths
    data_path = "data/test_conflict_data.jsonl"
    result_path = "data/results/conflict_detector_test.json"

    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please provide a valid data file to run this test.")
        return

    # Load data (limit to 5 samples for quick testing)
    print(f"Loading data from: {data_path}")
    eval_cases = load_eval_cases(file_path=data_path, max_samples=5)

    if not eval_cases:
        print("No data samples loaded.")
        return

    print(f"Loaded {len(eval_cases)} samples")

    # Initialize model
    model = OpenAIChatModel(
        model_name="gpt-4o-mini",
        generate_kwargs={"temperature": 0.1},
    )

    # Run evaluation using evaluate_async helper
    print("\nRunning evaluation...")
    report = await evaluate_async(
        eval_cases=eval_cases,
        model=model,
        metrics=["accuracy", "conflict_rate"],
    )

    # Print results
    print("\n" + "=" * 80)
    print("CONFLICT DETECTOR EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nModel: {report.get('model_name', 'Unknown')}")
    print(f"Total samples: {report.get('total_samples', 0)}")
    print(f"Valid samples: {report.get('valid_samples', 0)}")

    for metric_name, metric_result in report.get("metrics", {}).items():
        print(f"\n{metric_name}: {metric_result.get('value', 0):.4f}")
        print(f"  Details: {metric_result.get('details', {})}")

    # Save results
    import json

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to: {result_path}")


async def test_conflict_detector_custom_runner():
    """Test using PairwiseComparisonRunner directly with custom metrics."""
    data_path = "data/test_conflict_data.jsonl"

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    eval_cases = load_eval_cases(file_path=data_path, max_samples=3)

    if not eval_cases:
        print("No data samples loaded.")
        return

    # Initialize model
    model = OpenAIChatModel(
        model_name="gpt-4o-mini",
        generate_kwargs={"temperature": 0.1},
    )

    # Create runner with custom metrics
    runner = PairwiseComparisonRunner(
        model=model,
        max_workers=4,
        metrics=[
            AccuracyMetric(name="custom_accuracy"),
            ConflictMetric(name="custom_conflict"),
        ],
    )

    # Run evaluation
    print("\nRunning custom evaluation...")
    report = await runner(eval_cases)

    # Print results
    print("\n" + "=" * 80)
    print("CUSTOM RUNNER RESULTS")
    print("=" * 80)
    print(f"Model: {report.model_name}")
    print(f"Total samples: {report.total_samples}")
    print(f"Valid samples: {report.valid_samples}")

    for metric_name, metric_result in report.metrics.items():
        print(f"\n{metric_name}: {metric_result.value:.4f}")


def main():
    """Main function to run tests."""
    print("Testing Conflict Detector Evaluation\n")

    print("=" * 80)
    print("Test 1: Basic evaluation with evaluate_async")
    print("=" * 80)
    asyncio.run(test_conflict_detector_basic())

    print("\n\n")
    print("=" * 80)
    print("Test 2: Custom runner with custom metrics")
    print("=" * 80)
    asyncio.run(test_conflict_detector_custom_runner())


if __name__ == "__main__":
    main()

