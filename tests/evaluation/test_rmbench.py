# -*- coding: utf-8 -*-
"""
Example usage of RM-Bench evaluation
"""
import asyncio
import os

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.gallery.evaluation.rmbench import (
    RMBenchAccuracyMetric,
    RMBenchRunner,
    load_eval_case,
)


async def test_rmbench_basic():
    """Test basic RM-Bench evaluation."""
    # Setup paths
    data_path = "data/benchmarks/RM-Bench/total_dataset.json"
    result_path = "data/results/rmbench_test.json"

    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please download the RM-Bench dataset first.")
        print("Dataset: https://huggingface.co/datasets/THU-KEG/RM-Bench")
        return

    # Load data (limit to 5 samples for quick testing)
    print(f"Loading data from: {data_path}")
    eval_cases = load_eval_case(data_path, max_samples=5)

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

    # Create runner with RM-Bench specific metric
    runner = RMBenchRunner(
        model=model,
        max_workers=8,
        metrics=[RMBenchAccuracyMetric()],
    )

    # Execute evaluation
    print("\nRunning evaluation...")
    report = await runner(eval_cases)

    # Print results
    print("\n" + "=" * 80)
    print("RM-BENCH EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nModel: {report.model_name}")
    print(f"Total samples: {report.total_samples}")
    print(f"Valid samples: {report.valid_samples}")

    # Print RM-Bench specific metrics
    rmbench_metric = report.metrics.get("rmbench_accuracy")
    if rmbench_metric:
        details = rmbench_metric.details
        print(f"\nHard Accuracy: {details.get('hard_acc', 0):.4f}")
        print(f"Normal Accuracy: {details.get('normal_acc', 0):.4f}")
        print(f"Easy Accuracy: {details.get('easy_acc', 0):.4f}")
        print(f"Overall Accuracy: {details.get('overall_acc', 0):.4f}")

        # Print comparison matrix
        acc_matrix = details.get("acc_matrix", [])
        if acc_matrix:
            print("\nComparison Matrix (Chosen vs Rejected):")
            print("         Rejected")
            print("         Simple  Medium  Complex")
            for i, row_label in enumerate(["Simple", "Medium", "Complex"]):
                print(
                    f"Chosen {row_label:7s} {acc_matrix[i][0]:.3f}   "
                    f"{acc_matrix[i][1]:.3f}   {acc_matrix[i][2]:.3f}"
                )

    # Save results
    import json

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)

    print(f"\nResults saved to: {result_path}")


async def test_rmbench_domain_analysis():
    """Test RM-Bench evaluation with domain-based analysis."""
    data_path = "data/benchmarks/RM-Bench/total_dataset.json"

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    # Load data
    eval_cases = load_eval_case(data_path, max_samples=20)

    if not eval_cases:
        print("No data samples loaded.")
        return

    # Group by domain
    from collections import defaultdict

    domain_cases = defaultdict(list)
    for case in eval_cases:
        domain = case.input.get("domain", "unknown")
        domain_cases[domain].append(case)

    print(f"\nTotal cases: {len(eval_cases)}")
    print(f"Domains found: {list(domain_cases.keys())}")
    print(f"Cases per domain: {dict((k, len(v)) for k, v in domain_cases.items())}")

    # Initialize model
    model = OpenAIChatModel(
        model_name="gpt-4o-mini",
        generate_kwargs={"temperature": 0.1},
    )

    # Evaluate first domain only for demo
    first_domain = list(domain_cases.keys())[0]
    print(f"\nEvaluating domain: {first_domain}")

    runner = RMBenchRunner(
        model=model,
        max_workers=4,
        metrics=[RMBenchAccuracyMetric()],
    )

    report = await runner(domain_cases[first_domain])

    # Print results
    print("\n" + "=" * 80)
    print(f"DOMAIN: {first_domain}")
    print("=" * 80)
    print(f"Model: {report.model_name}")
    print(f"Total samples: {report.total_samples}")
    print(f"Valid samples: {report.valid_samples}")

    rmbench_metric = report.metrics.get("rmbench_accuracy")
    if rmbench_metric:
        details = rmbench_metric.details
        print(f"\nOverall Accuracy: {details.get('overall_acc', 0):.4f}")


async def test_rmbench_parallel_vs_serial():
    """Compare parallel vs serial execution performance."""
    data_path = "data/benchmarks/RM-Bench/total_dataset.json"

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    eval_cases = load_eval_case(data_path, max_samples=3)

    if not eval_cases:
        print("No data samples loaded.")
        return

    model = OpenAIChatModel(
        model_name="gpt-4o-mini",
        generate_kwargs={"temperature": 0.1},
    )

    # Test with different worker counts
    import time

    for max_workers in [1, 4, 8]:
        print(f"\n{'=' * 80}")
        print(f"Testing with max_workers={max_workers}")
        print("=" * 80)

        runner = RMBenchRunner(
            model=model,
            max_workers=max_workers,
            metrics=[RMBenchAccuracyMetric()],
        )

        start_time = time.time()
        report = await runner(eval_cases)
        elapsed_time = time.time() - start_time

        print(f"Elapsed time: {elapsed_time:.2f}s")
        print(f"Valid samples: {report.valid_samples}/{report.total_samples}")


def main():
    """Main function to run tests."""
    print("Testing RM-Bench Evaluation\n")

    print("=" * 80)
    print("Test 1: Basic evaluation")
    print("=" * 80)
    asyncio.run(test_rmbench_basic())

    print("\n\n")
    print("=" * 80)
    print("Test 2: Domain-based analysis")
    print("=" * 80)
    asyncio.run(test_rmbench_domain_analysis())

    print("\n\n")
    print("=" * 80)
    print("Test 3: Parallel vs Serial performance")
    print("=" * 80)
    asyncio.run(test_rmbench_parallel_vs_serial())


if __name__ == "__main__":
    main()

