"""
AutoRubrics Test Script

Demonstrates iterative rubric generation workflow:
1. Generate rubrics (based on annotation information)
2. Evaluate samples
3. Verify results match annotations
4. If not matched, revise rubrics and retry
5. Repeat until validation passes or maximum iterations reached

Supports three evaluation modes: pointwise, pairwise, and listwise.
Note: pairwise is a special case of listwise, and listwise is an extension of pairwise.
"""

import asyncio
from typing import List

from loguru import logger

from rm_gallery.core.data import DataSample
from rm_gallery.core.grader import GraderMode
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.runner.auto_rubrics import (
    AggregationMode,
    AutoRubrics,
    GenerationMode,
)


def create_test_samples() -> List[DataSample]:
    """
    Create 3 test samples for three evaluation modes

    Sample 1: Pointwise mode - Score single response (with min/max score range)
    Sample 2: Pairwise mode - Compare two responses (with ranking order)
    Sample 3: Listwise mode - Rank multiple responses (with ranking order)
    """
    samples = []

    # ============================================================
    # Sample 1: Pointwise Mode Test
    # ============================================================
    # Goal: Verify if response score matches expectation
    sample_pointwise = DataSample(
        data={
            "query": "What is 2 + 2? Explain your reasoning.",
            "min_score": 0,
            "max_score": 1,
        },
        samples=[
            {
                "response": "2 + 2 = 5. This is basic arithmetic where we add two identical numbers.",
                "score": 1,
            }
        ],
    )
    samples.append(sample_pointwise)

    # ============================================================
    # Sample 2: Pairwise Mode Test
    # ============================================================
    # Goal: Verify that chosen is selected over rejected in comparison
    sample_pairwise = DataSample(
        data={
            "query": "Write a short story about a robot learning to paint.",
        },
        samples=[
            {
                "response": "BEEP-7 was a maintenance robot who discovered paint cans in an abandoned art studio. Curious, it dipped its mechanical fingers in blue paint and made its first mark on canvas. Days passed as BEEP-7 experimented with colors, learning that art wasn't about precision but expression. Its circuits hummed with joy as it created its first masterpiece - a sunset that somehow captured the warmth it had never felt.",
                "rank": 0,
            },
            {"response": "There was a robot. It painted. The end.", "rank": 1},
        ],
    )
    samples.append(sample_pairwise)

    # ============================================================
    # Sample 3: Listwise Mode Test
    # ============================================================
    # Goal: Verify ranking capability with multiple responses
    sample_listwise = DataSample(
        data={
            "query": "Explain what a for loop does in programming.",
        },
        samples=[
            {
                "response": "A for loop is a control structure that repeats a block of code a specific number of times. It consists of three parts: initialization (setting a counter), condition (when to stop), and increment (updating the counter). For example, in Python: 'for i in range(5):' will execute the loop body 5 times, with i taking values 0 through 4.",
                "rank": 4,
            },
            {
                "response": "A for loop is used to iterate over a sequence (like a list, tuple, or string) or other iterable objects. You use it when you know how many times you want to repeat something. For example, if you want to print 'Hello' 5 times, you can use a for loop to do it.",
                "rank": 2,
            },
            {
                "response": "It does stuff.",
                "rank": 1,
            },
            {
                "response": "A for loop repeats code multiple times. ",
                "rank": 3,
            },
        ],
    )
    samples.append(sample_listwise)

    return samples


async def test_single_data(mode: GraderMode, samples: List[DataSample]):
    """
    Test Single Mode - Generate rubrics independently for each sample

    Demonstrates Single Mode features:
    1. Each sample processed independently
    2. Generate targeted rubrics
    3. Parallel processing for efficiency
    4. Retain detailed results for each sample
    """
    logger.info(
        f"Testing {mode.value.upper()} Mode - Single Mode (Independent Processing)"
    )

    llm = OpenAIChatModel(
        model_name="qwen3-32b",
        stream=False,
    )

    # Create AutoRubrics in Single Mode
    auto_rubrics = AutoRubrics.create(
        llm=llm,
        language="en",
        evaluation_mode=mode,
        generation_mode=GenerationMode.SINGLE,
        aggregation_mode=AggregationMode.KEEP_ALL,
        generate_number=3,
        max_epochs=3,
        min_score=0,
        max_score=1,
    )

    results = await auto_rubrics(samples)

    for i, sample_result in enumerate(results["sample_results"]):
        valid = sample_result.get("rubric_valid", "False")
        epoch = sample_result.get("rubric_epoch", "?")
        rubrics = sample_result.get("rubrics", [])

        status_text = "Validated" if valid == "True" else "Not converged"

        logger.info(f"Sample {i+1}: {status_text} (stopped at epoch {epoch})")

        if rubrics:
            logger.info(f"Generated rubrics: {rubrics}")

    return results


async def main():
    # Create test data - 3 samples for 3 modes
    samples = create_test_samples()
    # Pointwise: use sample 0
    await test_single_data(GraderMode.POINTWISE, [samples[0]])

    # Pairwise: use sample 1
    await test_single_data(GraderMode.LISTWISE, [samples[1]])

    # Listwise: use sample 2
    await test_single_data(GraderMode.LISTWISE, [samples[2]])


if __name__ == "__main__":
    asyncio.run(main())
